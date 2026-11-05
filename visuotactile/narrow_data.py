"""Initial-state selection and StateDP data adaptation; no simulator imports."""
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from .compact import sha256
from .native_data import geometry, object_dimensions, robot_rows, limits_affine
from .position_control import contract, position_robot_history
from .checkpointing import save
from .training_resources import atomic_json

GEOMETRY_FIELDS = ('base_pose_w','object_pose_w','goal_pose_w','hole_pose_w','table_pose_w',
    'palm_body_pose_w','object_velocity_w_m_s_rad_s','goal_velocity_w_m_s_rad_s',
    'finger_position_w','finger_quat_wxyz')


def read(path):
    return json.loads(Path(path).read_text())


def initial_row(identity, diagnostics, origin):
    d = {k: np.asarray(v) for k,v in diagnostics.items()}
    origin = np.asarray(origin)
    obj = d['object_pose_w'][:3]-origin
    hole = d['hole_pose_w'][:3]-origin
    quat = d['object_pose_w'][3:]/np.linalg.norm(d['object_pose_w'][3:])
    if quat[0] < 0: quat = -quat
    q = d['joint_position_rad']; qd = d['joint_velocity_rad_s']
    fingerprint = np.concatenate((obj,hole,quat,q,qd,d['table_pose_w'][:3]-origin)).astype('<f8')
    return dict(identity=identity,object_xyz=obj.tolist(),hole_xyz=hole.tolist(),relative_xyz=(hole-obj).tolist(),
        quaternion=quat.tolist(),arm=q[:7].tolist(),hand=q[7:].tolist(),velocity=qd.tolist(),
        object_velocity=d['object_velocity_w_m_s_rad_s'].tolist(),
        table_z=float(d['table_pose_w'][2]-origin[2]),
        fingerprint=hashlib.sha256(fingerprint.tobytes()).hexdigest())


def angle(a,b):
    return float(np.degrees(2*np.arccos(np.clip(abs(np.dot(a,b)),0,1))))


def inside(row, domain):
    c = domain['center']
    return bool(np.all(np.abs(np.array(row['object_xyz'][:2])-c['object_xyz'][:2]) <= domain['object_xy_halfwidth'])
        and np.all(np.abs(np.array(row['relative_xyz'][:2])-c['relative_xyz'][:2]) <= domain['relative_xy_halfwidth'])
        and angle(row['quaternion'],c['quaternion']) <= domain['rotation_radius_degrees']
        and np.sqrt(np.mean((np.array(row['arm'])-c['arm'])**2)) <= domain['arm_rms_radius_rad'])


def near(a,b):
    return (np.linalg.norm(np.array(a['object_xyz'])-b['object_xyz']) < .01
        and np.linalg.norm(np.array(a['hole_xyz'])-b['hole_xyz']) < .01
        and angle(a['quaternion'],b['quaternion']) < 5
        and np.max(np.abs(np.array(a['arm'])-b['arm'])) < .02)


def quantiles(rows, center):
    out = {k: np.quantile([r[k] for r in rows],[0,.1,.5,.9,1],axis=0).tolist()
        for k in ('object_xyz','hole_xyz','relative_xyz','arm','hand','velocity','object_velocity','table_z')}
    out['rotation_to_center_deg'] = np.quantile([angle(r['quaternion'],center['quaternion']) for r in rows],[0,.1,.5,.9,1]).tolist()
    return out


def select(source):
    data = read(Path(source)/'data.json'); rows = []
    splits = {i:k for k,v in data['splits'].items() for i in v}
    for identity,e in sorted(data['entries'].items()):
        with h5py.File(e['path']) as f:
            m=json.loads(f.attrs['metadata_json']); g=f['episodes'][e['episode']]
            d={k:v[0] for k,v in g['transition/diagnostics_before'].items()}
            row=initial_row(identity,d,m['env_origins_w'][0]); row['split']=splits[identity]
            row['force_probability']=float(g['transition/external_force_probability'][0])
            row['torque_probability']=float(g['transition/external_torque_probability'][0])
            rows.append(row)
    # Fixed physically interpretable widths, then densest observed center. No
    # lengths, rewards, student results or speed outcomes enter this selection.
    domains=[dict(center=r,object_xy_halfwidth=[.08,.08],relative_xy_halfwidth=[.12,.10],
        rotation_radius_degrees=75.,arm_rms_radius_rad=.22) for r in rows]
    counts=[sum(inside(r,d) for r in rows) for d in domains]
    domain=domains[int(np.argmax(counts))]
    selected=[r for r in rows if inside(r,domain)]
    for i,a in enumerate(selected):
        for b in selected[:i]:
            if near(a,b): raise ValueError('Near duplicate requires family merge before split: '+a['identity'])
    return domain,selected,dict(original_families=len(rows),selected_families=len(selected),
        quantile_order=[0,.1,.5,.9,1],before=quantiles(rows,domain['center']),after=quantiles(selected,domain['center']),
        selection='densest observed center under fixed joint/relative-pose bounds, initial-state only',
        rotation='SO(3) geodesic with q/-q equivalence; asymmetric L-peg, no extra symmetry assumed',
        original_splits_preserved=True,near_duplicate_rule='object/hole <1cm, rotation <5deg, all arm joints <.02rad',
        configured_dr=m['config']['domain_randomization'],configured_reset=m['config']['reset'],
        source_data_sha256=sha256(Path(source)/'data.json'))


def record(path):
    """Actual before-state/target labels and unchanged strict-past radian history."""
    with h5py.File(path) as f:
        key=next(iter(f['episodes'])); g=f['episodes'][key]; m=json.loads(f.attrs['metadata_json'])
        t=lambda p: torch.from_numpy(g[p][:])
        time=t('obs/time_s'); target=t('transition/target_after_rad'); before=t('transition/target_before_rad')
        n=len(time); limits=torch.tensor(m['joint_limits_rad'][0])
        assert n and bool(g.attrs['complete']) and torch.equal(t('obs/step_id'),torch.arange(n))
        assert torch.equal(before,t('obs/actuator_target_rad'))
        assert torch.equal(before[1:],target[:-1])
        assert torch.allclose(t('transition/time_after_s')-time,torch.full_like(time,1/60),atol=1e-8,rtol=0)
        assert abs(m['physics_dt']-1/120)<1e-8 and abs(m['control_dt']-1/60)<1e-8
        assert g['frames/rgb_u8'].shape==(n,3,3,252,448) and g['frames/depth_m'].shape==(n,3,1,252,448)
        assert torch.allclose(t('obs/camera_time_s'),time[:,None].expand(-1,3),atol=1e-8,rtol=0)
        assert t('obs/camera_valid').all()
        assert not bool(t('obs/contact_valid')[0]) and t('obs/contact_valid')[1:].all()
        assert g['transition/contact_substeps_after_w_N'].shape[:3]==(n,2,5)
        assert np.array_equal(g['frames/post_index'][:],np.r_[np.arange(1,n),-1])
        diag={k:t('transition/diagnostics_before/'+k) for k in GEOMETRY_FIELDS}
        q=t('transition/diagnostics_before/joint_position_rad'); qd=t('transition/diagnostics_before/joint_velocity_rad_s')
        robot=robot_rows(q,qd,before,torch.zeros(n,61),time-time[0])
        result=dict(time=time,command=target,robot=position_robot_history(dict(robot=robot,time=time),target),
            actor=t('transition/expert_actor_input_diagnostic'),geometry=geometry(diag,object_dimensions(m)),joint_limits=limits)
        assert result['actor'].shape==(n,141)
        assert all(torch.isfinite(v).all() for v in result.values())
        assert (target>=limits[:,0]-1e-6).all() and (target<=limits[:,1]+1e-6).all()
        # Both physics substeps must receive the recorded post-slowdown target.
        assert torch.equal(t('transition/actuator_substep_target_rad'),target[:,None,:].expand(-1,2,-1))
        return key,m,result


def build_cache(root, *, partial=False):
    root=Path(root); manifest=read(root/'manifest.json')
    if not partial and not manifest['collection_finished']: raise ValueError('Collection is not finished')
    out=root/('smoke_data' if partial else 'state_data'); out.mkdir(exist_ok=True)
    records={}; entries={}; splits={k:[] for k in ('train','validation','test')}; families={}
    for e in manifest['episodes']:
        if partial and e['split']!='train': continue
        key,meta,rec=record(e['path']); identity=e['id']; records[identity]=rec
        entries[identity]=dict(path=e['path'],episode=key,length=len(rec['time']),family=e['family'],source_sha256=e['sha256'])
        splits[e['split']].append(identity)
        old=families.setdefault(e['family'],e['split'])
        assert old==e['split'],'Family split leakage'
    train=splits['train']
    if not train: raise ValueError('No accepted training data')
    stats={}
    for k in ('robot','geometry'):
        # Preserve the approved statistics sampling cadence: robot every tick,
        # geometry at native replan anchors, although the State GRU sees 60Hz.
        x=torch.cat([records[i][k][::12] if k=='geometry' else records[i][k] for i in train])
        stats[k+'_mean']=x.mean(0);stats[k+'_std']=x.std(0).clamp_min(1e-4)
    x=torch.cat([records[i]['command'] for i in train]);stats['action_scale'],stats['action_offset']=limits_affine(x)
    assert float((((x*stats['action_scale']+stats['action_offset'])-stats['action_offset'])/stats['action_scale']-x).abs().max())<1e-6
    atomic_json(out/'data.json',dict(contract=contract(24,True),splits=splits,entries=entries,families=families,
        manifest_sha256=sha256(root/'manifest.json'),normalization_scope='new successful train families only'))
    save(out/'numeric.pt',dict(contract=contract(24,True),data_sha256=sha256(out/'data.json'),records=records))
    save(out/'statistics.pt',stats)
    atomic_json(out/'validation.json',dict(passed=True,episodes=len(entries),split_counts={k:len(v) for k,v in splits.items()},
        train_statistics_ids=train,statistics_sha256=sha256(out/'statistics.pt'),dino_cache_created=False,
        labels='transition/target_after_rad',continuous_history=True,partial=partial))
    return out
