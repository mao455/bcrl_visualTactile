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


def direct_configuration(domain, seed):
    """One joint relative-pose draw; native reset remains the only state writer.

    A separately seeded generator makes a family reproducible without consuming
    another lane's RNG. Native observation/action/force draws are unchanged.
    """
    rng=np.random.default_rng(seed)
    center=domain['center']
    def bounds(key, default=None, size=2):
        value=np.asarray(domain[key] if key in domain else default,dtype=float)
        if value.shape!=(size,) or not np.isfinite(value).all():
            raise ValueError('Missing or invalid sampling range: '+key)
        return value
    xy_width=bounds('object_xy_halfwidth');relative_width=bounds('relative_xy_halfwidth')
    table_range=bounds('table_z_m',[.376,.384])
    height_range=bounds('object_height_above_table_origin_m',[.244,.256])
    rotation=float(domain['rotation_radius_degrees'])
    lerp=float(domain['native_joint_lerp']);velocity=float(domain.get('joint_velocity_halfwidth',.15))
    if (np.any(xy_width<0) or np.any(relative_width<0) or table_range[0]>table_range[1]
        or height_range[0]>height_range[1] or not np.isfinite([rotation,lerp,velocity]).all()
        or not 0<=rotation<=180 or not 0<=lerp<=1 or velocity<0):
        raise ValueError('Invalid direct reset sampling bounds')
    table=float(rng.uniform(*table_range))
    xy=np.asarray(center['object_xyz'][:2])+rng.uniform(-xy_width,xy_width)
    relative_center=np.asarray(center['relative_xyz'][:2])
    # Sample directly inside the intersection with original hole bounds.
    lower=np.maximum(relative_center-relative_width,[-.1875,-.1]-xy)
    upper=np.minimum(relative_center+relative_width,[.1875,.1]-xy)
    if np.any(lower>upper) or not np.all(np.abs(xy)<=.1):
        raise ValueError('Configured domain exceeds native initial bounds')
    relative=rng.uniform(lower,upper)
    hole=xy+relative
    axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
    theta=np.deg2rad(rotation)*rng.random()**(1/3)
    a=np.r_[np.cos(theta/2),axis*np.sin(theta/2)];b=np.asarray(center['quaternion'])
    q=np.r_[a[0]*b[0]-a[1:]@b[1:],a[0]*b[1:]+b[0]*a[1:]+np.cross(a[1:],b[1:])]
    assert np.all(np.abs(xy)<=.1) and abs(hole[0])<=.1875 and abs(hole[1])<=.1
    return dict(fixed_start_pose=[*xy,table+float(rng.uniform(*height_range)),*q],
        table_reset_z=table,table_reset_z_range=0.,
        reset_dof_pos_random_interval_arm=lerp,reset_dof_pos_random_interval_fingers=lerp,
        reset_dof_vel_random_interval=velocity,hole_x_range=[float(hole[0])]*2,hole_y_range=[float(hole[1])]*2)


def configure_direct_reset(env, configuration):
    """Opt-in per-family native reset parameters, used identically by all speeds."""
    if env.num_envs!=1:raise ValueError('Direct reset v1 is single-lane; no shared per-lane configuration')
    for key,value in configuration.items():
        setattr(env.cfg.precise_assembly if key.startswith('hole_') else env.cfg.reset,key,value)


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


def admission_conflict(first, second, first_legal, second_legal):
    """Detect contradictory query results; never infer that the new hits are real."""
    keys=('joint_position_rad','joint_velocity_rad_s','object_pose_w','hole_pose_w','table_pose_w',
        'palm_body_pose_w','finger_position_w','finger_quat_wxyz')
    errors={k:float(np.max(np.abs(np.asarray(first[k])-np.asarray(second[k])))) for k in keys}
    return dict(conflict=bool(first_legal!=second_legal and max(errors.values())<=1e-6),
        observable_max_abs=errors,cause='unresolved simulator scene-query/reset consistency')


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


def build_cache(root, *, partial=False, reuse_source=None):
    root=Path(root); manifest=read(root/'manifest.json')
    current=(root/'configuration.json').exists() and read(root/'configuration.json').get('version')=='tight_state_foundation_v1'
    if current:
        from .tight_gate import validate_formal_manifest
        if partial:raise ValueError('Current formal dataset cannot be frozen from a partial subset')
        validate_formal_manifest(manifest,read(root/'configuration.json'))
    if manifest.get('quarantined') and not partial: raise ValueError('Quarantined collision admission; data cannot be marked ready')
    if not partial and not manifest['collection_finished']: raise ValueError('Collection is not finished')
    out=root/('smoke_data' if partial else 'state_data'); out.mkdir(exist_ok=True)
    reused={};reused_entries={}
    if reuse_source is not None:
        previous=Path(reuse_source)/'state_data';old_data=read(previous/'data.json')
        old_cache=torch.load(previous/'numeric.pt',map_location='cpu',weights_only=False)
        if old_cache['data_sha256']!=sha256(previous/'data.json') or old_cache['contract']!=contract(24,True):
            raise ValueError('Accepted reusable numeric cache identity differs')
        reused=old_cache['records'];reused_entries=old_data['entries']
    records={}; entries={}; splits={k:[] for k in ('train','validation','test')}; families={}
    for e in manifest['episodes']:
        if partial and e['split']!='train': continue
        if e['id'] in reused:
            previous=reused_entries[e['id']]
            assert previous['source_sha256']==e['sha256'] and previous['path']==e['path']
            assert Path(e['path']).is_file() and Path(e['path']).stat().st_size==e['size']
            identity=e['id'];records[identity]=reused[identity];entries[identity]=previous
            splits[e['split']].append(identity)
            assert families.setdefault(e['family'],e['split'])==e['split']
            continue
        if sha256(e['path'])!=e['sha256']:raise ValueError('Accepted capture source hash changed: '+e['id'])
        if current:
            from .camera_validity import inspect_capture
            from .tight_gate import numeric_admission,require
            require(inspect_capture(e['path'])['passed'],'Actual RGB-D content failed despite manifest flags')
            key,meta,rec,actual=numeric_admission(e['path'])
            require(actual['success'],'Formal slow capture did not complete native task')
        else:key,meta,rec=record(e['path'])
        identity=e['id']; records[identity]=rec
        assert meta['initial_family']==e['family'] and meta['dataset_split']==e['split'],'Capture family/split provenance mismatch'
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
