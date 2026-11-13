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


def configure_direct_reset_lanes(env, configurations):
    """Dispatch native resets with each lane's registered configuration.

    This changes only which existing reset parameters are active for a lane's
    native reset call. It writes no state and does not reseed the event stream.
    Native batched RNG order differs from a serial one-lane experiment.
    """
    import copy
    if len(configurations) != env.num_envs or hasattr(env, '_direct_lane_reset_original'):
        raise ValueError('One immutable configuration per lane, installed once')
    configurations = copy.deepcopy(configurations)
    expected = set(configurations[0])
    if not expected or any(set(c) != expected for c in configurations):
        raise ValueError('Each lane requires the same explicit native reset fields')
    original = env._reset_idx
    env._direct_lane_reset_original = original
    def reset(env_ids):
        ids = list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
        saved = {key:copy.deepcopy(getattr(env.cfg.precise_assembly if key.startswith('hole_') else env.cfg.reset,key))
            for key in expected}
        try:
            for lane in ids:
                for key,value in configurations[lane].items():
                    setattr(env.cfg.precise_assembly if key.startswith('hole_') else env.cfg.reset,key,value)
                original(torch.tensor([lane],device=env.device,dtype=torch.long))
        finally:
            for key,value in saved.items():
                setattr(env.cfg.precise_assembly if key.startswith('hole_') else env.cfg.reset,key,value)
    env._reset_idx = reset


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


def record(path, *, episode=None, privileged=True):
    """Actual before-state/target labels and unchanged strict-past radian history."""
    with h5py.File(path) as f:
        key=episode if episode is not None else next(iter(f['episodes'])); g=f['episodes'][key]; m=json.loads(f.attrs['metadata_json'])
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
        q=t('transition/diagnostics_before/joint_position_rad'); qd=t('transition/diagnostics_before/joint_velocity_rad_s')
        robot=robot_rows(q,qd,before,torch.zeros(n,61),time-time[0])
        result=dict(time=time,command=target,robot=position_robot_history(dict(robot=robot,time=time),target),joint_limits=limits)
        if privileged:
            diag={k:t('transition/diagnostics_before/'+k) for k in GEOMETRY_FIELDS}
            result.update(actor=t('transition/expert_actor_input_diagnostic'),geometry=geometry(diag,object_dimensions(m)))
            assert result['actor'].shape==(n,141)
        assert all(torch.isfinite(v).all() for v in result.values())
        assert (target>=limits[:,0]-1e-6).all() and (target<=limits[:,1]+1e-6).all()
        # Both physics substeps must receive the recorded post-slowdown target.
        assert torch.equal(t('transition/actuator_substep_target_rad'),target[:,None,:].expand(-1,2,-1))
        return key,m,result


def dagger_record(entry, *, include_geometry=True):
    """One accepted actual takeover episode, with loss-only expert labels.

    Numeric robot history is built from the complete actual execution, including
    the student prefix. Masked labels never replace issued/applied commands.
    Geometry stays in a separate loss context, not the policy record.
    """
    from .native_data import tactile_values
    from .anchor_sampling import supervision_rows
    if entry.get('trajectory_kind')!='expert_takeover_v1' or entry.get('split','train')!='train':
        raise ValueError('Only training-distribution expert takeovers enter aggregation')
    if sha256(entry['path'])!=entry['source_sha256']:raise ValueError('Takeover source changed')
    key,metadata,result=record(entry['path'],episode=entry['episode'],privileged=False)
    n=len(result['time'])
    if n!=entry['length'] or metadata.get('dataset_split')!='train':raise ValueError('Takeover length/split mismatch')
    actual_q,_=actual_configuration_labels(entry)
    with h5py.File(entry['path']) as stream:
        group=stream['episodes'][key];source=group['transition/dagger'];obs=group['obs']
        load=lambda name:torch.from_numpy(source[name][:])
        mask=load('expert_label_valid');segment=load('segment_id')
        control=load('control_source');clean=load('switch_clean')
        command=load('u_expert');reached=load('q_reached')
        if mask.dtype!=torch.bool or clean.dtype!=torch.bool or any(v.shape!=(n,) for v in (mask,segment,control,clean)):
            raise ValueError('Malformed takeover eligibility fields')
        if (mask & ((control!=1)|~clean)).any() or not mask.any() or mask[0]:
            raise ValueError('Expert labels require actual clean expert suffix after a student prefix')
        if not ((control==0)|(control==1)).all() or (control[1:]<control[:-1]).any():
            raise ValueError('Exactly one student-to-expert authority transition required')
        if command.shape!=(n,29) or reached.shape!=(n,29):raise ValueError('Takeover labels must be canonical29D')
        if not torch.equal(command[mask],result['command'][mask]) or not torch.equal(reached[mask],actual_q[mask]):
            raise ValueError('Expert labels must be actual issued targets and reached configurations')
        previous=load('previous_issued')
        if previous.shape!=(n,29) or not torch.equal(previous[1:],result['command'][:-1]):
            raise ValueError('Past issued targets differ from the actual mixed execution')
        if any(not torch.isfinite(v).all() for v in (command,reached,previous)):
            raise ValueError('Nonfinite takeover labels/history')
        changes=(control[1:]!=control[:-1])|(mask[1:]!=mask[:-1])
        if (changes&(segment[1:]==segment[:-1])).any():raise ValueError('Authority/eligibility switch lacks a new segment')
        terminal=torch.from_numpy(group['terminated'][:]).bool()|torch.from_numpy(group['truncated'][:]).bool()
        if terminal[:-1].any():raise ValueError('Takeover record crosses a reset/terminal boundary')
        values=[torch.from_numpy(obs[k][:]) for k in
            ('contact_substeps_w_N','contact_substeps_local_N','contact_time_s','time_s','contact_valid')]
        touch=tactile_values(*values)
    result.update(u_expert=torch.where(mask[:,None],command,0.),q_reached=torch.where(mask[:,None],reached,0.),
        supervision_mask=mask,control_segment=segment)
    supervision_rows(result)
    context=None
    if include_geometry:
        from .state_image_loss import geometry_loss_context
        context=geometry_loss_context(entry)
    return result,touch,context


def _pool_write(path,value):
    """Durable atomic metadata replacement; callers serialize with pool.lock."""
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value,stream,indent=2,sort_keys=True,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    descriptor=os.open(path.parent,os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def _pool_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _pool_core(data):
    """Metadata-only Core identity: no held-out trajectory is opened."""
    splits={k:list(data.get('normalization_train_ids',data['splits']['train']) if k=='train' else data['splits'][k])
        for k in ('train','validation','test')}
    if [len(splits[k]) for k in splits]!=[166,30,22]:raise ValueError('Persistent pool requires unchanged Core166/validation30/test22')
    all_ids=sum(splits.values(),[])
    if len(set(all_ids))!=len(all_ids):raise ValueError('Core split episode overlap')
    families={k:{data['entries'][i]['family'] for i in values} for k,values in splits.items()}
    if any(families[a]&families[b] for a,b in (('train','validation'),('train','test'),('validation','test'))):
        raise ValueError('Core split family overlap')
    rows={i:{k:data['entries'][i][k] for k in ('family','source_sha256')} for i in all_ids}
    value=dict(splits=splits,entries=rows)
    return dict(identity=_pool_digest(value),**value)


def _pool_id(run_id,group,round_index,source_identity):
    import re
    origin=dict(run_id=run_id,group=group,round_index=round_index,source_identity=source_identity)
    readable='__'.join(re.sub(r'[^A-Za-z0-9_.-]','_',str(v)) for v in (run_id,group,'r'+str(round_index),source_identity))
    return readable+'__'+_pool_digest(origin)[:12],origin


# Complete-policy consumers have explicit supervision identities. G01 and G11
# need the same command/tactile fields in a correction record; their frontends'
# observation modalities differ, so this is a data qualification alias only.
_DAGGER_CONSUMERS={'G11':'u_expert','QG01':'q_reached','G01':'u_expert'}


def _pool_compatibility(entry,consumer):
    receipts=entry.get('compatibility',{})
    source=consumer
    if consumer=='G01' and consumer not in receipts:source='G11'
    receipt=receipts.get(source)
    if (not isinstance(receipt,dict) or not isinstance(receipt.get('compatible'),bool)
            or not isinstance(receipt.get('reasons'),list)):
        raise ValueError('Missing explicit correction compatibility receipt for '+consumer)
    return receipt,source


def _execution_content_sha(record,touch):
    digest=hashlib.sha256()
    for key,value in sorted(dict(record,tactile=touch).items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(key.encode()+b'\0'+str(array.dtype).encode()+b'\0'+str(array.shape).encode()+b'\0')
        digest.update(array.tobytes())
    return digest.hexdigest()


def ingest_dagger_pool(pool_root,collection_root,*,run_id,source_group,round_index,core_data_root,development_ids=()):
    """Idempotently admit a real collector's successes into one shared pool.

    Every started attempt is retained. Complete HDF5 files remain at their
    original paths; both labels share one record. Development is assigned by
    the collection plan before rollout and never promoted into training.
    """
    import fcntl
    from .anchor_sampling import supervision_population
    from .native_data import CAMERAS
    pool=Path(pool_root).resolve();source=Path(collection_root).resolve();pool.mkdir(parents=True,exist_ok=True)
    if not isinstance(round_index,int) or isinstance(round_index,bool) or round_index<1:raise ValueError('Positive source round required')
    if not run_id or source_group not in _DAGGER_CONSUMERS:raise ValueError('Explicit run and supported source group required')
    core=_pool_core(read(Path(core_data_root)/'data.json'))
    plan=read(source/'collection_plan.json');planned={r['identity']:r for r in plan}
    if len(plan)!=len(planned):raise ValueError('Duplicate source collection identity')
    dev={i for i,p in planned.items() if p.get('pool_partition')=='correction_dev' or p.get('cohort')=='correction_dev'}
    if development_ids and set(development_ids)!=dev:raise ValueError('Development identities must be preregistered in collection plan')
    entries=read(source/'correction_entries.json') if (source/'correction_entries.json').exists() else {}
    rows=read(source/'collection_results.json') if (source/'collection_results.json').exists() else []
    by_id={r['identity']:r for r in rows}
    if len(rows)!=len(by_id) or not set(by_id)<=set(planned) or not set(entries)<=set(planned):raise ValueError('Unregistered/duplicate collected episode')
    started={}
    if (source/'attempts.jsonl').exists():
        for line in (source/'attempts.jsonl').read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row['identity'] not in planned:raise ValueError('Unregistered started attempt')
            if row['identity'] in started:raise ValueError('Duplicate started attempt')
            started[row['identity']]=row
    pair=read(source/'initial_pair.json')
    for key in ('frontend_sha256','backend_sha256'):
        if not isinstance(pair.get(key),str) or len(pair[key])!=64:raise ValueError('Collection student pair SHA required')
    receipt=dict(run_id=run_id,source_group=source_group,round_index=round_index,planned=len(plan),
        attempted=len(set(by_id)|set(started)),inserted=[],duplicates=[],excluded=[],data_errors=[],compatibility_exclusions=[],integrity_passed=True,
        source_root=str(source),plan_sha256=sha256(source/'collection_plan.json'))
    with (pool/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        manifest=read(pool/'manifest.json') if (pool/'manifest.json').exists() else dict(
            version='persistent_shared_dagger_pool_v1',revision=0,core=core,entries={},attempts={},
            content_index={},file_index={},ingestions={},raw_payloads_copied=0)
        if manifest['core']!=core:raise ValueError('Persistent pool Core identity changed')
        for source_id in planned:
            if source_id not in by_id and source_id not in started:continue
            identity,origin=_pool_id(run_id,source_group,round_index,source_id)
            row=by_id.get(source_id,{});result=row.get('result',{});takeover=result.get('dagger',{})
            cohort='correction_dev' if source_id in dev else 'train'
            origin.update(source_frontend_sha256=pair['frontend_sha256'],source_backend_sha256=pair['backend_sha256'])
            attempt=dict(pool_identity=identity,pool_source=origin,cohort=cohort,seed=row.get('seed',planned[source_id].get('seed')),
                family=planned[source_id].get('family'),takeover_tick=row.get('takeover_tick',planned[source_id].get('takeover_tick')),
                receipt_sha256=_pool_digest(row or started[source_id]),classification='pending',accepted_pool_identity=None)
            old=manifest['attempts'].get(identity)
            if old and old['receipt_sha256']==attempt['receipt_sha256']:
                if old['cohort']!=cohort or old['pool_source']!=origin:raise ValueError('Existing source/cohort cannot change')
                if old.get('accepted_pool_identity'):receipt['duplicates'].append(dict(identity=identity,existing=old['accepted_pool_identity'],reason='same_source_receipt'))
                elif old.get('integrity_error'):receipt['integrity_passed']=False;receipt['excluded'].append(old);receipt['data_errors'].append(old)
                continue
            if old and old['classification']!='pending':raise ValueError('Completed source receipt changed')
            reason=None
            if not row:reason='pending_no_result'
            elif result.get('legal') is False:reason='illegal_initial_state'
            elif row.get('error') or not result.get('software_complete'):reason='software_or_admission_error'
            elif takeover.get('student_autonomous_success'):reason='student_success_without_expert_labels'
            elif not result.get('success') or not takeover.get('expert_takeover_success'):
                reason='expert_timeout' if result.get('reasons',{}).get('timeout') else 'expert_takeover_failed'
            elif not row.get('accepted') or not row.get('label_contract_checked') or source_id not in entries:reason='collector_not_admitted'
            if reason:
                attempt.update(classification='pending' if not row else 'excluded',reason=reason)
                receipt['excluded'].append(attempt)
            else:
                try:
                    if result.get('reasons',{}).get('max_successes') is not True or not takeover.get('provenance_passed') or not takeover.get('position_executor_continuous'):
                        raise ValueError('Native success/executed takeover provenance missing')
                    if result.get('checkpoint_sha256')!=pair['backend_sha256']:raise ValueError('Collection student backend SHA differs')
                    entry=dict(entries[source_id]);family=entry.get('family')
                    train_families={core['entries'][i]['family'] for i in core['splits']['train']}
                    if family not in train_families or family!=planned[source_id].get('family'):raise ValueError('Correction family outside fixed training families')
                    record_,touch,_=dagger_record(entry,include_geometry=False)
                    exposure=supervision_population({identity:record_})
                    if exposure['supervised_anchors']<1 or exposure['supervised_ticks']!=takeover.get('expert_valid_ticks'):
                        raise ValueError('Recorded supervised population differs or no native anchor')
                    with h5py.File(entry['path']) as stream:
                        metadata=json.loads(stream.attrs['metadata_json']);group=stream['episodes'][entry['episode']]
                        if metadata.get('camera_ids')!=CAMERAS:raise ValueError('Incompatible camera modality/order')
                        if metadata.get('policy_checkpoint_sha256')!=pair['backend_sha256']:raise ValueError('Recorded source student SHA differs')
                        missing_touch=[name for name in ('contact_w_N','finger_position_base_m','finger_rotation_base') if name not in group['obs']]
                        missing_geometry=[name for name in ('object_pose_w','base_pose_w') if name not in group['transition/diagnostics_after']]
                        missing_geometry += [name for name in ('contact_time_after_s','object_contact_force_w') if name not in group['transition']]
                        control=group['transition/dagger/control_source'][:];clean=group['transition/dagger/switch_clean'][:]
                    mask=record_['supervision_mask'];valid_ticks=mask.nonzero().flatten();anchors=[t for t in range(0,len(mask),12) if bool(mask[t])]
                    first=int(valid_ticks[0]);first_anchor=anchors[0]
                    boundary=dict(takeover_tick=int(np.flatnonzero(control==1)[0]),queue_clean_tick=int(np.flatnonzero((control==1)&clean)[0]),
                        first_expert_label_tick=first,first_backend_supervised_tick=first,first_dp_supervised_anchor=first_anchor,
                        valid_expert_ticks_before_first_dp_anchor=int(mask[:first_anchor].sum()),
                        first_24_valid_expert_ticks=valid_ticks[:24].tolist())
                    compatibility={g:dict(compatible=not missing_touch and (g!='QG01' or not missing_geometry),
                        reasons=(['missing_backend_tactile_fields:'+','.join(missing_touch)] if missing_touch else [])+
                            (['missing_loss_geometry_fields:'+','.join(missing_geometry)] if g=='QG01' and missing_geometry else [])) for g in _DAGGER_CONSUMERS}
                    if not missing_geometry:
                        from .state_image_loss import geometry_loss_context
                        geometry_loss_context(entry)
                    content=_execution_content_sha(record_,touch);file_sha=entry['source_sha256']
                    duplicate=manifest['file_index'].get(file_sha) or manifest['content_index'].get(content)
                    if duplicate:
                        existing=manifest['entries'][duplicate]
                        if existing['cohort']!=cohort:raise ValueError('Duplicate execution cannot cross train/development boundary')
                        attempt.update(classification='duplicate',accepted_pool_identity=duplicate,reason='same_file_or_numeric_execution_content')
                        receipt['duplicates'].append(dict(identity=identity,existing=duplicate,reason=attempt['reason']))
                    else:
                        stored=dict(entry,path=str(Path(entry['path']).resolve()),split='train',pool_source=origin,cohort=cohort,
                            execution_content_sha256=content,content_hash_scope='Complete actual numeric robot/command/time/limits, both labels/masks/segments and tactile history; images protected by whole-file SHA and original collector acceptance',
                            compatibility=compatibility,exposure=exposure,supervision_boundary=boundary,
                            acquisition=dict(seed=attempt['seed'],event_seed=row.get('event_seed',planned[source_id].get('event_seed')),
                                takeover_reason=takeover.get('takeover_reason',planned[source_id].get('takeover_reason','registered_tick')),
                                actual_takeover_tick=boundary['takeover_tick'],native_success=True),
                            admission_receipt_sha256=attempt['receipt_sha256'],source_collection_plan_sha256=receipt['plan_sha256'])
                        manifest['entries'][identity]=stored;manifest['file_index'][file_sha]=identity;manifest['content_index'][content]=identity
                        attempt.update(classification='admitted',accepted_pool_identity=identity);receipt['inserted'].append(identity)
                except (ValueError,KeyError,AssertionError,OSError) as error:
                    incompatible=str(error).startswith('Incompatible camera')
                    attempt.update(classification='incompatible' if incompatible else 'data_error',reason=str(error),integrity_error=not incompatible)
                    receipt['integrity_passed'] &= incompatible;receipt['excluded'].append(attempt)
                    receipt['compatibility_exclusions' if incompatible else 'data_errors'].append(attempt)
            manifest['attempts'][identity]=attempt;manifest['revision']+=1
            _pool_write(pool/'manifest.json',manifest)
        key,_=_pool_id(run_id,source_group,round_index,'ingestion')
        receipt.update(pool_entries=len(manifest['entries']),training_entries=sum(e['cohort']=='train' for e in manifest['entries'].values()),
            development_entries=sum(e['cohort']=='correction_dev' for e in manifest['entries'].values()),pool_revision=manifest['revision'])
        manifest['ingestions'][key]=receipt;_pool_write(pool/'manifest.json',manifest)
        _pool_write(pool/'ingestions'/(key+'.json'),receipt)
    return receipt


def read_dagger_snapshot(path):
    value=read(path);identity=value.pop('snapshot_id',None)
    if value.get('version')!='immutable_shared_dagger_snapshot_v1' or identity!=_pool_digest(value):raise ValueError('Snapshot identity changed')
    return dict(value,snapshot_id=identity)


def freeze_dagger_snapshot(pool_root,snapshot_path,*,run_id,round_index):
    """Freeze a shared population once; subsequent ingestion never mutates it."""
    import fcntl
    pool=Path(pool_root).resolve();path=Path(snapshot_path).resolve()
    with (pool/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if path.exists():
            result=read_dagger_snapshot(path)
            if result['run_id']!=run_id or result['round_index']!=round_index or result['pool_root']!=str(pool):raise ValueError('Snapshot registration changed')
            return result
        manifest=read(pool/'manifest.json')
        value=dict(version='immutable_shared_dagger_snapshot_v1',run_id=run_id,round_index=round_index,pool_root=str(pool),
            pool_manifest_sha256=sha256(pool/'manifest.json'),pool_revision=manifest['revision'],core=manifest['core'],entries=manifest['entries'],
            train_ids=[i for i,e in manifest['entries'].items() if e['cohort']=='train'],
            correction_dev_ids=[i for i,e in manifest['entries'].items() if e['cohort']=='correction_dev'])
        value['snapshot_id']=_pool_digest(value);_pool_write(path,value)
        return value


def _shared_pool_tactile(frozen,touch):
    """One immutable derived tactile tensor payload, linked by both consumers."""
    import fcntl
    digest=hashlib.sha256()
    for identity,value in sorted(touch.items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(identity.encode()+b'\0'+str(array.dtype).encode()+str(array.shape).encode()+array.tobytes())
    key=digest.hexdigest();pool=Path(frozen['pool_root']);path=pool/'tactile_payloads'/(key+'.pt')
    receipt=path.with_suffix('.json')
    with (pool/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if receipt.exists():
            prior=read(receipt)
            if prior['content_sha256']!=key or sha256(path)!=prior['file_sha256']:raise ValueError('Shared tactile payload changed')
        else:
            if path.exists():raise ValueError('Incomplete shared tactile transaction; preserve for recovery')
            path.parent.mkdir(parents=True,exist_ok=True);save(path,touch)
            with path.open('rb') as stream:
                import os
                os.fsync(stream.fileno())
            _pool_write(receipt,dict(content_sha256=key,file_sha256=sha256(path),episode_ids=sorted(touch),
                scope='Derived tactile values only; one payload shared by compatible snapshot consumers'))
    return path


def build_dagger_aggregate(source_root,new_entries,output_root,supervision,*,manifest_sha256=None,include_geometry=True,
        snapshot=None,consumer_group=None,include_frontend_tactile=True):
    """Append accepted train takeovers; keep original validation/test and affine.

    Both roots name state_data directories. Selection/native success admission
    belongs to the owner. This layer verifies source identities and labels, never
    reads test trajectories, and never calculates new statistics from aggregation.
    """
    import shutil
    from .native_data import target_supervision
    source=Path(source_root).resolve();out=Path(output_root).resolve()
    frozen=read_dagger_snapshot(snapshot) if snapshot is not None else None
    if frozen and not include_frontend_tactile:raise ValueError('Shared historical pools require their registered tactile artifacts')
    if source==out:raise ValueError('Aggregate must be a new independent dataset')
    kind=target_supervision(supervision);old=read(source/'data.json')
    if target_supervision(old.get('supervision','u_expert'))!=kind:raise ValueError('Aggregate target semantics changed')
    exclusions={};compatibility_aliases={}
    if frozen:
        if consumer_group not in _DAGGER_CONSUMERS or new_entries:raise ValueError('Snapshot aggregate requires explicit consumer and no extra mutable entries')
        if _DAGGER_CONSUMERS[consumer_group]!=kind:raise ValueError('Snapshot consumer supervision semantics changed')
        if frozen['core']!=_pool_core(old):raise ValueError('Snapshot and source Core identities differ')
        selected={}
        for identity,entry in frozen['entries'].items():
            compatibility,qualification=_pool_compatibility(entry,consumer_group)
            if qualification!=consumer_group:compatibility_aliases[identity]=qualification
            if compatibility['compatible']:selected[identity]=entry
            else:exclusions[identity]=compatibility['reasons']
        new_entries=selected
        request=dict(snapshot_id=frozen['snapshot_id'],consumer_group=consumer_group,supervision=kind,
            source_data_sha256=sha256(source/'data.json'),source_statistics_sha256=sha256(source/'statistics.pt'),include_geometry=include_geometry)
        if (out/'aggregation_complete.json').exists():
            completed=read(out/'aggregation_complete.json')
            if completed['request']!=request or any(not (out/name).is_file() or sha256(out/name)!=digest for name,digest in completed['artifacts'].items()):
                raise ValueError('Completed immutable aggregate identity changed')
            return read(out/'data.json')
        if out.exists() and any(out.iterdir()):raise ValueError('Partial aggregate has no completion receipt; preserve for transaction recovery')
    elif (out/'data.json').exists():raise ValueError('Aggregate must be a new independent dataset')
    raw=torch.load(source/'numeric.pt',map_location='cpu',weights_only=False)
    if raw['data_sha256']!=sha256(source/'data.json'):raise ValueError('Source numeric cache identity differs')
    touched=torch.load(source/'tactile.pt',map_location='cpu',weights_only=False) if include_frontend_tactile else None
    base_splits=frozen['core']['splits'] if frozen else old['splits']
    ids=base_splits['train']+base_splits['validation']
    records={};touch={};contexts={}
    old_context=torch.load(source/'loss_context.pt',map_location='cpu',weights_only=False) if include_geometry and (source/'loss_context.pt').exists() else {}
    for identity in ids:
        row={k:raw['records'][identity][k] for k in ('robot','command','time','joint_limits')}
        # Read q labels only for train/validation when the historical command
        # cache did not yet carry them. No source episode in test is opened.
        row['q_reached']=(raw['records'][identity]['q_reached'] if 'q_reached' in raw['records'][identity]
            else actual_configuration_labels(old['entries'][identity])[0])
        n=len(row['time']);row.update(u_expert=row['command'],supervision_mask=torch.ones(n,dtype=torch.bool),
            control_segment=torch.zeros(n,dtype=torch.long))
        # Aggregating an already aggregated pool retains its real masks/labels.
        for key in ('u_expert','supervision_mask','control_segment'):
            if key in raw['records'][identity]:row[key]=raw['records'][identity][key]
        records[identity]=row
        if include_frontend_tactile:touch[identity]=touched[identity]
        if include_geometry:
            if identity in old_context:contexts[identity]=old_context[identity]
            else:
                from .state_image_loss import geometry_loss_context
                contexts[identity]=geometry_loss_context(old['entries'][identity])
    entries=({i:old['entries'][i] for values in base_splits.values() for i in values} if frozen else dict(old['entries']))
    splits={k:list(v) for k,v in base_splits.items()};families=dict(old.get('families',{}))
    if frozen:splits['correction_dev']=[]
    for identity,entry in new_entries.items():
        if identity in entries:raise ValueError('Duplicate aggregate episode: '+identity)
        family=entry.get('family',identity)
        if families.get(family,'train')!='train':raise ValueError('Takeover family leaks a held-out split')
        records[identity],touch[identity],context=dagger_record(entry,include_geometry=include_geometry)
        if include_geometry:contexts[identity]=context
        entries[identity]=dict(entry);splits['correction_dev' if frozen and entry['cohort']=='correction_dev' else 'train'].append(identity);families[family]='train'
    train_ids=old.get('normalization_train_ids',old['splits']['train'])
    if not set(train_ids)<=set(old['splits']['train']):raise ValueError('Source normalizer population is not original training data')
    data=dict(old,entries=entries,splits=splits,families=families,visual_pair=True,supervision=kind,
        normalization_identity=old.get('normalization_identity',kind+'_train166_affine_v1'),
        normalization_train_ids=train_ids,normalization_scope='unchanged original training affine; no aggregate refit',
        aggregation=dict(version='expert_takeover_aggregate_v1',selection='successful_expert_executed_corrections',
            student_prefix_supervised=False,new_training_ids=list(new_entries)),
        source_data_sha256=sha256(source/'data.json'),
        policy_inputs=['robot178','RGB_DINO_patches','deterministic_depth_grid'],privileged_inputs=False,
        source_statistics_sha256=sha256(source/'statistics.pt'))
    if manifest_sha256 is not None:data['manifest_sha256']=manifest_sha256
    if frozen:
        current=[i for i,e in new_entries.items() if e['cohort']=='train' and e['pool_source']['run_id']==frozen['run_id'] and e['pool_source']['round_index']==frozen['round_index']]
        historical=[i for i,e in new_entries.items() if e['cohort']=='train' and i not in current]
        data.update(snapshot_id=frozen['snapshot_id'],snapshot_path=str(Path(snapshot).resolve()),round_index=frozen['round_index'],
            sampling_strata=dict(core=list(base_splits['train']),historical_correction=historical,current_correction=current),
            snapshot_exclusions=exclusions,consumer_group=consumer_group)
        data['aggregation'].update(snapshot_id=frozen['snapshot_id'],pool_manifest_sha256=frozen['pool_manifest_sha256'],
            new_training_ids=current,all_correction_training_ids=historical+current,correction_dev_ids=list(splits['correction_dev']),
            shared_pool=True,source_group_contributions={g:sum(e['pool_source']['group']==g for e in new_entries.values())
                for g in ('G11','QG01')+tuple(sorted({e['pool_source']['group'] for e in new_entries.values()}-{'G11','QG01'}))})
        if compatibility_aliases:data['aggregation']['legacy_compatibility_qualifications']=compatibility_aliases
    out.mkdir(parents=True,exist_ok=True)
    for name in ('statistics.pt','tactile_statistics.pt') if include_frontend_tactile else ('statistics.pt',):
        shutil.copyfile(source/name,out/name)
        if sha256(source/name)!=sha256(out/name):raise ValueError('Fixed aggregate affine copy changed')
    atomic_json(out/'data.json',data)
    save(out/'numeric.pt',dict(contract=raw['contract'],data_sha256=sha256(out/'data.json'),records=records))
    if frozen:(out/'tactile.pt').symlink_to(_shared_pool_tactile(frozen,touch))
    elif include_frontend_tactile:save(out/'tactile.pt',touch)
    if include_geometry:save(out/'loss_context.pt',contexts)
    if frozen:
        names=['data.json','numeric.pt','tactile.pt','statistics.pt','tactile_statistics.pt']+(['loss_context.pt'] if include_geometry else [])
        _pool_write(out/'aggregation_complete.json',dict(complete=True,request=request,artifacts={name:sha256(out/name) for name in names},test_trajectories_opened=0))
    return data


def actual_configuration_labels(entry):
    """Read q(t+1) directly, with u(t) kept distinct and reset boundaries checked.

    This is loss-only supervision. Neither future configuration nor future object
    geometry is an observation input. A terminal row is the recorded pre-reset
    after-state; its missing post image is never replaced by a reset image.
    """
    with h5py.File(entry['path']) as f:
        g=f['episodes'][entry['episode']];n=entry['length']
        q=torch.from_numpy(g['transition/diagnostics_after/joint_position_rad'][:])
        before=torch.from_numpy(g['transition/diagnostics_before/joint_position_rad'][:])
        u=torch.from_numpy(g['transition/target_after_rad'][:])
        t=torch.from_numpy(g['obs/time_s'][:]);after=torch.from_numpy(g['transition/time_after_s'][:])
        time_before=torch.from_numpy(g['transition/time_before_s'][:])
        if q.shape!=(n,29) or u.shape!=q.shape or not torch.isfinite(q).all():raise ValueError('Invalid actual joint labels')
        if not torch.equal(q[:-1],before[1:]):raise ValueError('Actual configuration crosses reset or loses after/before alignment')
        if not torch.allclose(after-t,torch.full_like(t,1/60),atol=1e-8,rtol=0):raise ValueError('After-state clock differs')
        if not torch.equal(time_before,t) or not torch.allclose(after[:-1],t[1:],atol=1e-8,rtol=0):
            raise ValueError('Actual configuration observation/transition clocks lose causal alignment')
        if not np.array_equal(g['frames/post_index'][:],np.r_[np.arange(1,n),-1]):raise ValueError('Terminal image/reset boundary differs')
        if not bool(g.attrs['complete']):raise ValueError('Incomplete source trajectory')
        if not torch.equal(torch.from_numpy(g['obs/step_id'][:]),torch.arange(n)):raise ValueError('Native tick sequence differs')
        report=dict(ticks=n,after_before_exact=True,terminal_after_preserved=True,terminal_post_image=-1,
            command_reached_max_abs_rad=float((u-q).abs().max()),
            command_reached_arm_rmse_rad=float((u[:,:7]-q[:,:7]).square().mean().sqrt()),
            command_reached_hand_rmse_rad=float((u[:,7:]-q[:,7:]).square().mean().sqrt()),
            q_reached_field='transition/diagnostics_after/joint_position_rad',u_expert_field='transition/target_after_rad')
        return q,report


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
