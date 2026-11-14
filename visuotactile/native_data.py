"""Current immutable train320/validation40/test40 data and causal image cache reader."""
import json
import xml.etree.ElementTree as element_tree
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as functional
from .compact import sha256
from .models.native_vision import DINO_REVISION
CAMERAS=["fixed_high","fixed_side","wrist_rgbd"]
MANIFEST=Path("outputs/visuotactile/data/close_v4/manifest.json")
CACHE_SOURCE = Path('outputs/visuotactile/run/long_horizon_20261030_v2')
ROBOT_DIM = 178
GEOMETRY_DIM = 85
GROUPS = ('S-NativeDP', 'I-OfficialDP', 'I-HistoryDP')
CONTRACT = dict(version=1, action='native_command_clipped_pre_delay', horizon=64,
    commit=12, native_history=13, visual_history=3, native_hz=60, physics_hz=120,
    observation='native_robot178_geometry85_rgbd_v1', prediction_type='sample',
    loss='B', train_noise_steps=100, inference_steps=16,
    history_training='current_weight_causal_prefix_no_grad_then_96_ticks',
    history_source='past_expert_executed_teacher_forcing; student_executed_at_deployment')

def depth_grid(depth, valid):
    depth = torch.as_tensor(depth).float()
    valid = torch.as_tensor(valid).bool()
    leading = depth.shape[:-3]
    metric = torch.where(valid,depth,0.).reshape(-1,1,252,448)
    mask = valid.float().reshape_as(metric)
    coverage = functional.avg_pool2d(mask,14,14)
    average = functional.avg_pool2d(metric,14,14)/coverage.clamp_min(1e-6)
    return torch.cat((average/2.,coverage),1).reshape(*leading,2,18,32).half()


def proprio_radians(proprio, limits):
    result = proprio.clone()
    result[..., :29] = limits[:,0]+.5*(proprio[..., :29]+1)*(limits[:,1]-limits[:,0])
    return result
def multiply_quaternion(left, right):
    scalar = left[..., :1]*right[..., :1]-(left[..., 1:]*right[..., 1:]).sum(-1, keepdim=True)
    vector = left[..., :1]*right[..., 1:]+right[..., :1]*left[..., 1:]
    return torch.cat((scalar, vector+torch.cross(left[..., 1:], right[..., 1:], dim=-1)), -1)


def rotate(quaternion, vector):
    imaginary = quaternion[..., 1:].expand_as(vector)
    return vector+2*torch.cross(imaginary, torch.cross(imaginary, vector, dim=-1)
                              +quaternion[..., :1]*vector, dim=-1)


def geometry(diagnostics, dimensions):
    base = diagnostics['base_pose_w']
    inverse = base[..., 3:].clone()
    inverse[..., 1:] *= -1
    parts = []
    for name in ('object_pose_w', 'goal_pose_w', 'hole_pose_w', 'table_pose_w', 'palm_body_pose_w'):
        pose = diagnostics[name]
        orientation = multiply_quaternion(inverse, pose[..., 3:])
        orientation = torch.where(orientation[..., :1] < 0, -orientation, orientation)
        parts.extend((rotate(inverse, pose[..., :3]-base[..., :3]), orientation))
    for name in ('object_velocity_w_m_s_rad_s', 'goal_velocity_w_m_s_rad_s'):
        velocity = diagnostics[name]
        parts.extend((rotate(inverse, velocity[..., :3]), rotate(inverse, velocity[..., 3:])))
    position = rotate(inverse[..., None, :], diagnostics['finger_position_w']-base[..., None, :3])
    orientation = multiply_quaternion(inverse[..., None, :], diagnostics['finger_quat_wxyz'])
    orientation = torch.where(orientation[..., :1] < 0, -orientation, orientation)
    parts.extend((position.flatten(-2), orientation.flatten(-2), dimensions.expand(*base.shape[:-1], 3)))
    result = torch.cat(parts, -1).float()
    if result.shape[-1] != GEOMETRY_DIM or not torch.isfinite(result).all():
        raise ValueError('Invalid explicitly allowlisted state geometry')
    return result


def object_dimensions(metadata):
    path = Path(metadata['config']['assets']['object_urdf'])
    lower, upper = [], []
    for collision in element_tree.parse(path).getroot().findall('.//collision'):
        box, origin = collision.find('geometry/box'), collision.find('origin')
        if box is None:
            raise ValueError('Object dimensions require the audited box asset')
        position = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        angles = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
        if np.any(angles):
            raise ValueError('Rotated collider dimensions require explicit implementation')
        half = np.fromstring(box.get('size'), sep=' ')/2
        lower.append(position-half)
        upper.append(position+half)
    return torch.tensor(np.max(upper, axis=0)-np.min(lower, axis=0), dtype=torch.float32)


def robot_rows(position, velocity, target, action, elapsed):
    result = torch.cat((position, velocity, target, target-position, action,
                        elapsed[..., None].float()), -1).float()
    if result.shape[-1] != ROBOT_DIM or not torch.isfinite(result).all():
        raise ValueError('Invalid native robot observation')
    return result


def labels(record, anchors):
    from .anchor_sampling import labels as anchor_labels
    return anchor_labels(record,anchors,64)


def target_supervision(value):
    """Semantic targets, independent of experiment/group names."""
    aliases={'command':'u_expert','configuration':'q_reached','u_expert':'u_expert','q_reached':'q_reached'}
    if value not in aliases:raise ValueError('Explicit command/configuration supervision required')
    return aliases[value]


def limits_affine(values):
    minimum, maximum = values.amin(0), values.amax(0)
    span = maximum-minimum
    constant = span < 1e-4
    scale = 2/torch.where(constant, torch.full_like(span, 2.), span)
    offset = -1-scale*minimum
    offset = torch.where(constant, -minimum, offset)
    return scale, offset


class Episode:
    def __init__(self, root, identity, group):
        self.group = group
        root = Path(root)
        info = json.loads((root/'data.json').read_text())
        index = info['entries'][identity]
        self.visual_stride = info.get('visual_stride',12)
        if self.visual_stride not in (1,12):raise ValueError('Unknown visual sampling stride')
        if info.get('visual_pair'):
            shard = torch.load(root/'perception'/(identity+'.pt'),map_location='cpu',weights_only=False,mmap=True)
            self.record = {k:shard[k] for k in ('camera_time','camera_valid')}
            self.perception = {k:shard[k] for k in ('patches','depth')}
            self.index = index
            return
        self.record = torch.load(root/'numeric'/(identity+'.pt'), map_location='cpu', weights_only=False)
        self.index = index
        self.perception = None
        if group == 'I-HistoryDP':
            old = torch.load(index['perception'], map_location='cpu', weights_only=False)
            self.perception = {name: old[name] for name in ('patches', 'depth')}
            perception_index = json.loads((root/'native_perception.json').read_text())
            overlay = perception_index['entries'][identity].get('tail')
            if overlay:
                self.perception['patches'][-1] = torch.load(overlay['path'], map_location='cpu', weights_only=False)

    def __len__(self):
        return len(self.record['time'])

    def frame(self, tick, device):
        if tick % self.visual_stride:raise ValueError('Frame requested outside cache clock')
        ordinal = tick//self.visual_stride
        result = dict(time=self.record['camera_time'][ordinal].to(device),
                      valid=self.record['camera_valid'][ordinal].to(device))
        if self.group == 'S-NativeDP':
            result['geometry'] = self.record['geometry'][ordinal].to(device)
        elif self.group == 'I-HistoryDP':
            result.update({name: value[ordinal].to(device) for name, value in self.perception.items()})
        else:
            with h5py.File(self.index['path'], 'r') as stream:
                frames = stream['episodes'][self.index['episode']]['frames']
                result.update(rgb=torch.from_numpy(frames['rgb_u8'][tick]).to(device),
                    depth=torch.from_numpy(frames['depth_m'][tick]).to(device),
                    depth_valid=torch.from_numpy(frames['depth_valid'][tick]).to(device))
        return result


def tactile_values(world, local, stamps, observed, valid):
    """Preserved G10 long_data convention (3e11d62): 2x32 per control tick."""
    values = torch.cat((world,local),-1).flatten(-2)
    values = values.sign()*torch.log1p(values.abs())
    age = (observed[...,None]-stamps).float()
    mask = valid[...,None].expand_as(stamps)
    if ((age < -1e-6)&mask).any():raise ValueError('Future tactile sample')
    return torch.where(mask[...,None],torch.cat((values,age[...,None],mask[...,None].float()),-1),0.)


def visual_records(data_root, identities, modality, device='cpu', *, target_window_contract=None):
    """Strict dataset boundary: G00 never loads the separate tactile file."""
    root=Path(data_root)
    cache=torch.load(root/'numeric.pt',map_location='cpu',weights_only=False)
    if cache['data_sha256']!=sha256(root/'data.json'):raise ValueError('Numeric identity differs')
    touch=torch.load(root/'tactile.pt',map_location='cpu',weights_only=False) if modality=='G10' else None
    records={}
    data=json.loads((root/'data.json').read_text())
    supervision=target_supervision(data.get('supervision','u_expert'))
    for identity in identities:
        raw=cache['records'][identity]
        expected={'robot','command','time','joint_limits'}
        if data.get('supervision') is not None:expected|={'q_reached'}
        extra={'u_expert','supervision_mask','control_segment'}
        if not expected<=set(raw) or set(raw)-expected-extra:raise ValueError('Privileged visual record')
        if set(raw)&extra and not extra<=set(raw):raise ValueError('Incomplete supervised segment contract')
        records[identity]={k:raw[k].to(device) for k in ('robot','command','time','joint_limits')}
        if data.get('supervision') is not None:
            records[identity]['clean_sample']=raw['q_reached'].to(device) if supervision=='q_reached' else raw.get('u_expert',raw['command']).to(device)
        if 'supervision_mask' in raw:
            from .anchor_sampling import supervision_rows
            supervision_rows(raw)
            records[identity].update({k:raw[k].to(device) for k in ('supervision_mask','control_segment')})
            records[identity].update(supervision=supervision,expert_command=raw.get('u_expert',raw['command']).to(device))
        records[identity].update(identity=identity,source_root=str(root))
        if touch is not None:records[identity]['tactile']=touch[identity].to(device)
    from .target_windows import with_target_window
    return with_target_window(records,target_window_contract)


def residual_observations(entry, robot, time):
    """Existing observed force/FK fields only; targets stay outside the inputs."""
    with h5py.File(entry['path']) as f:
        g=f['episodes'][entry['episode']];n=len(time);obs=g['obs']
        if not torch.equal(torch.from_numpy(obs['time_s'][:n]),time.cpu()):
            raise ValueError('Residual observations and actual robot time differ')
        tactile={target:torch.from_numpy(obs[source][:n]).flatten(0,1) for source,target in (
            ('contact_substeps_w_N','force_w_N'),('contact_substeps_local_N','force_local_N'),
            ('finger_position_base_m','position_base_m'),('finger_rotation_base','rotation_base'),('contact_time_s','time_s'))}
        valid=torch.from_numpy(obs['contact_valid'][:n]).bool()
        force=torch.from_numpy(obs['contact_w_N'][:n])
        before=torch.from_numpy(g['transition/time_before_s'][:n]);after=torch.from_numpy(g['transition/time_after_s'][:n])
        if not torch.equal(before,time.cpu()) or not torch.allclose(after[:-1],time[1:].cpu(),atol=1e-6,rtol=0):
            raise ValueError('Expert execution acknowledgement is not strictly past')
        if not torch.allclose(time[1:]-time[:-1],torch.full_like(time[1:],1/60),atol=1e-6,rtol=0):
            raise ValueError('Residual requires native 60Hz data')
        stamps=tactile['time_s'].reshape(n,2)
        if ((stamps-time.cpu()[:,None]>1e-6)&valid[:,None]).any():raise ValueError('Future tactile sample')
        if not torch.allclose(stamps[valid,1]-stamps[valid,0],torch.full_like(stamps[valid,0],1/120),atol=1e-6,rtol=0):
            raise ValueError('Residual requires two native 120Hz samples')
    return dict(robot=robot.cpu(),tactile=tactile,tactile_valid=valid.repeat_interleave(2),
        gate=valid&(force.norm(dim=-1).amax(-1)>.1))


def residual_visual(model,memory,stamp):
    """Frozen DP's existing visual-only readout, before proprio/touch fusion."""
    return model.vision.read(torch.stack([f[0] for f in memory.frames]),
        torch.stack([stamp-f[1] for f in memory.frames]),torch.stack([f[2] for f in memory.frames]))


def remaining_target_plan(plan,cursor):
    if plan.shape[-2:]!=(24,29) or not 0<=cursor<12:raise ValueError('H24/commit12 nominal cursor differs')
    index=torch.arange(24,device=plan.device)+cursor
    return plan[...,index.clamp_max(23),:],index<24


def residual_batch(records,start,stop,device):
    """Same allowlisted builder for training and real one-tick deployment."""
    def tensor(values):return torch.stack(values).to(device)
    def padded(value,a,b):
        out=value.new_zeros((b-a,*value.shape[1:]));count=max(0,min(b,len(value))-a)
        if count:out[:count]=value[a:a+count]
        return out
    keys=('visual','robot','plans','plan_valid')
    extra=('plan_cursor','previous_nominal','previous_nominal_valid')
    if any(k in r for r in records for k in extra):
        if not all(all(k in r for k in extra) for r in records):raise ValueError('Incomplete previous nominal/cursor contract')
        keys+=extra
    inputs={k:tensor([padded(r[k],start,stop) for r in records]) for k in keys}
    inputs['tactile']={k:tensor([padded(r['tactile'][k],2*start,2*stop) for r in records]) for k in records[0]['tactile']}
    inputs['tactile_valid']=tensor([padded(r['tactile_valid'],2*start,2*stop) for r in records])
    inputs['valid']=tensor([torch.arange(start,stop)<len(r['robot']) for r in records])
    return inputs


def residual_previous_targets(inputs, limits):
    """Give padding a legal target without changing any actual observation.

    Padded robot fields are zero, which need not lie inside a joint's limits.
    The execution map validates its entire batch before the supervision mask
    is applied. Only nonexistent ticks receive a joint-limit placeholder.
    """
    previous = inputs['robot'][..., 58:87]
    return torch.where(inputs['valid'][..., None], previous, limits[..., 0])


def residual_supervision_mask(records,start,stop,device):
    """Loss/exposure mask only; residual_batch.valid keeps the real student prefix."""
    from .anchor_sampling import supervision_rows
    rows=[]
    for record in records:
        mask,_=supervision_rows(record)
        out=torch.zeros(stop-start,dtype=torch.bool,device=device)
        count=max(0,min(stop,len(mask))-start)
        if count:out[:count]=mask[start:start+count].to(device)
        rows.append(out)
    return torch.stack(rows)


def supervision_mask_sha256(records,identities=None):
    """Bind actual loss masks AND segment boundaries, not a renamed dataset hash."""
    import hashlib
    from .anchor_sampling import supervision_rows
    digest=hashlib.sha256()
    for identity in sorted(records if identities is None else identities):
        mask,segment=supervision_rows(records[identity])
        encoded=identity.encode('utf-8')
        digest.update(len(encoded).to_bytes(8,'little'));digest.update(encoded)
        digest.update(len(mask).to_bytes(8,'little'))
        digest.update(mask.cpu().numpy().tobytes());digest.update(segment.cpu().numpy().astype('<i8').tobytes())
    return digest.hexdigest()


@torch.no_grad()
def cache_residual_episode(base,record,entry,lease,*,progress=lambda **kw:None,cache_identity=None,transfer_mode='tick',scheduler_index_mode='legacy',offline_planning_profile=None):
    """Replay every native observation and optionally batch independent plans.

    ``tick`` retains the historical implementation for measured comparisons.
    ``episode`` transfers numeric inputs once and copies only complete replans
    back to CPU. Cursor expansion changes no inference, recurrence or RNG draws.
    Neither mode reuses a prediction from another checkpoint.
    The explicit offline profile defers only DDIM until this episode's full
    real history has been replayed. Current model conditions and per-tick visual
    features are recomputed; learned hidden states never become a reusable cache.
    """
    from .position_control import bounded_targets
    device=next(base.parameters()).device
    if transfer_mode not in ('tick','episode'):raise ValueError('Unknown residual cache transfer implementation')
    if scheduler_index_mode not in ('legacy','host'):raise ValueError('Unknown DDIM scheduler index implementation')
    if offline_planning_profile is not None:
        from .offline_planning import validate_profile,predict_contiguous
        validate_profile(offline_planning_profile)
        if (transfer_mode!='episode' or scheduler_index_mode!='host' or base.group!='I-HistoryDP'
                or base.modality!='G00' or base.horizon!=24 or base.contract.get('visual_stride')!=1):
            raise ValueError('Offline cache planning requires explicit episode/host H24 G00 every-tick history')
    if base.training or any(p.requires_grad for p in base.parameters()):raise ValueError('Base must be completely frozen')
    binding={'base_checkpoint_sha256','statistics_sha256','data_sha256','residual_statistics_sha256'}
    if entry.get('trajectory_kind')=='expert_takeover_v1' and cache_identity is None:
        raise ValueError('DAgger nominal caches require current DP, base/residual statistics and dataset SHA bindings')
    if cache_identity is not None:
        if set(cache_identity)!=binding or any(not isinstance(v,str) or len(v)!=64 for v in cache_identity.values()):
            raise ValueError('Incomplete nominal cache identity')
        if cache_identity['data_sha256']!=sha256(Path(record['source_root'])/'data.json'):
            raise ValueError('Nominal cache dataset changed')
    episode=Episode(record['source_root'],record['identity'],base.group)
    generator=torch.Generator(device=device).manual_seed(7091);memory=None
    result=residual_observations(entry,record['robot'],record['time'])
    plans=[];masks=[];commands=[];features=[]
    conditions=[] if offline_planning_profile is not None else None
    inputs=record
    if transfer_mode=='episode':
        # At most a few MB of numeric observations; frozen image shards continue
        # to transfer only at their actual observation tick. This is not a batch
        # of recurrent updates and never writes back to issued/applied history.
        with lease.block():
            inputs={key:record[key].to(device) for key in ('robot','time','joint_limits')}
            if base.modality=='G10':inputs['tactile']=record['tactile'].to(device)
    for tick in range(len(record['time'])):
        with lease.block():
            stamp=inputs['time'][tick].to(device)
            memory=base.observe(inputs['robot'][tick].to(device),stamp,tick,episode.frame(tick,device) if tick%base.contract.get('visual_stride',12)==0 else None,
                memory,tactile=inputs['tactile'][tick].to(device) if base.modality=='G10' else None)
            if tick%12==0:
                if conditions is not None:
                    # Condition is a new tensor; no learned hidden state is
                    # reused across episodes/checkpoints. Recorded robot
                    # histories never receive these future nominal plans.
                    conditions.append(base.condition(memory,stamp))
                else:
                    sampling={} if scheduler_index_mode=='legacy' else dict(scheduler_index_mode=scheduler_index_mode)
                    plan=base.predict(base.condition(memory,stamp),generator,**sampling)
                    plan=bounded_targets(plan[0],inputs['joint_limits'].to(device))[None]
                    if transfer_mode=='episode':plans.append(plan[0])
            if tick%base.contract.get('visual_stride',12)==0:
                visual=residual_visual(base,memory,stamp)[0]
            if transfer_mode=='episode':features.append(visual)
            else:visual=visual.cpu()
            if transfer_mode=='tick':
                remaining,valid=remaining_target_plan(plan,tick%12)
                plans.append(remaining[0].cpu());masks.append(valid.cpu());commands.append(plan[0,tick%12].cpu());features.append(visual)
        if tick%60==0:
            if conditions is None:progress(tick=tick+1,plans=tick//12+1)
            else:progress(tick=tick+1,plans=0,conditions=len(conditions),phase='complete_history_replay')
    if conditions is not None:
        with lease.block():
            condition=torch.cat(conditions)
            anchors=torch.arange(0,len(record['time']),12,device=device)
            predicted=predict_contiguous(base,condition,anchors,configuration=offline_planning_profile,
                check=getattr(lease,'check',None))
            predicted=bounded_targets(predicted.flatten(0,1),inputs['joint_limits']).reshape_as(predicted)
            plans=list(predicted.unbind())
        progress(tick=len(record['time']),plans=len(plans),conditions=len(conditions),phase='offline_planning_complete')
        del conditions,condition,predicted
    if transfer_mode=='episode':
        # Expand unchanged H24 plans on CPU instead of copying a 24x29 tensor
        # on every one of the 12 committed ticks. The final partial chunk is
        # selected by n, with the same repeated-final-row/valid-mask convention.
        with lease.block():
            compact=torch.stack(plans).cpu();visual=torch.stack(features).cpu()
        rows=torch.arange(len(record['time']));ordinal=rows//12;cursor=rows%12
        indices=cursor[:,None]+torch.arange(24)[None]
        planned=compact[ordinal[:,None],indices.clamp_max(23)]
        valid=indices<24;commands=compact[ordinal,cursor]
    else:
        planned=torch.stack(plans);valid=torch.stack(masks);commands=torch.stack(commands);visual=torch.stack(features)
    result.update(plans=planned,plan_valid=valid,base=commands,
        visual=visual,target=record.get('expert_command',record['command']).cpu(),limits=record['joint_limits'].cpu(),
        identity=record['identity'],time=record['time'].cpu(),
        nominal_source='frozen_EMA_DDIM16_seed7091_predictions_on_complete_recorded_observation_prefix',
        history_source='strictly_past_actual_executed_targets_including_student_prefix; never regenerated nominal labels')
    for key in ('supervision_mask','control_segment'):
        if key in record:result[key]=record[key].cpu()
    if cache_identity is not None:result['cache_identity']=dict(cache_identity)
    if offline_planning_profile is not None:result['cache_offline_planning_profile']=dict(offline_planning_profile)
    nominal=result['base'];n=len(nominal)
    if base.contract.get('supervision')=='q_reached' and 'clean_sample' not in record:
        raise ValueError('Actual-configuration nominal cache requires explicit reached labels')
    result.update(plan_cursor=torch.arange(n)%12,
        previous_nominal=torch.cat((torch.zeros_like(nominal[:1]),nominal[:-1])),
        previous_nominal_valid=torch.arange(n)>0,
        nominal_supervision=base.contract.get('supervision','u_expert'),
        q_reached=record['clean_sample'].cpu() if base.contract.get('supervision')=='q_reached' else None)
    return result


def cache_visual_pair(root, identities=None, *, reuse_index=None, payload_audit=None):
    """Single writer, immutable backbone features only; live uses the same call."""
    import time
    from .models.native_vision import NativeDinoSpatial
    from .state_resources import lease_for
    from .checkpointing import save
    from .training_resources import atomic_json
    if payload_audit is not None:
        from .frozen_payload_audit import FrozenPayloadAudit
        if not isinstance(payload_audit,FrozenPayloadAudit):
            raise TypeError('Explicit process-local frozen payload auditor required')
    payload_sha256=sha256 if payload_audit is None else payload_audit.sha256
    root=Path(root);out=root/'state_data';info=json.loads((out/'data.json').read_text())
    ids=identities or info['splits']['train']+info['splits']['validation']+info['splits'].get('correction_dev',[])
    plan=json.loads((root/'session.json').read_text())
    if sha256(plan['dino_checkpoint'])!=plan['dino_sha256']:raise ValueError('Frozen DINO weight file changed')
    lease=lease_for(root,deadline=plan['cache_deadline'])
    dino=None
    (out/'perception').mkdir(exist_ok=True);entries={};started=time.time();frames_done=0
    stride=info.get('visual_stride',12)
    if stride not in (1,12):raise ValueError('Unknown visual cache cadence')
    total=sum((info['entries'][i]['length']+stride-1)//stride for i in ids)
    cache_contract=dict(dino_revision=DINO_REVISION,dino_sha256=plan['dino_sha256'],data_sha256=sha256(out/'data.json'),
        cameras=CAMERAS,resolution=[448,252],patches=[384,18,32],depth=[2,18,32],stride=stride,
        random_augmentation=False,trainable_projection_cached=False)
    # Only a committed per-episode index proves a reusable payload. A worker
    # can die after any episode; the next worker must retain its measured work.
    prior={'entries':{}};orphans=[]
    for name in ('cache_acceptance_index.json','cache_progress_index.json','cache_index.json'):
        checkpoint=root/name
        if not checkpoint.is_file():continue
        previous=json.loads(checkpoint.read_text())
        if any(previous.get(k)!=v for k,v in cache_contract.items()):
            raise ValueError('Existing DINO cache identity differs: '+name)
        orphans.extend(previous.get('preserved_orphans',[]))
        for key,value in previous['entries'].items():
            if key in prior['entries'] and value!=prior['entries'][key]:
                raise ValueError('Conflicting committed DINO payload identity: '+key)
            prior['entries'][key]=value
    def checkpoint_progress():
        committed=dict(prior['entries']);committed.update(entries)
        atomic_json(root/'cache_progress_index.json',dict(cache_contract,entries=committed,
            complete=False,requested_ids=ids,preserved_orphans=orphans,utc=time.time()))
    reusable=json.loads(Path(reuse_index).read_text()) if reuse_index is not None else None
    if reusable and (reusable['dino_sha256']!=plan['dino_sha256'] or reusable.get('dino_revision')!=DINO_REVISION
            or reusable.get('stride')!=stride or reusable.get('cameras')!=CAMERAS
            or reusable.get('resolution')!=[448,252] or reusable.get('patches')!=[384,18,32]
            or reusable.get('depth')!=[2,18,32] or reusable.get('random_augmentation') is not False
            or reusable.get('trainable_projection_cached') is not False or reusable.get('complete') is not True):
        raise ValueError('Only matching frozen DINO features may be shared across policies')
    for identity in ids:
        entry=info['entries'][identity];patches=[];depth=[]
        path=out/'perception'/(identity+'.pt')
        if reusable and identity in reusable['entries'] and reusable['entries'][identity]['source_sha256']!=entry['source_sha256']:
            raise ValueError('Shared DINO payload/source identity changed: '+identity)
        old=prior['entries'].get(identity)
        if old is not None:
            if (not path.is_file() or old['sha256']!=payload_sha256(path) or old['source_sha256']!=entry['source_sha256']
                    or old['frames']!=(entry['length']+stride-1)//stride):
                raise ValueError('Committed DINO payload/source identity changed: '+identity)
            entries[identity]=old;total-=old['frames'];checkpoint_progress();continue
        shared=reusable['entries'].get(identity) if reusable else None
        if reusable and shared is None:
            # A pool identity includes run/group/round, whereas an earlier cache
            # may use the collector's local name. Only immutable source-content
            # and backbone identity permit reuse; never copy trainable features.
            matches=[row for row in reusable['entries'].values() if row.get('source_sha256')==entry['source_sha256']]
            if len(matches)>1 and len({row['sha256'] for row in matches})!=1:
                raise ValueError('Ambiguous source-content DINO cache identity')
            shared=matches[0] if matches else None
        if shared is not None:
            source=Path(shared['path'])
            if not source.is_absolute():
                candidates=(Path.cwd()/source,Path(reuse_index).resolve().parent/source)
                source=next((p for p in candidates if p.is_file()),source)
            if shared['source_sha256']!=entry['source_sha256'] or payload_sha256(source)!=shared['sha256']:
                raise ValueError('Shared DINO payload/source identity changed: '+identity)
            if shared['frames']!=(entry['length']+stride-1)//stride:raise ValueError('Shared DINO temporal coverage differs')
            if path.exists() and payload_sha256(path)!=shared['sha256']:raise ValueError('Existing DINO payload differs')
            if not path.exists():path.symlink_to(source.resolve())
            entries[identity]=dict(shared,path=str(path.resolve()));total-=shared['frames'];checkpoint_progress();continue
        if path.exists() or path.is_symlink():
            orphan=path.with_name(path.name+'.unindexed.'+str(time.time_ns()))
            record=dict(identity=identity,path=str(orphan),sha256=payload_sha256(path) if path.is_file() else None,
                reason='Payload exists without a committed matching index; preserved before rebuilding')
            path.replace(orphan);orphans.append(record);checkpoint_progress()
        with h5py.File(entry['path'],'r') as f:
            g=f['episodes'][entry['episode']];meta=json.loads(f.attrs['metadata_json'])
            if meta['camera_ids']!=CAMERAS:raise ValueError('Camera order differs')
            ticks=np.arange(0,entry['length'],stride)
            times=torch.from_numpy(g['obs/time_s'][ticks]);ct=torch.from_numpy(g['obs/camera_time_s'][ticks]);cv=torch.from_numpy(g['obs/camera_valid'][ticks])
            if not cv.all() or ((times[:,None]-ct)<-1e-6).any():raise ValueError('Invalid/future camera')
            for tick in ticks:
                if dino is None:
                    with lease.block():dino=NativeDinoSpatial().cuda().eval()
                    if any(p.requires_grad for p in dino.parameters()):raise ValueError('DINO must be frozen')
                with lease.block():
                    rgb=torch.from_numpy(g['frames/rgb_u8'][int(tick)]).cuda()
                    patches.append(dino(rgb).cpu())
                    depth.append(depth_grid(torch.from_numpy(g['frames/depth_m'][int(tick)]),torch.from_numpy(g['frames/depth_valid'][int(tick)])))
                frames_done+=1
                elapsed=time.time()-started
                atomic_json(root/'cache_progress.json',dict(episodes=len(entries),total_episodes=len(ids),frames=frames_done,total_frames=total,
                    seconds=elapsed,eta_seconds=(total-frames_done)*elapsed/frames_done,identity=identity,utc=time.time()))
        if path.exists():raise FileExistsError('No repeated cache writes: '+str(path))
        save(path,dict(patches=torch.stack(patches),depth=torch.stack(depth),camera_time=ct,camera_valid=cv))
        entries[identity]=dict(path=str(path),sha256=payload_sha256(path),source_sha256=entry['source_sha256'],frames=len(ticks),camera_time_sha256=__import__('hashlib').sha256(ct.numpy().tobytes()).hexdigest())
        checkpoint_progress()
    atomic_json(root/('cache_acceptance_index.json' if identities else 'cache_index.json'),dict(cache_contract,entries=entries,
        complete=True,preserved_orphans=orphans))
