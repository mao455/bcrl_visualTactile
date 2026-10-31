"""Frozen selected G00 replay, with isolated proposals and true teacher feedback."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import h5py
import torch

from .action_reference import reference_to_command
from .compact import sha256
from .long_data import MANIFEST, inputs_for, load_record, window
from .models.long_horizon import LongHorizonConfig, LongHorizonPolicy
from .models.residual import observed_contact
from .models.transformer_residual import HISTORY_PROTOCOL
from .private_history import private_action_rows
from .study_resources import StudyLease
from .train_long import move, save
from .training_resources import atomic_json


@torch.no_grad()
def audit_generated_episode(root, row, record, saved, lease, output_name='transformer_interface_audit.json'):
    from .long_controller import LongController
    from .models.transformer_residual import TransformerResidual
    from .private_history import PrivateProposalHistory
    original = load_record(root,row['id'])
    class CachedDino:
        def __init__(self):
            self.cursor = 0

        def __call__(self, rgb):
            value = original['patches'][self.cursor].to(rgb.device)
            self.cursor += 1
            return value
    with lease.block():
        controller = LongController(saved,CachedDino())
    controller.generator.manual_seed(record['policy_noise_seed'])
    command_error = plan_error = action_error = 0.
    per_plan = []
    length = min(25,len(record['time']))
    with h5py.File(MANIFEST.parent/row['directory']/'data.h5','r') as stream:
        group = stream['episodes'][row['episode']]
        def at(name,step):
            return torch.from_numpy(group[name][step:step+1])
        selected = torch.from_numpy(group['transition/control_trace/selected_command_time_s'][:len(record['time'])])
        restored = private_action_rows(original['command'],original['time'],selected,0,len(record['time']))
        actual_error = float((restored-original['action']).abs().max())
        for step in range(length):
            image = torch.cat((at('frames/rgb_u8',step).float()/255.,at('frames/depth_m',step).float()),2)
            packet = dict(image=image,depth_valid=at('frames/depth_valid',step),proprio=at('obs/proprio',step),
                time_s=at('obs/time_s',step),camera_time_s=at('obs/camera_time_s',step),
                camera_valid=at('obs/camera_valid',step),actuator_target_rad=at('obs/actuator_target_rad',step),
                episode_id=torch.tensor([1]),step_id=torch.tensor([step]))
            with lease.block():
                command = controller.act(packet)
            if step==0:
                controller.execution = PrivateProposalHistory()
            transition = dict(episode_id=packet['episode_id'],step_id=packet['step_id'],
                time_before_s=at('transition/time_before_s',step),time_after_s=at('transition/time_after_s',step),
                control_trace=dict(valid=at('transition/control_trace/valid',step),
                                   selected_command_time_s=at('transition/control_trace/selected_command_time_s',step)))
            controller.execution.observe_proposal(transition,command,packet)
            command_error = max(command_error,float((command[0]-record['base'][step]).abs().max()))
            action_error = max(action_error,float((controller.records['action'][-1]-record['private_action'][step]).abs().max()))
            if step%12==0:
                plan_error = max(plan_error,float((controller.plan[0]-record['plans'][step//12]).abs().max()))
                per_plan.append(dict(step=step,reference_error=float((controller.plan[0]-record['plans'][step//12]).abs().max()),
                    condition_error=float((controller.last_condition[0]-record['nominal_conditions'][step//12]).abs().max()),
                    state_error=float((controller.state.float().cpu()-record['nominal_states'][step//12]).abs().max()),
                    depth_error=float((controller.records['depth'][-1]-original['depth'][step//12]).abs().max())))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(471)
        residual = TransformerResidual().eval()
        torch.nn.init.normal_(residual.output[-1].weight,std=.01)
        for name in ('proprio_mean','proprio_std','target_scale','target_offset'):
            getattr(residual,name).copy_(saved['model'][name])
    batched,memory = residual(residual_inputs(record,0,length))
    state = None
    predictions = []
    for step in range(length):
        anchors = torch.tensor([step]) if step%6==0 else torch.empty(0,dtype=torch.long)
        prediction,state = residual(residual_inputs(record,step,step+1,anchors),state)
        if len(anchors):
            predictions.append(prediction)
    residual_error = float((batched-torch.cat(predictions)).abs().max())
    memory_error = float((memory-state).abs().max())
    with lease.block():
        residual = residual.to('cuda:0')
        gpu_batched,gpu_memory = residual(move(residual_inputs(record,0,length),'cuda:0'))
    state = None
    gpu_predictions = []
    for step in range(length):
        anchors = torch.tensor([step]) if step%6==0 else torch.empty(0,dtype=torch.long)
        with lease.block():
            prediction,state = residual(move(residual_inputs(record,step,step+1,anchors),'cuda:0'),state)
            if len(anchors):
                gpu_predictions.append(prediction)
    with lease.block():
        gpu_error = float((gpu_batched-torch.cat(gpu_predictions)).abs().max())
        gpu_memory_error = float((gpu_memory-state).abs().max())
    result = dict(passed=command_error<=1e-3 and plan_error<=1e-3 and action_error<=1e-3 and actual_error<=1e-6
        and residual_error<=1e-5 and memory_error<=1e-5 and gpu_error<=1e-5 and gpu_memory_error<=1e-5,
        observations=length,episode=row['id'],
        nominal_command_max_error=command_error,nominal_reference_max_error_rad=plan_error,
        private_history_max_error=action_error,actual_private_no_correction_max_error=actual_error,
        residual_streaming_max_error=residual_error,residual_memory_max_error=memory_error,
        residual_gpu_streaming_max_error=gpu_error,residual_gpu_memory_max_error=gpu_memory_error,
        residual_parameters=sum(parameter.numel() for parameter in residual.parameters()),
        scope='real expert observations; cached audited DINO; GPU nominal private replay, CPU/GPU FP32 nonzero-head residual parity',
        thresholds=dict(nominal_bf16=1e-3,residual_fp32=1e-5),base_sha256=record['base_sha256'],per_plan=per_plan)
    atomic_json(root/output_name,result)
    if not result['passed']:
        raise ValueError('New residual/private-history real data parity failed')


def tactile_features(world, local, position, rotation, timestamps, observed, valid):
    if world.shape[-3:] != (2,5,3) or local.shape != world.shape:
        raise ValueError('Expected two actual120Hz samples at each native observation')
    age = (observed[...,None]-timestamps).float()
    mask = valid[...,None,None].expand(*age.shape,5)
    if ((age < -1e-6)&valid[...,None]).any():
        raise ValueError('Future measured tactile input')
    values = torch.cat((world.float(),local.float(),position.float(),rotation.float().flatten(-2),
        age[...,None,None].expand(*age.shape,5,1),mask[...,None].float()),-1)
    return torch.where(mask[...,None],values,0.)


def remaining_plans(plans, anchors):
    cursors = anchors%12
    positions = cursors[:,None]+torch.arange(64)
    valid = positions<64
    values = plans[anchors//12].gather(1,positions.clamp_max(63)[...,None].expand(-1,-1,29))
    return torch.where(valid[...,None],values,0.),valid


def residual_inputs(record, start, stop, anchors=None):
    if anchors is None:
        anchors = torch.arange(start,stop)
        anchors = anchors[anchors%6==0]
    recent,valid = window(record['tactile'],anchors,13)
    times,_ = window(record['time'],anchors,13)
    age = (record['time'][anchors,None]-times).float()
    recent = recent.clone()
    recent[...,-2] += age[...,None,None]
    recent = torch.where(valid[...,None,None,None],recent,0.)
    plans,plan_valid = remaining_plans(record['plans'],anchors)
    result = {name:record[name][start:stop].float() for name in ('proprio','action','target','tracking','tactile','base')}
    result.update(visual=record['visual'][torch.arange(start,stop)//12].float(),anchors=anchors-start,
        plan=plans.float(),plan_valid=plan_valid,recent_tactile=recent.flatten(1,2).float())
    return result


def residual_labels(record, anchors):
    positions = anchors[:,None]+torch.arange(6)
    valid = positions<len(record['time'])
    selected = positions.clamp_max(len(record['time'])-1)
    return dict(delta=record['expert'][selected]-record['base'][selected],base=record['base'][selected],
                expert=record['expert'][selected],gate=record['gate'][selected],valid=valid)


def load_residual_record(root, identity):
    record = torch.load(Path(root)/'transformer_cache'/f'{identity}.pt',map_location='cpu',weights_only=False)
    if record.get('history_protocol')!=HISTORY_PROTOCOL or record.get('version')!=1:
        raise ValueError('Residual cache history/version differs')
    return record


@torch.no_grad()
def generate_episode(root, row, model, metadata, lease):
    identity = row['id']
    original = load_record(root,identity)
    length = len(original['time'])
    path = MANIFEST.parent/row['directory']/'data.h5'
    if sha256(path)!=row['files']['data.h5']['sha256']:
        raise ValueError('Original expert data hash changed')
    with h5py.File(path,'r') as stream:
        group = stream['episodes'][row['episode']]
        def read(name):
            return torch.from_numpy(group[name][:length])
        times,before,after = read('obs/time_s'),read('transition/time_before_s'),read('transition/time_after_s')
        selected = read('transition/control_trace/selected_command_time_s')
        if not torch.equal(before,times) or not torch.allclose(after[:-1],times[1:],atol=1e-6,rtol=0):
            raise ValueError('Actual native teacher feedback is not aligned')
        if not read('transition/control_trace/valid').all():
            raise ValueError('Missing native delay timing')
        tactile = tactile_features(read('obs/contact_substeps_w_N'),read('obs/contact_substeps_local_N'),
            read('obs/finger_position_base_m'),read('obs/finger_rotation_base'),read('obs/contact_time_s'),
            times,read('obs/contact_valid'))
        tracking = read('obs/tracking_error_rad')
        gate = observed_contact(read('obs/contact_w_N'),read('obs/contact_valid'))
    if not torch.equal(times,original['time']):
        raise ValueError('Frozen visual cache and expert observations differ')
    limits = torch.tensor(metadata['joint_limits_rad'][0])
    speed,dt = metadata['config']['action']['dof_speed_scale'],metadata['control_dt']
    private = dict(original)
    private['action'] = torch.zeros_like(original['action'])
    bases = torch.zeros(length,29)
    plans,visuals,conditions,states = [],[],[],[]
    state = None
    seed = int(hashlib.sha256(identity.encode()).hexdigest()[:8],16)
    generator = torch.Generator(device='cuda:0').manual_seed(seed)
    plan = None
    for step in range(length):
        private['action'][step:step+1] = private_action_rows(bases,times,selected,step,step+1)
        inputs = inputs_for(private,step,step+1,False,torch.tensor([step]))
        with lease.block():
            with torch.autocast('cuda',dtype=torch.bfloat16):
                condition,state = model.encode(move(inputs,'cuda:0'),state)
                if step%12==0:
                    plan = model.predict_from_condition(condition,generator).float().cpu()[0]
                    visual = model.project_frames(original['patches'][step//12:step//12+1].to('cuda:0'),
                        original['depth'][step//12:step//12+1].to('cuda:0'),
                        original['camera_valid'][step//12:step//12+1].to('cuda:0')).float().cpu()[0]
        if step%12==0:
            plans.append(plan)
            visuals.append(visual)
            conditions.append(condition[0].float().cpu())
            states.append(state.float().cpu().clone())
        bases[step:step+1] = reference_to_command(plan[step%12:step%12+1],original['target_before'][step:step+1],limits,
            speed_scale=speed,control_dt=dt)
    return dict(version=1,identity=identity,history_protocol=HISTORY_PROTOCOL,
        actual_history_source='past_executed_expert_teacher_forcing',policy_noise_seed=seed,
        time=times,proprio=original['proprio'],action=original['action'],private_action=private['action'],
        target=original['target_before'],tracking=tracking,tactile=tactile,gate=gate,
        expert=original['command'],base=bases,plans=torch.stack(plans),visual=torch.stack(visuals),
        nominal_conditions=torch.stack(conditions),nominal_states=states,
        source_sha256=row['files']['data.h5']['sha256'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    root = args.root
    if args.preflight:
        session = json.loads((root/'session.json').read_text())
        source = Path(session['source_root'])/'G00-new_full/best.pt'
        saved = torch.load(source,map_location='cpu',weights_only=False)
        lease = StudyLease(root)
        with lease.block():
            model = LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval().requires_grad_(False)
            model.load_state_dict(saved['model'])
        identity = json.loads((root/'splits.json').read_text())['train'][0]
        row = next(row for row in json.loads(MANIFEST.read_text())['episodes'] if row['id']==identity)
        record = generate_episode(root,row,model,saved['metadata'],lease)
        record['base_sha256'] = sha256(source)
        audit_generated_episode(root,row,record,saved,lease,f'transformer_preflight_{time.time_ns()}.json')
        return
    selection = json.loads((root/'selected.json').read_text())
    checkpoint = root/selection['selected']['G00']/'best.pt'
    checkpoint_hash = sha256(checkpoint)
    if checkpoint_hash!=selection['checkpoint_sha256']['G00']:
        raise ValueError('Selected G00 checkpoint changed')
    admission = json.loads((root/'dp_closed_loop/report.json').read_text())
    if not admission['complete']:
        raise ValueError('Base closed-loop contact assessment is incomplete')
    from .evaluate_study import allocation
    residual_trials = allocation(root,'residual')
    atomic_json(root/'residual_trial_allocation.json',dict(trials=residual_trials,allocated_before_residual_training=True))
    directory = root/'transformer_cache'
    directory.mkdir(exist_ok=False)
    lease = StudyLease(root)
    saved = torch.load(checkpoint,map_location='cpu',weights_only=False)
    with lease.block():
        model = LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval().requires_grad_(False)
        model.load_state_dict(saved['model'])
    if model.cfg.tactile:
        raise ValueError('G01 nominal must be vision-only')
    splits = json.loads((root/'splits.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    rows = {row['id']:row for row in manifest['episodes']}
    index = dict(version=1,base=str(checkpoint),base_sha256=checkpoint_hash,history_protocol=HISTORY_PROTOCOL,
        splits=splits,entries={},started_epoch=time.time(),source_hashes={str(path):sha256(path) for path in (
            Path(__file__),Path('visuotactile/private_history.py'),Path('visuotactile/models/long_horizon.py'),
            Path('visuotactile/long_data.py'),Path('visuotactile/models/transformer_residual.py'))})
    atomic_json(directory/'index.json',index)
    for identity in splits['train']+splits['validation']+splits['test']:
        lease.check()
        record = generate_episode(root,rows[identity],model,saved['metadata'],lease)
        record['base_sha256'] = checkpoint_hash
        if not index['entries']:
            audit_generated_episode(root,rows[identity],record,saved,lease)
        path = directory/f'{identity}.pt'
        save(path,record)
        index['entries'][identity] = dict(sha256=sha256(path),steps=len(record['time']),
            contact_steps=int(record['gate'].sum()),valid_tactile_steps=int(record['tactile'][...,-1].any(-1).all(-1).sum()))
        atomic_json(directory/'index.json',index)
        atomic_json(directory/'status.json',dict(cached=len(index['entries']),total=400,last=identity))
        if len(index['entries'])%20==0:
            print('RESIDUAL_CACHE',len(index['entries']),flush=True)
    if any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError('Frozen nominal received gradients')
    if sha256(checkpoint)!=checkpoint_hash:
        raise ValueError('Frozen checkpoint mutated')
    atomic_json(root/'transformer_cache_complete.json',dict(complete=True,episodes=len(index['entries']),
        base_sha256=checkpoint_hash,index_sha256=sha256(directory/'index.json'),finished_epoch=time.time()))


if __name__=='__main__':
    main()
