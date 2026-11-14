"""Real RGB-D lane parity with independent reset/history boundaries, no physics."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
import h5py
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256


TAIL_TICKS = 25
TAIL_LANES = 8
TARGET_ATOL = 2e-5
STATE_ATOL = 2e-5
STATE_RTOL = 2e-5


def tail_schedule(tick):
    if type(tick) is not int or not 0 <= tick < TAIL_TICKS:
        raise ValueError('Exact-tail gate has exactly 25 registered observations')
    count = 8 if tick < 8 else 7 if tick < 14 else 5 if tick < 20 else 1
    return [lane < count for lane in range(TAIL_LANES)], [tick if tick < 13 else tick-13]+[tick]*7


class KernelObservedDino:
    """Observe actual backbone input shapes; delegate unchanged tensor math."""
    def __init__(self, model, *, cuda_timing=False):
        self.model, self.kernel_image_batch_counts = model, {}
        self.cuda_timing, self.kernel_timings = cuda_timing, []

    @property
    def last_lane_execution(self):
        return self.model.last_lane_execution

    def forward_lanes(self, rgb, **kwargs):
        original = self.model.backbone.forward_features
        def observe(images, *args, **kw):
            key = str(len(images))
            self.kernel_image_batch_counts[key] = self.kernel_image_batch_counts.get(key, 0)+1
            if not self.cuda_timing:return original(images, *args, **kw)
            torch.cuda.synchronize(images.device)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            wall=time.perf_counter();begin.record()
            result=original(images,*args,**kw)
            end.record();end.synchronize()
            self.kernel_timings.append(dict(images=len(images),cuda_ms=begin.elapsed_time(end),
                synchronized_wall_s=time.perf_counter()-wall))
            return result
        self.model.backbone.forward_features = observe
        try:
            return self.model.forward_lanes(rgb, **kwargs)
        finally:
            self.model.backbone.forward_features = original


def tail_packets(sources, tick):
    """Only lane0 restarts, while active; terminated lanes never reactivate."""
    active, indices = tail_schedule(tick)
    packets, diagnostics, targets = [], [], []
    for lane, index in enumerate(indices):
        packet, diag, target = sources[lane][index]
        packet = {key:value.clone() for key,value in packet.items()}
        if lane == 0 and tick >= 13:
            packet['episode_id'] += 100000
        packets.append(packet); diagnostics.append(diag); targets.append(target)
    return ({key:torch.cat([p[key] for p in packets]) for key in packets[0]},
        {key:torch.cat([d[key] for d in diagnostics]) for key in diagnostics[0]},
        torch.cat(targets), active, indices)


def history_shape(controller, step, episode):
    base, state = controller.base, controller.state
    memory = base.memory
    valid = state['history_valid'].detach().cpu()
    expected = dict(next_step=step+1, memory_next_tick=step+1, episode=episode,
        robot_rows=min(step+1, 13), visual_rows=min(step+1, 3),
        backend_valid=min(step+1, valid.shape[-1]))
    actual = dict(next_step=base.next_step, memory_next_tick=memory.next_tick,
        episode=base.episode, robot_rows=len(memory.robot), visual_rows=len(memory.frames),
        backend_valid=int(valid.sum()))
    if actual != expected or memory.native is None or memory.visual is None:
        raise ValueError('Every real frame must update native, visual and backend history')
    return actual


def numeric_history_comparison(left,right):
    """All frozen state fields, including both GRUs/plans; discrete fields exact."""
    maximum=0.;leaves=0
    if set(left)!=set(right):raise ValueError('Complete history state fields differ')
    for key,a in left.items():
        b=right[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Complete history tensor contract differs: '+key)
            if a.is_floating_point():
                if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                    raise ValueError('Nonfinite complete history: '+key)
                delta=float((a-b).abs().max()) if a.numel() else 0.
                maximum=max(maximum,delta)
                if not torch.allclose(a,b,atol=STATE_ATOL,rtol=STATE_RTOL):
                    raise ValueError('Complete history numerical parity failed: '+key+' max_abs='+str(delta))
            elif not torch.equal(a,b):raise ValueError('Discrete complete history differs: '+key)
            leaves+=1
        elif torch.is_tensor(a) or torch.is_tensor(b) or type(a) is not type(b) or a!=b:
            raise ValueError('Complete history structure/value differs: '+key)
    return dict(tensor_leaves=leaves,max_abs_float_state=maximum,atol=STATE_ATOL,rtol=STATE_RTOL)


def exact_state_equal(left,right):
    """Compare owned snapshots, including singleton tensors with nonunit strides."""
    if set(left)!=set(right):return False
    for key,a in left.items():
        b=right[key]
        if torch.is_tensor(a) and torch.is_tensor(b):
            if a.shape!=b.shape or a.dtype!=b.dtype:return False
            raw=lambda x:x.reshape(-1).clone(memory_format=torch.contiguous_format).view(torch.uint8)
            if not torch.equal(raw(a),raw(b)):return False
        elif torch.is_tensor(a) or torch.is_tensor(b) or type(a) is not type(b) or a!=b:return False
    return True


def run_tail_sequence(fixed, exact, sources, *, lease, cuda, exact_first=False):
    """CPU-testable complete-student gate; CUDA admission is the caller's job."""
    from tools.tight_fresh_forensics_gate import controller_state, freeze, rng_state
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    if len(sources) != TAIL_LANES or any(len(source) != TAIL_TICKS for source in sources):
        raise ValueError('Eight distinct registered lanes need all 25 source observations')
    rows = []; counts = [0]*TAIL_LANES; acknowledgement_counts = [0]*TAIL_LANES
    previous = None; last_states = {}; caught = None
    try:
        for tick in range(TAIL_TICKS):
            packet, diag, target, active, indices = tail_packets(sources, tick)
            if previous is not None and any(now and not before for now,before in zip(active,previous)):
                raise ValueError('Terminated gate lanes cannot reactivate')
            previous = active
            before_inputs = freeze(dict(packet=packet, diagnostics=diag, target=target))
            order = [('exact',exact),('fixed',fixed)] if exact_first else [('fixed',fixed),('exact',exact)]
            predictions = {}; histories = []
            with lease.block():
                global_before = rng_state(cuda)
                for name,controller in order:
                    predictions[name] = controller.act(packet,diag,active).detach().cpu().clone()
                    if not exact_state_equal(before_inputs,freeze(dict(packet=packet,diagnostics=diag,target=target))):
                        raise ValueError('Gate controller mutated shared recorded input')
                    if not exact_state_equal(global_before,rng_state(cuda)):
                        raise ValueError('Gate inference consumed shared RNG')
                difference = (predictions['fixed']-predictions['exact']).abs()
                error = float(difference.max())
                if not torch.isfinite(difference).all() or error > TARGET_ATOL:
                    raise ValueError('Complete fixed-M8/exact-tail target parity failed: '+str(error))
                for lane,running in enumerate(active):
                    left,right = fixed.lanes[lane],exact.lanes[lane]
                    if not torch.equal(left.base.generator.get_state(),right.base.generator.get_state()):
                        raise ValueError('Independent lane sampler streams diverged')
                    if running:
                        counts[lane] += 1
                        shape = history_shape(left,indices[lane],int(packet['episode_id'][lane]))
                        if shape != history_shape(right,indices[lane],int(packet['episode_id'][lane])):
                            raise ValueError('Complete recurrent history counts differ')
                        if not torch.equal(left.state['history_valid'],right.state['history_valid']):
                            raise ValueError('Backend history masks differ')
                        states=numeric_history_comparison(controller_state(left),controller_state(right))
                        histories.append(dict(lane=lane,**shape,numerical_state=states))
                        for controller in (left,right):
                            # Real recorded execution, explicitly teacher forced;
                            # predictions above are never claimed executed actions.
                            controller.base.command = target[lane:lane+1].clone()
                for name,controller in order:
                    controller.observe_execution(dict(position_applied_rad=target),active)
                    if controller.visual_frames != counts:
                        raise ValueError('Per-lane visual consumption differs from real active ticks')
                    for lane,running in enumerate(active):
                        state = controller_state(controller.lanes[lane])
                        if not running and not exact_state_equal(last_states[name,lane],state):
                            raise ValueError('Inactive lane history or sampler changed')
                        if running:
                            previous_target = controller.lanes[lane].base.position_previous
                            expected = (target[lane],target[lane],packet['time_s'][lane])
                            if not exact_state_equal(freeze(previous_target),freeze(expected)):
                                raise ValueError('Source execution acknowledgement was not retained exactly')
                        last_states[name,lane] = state
                for lane,running in enumerate(active):
                    if running:numeric_history_comparison(last_states['fixed',lane],last_states['exact',lane])
                acknowledgement_counts = counts.copy()
            rows.append(dict(tick=tick,active_lanes=sum(active),active=active,source_steps=indices,
                max_abs_command_rad=error,history=histories,consumed_observations=counts.copy(),
                actual_source_acknowledgements=acknowledgement_counts.copy(),first_execution_order=order[0][0]))
    except Exception as exc:
        caught = dict(type=type(exc).__name__,message=str(exc),tick=len(rows))
    reports = {}
    if caught is None:
        try:
            for name,controller in [('fixed',fixed),('exact',exact)]:
                report = controller.vision_execution_report()
                histogram = dict(controller.dino.kernel_image_batch_counts)
                expected = {'24':25} if name=='fixed' else {'24':8,'21':6,'15':6,'3':5}
                if histogram != expected or report['consumed_observations_per_lane'] != counts:
                    raise ValueError('Actual backbone shape or visual count differs from schedule')
                if report['real_lanes'] != sum(counts) or report['forward_features_calls'] != 25:
                    raise ValueError('Vision work denominator differs from complete real observations')
                if name=='exact' and (report['vision_tail_profile'] != EXACT_TAIL_PROFILE
                        or report['kernel_image_batch_counts'] != histogram or report['padding_lanes'] != 0
                        or report['processed_images'] != 3*sum(counts)):
                    raise ValueError('Exact tail padded or lost a real observation')
                if name=='fixed' and (report['padding_lanes'] != 8*25-sum(counts)
                        or report['processed_images'] != 3*8*25):
                    raise ValueError('Fixed M8 reference changed its original padding contract')
                reports[name] = dict(report,observed_backbone_kernel_image_batch_counts=histogram,
                    synchronized_backbone_timings=controller.dino.kernel_timings,
                    timing_scope='Isolated backbone CUDA intervals with explicit synchronization, excluding transfer, normalization, history and physics; not formal collection throughput')
        except Exception as exc:
            caught = dict(type=type(exc).__name__,message=str(exc),tick=len(rows))
    if caught is not None:
        for name,controller in [('fixed',fixed),('exact',exact)]:
            try:partial=controller.vision_execution_report()
            except Exception as exc:partial=dict(accounting_error=str(exc))
            reports[name]=dict(partial,partial=True,
                observed_backbone_kernel_image_batch_counts=dict(controller.dino.kernel_image_batch_counts),
                synchronized_backbone_timings=controller.dino.kernel_timings)
    return dict(passed=caught is None and len(rows)==TAIL_TICKS,rows=rows,failure=caught,
        target_max_abs_tolerance_rad=TARGET_ATOL,
        full_history_numeric_tolerance=dict(atol=STATE_ATOL,rtol=STATE_RTOL,discrete='exact'),
        max_abs_command_rad=max((row['max_abs_command_rad'] for row in rows),default=None),
        vision_execution=reports,consumed_observations_per_lane=counts,
        actual_source_acknowledgements_per_lane=acknowledgement_counts,
        scope='Same recorded training RGB-D, full DP plus backend, identical teacher-forced actual source target history; no physics, success-rate or speed claim')


def exact_tail_gate(root,source_root,output):
    from tools.tight_fresh_rollouts import completed_pair,read
    from tools.tight_fresh_forensics_gate import load_source
    from visuotactile.batched_controller import BatchedResidualController
    from visuotactile.models.native_vision import NativeDinoSpatial,EXACT_TAIL_PROFILE
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32,snapshot
    if output.exists():raise FileExistsError('Preserve exact-tail gate identity and evidence')
    session=read(root/'session.json');lease=lease_for(root,deadline=session['deadline'])
    pairs={role:completed_pair(source_root,role) for role in ('A','B')}
    datasets={role:read(source_root/role/'state_data/data.json') for role in pairs}
    selected={}
    for role,data in datasets.items():
        original=data['normalization_train_ids']
        if (len(original)!=30 or len(set(original))!=30 or not set(original)<=set(data['splits']['train'])
                or set(original)&set(data['splits'].get('validation',[])+data['splits'].get('test',[]))):
            raise ValueError('Original train30 only; no held-out source may enter this gate')
        selected[role]=sorted(original)[:TAIL_LANES]
        if any(data['entries'][i].get('trajectory_kind')=='expert_takeover_v1' or
                data['entries'][i].get('split','train')!='train' for i in selected[role]):
            raise ValueError('Eight original training trajectories required')
    output.mkdir(parents=True)
    registration=dict(version='fixed_m8_exact_tail_complete_student_gate_v1',pairs=pairs,
        source_root=str(source_root.resolve()),selected_training_ids=selected,lanes=TAIL_LANES,ticks=TAIL_TICKS,
        active_schedule=[dict(tick=t,active=tail_schedule(t)[0],steps=tail_schedule(t)[1]) for t in range(TAIL_TICKS)],
        vision_microbatch=8,vision_tail_profile=EXACT_TAIL_PROFILE,neural_batch=True,scheduler_index_mode='host',
        target_max_abs_tolerance_rad=TARGET_ATOL,full_history_numeric_tolerance=dict(atol=STATE_ATOL,rtol=STATE_RTOL),
        session_sha256=sha256(root/'session.json'),
        dataset_sha256={r:sha256(source_root/r/'state_data/data.json') for r in pairs},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ('tools','visuotactile') for p in sorted(Path(folder).rglob('*.py'))},
        test_used=False,simulator_started=False,utc=time.time())
    atomic_json(output/'registration.json',registration)
    began=time.perf_counter();runs=[];caught=None;arithmetic=None
    try:
        with lease.block():
            arithmetic=state_fp32()
            if not torch.cuda.is_available():raise ValueError('Actual CUDA is required; CPU mocks do not pass this gate')
            dino=NativeDinoSpatial().cuda().eval()
        for role in ('A','B'):
            (output/role).mkdir()
            loaded=[load_source(datasets[role]['entries'][identity],identity,TAIL_TICKS) for identity in selected[role]]
            metadata=copy.deepcopy(loaded[0][0])
            metadata['joint_limits_rad']=[copy.deepcopy(item[0]['joint_limits_rad'][0]) for item in loaded]
            if any(item[0]['action_control_mode']!=metadata['action_control_mode'] for item in loaded):
                raise ValueError('Offline target control metadata differs across lanes')
            saved=torch.load(pairs[role]['backend'],map_location='cpu',weights_only=False)
            with lease.block():
                fixed=BatchedResidualController(saved,metadata,KernelObservedDino(dino,cuda_timing=True),TAIL_LANES,
                    neural_batch=True,scheduler_index_mode='host',vision_microbatch=8)
                exact=BatchedResidualController(saved,metadata,KernelObservedDino(dino,cuda_timing=True),TAIL_LANES,
                    neural_batch=True,scheduler_index_mode='host',vision_microbatch=8,vision_tail_profile=EXACT_TAIL_PROFILE)
            result=run_tail_sequence(fixed,exact,[item[1] for item in loaded],lease=lease,cuda=True,exact_first=role=='B')
            result.update(role=role,pair=pairs[role],sources=[item[2] for item in loaded])
            atomic_json(output/role/'parity.json',result)
            runs.append(dict(role=role,passed=result['passed'],max_abs_command_rad=result['max_abs_command_rad'],
                parity_path=role+'/parity.json',parity_sha256=sha256(output/role/'parity.json')))
            del fixed,exact,saved,loaded
            if not result['passed']:raise ValueError('Complete exact-tail parity failed; inspect '+role+'/parity.json')
    except Exception as exc:
        caught=dict(type=type(exc).__name__,message=str(exc))
    report=dict(passed=caught is None and len(runs)==2 and all(r['passed'] for r in runs),
        registration_sha256=sha256(output/'registration.json'),runs=runs,failure=caught,
        seconds=time.perf_counter()-began,resource_wait_s=lease.wait_s,precision_before=arithmetic,precision_after=snapshot(),
        scope='Actual CUDA complete selected A/B students, recorded train-only observations, same source execution acknowledgements; no simulator, training, task success or throughput claim')
    atomic_json(output/'result.json',report)
    if not report['passed']:raise RuntimeError('Exact-tail gate did not pass: '+str(output/'result.json'))
    return report


def gate(root,output,*,role='A',count=2,neural_batch=False,vision_microbatch=2):
    from tools.tight_fresh_rollouts import completed_pair
    from visuotactile.native_controller import ResidualController
    from visuotactile.batched_controller import BatchedResidualController,lane_tree
    from visuotactile.models.native_vision import NativeDinoSpatial,validate_vision_microbatch
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32
    if output.exists():raise FileExistsError('Preserve gate evidence')
    vision_microbatch=validate_vision_microbatch(vision_microbatch)
    if role not in ('A','B') or type(count) is not int or count<2:raise ValueError('Registered A/B role and at least two lanes required')
    state_fp32();s=json.loads((root/'session.json').read_text());lease=lease_for(root,deadline=s['deadline'])
    pair=completed_pair(root,role);saved=torch.load(pair['backend'],weights_only=False,map_location='cpu')
    data=json.loads((root/role/'state_data/data.json').read_text());population=data['splits']['train']
    ids=[population[i%len(population)] for i in range(count)]
    sources=[]
    for identity in ids:
        e=data['entries'][identity]
        with h5py.File(e['path']) as f:
            g=f['episodes'][e['episode']];metadata=json.loads(f.attrs['metadata_json'])
            read=lambda group:{k:torch.from_numpy(v[:25]) for k,v in group.items()}
            sources.append(dict(obs=read(g['obs']),diag=read(g['transition/diagnostics_before']),
                image=torch.cat((torch.from_numpy(g['frames/rgb_u8'][:25]).float()/255,torch.from_numpy(g['frames/depth_m'][:25])),2),
                depth_valid=torch.from_numpy(g['frames/depth_valid'][:25]),
                target=torch.from_numpy(g['transition/target_after_rad'][:25])))
    source_control_mode=metadata['action_control_mode']
    # Offline controller comparison only: source expert labels were postprocessed
    # native targets. No simulator is present or being reconfigured by this gate.
    metadata['action_control_mode']='absolute_joint_position_target_rad'
    metadata['joint_limits_rad']=metadata['joint_limits_rad']*count
    rows=[]
    with lease.block():
        dino=NativeDinoSpatial().cuda().eval()
        batch=BatchedResidualController(saved,metadata,dino,count,neural_batch=neural_batch,scheduler_index_mode='host',
            vision_microbatch=vision_microbatch)
        serial=[ResidualController(saved,metadata,dino,scheduler_index_mode='host') for _ in ids]
    for tick in range(25):
        # Restart only lane0 at the 13th observation; lane1's prefix continues.
        indices=[tick if tick<13 else tick-13]+[tick]*(count-1)
        packets=[];diagnostics=[];targets=[]
        for lane,j in enumerate(indices):
            source=sources[lane];p={k:v[j:j+1].clone() for k,v in source['obs'].items()}
            if lane==0 and tick>=13:p['episode_id']+=100000
            p.update(image=source['image'][j:j+1],depth_valid=source['depth_valid'][j:j+1])
            packets.append(p);diagnostics.append({k:v[j:j+1] for k,v in source['diag'].items()});targets.append(source['target'][j:j+1])
        packet={k:torch.cat([p[k] for p in packets]) for k in packets[0]}
        diag={k:torch.cat([d[k] for d in diagnostics]) for k in diagnostics[0]}
        with lease.block():
            prediction=batch.act(packet,diag,[True]*count)
            reference=torch.cat([c.act(p,d) for c,p,d in zip(serial,packets,diagnostics)])
            error=float((prediction-reference).abs().max())
            if error>2e-5:raise ValueError('Real complete-policy lane parity failed: '+str(error))
            for lane in range(count):
                serial[lane].base.command=targets[lane]
                batch.lanes[lane].base.command=targets[lane]
                serial[lane].observe_execution(dict(position_applied_rad=targets[lane]))
            batch.observe_execution(dict(position_applied_rad=torch.cat(targets)),[True]*count)
        rows.append(dict(tick=tick,max_abs_command_rad=error,steps=indices))
    atomic_json(output,dict(passed=True,pair=pair,role=role,lanes=count,neural_batch=neural_batch,scheduler_index_mode='host',source_ids=ids,rows=rows,visual_frames=batch.visual_frames,
        vision_microbatch=vision_microbatch,serial_reference_vision_microbatch=2,
        vision_execution=batch.vision_execution_report(),
        source_control_mode=source_control_mode,offline_controller_mode=metadata['action_control_mode'],
        max_abs_command_rad=max(row['max_abs_command_rad'] for row in rows),
        scope='Real recorded RGB-D and expert actual-history acknowledgements; lane0 reset while lane1 continues; no simulation outcome or speed claim'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--role',choices=['A','B'],default='A');p.add_argument('--lanes',type=int,default=2)
    p.add_argument('--neural-batch',action='store_true')
    p.add_argument('--vision-microbatch',type=int,default=2)
    p.add_argument('--compare-exact-tail',action='store_true',help='Separate A/B eight-lane fixed-M8 versus exact-tail complete-student gate')
    p.add_argument('--source-root',type=Path,help='Required selected complete A/B source for --compare-exact-tail; --root retains the active resource lease')
    a=p.parse_args();torch.set_num_threads(2)
    if a.compare_exact_tail:
        if a.source_root is None:p.error('--compare-exact-tail requires --source-root')
        exact_tail_gate(a.root.resolve(),a.source_root.resolve(),a.output.resolve())
    else:
        if a.source_root is not None:p.error('--source-root requires --compare-exact-tail')
        gate(a.root,a.output,role=a.role,count=a.lanes,neural_batch=a.neural_batch,
            vision_microbatch=a.vision_microbatch)
