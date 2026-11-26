"""Persistent complete-student queue with native indexed reset and bounded capture.

Terminal records close asynchronously. A released lane receives the next registered
case immediately; A/B share DINO and group their neural work by frozen weights.
This changes batch RNG order, never claims exact hidden-state replay.
"""
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from .batched_rollout import (lane_metadata,finish_lane,trace_row,expert_actor_input,_PendingCapture)
from .batched_controller import lane_tree
from .batch_recording import snapshot,lane_arrays
from .gpu_batch_controller import MultiPolicyController,PROFILE as GPU_PROFILE
from .training_resources import atomic_json
from .recording import flatten,BufferedNumericWriter,post_observation

PROFILE='independent_lane_queue_v1'


def ready_reactivated_cameras(env, lanes, *, max_extra_renders=4):
    """Bounded Hydra warmup at the SAME physical state after product enable.

    Disabled products can return an empty annotator buffer on their first render
    after reactivation. Do not feed it to Camera's fixed-size tensor assignment,
    substitute a stale image, or advance physics to warm up the renderer.
    """
    lanes=list(lanes);stamp=int(env.sim.current_time_step_index)
    for extra in range(max_extra_renders+1):
        missing=[]
        for view,camera in enumerate(env.research_cameras):
            for name,annotators in camera._rep_registry.items():
                for lane in lanes:
                    output=annotators[lane].get_data()
                    raw=output.get('data') if isinstance(output,dict) else output
                    shape=getattr(raw,'shape',())
                    if not shape or int(np.prod(shape))==0:
                        missing.append((view,name,lane));continue
                    data,_=camera._process_annotator_output(name,output)
                    if tuple(data.shape)!=tuple(camera._data.output[name].shape[1:]):
                        missing.append((view,name,lane))
        if not missing:
            for camera in env.research_cameras:camera._is_outdated[lanes]=True
            return dict(lanes=lanes,extra_renders=extra,physics_step=stamp,physics_steps_added=0)
        if extra==max_extra_renders:
            raise RuntimeError('Reactivated camera buffers remained unavailable: '+str(missing))
        env.sim.render()
        if int(env.sim.current_time_step_index)!=stamp:
            raise RuntimeError('Camera warmup unexpectedly advanced physics')


def replace_reset_observations(previous,current,lanes,count):
    """Keep continuing lanes' already observed control-time sample exactly.

    Indexed reset needs an extra render, not a new control observation for peers.
    Only reset lanes consume that render; no continuing history frame is replaced.
    """
    if isinstance(current,dict):
        return {k:replace_reset_observations(previous[k],v,lanes,count) for k,v in current.items()}
    if torch.is_tensor(current) and current.ndim and current.shape[0]==count:
        result=previous.clone();result[lanes]=current[lanes];return result
    return current


def reset_expert_lanes(player,lanes,count):
    """Released expert LSTM history is independent on its second (batch) axis."""
    if not player.is_rnn:return
    if not player.states or any(s.ndim!=3 or s.shape[1]!=count for s in player.states):
        raise ValueError('Unexpected expert recurrent-state layout')
    for state in player.states:state[:,lanes]=0


def active_expert_action(player,actor,ids):
    """Only allocated lanes update the expert's recurrent state."""
    original=player.states
    if player.is_rnn:player.states=[s[:,ids].contiguous() for s in original]
    try:
        action=player.get_action(actor[ids],is_deterministic=True).clamp(-1,1)
        if player.is_rnn:
            for full,part in zip(original,player.states):full[:,ids]=part
    finally:player.states=original
    return action


def record_batch_transition(controller,transition,active,previous):
    """Validate the physical command/label chain once for all current lanes."""
    from .state_benchmark import record_dagger_transition
    ids=[i for i,v in enumerate(active) if v];n=controller.count
    def select(value):
        if isinstance(value,dict):return {k:select(v) for k,v in value.items()}
        return value[ids] if torch.is_tensor(value) and value.ndim and value.shape[0]==n else value
    tr=select(transition);cs=[controller.lanes[i] for i in ids]
    cat=lambda get:torch.cat([get(c).to(tr['target_after_rad'].device) for c in cs])
    base=SimpleNamespace(limits=torch.stack([c.base.limits for c in cs]))
    proxy=SimpleNamespace(base=base,residual=cs[0].residual,gpu_pipeline=True,
        nominal=cat(lambda c:c.nominal),command=cat(lambda c:c.command),
        residual_diagnostics={k:cat(lambda c:c.residual_diagnostics[k]) for k in
            ('applied_rad','execution_projection_rad','execution_preprojection_target_rad','execution_previous_target_rad')},
        action_scale_cpu=torch.stack([c.action_scale_cpu for c in cs]),
        residual_scales_cpu=torch.stack([c.residual_scales_cpu for c in cs]),observe_execution=lambda tr:None)
    prior=torch.cat([torch.zeros_like(c.nominal) if previous[i] is None else previous[i]
                     for i,c in zip(ids,cs)])
    record_dagger_transition(proxy,tr,prior)
    tr['dagger']['previous_nominal_valid']=torch.tensor([previous[i] is not None for i in ids],device=prior.device)
    result={}
    for j,(i,c) in enumerate(zip(ids,cs)):
        local=lane_tree(tr,j,len(ids));result[i]=local
        c.base.command=local['position_requested_rad'].clone();c.observe_execution(local)
        previous[i]=c.nominal.clone()
    return result


class RenderLanes:
    """Disable only unassigned standard products, restore all on every exit."""
    def __init__(self,env):
        self.handles=[];self.active=[True]*env.num_envs;self.changes=[];self.readers=[];self.pose_readers=[]
        self.env=env;self.pending_reactivation=set();self.reactivation_receipts=[]
        if env.cfg.student_obs.camera_backend!='standard':return
        from omni.replicator.core.scripts.utils import viewport_manager
        from .inactive_render_products import existing_product
        manager=viewport_manager.ViewportManager()
        for camera in env.research_cameras:
            if len(camera.render_product_paths)!=env.num_envs:raise ValueError('Expected lane products')
            self.handles.append([existing_product(manager,p).hydra_texture for p in camera.render_product_paths])
            # A disabled Hydra product has no new image to copy. The stock
            # sensor still visits every outdated lane, and can block in Warp
            # copying that disabled product (observed08 thread dump). Filter
            # the read as well as the render. Inactive buffers are never used
            # as observations or recordings; reactivation forces a fresh read.
            if not camera._data.output:raise RuntimeError('Lane filtering requires initialized camera buffers')
            original=camera._update_buffers_impl
            def update(env_ids,original=original):
                ids=env_ids.tolist() if torch.is_tensor(env_ids) else list(env_ids)
                ids=[i for i in ids if self.active[i]]
                if ids:original(ids)
            camera._update_buffers_impl=update
            self.readers.append((camera,original))
        for annotators in env._camera_pose_annotators:
            if len(annotators)!=env.num_envs:raise ValueError('Expected individual standard camera poses')
            for lane,annotator in enumerate(annotators):
                original=annotator.get_data;cached=[original()]
                def pose(*args,lane=lane,original=original,cached=cached,**kwargs):
                    if self.active[lane]:cached[0]=original(*args,**kwargs)
                    return cached[0]
                annotator.get_data=pose;self.pose_readers.append((annotator,original))
    def set(self,lane,enabled):
        if self.active[lane]==enabled:return
        for view in self.handles:
            view[lane].set_updates_enabled(enabled)
            if view[lane].get_updates_enabled()!=enabled:raise RuntimeError('Lane renderer setting failed')
        self.active[lane]=enabled;self.changes.append(dict(lane=lane,enabled=enabled,utc=time.time()))
        if enabled and self.handles:self.pending_reactivation.add(lane)
        else:self.pending_reactivation.discard(lane)
    def ready(self,lanes):
        pending=sorted(set(lanes)&self.pending_reactivation)
        if pending:
            self.reactivation_receipts.append(ready_reactivated_cameras(self.env,pending))
            self.pending_reactivation.difference_update(pending)
    def close(self):
        for lane in range(len(self.active)):self.set(lane,True)
        for camera,original in self.readers:camera._update_buffers_impl=original
        for annotator,original in self.pose_readers:annotator.get_data=original


@torch.no_grad()
def run_queue(env,player,lease,output,cases,pairs,saved,dino,plan,*,capture_pool=None):
    from .environment_rng import IsolatedEnvironment
    from .narrow_data import configure_direct_reset_lanes
    from .live_collision import clearance
    from .state_benchmark import dagger_configuration,record_dagger_transition
    from .takeover_trigger import make_trigger,observe_transition
    from .position_control import ExpertPositionBridge
    from .vector_position_bridge import VectorExpertPositionBridge
    from .async_recording import SharedCaptureWriter,AsyncLaneRecorder
    from tools.tight_fresh_stage_takeover import native_bound_guard
    if plan['batch'].get('gpu_pipeline')!=GPU_PROFILE:raise ValueError('Queue requires GPU batch profile')
    n=env.num_envs;capture=plan['kind']=='takeover';started=time.perf_counter()
    queue=list(cases);slots=[None]*n;active=[False]*n;finished=[];pending=[]
    rows=[[] for _ in range(n)];plans=[[] for _ in range(n)];previous=[None]*n
    directories=[None]*n;recorders=[None]*n;numerics=[None]*n;writers=[None]*n
    daggers=[None]*n;triggers=[None]*n;initials=[None]*n;lane_started=[None]*n
    lane_batches=[None]*n;assigned=[]
    batch=dict(version=PROFILE,lanes=n,vision_microbatch=plan['batch']['vision_microbatch'],
        physics_seed=plan['batch']['seed'],event_seed=plan['batch']['seed']+100000,
        native_rng='Persistent batch event stream; native indexed reset without reseeding other lanes; no hidden-state parity claim',
        gpu_pipeline=GPU_PROFILE)
    if hasattr(env,'_direct_lane_reset_original'):env._reset_idx=env._direct_lane_reset_original;del env._direct_lane_reset_original
    initial_cases=queue[:n];del queue[:n]
    configs=[c['spec']['narrow_reset'] for c in initial_cases]
    configure_direct_reset_lanes(env,configs+[configs[-1]]*(n-len(configs)))
    env.batch_static_calibration_cache=True;env._fixed_renderer_pose_cache=None
    isolated=IsolatedEnvironment(env,batch['physics_seed']);render=None;bridge=None;hook=None
    pool=capture_pool or (SharedCaptureWriter(max_pending_bytes=plan['batch']['writer_bytes'],
        verification_workers=plan['batch'].get('verification_workers',0),
        verification_max_pending=plan['batch'].get('verification_max_pending',4)) if capture else None)
    profile=dict(controller_s=0.,environment_s=0.,expert_s=0.,record_submit_s=0.,finish_audit_s=0.,
        reset_refill_s=0.,active_lane_steps=0,allocated_lane_steps=0,refills=0,finished_before_queue_empty=0)
    raw={}
    render_reuse=None
    if plan['batch'].get('render_profile'):
        from .current_render import PROFILE as RENDER_PROFILE,CurrentRenderReuse
        if plan['batch']['render_profile']!=RENDER_PROFILE:raise ValueError('Unknown render reuse profile')
        render_reuse=CurrentRenderReuse(env)
    try:
        with lease.block():
            env.clear_recorded_events();env.action_control_mode='native_command_clipped_pre_delay'
            teacher,_=isolated.reset(batch['physics_seed']);isolated.begin_events(batch['event_seed'])
            env.action_control_mode='absolute_joint_position_target_rad';player.reset()
            if capture:atomic_json(output/'queue_native_bound.json',native_bound_guard(env))
            metadata=env.research_metadata();metadata['config']=env.cfg.to_dict()
            metadata.update(env_origins_w=env.scene.env_origins.cpu().tolist(),expert_model_sha256=plan['expert_sha256'],
                inference_scheduler_index_mode='host')
            # Same physical executor for both matched complete policies.
            metadata['execution_adapter']=next(iter(saved.values()))['configuration']['execution_adapter']
            controller=MultiPolicyController(saved,metadata,dino,n,neural_batch=True,scheduler_index_mode='host',
                vision_microbatch=batch['vision_microbatch'],vision_tail_profile=plan['batch'].get('vision_tail_profile'))
            render=RenderLanes(env)
        if capture:
            bridge=VectorExpertPositionBridge(env,[10**9]*n,ExpertPositionBridge.speed_scheme)
            bridge.dynamic=True;env._dagger_position_bridge=bridge
            def capture_raw(module,inputs,result):raw['mean']=result['mus'].detach().clone()
            hook=player.model.register_forward_hook(capture_raw)
        def admit_lane(lane,case):
            directory=output/case['name'];directory.mkdir();directories[lane]=directory
            slots[lane]=case;controller.assign(lane,case['role']);lane_started[lane]=time.perf_counter()
            reset_expert_lanes(player,[lane],n)
            rows[lane]=[];plans[lane]=[];previous[lane]=None
            admission=clearance(env,lane=lane);initial=env._diagnostics()
            initials[lane]={key:value[lane:lane+1].cpu().tolist() for key,value in initial.items()}
            localbatch=dict(batch,lane=lane,queue_ordinal=len(assigned),global_start_tick=env.research_tick)
            assigned.append(case['name']);lane_batches[lane]=localbatch
            atomic_json(directory/'started.json',dict(spec=case['spec'],batch_execution=localbatch,utc=time.time()))
            atomic_json(directory/'initial_collision.json',admission)
            atomic_json(directory/'initial_observable.json',initials[lane])
            atomic_json(directory/'restoration.json',dict(kind='fresh_native_indexed_reset',**localbatch))
            dagger=dagger_configuration(case['dagger'],case['spec'],json.loads(Path(case['source_data']).read_text())) if capture else None
            daggers[lane]=dagger;triggers[lane]=make_trigger(dagger['trigger'],dagger['takeover_tick']) if dagger and dagger.get('trigger') else None
            localmeta=lane_metadata(metadata,lane,n,case,pairs[case['role']],dagger,localbatch)
            atomic_json(directory/'metadata.json',localmeta)
            if plan.get('record_sampler_rng'):
                from .sampler_audit import record as record_sampler_rng
                record_sampler_rng(directory,controller.lanes[lane].base,'start',identity=case['name'],
                    vision_arithmetic=plan.get('visual_arithmetic'))
            if not admission['legal']:
                result=dict(**case['spec'],software_complete=True,legal=False,success=False,not_started=True,
                    reason='pre-action collision rejection; no outcome replacement',batch_execution=localbatch)
                atomic_json(directory/'result.json',result);finished.append((case,result));active[lane]=False
                return
            active[lane]=True
            if capture:
                recorders[lane]=AsyncLaneRecorder(directory,localmeta,case['spec']['identity'],writer=pool,block_size=1,numeric_buffer_steps=8)
                bridge.replace_lane(lane,dagger['takeover_tick'])
            else:
                f=h5py.File(directory/'numeric.h5','x');f.attrs['metadata_json']=json.dumps(localmeta)
                numerics[lane]=f;writers[lane]=BufferedNumericWriter(f.create_group('episode'),32)
        for lane,case in enumerate(initial_cases):admit_lane(lane,case)
        bound=(int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)*len(cases)
        for global_tick in range(bound):
            # A different registered task is not a retry of the preceding outcome.
            refill=[i for i,running in enumerate(active) if not running][:len(queue)]
            if refill:
                t=time.perf_counter()
                replacements={lane:queue.pop(0) for lane in refill}
                for lane,case in replacements.items():
                    env._direct_lane_reset_configurations[lane]=copy.deepcopy(case['spec']['narrow_reset']);render.set(lane,True)
                with lease.block():
                    continuing_packet=copy.deepcopy(env.research_packet);continuing_teacher=copy.deepcopy(teacher)
                    isolated.reset_lanes(refill);env.scene.write_data_to_sim();env.sim.forward();env.sim.render()
                    env.scene.update(0.0)
                    render.ready(refill)
                    with isolated.stream.scope():teacher=env._get_observations()
                    env.research_packet=replace_reset_observations(continuing_packet,env.research_packet,refill,n)
                    teacher=replace_reset_observations(continuing_teacher,teacher,refill,n)
                    for lane,case in replacements.items():admit_lane(lane,case)
                profile['reset_refill_s']+=time.perf_counter()-t;profile['refills']+=len(refill)
                if not any(active) and queue:continue
            if not any(active):break
            # A rejected replacement can leave a lane vacant for just one tick.
            # With queued work it will be refilled on the next loop: retiring
            # its Hydra product here creates avoidable disable/enable churn.
            # Once the finite task queue is empty, retire unused products.
            for lane,running in enumerate(active):render.set(lane,running or bool(queue))
            packet=env.research_packet
            before_active=list(active);local_ticks=[len(row) for row in rows]
            with lease.block():
                t=time.perf_counter();actor=expert_actor_input(teacher,player)
                expert=torch.zeros(n,29,device=actor.device)
                if capture:
                    ids=[i for i,running in enumerate(active) if running]
                    expert[ids]=active_expert_action(player,actor,ids)
                    raw_batch=torch.zeros_like(expert);raw_batch[ids]=raw['mean'];raw['mean']=raw_batch
                profile['expert_s']+=time.perf_counter()-t
                t=time.perf_counter();commands=controller.act(packet,env._diagnostics(),active)
                profile['controller_s']+=time.perf_counter()-t
                if bridge is not None:bridge.submit(raw['mean'],expert,bridge.tick,active)
                before_rng=isolated.stream.state[2].clone()
                before_cuda=isolated.stream.state[3]['state'].clone() if len(isolated.stream.state)>3 else None
                t=time.perf_counter();teacher,_,terminated,truncated,_=isolated.step(commands)
                profile['environment_s']+=time.perf_counter()-t
                tr=env.last_transition
                tr.update(expert_actor_input_diagnostic=actor.clone(),expert_shadow_command=expert.clone(),
                    torch_rng_before_step=before_rng[None].expand(n,-1).clone())
                if before_cuda is not None:tr['torch_cuda_rng_before_step']=before_cuda[None].expand(n,-1).clone()
                if capture:tr['expert_raw_policy_mean_before_clip']=raw['mean']
                locals_=record_batch_transition(controller,tr,active,previous) if capture else {}
                for lane,running in enumerate(active):
                    if not running:continue
                    c=controller.lanes[lane]
                    local=locals_[lane] if capture else lane_tree(tr,lane,n);locals_[lane]=local
                    if capture:
                        if triggers[lane] is not None and not bool(local['dagger']['control_source'][0]):
                            schedule=observe_transition(triggers[lane],local_ticks[lane],local)
                            if schedule is not None:bridge.schedule_takeover(lane,schedule)
                    else:c.observe_execution(local)
                t=time.perf_counter()
                if capture:
                    template=next(iter(locals_.values()))['dagger']
                    tr['dagger']={k:torch.cat([locals_[i]['dagger'][k] if i in locals_ else torch.zeros_like(v) for i in range(n)]) for k,v in template.items()}
                def combine(get):
                    template=get(controller.lanes[next(iter(locals_))])
                    return torch.cat([get(controller.lanes[i]) if i in locals_ else torch.zeros_like(template) for i in range(n)])
                diagkeys=controller.lanes[next(iter(locals_))].residual_diagnostics
                diag={k:combine(lambda c:c.residual_diagnostics[k].reshape(1) if c.residual_diagnostics[k].ndim==0 else c.residual_diagnostics[k]) for k in diagkeys}
                shared=snapshot(dict(obs={k:v for k,v in packet.items() if torch.is_tensor(v)},
                    transition=dict(tr,**(dict(observation_after=post_observation(env.research_packet,terminated|truncated)) if capture else {})),
                    terminated=terminated,truncated=truncated,controller=dict(nominal=combine(lambda c:c.nominal),
                        delta=combine(lambda c:c.delta),residual_diagnostics=diag)))
                profile['record_submit_s']+=time.perf_counter()-t
                reasons_cpu={k:v.cpu().tolist() for k,v in tr['termination_reasons'].items()}
                done=(terminated|truncated).cpu().tolist()
                actor_cpu=actor.cpu().numpy();expert_cpu=expert.cpu().numpy()
                for lane,running in enumerate(active):
                    if not running:continue
                    c=controller.lanes[lane];case=slots[lane];tick=local_ticks[lane]
                    local=lane_arrays(shared,lane,n);cpu_c=SimpleNamespace(**local.pop('controller'),contact_gate=c.contact_gate)
                    localtr=local['transition']
                    if tick%12==0:plans[lane].append(c.plan[0].cpu().numpy().copy())
                    rows[lane].append(trace_row(cpu_c,localtr,actor_cpu[lane:lane+1],expert_cpu[lane:lane+1]))
                    numeric=dict(obs={k:v for k,v in local['obs'].items() if k not in ('image','depth_m','depth_valid')},
                        transition={k:v for k,v in localtr.items() if k!='observation_after'})
                    t=time.perf_counter()
                    if capture:recorders[lane].submit(numeric,{k:v[0] for k,v in flatten(local).items()},tick)
                    else:writers[lane].append(numeric,tick)
                    profile['record_submit_s']+=time.perf_counter()-t
                    reasons={k:v[lane] for k,v in reasons_cpu.items()}
                    cutoff=case.get('grasp_evaluation_ticks')
                    censored=not done[lane] and cutoff is not None and tick+1>=cutoff
                    if done[lane] or censored:
                        if plan.get('record_sampler_rng'):
                            from .sampler_audit import record as record_sampler_rng
                            record_sampler_rng(directories[lane],c.base,'end',identity=case['name'],
                                vision_arithmetic=plan.get('visual_arithmetic'))
                        active[lane]=False
                        if controller.visual_frames[lane]!=len(rows[lane]):raise ValueError('Lost real camera frame')
                        if reasons['max_successes'] and not bool(terminated[lane]):raise ValueError('Native success mismatch')
                        t=time.perf_counter()
                        result=finish_lane(directories[lane],case,pairs[case['role']],saved[case['role']],rows[lane],plans[lane],
                            reasons,initials[lane],c,recorders[lane],numerics[lane],writers[lane],daggers[lane],triggers[lane],
                            lane_batches[lane],lane_started[lane],defer_capture=capture)
                        if censored:
                            result.update(grasp_stage_censored=True,full_task_evaluation=False,
                                termination_scope='Registered grasp observation horizon; not a native timeout/failure')
                            atomic_json(directories[lane]/'result.json',result)
                        if isinstance(result,_PendingCapture):pending.append((case,result))
                        else:finished.append((case,result))
                        profile['finish_audit_s']+=time.perf_counter()-t
                        if queue:profile['finished_before_queue_empty']+=1
                profile['active_lane_steps']+=sum(before_active);profile['allocated_lane_steps']+=n
            # Finalization is admitted only when already verified; never stall refill.
            remaining=[]
            for case,result in pending:
                if result.ticket.done():
                    t=time.perf_counter();finished.append((case,result.finalize()));profile['finish_audit_s']+=time.perf_counter()-t
                else:remaining.append((case,result))
            pending=remaining
            if global_tick%25==0:atomic_json(output/'batch_progress.json',dict(tick=global_tick,active=active,
                active_count=sum(active),remaining_tasks=len(queue),closed_results=len(finished),verifying=len(pending),
                active_lane_steps=profile['active_lane_steps'],utc=time.time()))
        if queue or any(active):raise RuntimeError('Registered natural bounds exhausted')
        for case,result in pending:
            t=time.perf_counter();finished.append((case,result.finalize()));profile['finish_audit_s']+=time.perf_counter()-t
        if pool is not None:profile['capture_writer']=pool.statistics()
        profile.update(seconds=time.perf_counter()-started,cases=len(cases),control_ticks=profile['active_lane_steps'],
            vision_execution=controller.vision_execution_report(),render_transitions=render.changes,
            camera_reactivation=render.reactivation_receipts)
        profile['aggregate_control_ticks_per_second']=profile['control_ticks']/profile['seconds']
        profile['active_lane_fraction']=profile['active_lane_steps']/max(1,profile['allocated_lane_steps'])
        if render_reuse is not None:profile['render_reuse']=dict(render_reuse.stats)
        return finished,profile
    finally:
        if render_reuse is not None:render_reuse.close()
        if hook is not None:hook.remove()
        if bridge is not None and getattr(env,'_dagger_position_bridge',None) is bridge:del env._dagger_position_bridge
        if render is not None:render.close()
        if pool is not None:pool.close()
        for writer,f in zip(writers,numerics):
            if f is not None and f.id.valid:writer.flush();f.close()
