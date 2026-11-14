"""Whole-wave complete-student execution with separately audited lane records.

The environment batch is a scheduling choice, not a new task or an exact replay
of serial event RNG. No lane is recycled until all registered lanes terminate.
"""
import copy
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from .batched_controller import lane_tree, BatchedResidualController
from .compact import sha256
from .recording import flatten, numpy_tree, BufferedNumericWriter, post_observation
from .training_resources import atomic_json


def waves(cases, lanes):
    if isinstance(lanes, bool) or not isinstance(lanes, int) or lanes < 1:
        raise ValueError('Positive explicit concurrency required')
    names = [case['name'] for case in cases]
    if not names or len(names) != len(set(names)) or any(Path(n).name != n for n in names):
        raise ValueError('Unique local case identities required')
    # Only identical weights share neural kernels. Preserve each role's order.
    for role in dict.fromkeys(case['role'] for case in cases):
        selected = [case for case in cases if case['role'] == role]
        for start in range(0, len(selected), lanes):
            yield selected[start:start+lanes]


def lane_metadata(metadata, lane, count, case, pair, dagger, batch):
    value = copy.deepcopy(metadata)
    if not 0 <= lane < count:
        raise ValueError('Metadata lane outside registered wave')
    def singleton(container, key):
        if len(container[key]) != count:
            raise ValueError('Metadata lane census mismatch: '+key)
        container[key] = [container[key][lane]]
    for key in ('joint_limits_rad', 'env_origins_w'):
        if key in value:
            singleton(value, key)
    for camera in value.get('cameras', []):
        for key in ('pos_env', 'quat_ros_wxyz', 'intrinsics'):
            if key in camera:
                singleton(camera, key)
    for identity in ('camera_calibration_bindings', 'tiledcamera_calibration_bindings'):
        for binding in value.get(identity, []):
            for key in ('camera_paths', 'render_products'):
                singleton(binding, key)
            if 'image_render_products' in binding:
                products = binding['image_render_products']
                if len(products) == count:
                    singleton(binding, 'image_render_products')
                elif len(products) != 1:
                    raise ValueError('Image render products require lane or shared tiled binding')
                binding['image_render_products_scope'] = (
                    'Shared tiled image product; camera_paths/render_products select this lane'
                    if count > 1 and len(products) == 1 else 'This lane image product')
    value.update(initial_family=case['spec']['family'], dataset_split=case['spec']['split'],
        policy_checkpoint_sha256=pair['backend_sha256'], batch_execution=batch)
    if dagger is not None:
        value.update(dagger=dagger, trajectory_kind='expert_takeover_v1',
            source_identity=case['spec']['source_identity'], complete_student_policy=pair)
        if case['spec'].get('training_expansion') is not None:
            value.update(training_expansion=copy.deepcopy(case['spec']['training_expansion']),
                initial_configuration_sha256=case['spec']['initial_configuration_sha256'])
    return value


def trace_row(controller, transition, actor, expert):
    take = lambda value: value[0].detach().cpu().numpy().copy()
    after = transition['diagnostics_after']
    diag = controller.residual_diagnostics
    row = dict(actor=take(actor), command=take(transition['position_requested_rad']),
        expert_shadow=take(expert), delayed=take(transition['position_applied_rad']),
        delay_index=take(transition['control_trace']['delay_index']),
        target=take(transition['target_after_rad']), joint=take(after['joint_position_rad']),
        actuator_targets=take(transition['actuator_substep_target_rad']),
        velocity=take(after['joint_velocity_rad_s']), object=take(after['object_pose_w']),
        palm=take(after['palm_body_pose_w']), contact=take(transition['contact_substeps_after_w_N']),
        object_contact=take(transition['object_contact_force_w']),
        force=take(transition['applied_wrench']['force_w_N']), torque=take(transition['applied_wrench']['torque_w_Nm']),
        lifted=take(after['lifted_object']), subgoals=take(after['successes']),
        retract=take(after['retract_succeeded']), alignment=take(transition['native_alignment_condition']),
        nominal=take(controller.nominal), residual_normalized=take(controller.delta),
        residual_contact_gate=np.asarray(controller.contact_gate),
        residual_contact_valid=np.asarray(bool(diag['contact_valid'])),
        fingertip_distance=np.linalg.norm(take(after['finger_position_w'])-take(after['object_pose_w'])[:3],axis=-1).min())
    for target, source in [('residual_proposed_rad','proposed_rad'),('residual_applied_rad','applied_rad'),
                           ('residual_clip_mask','clip_mask'),('residual_tanh_fraction','tanh_fraction')]:
        row[target] = take(diag[source])
    for key in ('execution_projection_rad','execution_preprojection_target_rad','execution_previous_target_rad'):
        if key in diag: row[key] = take(diag[key])
    if 'dagger' in transition:
        for key in ('control_source','segment_id','expert_label_valid','switch_clean','expert_source_step',
                    'expert_delay_index','required_expert_residual_bound_fraction','required_expert_residual_out_of_bounds'):
            row['dagger_'+key] = take(transition['dagger'][key])
    return row


def expert_actor_input(teacher, player):
    """Bridge the simulator observation device to the frozen expert device.

    The released expert remains on CPU even when physics uses CUDA. Its block
    embedding is already on the player device; raw environment observations
    bypass the RL wrapper and therefore require this explicit transfer.
    """
    embedding = player.intr_reward_coef_embd
    return torch.cat((teacher['policy'].to(embedding.device), embedding), -1)


def _publish_result(directory, result, elapsed):
    result['wall_profile']=dict(through_result_s=time.perf_counter()-elapsed,
        scope='Overlapping lane lifetime, including final drain and validation; use wave wall for throughput')
    atomic_json(directory/'result.json',result)
    return result


class _PendingCapture:
    """A terminated lane can drain while other lanes continue real control."""
    def __init__(self, directory, result, arrays, ticket, elapsed):
        self.directory, self.result, self.arrays = directory, result, arrays
        self.ticket, self.elapsed = ticket, elapsed

    def finalize(self):
        from .state_benchmark import dagger_recovery_coverage
        capture=self.ticket.result()
        with h5py.File(self.directory/'numeric.h5','r') as stream:
            self.result['dagger']['recovery_coverage']=dagger_recovery_coverage(self.arrays,stream['episode'])
        self.result['dataset_capture']=capture['dataset_capture']
        self.result['async_capture']=capture
        return _publish_result(self.directory,self.result,self.elapsed)


def finish_lane(directory, case, pair, saved, rows, plans, reasons, initial, controller,
                recorder, numeric, writer, dagger, trigger, batch, elapsed, *, defer_capture=False):
    from .native_evaluation import stable_grasp
    from .state_benchmark import residual_episode_metrics, dagger_recovery_coverage
    arrays = {key:np.stack([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(directory/'trace.npz', **arrays)
    np.savez_compressed(directory/'plans.npz', plans=np.stack(plans))
    count = len(rows)
    result = dict(**case['spec'], software_complete=True, legal=True, success=reasons['max_successes'],
        reasons=reasons, steps=count, native_lift=bool(arrays['lifted'].any()),
        native_subgoals=int(arrays['subgoals'].max()), native_alignment=bool(arrays['alignment'].any()),
        checkpoint_sha256=pair['backend_sha256'], complete_student_policy=pair,
        simulation_step_hz=60, physics_hz=120, plan_hz=5, residual_hz=60,
        action_control_mode='absolute_joint_position_target_rad', action_exhausted=False,
        ever_complete_success=bool(reasons['max_successes']), first_success_tick=count if reasons['max_successes'] else None,
        termination_tick=count, duration_sim_s=count/60., post_success_actions=0,
        tick_index='1-based completed control transitions', batch_execution=batch,
        actuator_target_max_error=float(np.abs(arrays['actuator_targets']-arrays['target'][:,None]).max()),
        submitted_target_max_error=float(np.abs(arrays['command']-arrays['target']).max()),
        tracking_rmse_rad=float(np.sqrt(np.mean((arrays['joint']-arrays['target'])**2))),
        residual_version=saved['residual_version'], residual_ticks=count,
        contact_gate_ticks=int(arrays['residual_contact_gate'].sum()),
        contact=stable_grasp(arrays,float(initial['object_pose_w'][0][2])),
        residual_diagnostics=residual_episode_metrics(arrays,output_contract=saved['configuration'].get('output_contract')),
        expert_shadow_computed=dagger is not None, expert_shadow_placeholder_zero=dagger is None)
    timing = controller.inference_metrics()
    atomic_json(directory/'native_inference_timing.json',timing)
    result['native_inference'] = {key:timing[key] for key in
        ('summary','pending_cuda_events','timing_scope','wall_scope','excludes')}
    if dagger is not None:
        expert_ticks = int(arrays['dagger_control_source'].sum())
        valid = arrays['dagger_expert_label_valid'].astype(bool)
        result['dagger'] = dict(configuration=dagger, expert_shadow_queries=count,
            student_ticks=count-expert_ticks, expert_ticks=expert_ticks, takeover_triggered=expert_ticks>0,
            expert_valid_ticks=int(valid.sum()), expert_switch_masked_ticks=expert_ticks-int(valid.sum()),
            student_autonomous_success=bool(result['success'] and expert_ticks==0),
            expert_takeover_success=bool(result['success'] and expert_ticks>0),
            position_executor_continuous=True, provenance_passed=True, autonomous_evaluation=False,
            admission='Only verified successful real expert suffixes enter training')
        if trigger is not None: result['dagger']['trigger_receipt']=trigger.receipt()
        result.update(residual_forward_ticks=count,residual_executed_ticks=count-expert_ticks)
        # Preserve execution independently of writer verification/admission.
        # Coverage requires closed numeric fields. The compact file embeds this
        # factual outcome, never a premature admission to the training pool.
        atomic_json(directory/'execution_complete.json',result)
        if defer_capture:
            return _PendingCapture(directory,result,arrays,recorder.finish_async(result),elapsed)
        capture = recorder.finish(result)
        with h5py.File(directory/'numeric.h5','r') as stream:
            result['dagger']['recovery_coverage']=dagger_recovery_coverage(arrays,stream['episode'])
        result['dataset_capture']=capture['dataset_capture']
        result['async_capture']=capture
    else:
        writer.flush(); numeric.close()
    return _publish_result(directory,result,elapsed)


@torch.no_grad()
def run_wave(env, player, lease, output, cases, pair, saved, dino, plan, wave_index):
    from .environment_rng import IsolatedEnvironment
    from .narrow_data import configure_direct_reset_lanes
    from .live_collision import clearance
    from .state_benchmark import dagger_configuration, record_dagger_transition
    from .takeover_trigger import make_trigger,observe_transition
    from .position_control import ExpertPositionBridge
    from .batched_position_bridge import BatchedExpertPositionBridge
    from .async_recording import SharedCaptureWriter, AsyncLaneRecorder
    n=env.num_envs; capture=plan['kind']=='takeover'; start=time.perf_counter()
    if not cases or len(cases)>n or len({case['role'] for case in cases})!=1:
        raise ValueError('One nonempty same-policy wave within allocated lanes is required')
    if capture:
        from tools.tight_fresh_stage_takeover import native_bound_guard
        bound_receipt=native_bound_guard(env)
        atomic_json(output/f'wave_{wave_index:04d}_native_bound.json',bound_receipt)
    if hasattr(env,'_direct_lane_reset_original'):
        env._reset_idx=env._direct_lane_reset_original;del env._direct_lane_reset_original
    configs=[c['spec']['narrow_reset'] for c in cases]
    configure_direct_reset_lanes(env,configs+[configs[-1]]*(n-len(cases)))
    batch=dict(version='whole_wave_complete_student_v1',wave=wave_index,lanes=n,
        vision_microbatch=plan['batch'].get('vision_microbatch',2),
        physics_seed=plan['batch']['seed']+wave_index,event_seed=plan['batch']['seed']+100000+wave_index,
        native_rng='One batch stream; independent policy generators and lane histories; no serial hidden-state equivalence')
    # The opt-in backend provides CUDA stream isolation when requested.
    isolated=IsolatedEnvironment(env,batch['physics_seed'])
    with lease.block():
        env.clear_recorded_events();env.action_control_mode='native_command_clipped_pre_delay'
        teacher,_=isolated.reset(batch['physics_seed']);isolated.begin_events(batch['event_seed'])
        env.action_control_mode='absolute_joint_position_target_rad';player.reset()
        initial=env._diagnostics();metadata=env.research_metadata();metadata['config']=env.cfg.to_dict()
        metadata.update(env_origins_w=env.scene.env_origins.cpu().tolist(),
            expert_model_sha256=plan['expert_sha256'],inference_scheduler_index_mode='host')
        if saved['configuration'].get('execution_adapter'):
            metadata['execution_adapter']=saved['configuration']['execution_adapter']
        controller=BatchedResidualController(saved,metadata,dino,n,neural_batch=True,scheduler_index_mode='host',
            vision_microbatch=batch['vision_microbatch'])
        active=[clearance(env,lane=i)['legal'] if i<len(cases) else False for i in range(n)]
    daggers=[];triggers=[];directories=[];recorders=[];numerics=[];writers=[]
    rows=[[] for _ in range(n)];plans=[[] for _ in range(n)];previous=[None]*n;results={}
    pool=SharedCaptureWriter(max_pending_bytes=plan['batch']['writer_bytes'],
        verification_workers=plan['batch'].get('verification_workers',0),
        verification_max_pending=plan['batch'].get('verification_max_pending',4)) if capture else None
    raw={};hook=None;bridge=None
    profile=dict(controller_s=0.,environment_s=0.,expert_s=0.,record_submit_s=0.,finish_audit_s=0.)
    try:
        for lane,case in enumerate(cases):
            directory=output/case['name'];directory.mkdir();directories.append(directory)
            atomic_json(directory/'started.json',dict(spec=case['spec'],batch_execution=dict(batch,lane=lane),utc=time.time()))
            admission=clearance(env,lane=lane)
            atomic_json(directory/'initial_collision.json',admission)
            local_initial={key:value[lane:lane+1].cpu().tolist() for key,value in initial.items()}
            atomic_json(directory/'initial_observable.json',local_initial)
            atomic_json(directory/'restoration.json',dict(kind='fresh_native_batch_reset',**batch))
            dagger=dagger_configuration(case['dagger'],case['spec'],json.loads(Path(case['source_data']).read_text())) if capture else None
            daggers.append(dagger);triggers.append(make_trigger(dagger['trigger'],dagger['takeover_tick']) if dagger and dagger.get('trigger') else None)
            local_metadata=lane_metadata(metadata,lane,n,case,pair,dagger,dict(batch,lane=lane))
            atomic_json(directory/'metadata.json',local_metadata)
            if not active[lane]:
                result=dict(**case['spec'],software_complete=True,legal=False,success=False,not_started=True,
                    reason='pre-action collision rejection; no outcome replacement',batch_execution=dict(batch,lane=lane))
                atomic_json(directory/'result.json',result);results[lane]=result
                recorders.append(None);numerics.append(None);writers.append(None);continue
            if capture:
                recorders.append(AsyncLaneRecorder(directory,local_metadata,case['spec']['identity'],writer=pool,
                    block_size=1,numeric_buffer_steps=8))
                numerics.append(None);writers.append(None)
            else:
                f=h5py.File(directory/'numeric.h5','x');f.attrs['metadata_json']=json.dumps(local_metadata)
                numerics.append(f);writers.append(BufferedNumericWriter(f.create_group('episode'),32));recorders.append(None)
        if capture:
            bridge=BatchedExpertPositionBridge(env,[d['takeover_tick'] for d in daggers]+[10**9]*(n-len(cases)),ExpertPositionBridge.speed_scheme)
            env._dagger_position_bridge=bridge
            def capture_raw(module,inputs,result):raw['mean']=result['mus'].detach().clone()
            hook=player.model.register_forward_hook(capture_raw)
        bound=(int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
        for tick in range(bound):
            if not any(active):break
            packet=env.research_packet
            with lease.block():
                t=time.perf_counter()
                actor=expert_actor_input(teacher,player)
                expert=player.get_action(actor,is_deterministic=True).clamp(-1,1) if capture else torch.zeros((n,29),device=actor.device)
                profile['expert_s']+=time.perf_counter()-t
                t=time.perf_counter();commands=controller.act(packet,env._diagnostics(),active)
                profile['controller_s']+=time.perf_counter()-t
                if bridge is not None:bridge.submit(raw['mean'],expert,tick,active)
                before_rng=isolated.stream.state[2].clone()
                before_cuda=(isolated.stream.state[3]['state'].clone() if len(isolated.stream.state)>3 else None)
                t=time.perf_counter();teacher,_,terminated,truncated,_=isolated.step(commands)
                profile['environment_s']+=time.perf_counter()-t
                transition=env.last_transition
                transition.update(expert_actor_input_diagnostic=actor.clone(),expert_shadow_command=expert.clone(),
                    torch_rng_before_step=before_rng[None].expand(n,-1).clone())
                if before_cuda is not None:
                    transition['torch_cuda_rng_before_step']=before_cuda[None].expand(n,-1).clone()
                if capture:transition['expert_raw_policy_mean_before_clip']=raw['mean']
                after_packet=post_observation(env.research_packet,terminated|truncated) if capture else None
                for lane,running in enumerate(active):
                    if not running:continue
                    c=controller.lanes[lane];tr=lane_tree(transition,lane,n);p=lane_tree(packet,lane,n)
                    if capture:
                        previous[lane]=record_dagger_transition(c,tr,previous[lane])
                        if triggers[lane] is not None and not bool(tr['dagger']['control_source'][0]):
                            schedule=observe_transition(triggers[lane],tick,tr)
                            if schedule is not None:bridge.schedule_takeover(lane,schedule)
                    else:c.observe_execution(tr)
                    if tick%12==0:plans[lane].append(c.plan[0].cpu().numpy().copy())
                    rows[lane].append(trace_row(c,tr,actor[lane:lane+1],expert[lane:lane+1]))
                    numeric=dict(obs={k:v for k,v in p.items() if torch.is_tensor(v) and k not in ('image','depth_m','depth_valid')},transition=tr)
                    t=time.perf_counter()
                    if capture:
                        full_tr=dict(tr,observation_after=lane_tree(after_packet,lane,n))
                        compact=flatten(numpy_tree(dict(obs={k:v for k,v in p.items() if torch.is_tensor(v)},transition=full_tr,
                            terminated=terminated[lane:lane+1],truncated=truncated[lane:lane+1])))
                        recorders[lane].submit(numpy_tree(numeric),{k:v[0] for k,v in compact.items()},tick)
                    else:writers[lane].append(numeric,tick)
                    profile['record_submit_s']+=time.perf_counter()-t
                    reasons={k:bool(v[lane]) for k,v in transition['termination_reasons'].items()}
                    if reasons['max_successes'] and not bool(terminated[lane]):raise ValueError('Native success/termination mismatch')
                    if bool((terminated|truncated)[lane]):
                        active[lane]=False
                        if controller.visual_frames[lane]!=len(rows[lane]):raise ValueError('Dropped per-tick camera input')
                        t=time.perf_counter()
                        results[lane]=finish_lane(directories[lane],cases[lane],pair,saved,rows[lane],plans[lane],reasons,
                            json.loads((directories[lane]/'initial_observable.json').read_text()),c,recorders[lane],
                            numerics[lane],writers[lane],daggers[lane],triggers[lane],dict(batch,lane=lane),start,
                            defer_capture=capture)
                        profile['finish_audit_s']+=time.perf_counter()-t
            if tick%25==0:atomic_json(output/'batch_progress.json',dict(wave=wave_index,tick=tick,active=active,utc=time.time()))
        if any(active):raise RuntimeError('Native natural termination bound exhausted')
        for lane,result in list(results.items()):
            if isinstance(result,_PendingCapture):
                t=time.perf_counter();results[lane]=result.finalize()
                profile['finish_audit_s']+=time.perf_counter()-t
        if pool is not None:
            profile['capture_writer']=dict(pool.statistics(),
                stage='all_started_capture_lanes_verified_before_close',
                queue_scope='Shared writer jobs, not trajectories or observations; excludes the currently executing job',
                memory_scope='Owned payload and retained buffers, not process RAM; excludes source inputs and HDF5/compressor scratch')
        profile.update(seconds=time.perf_counter()-start,control_ticks=sum(len(r) for r in rows),cases=len(cases),
            vision_execution=controller.vision_execution_report())
        profile['aggregate_control_ticks_per_second']=profile['control_ticks']/profile['seconds']
        return [(case,results[lane]) for lane,case in enumerate(cases)],profile
    finally:
        if hook is not None:hook.remove()
        if bridge is not None and getattr(env,'_dagger_position_bridge',None) is bridge:del env._dagger_position_bridge
        if pool is not None:pool.close()
        for writer,f in zip(writers,numerics):
            if f is not None and f.id.valid:
                writer.flush();f.close()
