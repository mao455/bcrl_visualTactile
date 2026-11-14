"""Finite multi-environment complete-student development and actual throughput."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read, completed_pair


def probe_options(max_control_ticks=None, roles=None):
    if max_control_ticks is not None and (type(max_control_ticks) is not int or max_control_ticks<1):
        raise ValueError('Diagnostic max_control_ticks must be a positive integer')
    roles=['A','B'] if roles is None else list(roles)
    if not roles or len(roles)!=len(set(roles)) or any(role not in ('A','B') for role in roles):
        raise ValueError('Choose each explicit A/B role at most once')
    return dict(max_control_ticks=max_control_ticks,roles=roles,diagnostic_only=max_control_ticks is not None)


def execution_limit(native_bound,max_control_ticks):
    probe_options(max_control_ticks)
    if type(native_bound) is not int or native_bound<1:raise ValueError('Positive native termination bound required')
    return native_bound if max_control_ticks is None else min(native_bound,max_control_ticks)


def lane_outcome(*,legal,steps,reasons,still_active,max_control_ticks,native_bound):
    """A diagnostic cutoff is neither native timeout nor an autonomy outcome."""
    limit=execution_limit(native_bound,max_control_ticks)
    if still_active and (max_control_ticks is None or limit==native_bound):
        raise ValueError('Native natural-termination bound exhausted')
    if still_active and (not legal or reasons is not None or steps!=limit):
        raise ValueError('Diagnostic cutoff must follow every registered control tick')
    diagnostic=max_control_ticks is not None
    observed=bool(reasons and reasons.get('max_successes'))
    return dict(success=None if diagnostic else observed,observed_native_completion=observed,
        diagnostic_only=diagnostic,autonomous_evaluation_eligible=bool(legal and not diagnostic),
        accepted_for_training=False,censored=bool(still_active),
        censor_reason='diagnostic_control_limit' if still_active else None,
        naturally_ended=bool(reasons is not None),max_control_ticks=max_control_ticks,
        termination_classification=('invalid_initialization' if not legal else
            'diagnostic_censored' if still_active else 'native_natural_termination'))


def inference_counts(report,steps,visual_frames,*,strict=False):
    counts={}
    for row in report['rows']:counts[row['stage']]=counts.get(row['stage'],0)+1
    measured=dict(visual_frames=visual_frames,history_updates=counts.get('learned_visual_and_history',0),
        robot_observations=counts.get('robot_history_input',0),
        plans=counts.get('ddim_plan',0),backend_updates=counts.get('residual_forward_compose',0),
        expected_plans=(steps+11)//12,control_ticks=steps)
    if strict and (any(measured[key]!=steps for key in ('visual_frames','history_updates','robot_observations','backend_updates'))
                   or measured['plans']!=measured['expected_plans']):
        raise ValueError('Diagnostic did not run full real-frame student inference at every control tick')
    return measured


def check_sensor_clock(packet,active,previous):
    """Fail closed if a probe skips/reuses a real camera exposure or timestamp."""
    import torch
    ids=[i for i,value in enumerate(active) if value]
    if not ids:return previous
    if packet['image'].shape[:2]!=(len(active),3) or packet['camera_valid'].shape!=(len(active),3):
        raise ValueError('Three real camera views required for each lane')
    if not bool(packet['camera_valid'][ids].all()):raise ValueError('Invalid real RGB-D content during backend probe')
    times=packet['camera_time_s'][ids].detach().cpu()
    now=packet['time_s'][ids].detach().cpu()
    if not torch.isfinite(times).all() or not torch.allclose(times,now[:,None].expand_as(times),rtol=0,atol=1e-8):
        raise ValueError('Probe sensor clock differs from current control observation')
    frames=packet['camera_frame'].detach().cpu()
    for lane in ids:
        if previous[lane] is not None and not bool((frames[lane]>previous[lane]).all()):
            raise ValueError('Probe repeated a camera exposure')
        previous[lane]=frames[lane].clone()
    return previous


def check_sensor_clock_with_evidence(packet,active,previous,*,directory,env,control_tick):
    """Keep acceptance unchanged; save the first failed real observation locally."""
    try:
        return check_sensor_clock(packet,active,previous)
    except Exception as failure:
        try:
            from visuotactile.camera_failure_diagnostics import save_failure
            save_failure(directory,packet,active,env,failure,control_tick=control_tick)
        except Exception as diagnostic_error:
            failure.add_note('Sensor failure snapshot could not be saved: '+repr(diagnostic_error))
        raise


def resource_snapshot(root,samples):
    row=read(root/'shared_resources.json')
    if not samples or samples[-1]['utc_epoch']!=row['utc_epoch']:samples.append(row)


def prepare(root, output, count, seed, *, neural_batch=False,max_control_ticks=None,roles=None,execution_backend=None,
            vision_microbatch=2):
    import subprocess
    from visuotactile.narrow_data import direct_configuration
    from visuotactile.models.native_vision import validate_vision_microbatch
    vision_microbatch=validate_vision_microbatch(vision_microbatch)
    if type(count) is not int or count < 1:raise ValueError('Positive registered lanes; resource acceptance determines concurrency')
    options=probe_options(max_control_ticks,roles)
    if execution_backend is not None:
        from visuotactile.execution_backend import normalize_backend
        execution_backend=normalize_backend(execution_backend)
    if output.exists():raise FileExistsError('Preserve previous batch identity')
    session=read(root/'session.json')
    pairs={role:completed_pair(root,role) for role in options['roles']}
    cases=[dict(identity=f'batch_development_{seed+i}',seed=seed+i,
        narrow_reset=direct_configuration(session['domain'],seed+i)) for i in range(count)]
    output.mkdir(parents=True)
    atomic_json(output/'session.json',session)
    plan=dict(pairs=pairs,cases=cases,seed=seed,event_seed=seed+100000,
        resource_root=str(root),test_used=False,vision_microbatch=vision_microbatch,
        kind='engineering_batch_throughput_probe' if options['diagnostic_only'] else 'engineering_batch_development',neural_batch=neural_batch,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ['tools','visuotactile'] for p in Path(folder).rglob('*.py')},
        limitation='Native batched event RNG differs from serial; same registered opportunities are not exact hidden-state pairs')
    if max_control_ticks is not None:plan.update(max_control_ticks=max_control_ticks,diagnostic_only=True,
        inference_scheduler_index_mode='host',
        result_scope='Bounded throughput/backend diagnostic only; censored lanes are not task failures or successes; no new training data')
    if execution_backend is not None:
        plan['execution_backend']=execution_backend
    atomic_json(output/'plan.json',plan)


def run(output):
    import numpy as np
    import torch
    from tools.tight_state import environment
    from visuotactile.state_resources import lease_for
    from visuotactile.batched_controller import BatchedResidualController
    from visuotactile.models.native_vision import NativeDinoSpatial,validate_vision_microbatch
    from visuotactile.narrow_data import configure_direct_reset_lanes
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.live_collision import clearance
    from visuotactile.native_evaluation import stable_grasp,exit_completed_kit_worker
    from visuotactile.recording import BufferedNumericWriter
    import h5py
    plan=read(output/'plan.json');session=read(output/'session.json')
    vision_microbatch=validate_vision_microbatch(plan.get('vision_microbatch',2))
    max_ticks=probe_options(plan.get('max_control_ticks'),plan['pairs'])['max_control_ticks']
    diagnostic=max_ticks is not None
    if diagnostic and (plan['kind']!='engineering_batch_throughput_probe' or plan.get('diagnostic_only') is not True):
        raise ValueError('Censored diagnostics require an explicit probe plan identity')
    scheduler_mode=plan.get('inference_scheduler_index_mode','legacy')
    if scheduler_mode not in ('legacy','host') or diagnostic and scheduler_mode!='host':
        raise ValueError('Probe requires the registered host scheduler index mode used by formal batch execution')
    for path,digest in plan['source_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Frozen source changed: '+path)
    if (output/'started.json').exists():raise ValueError('No duplicate starts')
    atomic_json(output/'started.json',dict(pid=os.getpid(),utc=time.time(),plan_sha256=sha256(output/'plan.json')))
    resource_root=Path(plan['resource_root']);lease=lease_for(resource_root,deadline=session['deadline']);count=len(plan['cases'])
    backend_kwargs={'execution_backend':plan['execution_backend']} if 'execution_backend' in plan else {}
    if diagnostic:backend_kwargs['camera_content_validation']=True
    startup_began=time.perf_counter();resources=[]
    if diagnostic:resource_snapshot(resource_root,resources)
    app,env,player,lease=environment(output,evaluation=True,lease=lease,num_envs=count,**backend_kwargs)
    startup_seconds=time.perf_counter()-startup_began
    atomic_json(output/'startup_timing.json',dict(seconds=startup_seconds,vision_microbatch=vision_microbatch))
    configure_direct_reset_lanes(env,[c['narrow_reset'] for c in plan['cases']])
    isolated=IsolatedEnvironment(env,plan['seed'])
    with lease.block():dino=NativeDinoSpatial().cuda().eval()
    all_results=[]
    for role,pair in plan['pairs'].items():
        role_began=time.perf_counter()
        for key in ['frontend','backend']:
            if sha256(pair[key])!=pair[key+'_sha256']:raise ValueError('Frozen pair changed')
        directory=output/role;directory.mkdir()
        with lease.block():
            env.action_control_mode='native_command_clipped_pre_delay'
            teacher,_=isolated.reset(plan['seed']);isolated.begin_events(plan['event_seed'])
            env.action_control_mode='absolute_joint_position_target_rad'
            initial=env._diagnostics()
            atomic_json(directory/'initial_observable.json',{k:v.cpu().tolist() for k,v in initial.items()})
            admissions=[clearance(env,lane=i) for i in range(count)]
            atomic_json(directory/'initial_collision.json',admissions)
            active=[row['legal'] for row in admissions]
            saved=torch.load(pair['backend'],map_location='cpu',weights_only=False)
            metadata=env.research_metadata();metadata['config']=env.cfg.to_dict()
            controller=BatchedResidualController(saved,metadata,dino,count,neural_batch=plan.get('neural_batch',False),
                scheduler_index_mode=scheduler_mode,vision_microbatch=vision_microbatch)
        rows=[[] for _ in range(count)];outcomes=[None]*count;native_flags=[None]*count;camera_frames=[None]*count
        files=[h5py.File(directory/f'lane_{i}_numeric.h5','w') for i in range(count)]
        writers=[BufferedNumericWriter(f.create_group('episode'),32) for f in files]
        profile=dict(controller_s=0.,environment_s=0.,recording_s=0.,resource_wait_s=0.)
        began=time.perf_counter();prior_wait=lease.wait_s
        try:
            bound=(int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
            limit=execution_limit(bound,max_ticks)
            for tick in range(limit):
                if not any(active):break
                packet=env.research_packet
                if diagnostic:check_sensor_clock_with_evidence(packet,active,camera_frames,
                    directory=directory,env=env,control_tick=tick)
                with lease.block():
                    start=time.perf_counter();command=controller.act(packet,env._diagnostics(),active)
                    if diagnostic:torch.cuda.synchronize()
                    profile['controller_s']+=time.perf_counter()-start
                    start=time.perf_counter();teacher,_,terminated,truncated,_=isolated.step(command)
                    if diagnostic:torch.cuda.synchronize()
                    profile['environment_s']+=time.perf_counter()-start
                    transition=env.last_transition;controller.observe_execution(transition,active)
                    start=time.perf_counter()
                    for lane,running in enumerate(active):
                        if not running:continue
                        from visuotactile.batched_controller import lane_tree
                        numeric=dict(obs={k:v for k,v in packet.items() if torch.is_tensor(v) and k not in ['image','depth_m','depth_valid']},transition=transition)
                        writers[lane].append(lane_tree(numeric,lane,count),len(rows[lane]))
                        after=transition['diagnostics_after']
                        take=lambda v:v[lane].cpu().numpy().copy()
                        rows[lane].append(dict(command=take(command),target=take(transition['target_after_rad']),
                            joint=take(after['joint_position_rad']),object=take(after['object_pose_w']),palm=take(after['palm_body_pose_w']),
                            object_contact=take(transition['object_contact_force_w']),lifted=take(after['lifted_object']),
                            alignment=take(transition['native_alignment_condition']),subgoals=take(after['successes'])))
                        reasons={key:bool(value[lane]) for key,value in transition['termination_reasons'].items()}
                        if reasons['max_successes'] and not bool(terminated[lane]):raise ValueError('Success/termination disagreement')
                        if bool((terminated|truncated)[lane]):
                            active[lane]=False;outcomes[lane]=reasons
                            native_flags[lane]=dict(terminated=bool(terminated[lane]),truncated=bool(truncated[lane]))
                    profile['recording_s']+=time.perf_counter()-start
                if tick%100==0:atomic_json(directory/'progress.json',dict(tick=tick,active=active,utc=time.time()))
                if diagnostic:
                    resource_snapshot(resource_root,resources)
                    atomic_json(directory/'progress.json',dict(completed_control_ticks=tick+1,active=active,
                        max_control_ticks=max_ticks,diagnostic_only=True,utc=time.time()))
            for lane in range(count):
                lane_outcome(legal=admissions[lane]['legal'],steps=len(rows[lane]),reasons=outcomes[lane],
                    still_active=active[lane],max_control_ticks=max_ticks,native_bound=bound)
        finally:
            for writer,f in zip(writers,files):writer.flush();f.close()
        profile.update(total_s=time.perf_counter()-began,resource_wait_s=lease.wait_s-prior_wait,
            steps=sum(map(len,rows)),scope=('Current-frame DINO, DDIM and backend batches; per-lane histories and generators' if plan.get('neural_batch',False) else 'current-frame DINO batches; DP and backend still per lane; native dt unchanged'))
        profile['aggregate_control_ticks_per_second']=profile['steps']/profile['total_s']
        profile['control_loop_and_numeric_flush_s']=profile['total_s']
        profile['vision_execution']=controller.vision_execution_report()
        results=[]
        for lane in range(count):
            result=dict(lane=lane,identity=plan['cases'][lane]['identity'],legal=admissions[lane]['legal'],steps=len(rows[lane]),
                success=bool(outcomes[lane] and outcomes[lane]['max_successes']),reasons=outcomes[lane],pair=pair)
            result.update(lane_outcome(legal=result['legal'],steps=result['steps'],reasons=outcomes[lane],
                still_active=active[lane],max_control_ticks=max_ticks,native_bound=bound),native_termination_flags=native_flags[lane])
            if rows[lane]:
                arrays={key:np.stack([row[key] for row in rows[lane]]) for key in rows[lane][0]}
                np.savez_compressed(directory/f'lane_{lane}_trace.npz',**arrays)
                result.update(native_lift=bool(arrays['lifted'].any()),native_alignment=bool(arrays['alignment'].any()),
                    native_subgoals=int(arrays['subgoals'].max()),contact=stable_grasp(arrays,float(initial['object_pose_w'][lane,2])),
                    visual_frames=controller.visual_frames[lane],position_target_max_error=float(np.abs(arrays['command']-arrays['target']).max()))
                if result['visual_frames']!=result['steps']:raise ValueError('Dropped native visual frame')
                inference=controller.inference_metrics(lane)
                result['inference_counts']=inference_counts(inference,result['steps'],result['visual_frames'],strict=diagnostic)
                atomic_json(directory/f'lane_{lane}_inference.json',inference)
            results.append(result)
        if diagnostic:
            resource_snapshot(resource_root,resources)
            profile.update(total_s=time.perf_counter()-began,role_wall_s=time.perf_counter()-role_began,
                startup_seconds=startup_seconds,worker_memory=lease.memory(),
                explicit_stage_cuda_synchronization=True,
                throughput_scope='Control loop plus numeric flush, trace compression and inference-count validation; startup/model load separate')
            profile['aggregate_control_ticks_per_second']=profile['steps']/profile['total_s']
        atomic_json(directory/'result.json',dict(software_complete=True,results=results,profile=profile,
            diagnostic_only=diagnostic,max_control_ticks=max_ticks,autonomous_evaluation_eligible=not diagnostic))
        all_results.append(dict(role=role,results=results,profile=profile))
        print(json.dumps(dict(role=role,profile=profile,successes=None if diagnostic else sum(r['success'] is True for r in results),
            observed_native_completions=sum(r['observed_native_completion'] for r in results),
            diagnostic_only=diagnostic,censored=sum(r['censored'] for r in results))),flush=True)
    if diagnostic:atomic_json(output/'resource_samples.json',dict(samples=resources,
        source=str(resource_root/'resources/shared.jsonl'),scope='Observed shared monitor samples; full original monitor remains authoritative'))
    atomic_json(output/'result.json',dict(software_complete=True,results=all_results,test_used=False,
        diagnostic_only=diagnostic,max_control_ticks=max_ticks,accepted_training_trajectories=0,
        autonomous_evaluation_eligible=not diagnostic,total_wall_seconds=time.perf_counter()-startup_began))
    exit_completed_kit_worker(output)


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=2);p.add_argument('--seed',type=int,default=28141000)
    p.add_argument('--neural-batch',action='store_true')
    p.add_argument('--max-control-ticks',type=int)
    p.add_argument('--roles',nargs='+',choices=['A','B'])
    p.add_argument('--backend-json',type=Path)
    p.add_argument('--vision-microbatch',type=int,default=2)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.root.resolve(),args.output.resolve(),args.count,args.seed,neural_batch=args.neural_batch,
        max_control_ticks=args.max_control_ticks,roles=args.roles,
        execution_backend=read(args.backend_json) if args.backend_json else None,vision_microbatch=args.vision_microbatch)
    else:run(args.output.resolve())


if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        traceback.print_exc()
        import sys
        sys.stdout.flush();sys.stderr.flush();os._exit(1)
