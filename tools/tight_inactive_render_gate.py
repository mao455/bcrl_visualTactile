"""Finite CPU-standard renderer retirement diagnostic, never training data.

Two separate Kit workers compare ordinary rendering and the candidate. Each
worker executes two 36-tick waves (A then B), with diagnostic active lanes
16 -> 8 -> 1. The owner preserves every failure and never retries an arm.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tools.tight_fresh_rollouts import read,completed_pair
from tools.tight_fresh_batch_run import validate_inactive_render_request,inactive_render_scope
from visuotactile.compact import sha256
from visuotactile.inactive_render_products import PROFILE
from visuotactile.training_resources import atomic_json


VERSION='inactive_render_two_wave_gate_v1'
TICKS=36
LANES=16
ARMS=('ordinary','inactive')


def check_bounds(plan,output):
    if time.time()>=plan['work_deadline']:raise TimeoutError('Original finite gate work deadline exhausted; cleanup reserve retained')
    if (Path(output)/'STOP').exists() or (Path(plan['resource_root'])/'STOP').exists():
        raise RuntimeError('Explicit gate/session STOP')


def scheduled_retirements(tick):
    if type(tick) is not int or not 0<=tick<TICKS:raise ValueError('Exactly 36 diagnostic ticks per wave')
    return list(range(8,16)) if tick==11 else list(range(1,8)) if tick==23 else [0] if tick==35 else []


def prepare(root,source_root,case_plan,output,*,execution_backend=None,max_seconds=600):
    if output.exists():raise FileExistsError('Preserve diagnostic attempt identity')
    if type(max_seconds) is not int or not 60<=max_seconds<=900:
        raise ValueError('Finite gate window must be explicitly 60..900 seconds')
    from visuotactile.execution_backend import normalize_backend
    source=read(case_plan);session=read(root/'session.json')
    if source.get('kind')!='takeover' or source.get('test_used') is not False:
        raise ValueError('Existing registered training-family reset plan required')
    cases=[copy.deepcopy(case['spec']) for case in source['cases'] if case['role']=='A'][:LANES]
    if len(cases)!=LANES or any(case['split']!='train' for case in cases):
        raise ValueError('Sixteen preregistered training reset opportunities required')
    backend=normalize_backend(execution_backend or source['execution_backend'])
    validate_inactive_render_request(PROFILE,backend)
    pairs={role:completed_pair(source_root,role) for role in ('A','B')}
    started=time.time();deadline=min(session['deadline'],started+max_seconds)
    if deadline<=started:raise TimeoutError('Original session expired')
    plan=dict(version=VERSION,kind='engineering_inactive_render_gate',resource_root=str(root.resolve()),
        source_root=str(source_root.resolve()),pairs=pairs,cases=cases,
        source_case_plan=dict(path=str(case_plan.resolve()),sha256=sha256(case_plan)),
        session_sha256=sha256(root/'session.json'),started=started,deadline=deadline,work_deadline=deadline-30,
        cleanup_reserve_seconds=30,max_seconds=max_seconds,
        startup_seconds=300,arms=list(ARMS),roles=['A','B'],lanes=LANES,ticks_per_wave=TICKS,
        seed=source['batch']['seed'],event_seed=source['batch']['seed']+100000,
        execution_backend=backend,inactive_render_profile=PROFILE,vision_microbatch=8,
        retirement_schedule={str(tick):scheduled_retirements(tick) for tick in (11,23,35)},
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ('tools','visuotactile') for p in sorted(Path(folder).rglob('*.py'))},
        diagnostic_only=True,test_used=False,accepted_for_training=False,
        illegal_lane_policy='retire_after_tick0_proof_no_action_no_replacement_v1',
        scope='Artificial lane censoring only; no task success, autonomous failure rate, or training admission. Equal registered resets do not imply equal hidden simulator state.')
    output.mkdir(parents=True);atomic_json(output/'plan.json',plan)
    return plan


def verify_plan(plan,tree):
    if (plan.get('version')!=VERSION or plan.get('diagnostic_only') is not True or plan.get('test_used') is not False
            or plan.get('accepted_for_training') is not False or plan['lanes']!=LANES
            or plan.get('illegal_lane_policy')!='retire_after_tick0_proof_no_action_no_replacement_v1'
            or plan['ticks_per_wave']!=TICKS or plan['arms']!=list(ARMS) or plan['roles']!=['A','B']):
        raise ValueError('Exact finite diagnostic contract required')
    if plan['retirement_schedule']!={str(t):scheduled_retirements(t) for t in (11,23,35)}:
        raise ValueError('Registered artificial retirement schedule changed')
    validate_inactive_render_request(plan['inactive_render_profile'],plan['execution_backend'])
    if (plan.get('cleanup_reserve_seconds')!=30 or plan['work_deadline']!=plan['deadline']-30
            or plan['deadline']>plan['started']+plan['max_seconds']
            or plan['deadline']>read(Path(plan['resource_root'])/'session.json')['deadline']):
        raise ValueError('Original gate/session deadline or cleanup reserve changed')
    if sha256(Path(plan['resource_root'])/'session.json')!=plan['session_sha256']:
        raise ValueError('Resource session bytes changed')
    if sha256(plan['source_case_plan']['path'])!=plan['source_case_plan']['sha256']:
        raise ValueError('Registered reset source changed')
    actual={str(p.relative_to(tree)):sha256(p) for folder in ('tools','visuotactile') for p in (tree/folder).rglob('*.py')}
    if actual!=plan['source_sha256'] or subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()!=plan['source_commit']:
        raise ValueError('Frozen worker sources changed')
    for pair in plan['pairs'].values():
        for key in ('frontend','backend'):
            if sha256(pair[key])!=pair[key+'_sha256']:raise ValueError('Complete selected student changed')


def camera_inventory(env):
    return [dict(sensor_id=id(camera),render_products=list(camera.render_product_paths),
        annotators={name:[id(item) for item in items] for name,items in camera._rep_registry.items()})
        for camera in env.research_cameras]


def run_wave(env,lease,guard,controller,isolated,directory,plan,arm,wave,*,synchronize,legal=None):
    """Finite, testable native loop. Guard already began before caller's reset."""
    import h5py
    import torch
    from visuotactile.recording import BufferedNumericWriter
    from visuotactile.batched_controller import lane_tree
    from tools.tight_fresh_batch_eval import check_sensor_clock,inference_counts
    legal=[True]*LANES if legal is None else list(legal)
    if len(legal)!=LANES or any(type(value) is not bool for value in legal):raise ValueError('Explicit 16-slot legal mask required')
    if not legal[0] or not any(legal[1:8]) or not any(legal[8:]):
        raise ValueError('Lane0 and at least one legal lane in each earlier censor block required; no replacement')
    directory.mkdir();active=legal.copy();seen=[None]*LANES
    files=[];writers=[];rows=[];counts=[0]*LANES;times=[];resources=[]
    inventory=camera_inventory(env)
    try:
        for lane in range(LANES):
            stream=h5py.File(directory/f'lane_{lane:02d}_numeric.h5','x');files.append(stream)
            stream.attrs['diagnostic_only']=True
            writers.append(BufferedNumericWriter(stream.create_group('episode'),8))
        for tick in range(TICKS):
            check_bounds(plan,directory.parent.parent)
            packet,proof=guard.observe(env.research_packet,control_tick=tick)
            if tick==0:
                if proof['active_lanes']!=list(range(LANES)):raise ValueError('First proof must cover every allocated product')
                illegal=[i for i,value in enumerate(legal) if not value]
                if illegal and arm=='inactive':guard.retire(illegal,reason='initial_illegal',control_tick=0)
                if arm=='inactive':packet=guard.mask_inactive(packet)
            expected=([i for i,a in enumerate(active) if a] if arm=='inactive' and tick else list(range(LANES)))
            proof=dict(proof,consumed_active_lanes=[i for i,a in enumerate(active) if a])
            if proof['active_lanes']!=expected:raise ValueError('Diagnostic renderer activity differs from registration')
            check_sensor_clock(packet,active,seen)
            active_before=sum(active);nested_before=dict(getattr(env,'_inactive_gate_timings',{}))
            with lease.block():
                before=time.perf_counter();command=controller.act(packet,env._diagnostics(),active);synchronize()
                inference_s=time.perf_counter()-before
                before=time.perf_counter();_,_,terminated,truncated,_=isolated.step(command);synchronize()
                environment_s=time.perf_counter()-before
                transition=env.last_transition;controller.observe_execution(transition,active)
                before=time.perf_counter()
                for lane,running in enumerate(active):
                    if not running:continue
                    numeric=dict(obs={key:value for key,value in packet.items() if torch.is_tensor(value)
                        and key not in ('image','depth_m','depth_valid')},transition=transition)
                    writers[lane].append(lane_tree(numeric,lane,LANES),counts[lane]);counts[lane]+=1
                write_s=time.perf_counter()-before
                if any(bool((terminated|truncated)[lane]) for lane,running in enumerate(active) if running):
                    raise ValueError('Native termination changed the finite diagnostic active schedule; no replacement')
                scheduled=scheduled_retirements(tick);retiring=[lane for lane in scheduled if active[lane]]
                if scheduled:
                    for lane in retiring:
                        writers[lane].flush()
                        active[lane]=False
                    # Ordinary arm keeps all products drawing through tick35;
                    # final retirement merely permits the next restored wave.
                    if arm=='inactive' and retiring:guard.retire(retiring,reason='diagnostic_censor',control_tick=tick)
                    elif arm=='ordinary' and tick==TICKS-1:
                        guard.retire(list(range(LANES)),reason='diagnostic_censor',control_tick=tick)
                if controller.visual_frames!=counts:raise ValueError('Real lane visual/history count differs')
            times.append(dict(tick=tick,active_before=active_before,
                inference_s=inference_s,environment_s=environment_s,numeric_append_s=write_s,
                nested_environment={key:value-nested_before.get(key,0.) for key,value in getattr(env,'_inactive_gate_timings',{}).items()}))
            rows.append(dict(tick=tick,active_after=active.copy(),consumed_observations=counts.copy(),renderer=proof))
            if scheduled:
                atomic_json(directory/f'render_state_{tick+1:02d}.json',guard.snapshot())
                telemetry=Path(plan['resource_root'])/'shared_resources.json'
                if telemetry.is_file():resources.append(read(telemetry))
            atomic_json(directory/'progress.json',dict(completed_ticks=tick+1,active=active,utc=time.time()))
        expected_counts=[value if legal[i] else 0 for i,value in enumerate([36]+[24]*7+[12]*8)]
        if counts!=expected_counts or any(active):raise ValueError('Incomplete allocated 16→8→1 finite schedule')
        if camera_inventory(env)!=inventory:raise ValueError('Sensor/product/annotator identity changed')
        for writer,stream in zip(writers,files):writer.flush();stream.close()
        inference=[inference_counts(controller.inference_metrics(lane),counts[lane],controller.visual_frames[lane],strict=True)
            for lane in range(LANES)]
        result=dict(software_complete=True,diagnostic_only=True,accepted_for_training=False,success=None,
            wave=wave,arm=arm,counts=counts,legal=legal,allocated_slots=LANES,
            rows=rows,timing=times,inference=inference,resource_samples=resources,
            vision_execution=controller.vision_execution_report(),camera_inventory=inventory,
            final_renderer_state=guard.snapshot(),
            scope='Natural 120/60Hz steps with diagnostic lane censoring after actual transitions; no task outcome or data admission',
            timing_scope='Environment includes nested physics/render/scene timers; never add overlapping values. GPU synchronization is diagnostic instrumentation.')
        atomic_json(directory/'result.json',result)
        return result
    except BaseException as error:
        atomic_json(directory/'failure.json',dict(error=repr(error),completed_ticks=len(rows),counts=counts,rows=rows,timing=times))
        try:
            from visuotactile.camera_failure_diagnostics import save_failure
            save_failure(directory,env.research_packet,active,env,error,control_tick=len(rows))
        except BaseException as capture_error:error.add_note('Failure packet could not be saved: '+repr(capture_error))
        raise
    finally:
        close_errors=[]
        for writer,stream in zip(writers,files):
            if stream.id.valid:
                try:writer.flush()
                except BaseException as error:close_errors.append(repr(error))
                finally:
                    try:stream.close()
                    except BaseException as error:close_errors.append(repr(error))
        if close_errors:
            atomic_json(directory/'numeric_cleanup_failure.json',dict(errors=close_errors))
            raise RuntimeError('Diagnostic numeric cleanup failed: '+str(close_errors))


def worker(root,output,arm):
    import torch
    from tools.tight_state import environment
    from visuotactile.state_resources import lease_for
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.batched_controller import BatchedResidualController
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.narrow_data import configure_direct_reset_lanes
    from visuotactile.live_collision import clearance
    from visuotactile.native_evaluation import exit_completed_kit_worker
    output=output.resolve();plan=read(output/'plan.json');directory=output/arm
    if arm not in ARMS or root.resolve()!=Path(plan['resource_root']):raise ValueError('Registered arm/resource root required')
    verify_plan(plan,Path.cwd());check_bounds(plan,output)
    with (directory/'run_started.json').open('x') as stream:json.dump(dict(pid=os.getpid(),utc=time.time(),plan_sha256=sha256(output/'plan.json')),stream)
    lease=lease_for(root,deadline=plan['work_deadline']);began=time.perf_counter()
    app,env,_,lease=environment(directory,evaluation=True,lease=lease,num_envs=LANES,
        camera_content_validation=True,startup_diagnostics=True,execution_backend=plan['execution_backend'])
    atomic_json(directory/'initialization.json',dict(seconds=time.perf_counter()-began))
    env._inactive_gate_timings={}
    def measure(obj,name,key):
        original=getattr(obj,name)
        def call(*args,**kwargs):
            before=time.perf_counter()
            try:return original(*args,**kwargs)
            finally:env._inactive_gate_timings[key]=env._inactive_gate_timings.get(key,0.)+time.perf_counter()-before
        setattr(obj,name,call)
    measure(env.sim,'step','physics_step_s');measure(env.sim,'render','render_s');measure(env.scene,'update','scene_update_s')
    with inactive_render_scope(env,plan['inactive_render_profile'],directory,gate_limit_ticks=TICKS) as guard:
        with lease.block():dino=NativeDinoSpatial().cuda().eval()
        configure_direct_reset_lanes(env,[case['narrow_reset'] for case in plan['cases']])
        inventory=camera_inventory(env);results=[]
        for wave,role in enumerate(plan['roles']):
            check_bounds(plan,output);isolated=IsolatedEnvironment(env,plan['seed']+wave)
            with lease.block():
                guard.begin_wave(wave)
                restored=guard.snapshot()
                if not all(p['enabled'] is True for p in restored['products']):raise ValueError('Wave reset preceded full product restoration')
                atomic_json(directory/f'wave_{wave}_before_reset.json',restored)
                env.clear_recorded_events();env.action_control_mode='native_command_clipped_pre_delay'
                isolated.reset(plan['seed']+wave);isolated.begin_events(plan['event_seed']+wave)
                env.action_control_mode='absolute_joint_position_target_rad'
                admissions=[clearance(env,lane=i) for i in range(LANES)]
                atomic_json(directory/f'wave_{wave}_initial_collision.json',admissions)
                atomic_json(directory/f'wave_{wave}_initial_observable.json',{key:value.cpu().tolist() for key,value in env._diagnostics().items()})
                metadata=env.research_metadata();metadata['config']=env.cfg.to_dict()
                saved=torch.load(plan['pairs'][role]['backend'],map_location='cpu',weights_only=False)
                if saved['configuration'].get('execution_adapter'):metadata['execution_adapter']=saved['configuration']['execution_adapter']
                controller=BatchedResidualController(saved,metadata,dino,LANES,neural_batch=True,
                    scheduler_index_mode='host',vision_microbatch=8)
            result=run_wave(env,lease,guard,controller,isolated,directory/role,plan,arm,wave,
                synchronize=torch.cuda.synchronize,legal=[bool(row['legal']) for row in admissions])
            if camera_inventory(env)!=inventory:raise ValueError('Products or sensor attachments changed across reset waves')
            results.append(dict(role=role,counts=result['counts'],legal=result['legal'],result_sha256=sha256(directory/role/'result.json')))
            del controller,saved
    atomic_json(directory/'result.json',dict(software_complete=True,arm=arm,waves=results,
        seconds=time.perf_counter()-began,diagnostic_only=True,accepted_for_training=False,success=None))
    exit_completed_kit_worker(directory)


def owner(output,worker_tree):
    from tools.tight_batch_owner import bind_worker_group,cleanup_owned_group
    from tools.tight_fresh_eval_recovery import stop_owned
    output=output.resolve();worker_tree=worker_tree.resolve();plan=read(output/'plan.json')
    verify_plan(plan,worker_tree);check_bounds(plan,output)
    if any((output/name).exists() for name in ('owner_started.json','result.json',*ARMS)):
        raise FileExistsError('No duplicate gate start or repeated arm')
    digest=sha256(output/'plan.json');started=time.time();receipts=[];evidence={};caught=None;process=None;binding=None
    with (output/'owner_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(),deadline=plan['deadline'],started=started,plan_sha256=digest),stream)
        stream.flush();os.fsync(stream.fileno())
    handlers={}
    def interrupted(signum,frame):raise SystemExit(128+signum)
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        for arm in ARMS:
            check_bounds(plan,output);directory=output/arm;directory.mkdir();binding=None;process=None
            command=[str(worker_tree/'.venv_isaacsim/bin/python'),'-m','tools.tight_inactive_render_gate','worker',
                '--root',plan['resource_root'],'--output',str(output),'--arm',arm]
            began=time.time()
            try:
                with (directory/'worker.log').open('x') as stream:
                    process=subprocess.Popen(command,cwd=worker_tree,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                        start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONFAULTHANDLER='1'))
                    binding=bind_worker_group(process.pid)
                    atomic_json(directory/'owner_worker_started.json',dict(command=command,worker_group=binding,started=began))
                    while process.poll() is None:
                        check_bounds(plan,output)
                        if not (directory/'initialization.json').exists() and time.time()-began>=plan['startup_seconds']:
                            raise TimeoutError('Bounded Kit startup exhausted')
                        time.sleep(min(.25,max(.001,plan['work_deadline']-time.time())))
                if process.returncode!=0:raise RuntimeError('Diagnostic worker failed: '+arm)
                result=read(directory/'result.json')
                if result.get('software_complete') is not True or len(result.get('waves',[]))!=2:
                    raise ValueError('Incomplete two-wave diagnostic evidence')
                for role,row in zip(('A','B'),result['waves']):
                    legal=row.get('legal',[])
                    if (row.get('role')!=role or len(legal)!=LANES or any(type(v) is not bool for v in legal)
                            or row.get('counts')!=[v if legal[i] else 0 for i,v in enumerate([36]+[24]*7+[12]*8)]
                            or sha256(directory/role/'result.json')!=row.get('result_sha256')):
                        raise ValueError('Finite wave population or evidence binding changed')
                evidence[arm]=result['waves']
                receipts.append(dict(arm=arm,result_sha256=sha256(directory/'result.json'),seconds=time.time()-began))
            finally:
                if process is not None:
                    stop_error=None
                    try:stop_owned(process)
                    except BaseException as error:stop_error=repr(error)
                    if binding is None:
                        atomic_json(directory/'owner_group_cleanup.json',dict(error='Missing worker group creation binding',stop_error=stop_error))
                        raise RuntimeError('Missing worker group creation binding')
                    cleanup=cleanup_owned_group(binding);cleanup['stop_error']=stop_error
                    atomic_json(directory/'owner_group_cleanup.json',cleanup)
                    if not cleanup['live_descendants_exited']:raise RuntimeError('Owned gate worker group remains live')
                    if stop_error is not None:raise RuntimeError('Direct worker stop failed: '+stop_error)
        if sha256(output/'plan.json')!=digest:raise ValueError('Gate registration changed')
        if time.time()>plan['deadline']:raise TimeoutError('Gate cleanup exceeded its original deadline')
    except BaseException as error:caught=dict(type=type(error).__name__,error=repr(error))
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        comparison=[]
        if len(receipts)==2:
            for ordinary,candidate in zip(evidence['ordinary'],evidence['inactive']):
                matched=ordinary['role']==candidate['role'] and ordinary['legal']==candidate['legal'] and ordinary['counts']==candidate['counts']
                comparison.append(dict(role=ordinary['role'],ordinary_legal=ordinary['legal'],candidate_legal=candidate['legal'],
                    same_observed_legal_cohort_and_counts=matched,timing_comparable=matched,
                    scope='Separate simulator workers; equal legal cohorts do not establish equal hidden physical states or a production throughput gain'))
        report=dict(software_complete=caught is None and len(receipts)==2,arms=receipts,error=caught,
            plan_sha256=digest,seconds=time.time()-started,deadline=plan['deadline'],ended=time.time(),
            diagnostic_only=True,accepted_for_training=False,success=None,comparison=comparison,
            scope='Finite renderer API/control regression only; review active freshness, restored waves and actual production before adopting or claiming throughput gains')
        atomic_json(output/'result.json',report)
    if not report['software_complete']:raise RuntimeError('Inactive renderer gate failed; original evidence retained')
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('stage',choices=['prepare','owner','worker'])
    p.add_argument('--root',type=Path);p.add_argument('--source-root',type=Path);p.add_argument('--case-plan',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--worker-tree',type=Path)
    p.add_argument('--backend-json',type=Path);p.add_argument('--max-seconds',type=int,default=600)
    p.add_argument('--arm',choices=ARMS);a=p.parse_args()
    if a.stage=='prepare':
        if any(value is None for value in (a.root,a.source_root,a.case_plan)):p.error('prepare needs root/source-root/case-plan')
        prepare(a.root.resolve(),a.source_root.resolve(),a.case_plan.resolve(),a.output.resolve(),
            execution_backend=read(a.backend_json) if a.backend_json else None,max_seconds=a.max_seconds)
    elif a.stage=='owner':
        if a.worker_tree is None:p.error('owner needs frozen worker-tree')
        owner(a.output,a.worker_tree)
    else:
        if a.root is None or a.arm is None:p.error('worker needs registered root and arm')
        worker(a.root,a.output,a.arm)


if __name__=='__main__':
    try:main()
    except SystemExit:raise
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
