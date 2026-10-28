"""Persistent released-expert capture, bounded audit pipeline and resumable manifest.

Only this task's process tree is managed. Native success/control/DR are unchanged.
"""
import argparse
from collections import Counter, deque
import copy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .compact import sha256
from .run_insertion_dataset import CHECKPOINT, CHECKPOINT_SHA
from .trial_utils import write_json

DEFAULT_ROOT=Path('outputs/visuotactile/data/close_v4')
DEFAULT_RUN=Path('outputs/visuotactile/run/close400')
DEADLINE=datetime.fromisoformat('2026-10-29T08:00:00+08:00').timestamp()


def audit_worker_init():
    """Keep CPU file audits off the simulation's four cores, nice inherited."""
    os.sched_setaffinity(0,{16,17,18,19})
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def classify_pre_action_rejection(path):
    path=Path(path)
    if (path/'trajectories.h5').exists() or not (path/'metadata.json').exists():return None
    if not (path/'FAILED.txt').exists():return None
    if 'Initial object/table/hole collision overlap' not in (path/'FAILED.txt').read_text():return None
    meta=json.loads((path/'metadata.json').read_text())
    return meta if meta.get('initial_geometry',{}).get('valid_box_initialization') is False else None


def episode_metadata(base, event, packet, index, args):
    """One real env's calibration and identity, no env0 image/state replication."""
    import numpy as np
    m=copy.deepcopy(base)
    for key in ('joint_limits_rad','env_origins_w','camera_pos_w','camera_pos_env',
                'camera_quat_ros_wxyz','camera_intrinsics'):
        if key in m:m[key]=[m[key][index]]
    origin=np.asarray(m['env_origins_w'][0])
    for v,c in enumerate(m['cameras']):
        c['intrinsics']=[c['intrinsics'][index]]
        pose=packet['camera_pose_w'][index,v].detach().cpu().numpy()
        c['pos_env']=[(pose[:3]-origin).tolist()];c['quat_ros_wxyz']=[pose[3:].tolist()]
    if 'wrist' in m['camera_rig']:
        m['camera_rig']['wrist']['parent_prim']=m['camera_rig']['wrist']['parent_prim'].replace('/env_0/',f'/env_{index}/')
    m.update(initial_geometry=event['initial_geometry'],initial_offset=event['initial_offset'],
        collection_protocol='persistent_native_v4_v1',persistent_run_id=args.run_id,
        source_env_index=index,source_episode_id=event['episode_id'],
        initial_global_control_tick=event['tick'],num_parallel_envs=args.num_envs,
        seed_semantics='One global run seed; per-env native resets consume the shared stream without reseeding',
        nominal_plan_source='absent',nominal_action_history='absent, masked; never teacher history',
        correction_mode='disabled',command_source_codes={'1':'expert','2':'nominal'},
        nominal_prediction_outcome_attribution='Only released expert commands executed',
        successful_expert_demonstrations_verified=False)
    return m


def collect_persistent(env,player,teacher_obs,metadata,args):
    """One Kit, batched expert, native per-env resets, one bounded disk writer.

    CPU audits run in spawned processes (never fork a running Kit/CUDA context).
    Source digests and all native/data checks must pass before manifest commit.
    """
    from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
    import multiprocessing
    import numpy as np
    import torch
    from .recording import CompactEpisodeStream, numpy_tree, flatten, post_observation
    from .finalize_collection import finalize_stream
    from .contracts import assert_policy_compatible
    root=args.production_root;run=DEFAULT_RUN;run.mkdir(parents=True,exist_ok=True)
    manifest_path=root/'manifest.json';manifest=json.loads(manifest_path.read_text())
    reference=json.loads((root/'reference_metadata.json').read_text())
    # cfg.to_dict contains tuples; stored JSON contains arrays. Normalize the
    # representation before semantic equality (not a physical config change).
    metadata=json.loads(json.dumps(metadata,default=str))
    assert_policy_compatible(reference,metadata)
    if metadata['expert_model_sha256']!=CHECKPOINT_SHA:raise ValueError('Released checkpoint hash differs')
    for section in ('action','assets','domain_randomization','obs','reset','reward','termination','precise_assembly'):
        if metadata['config'][section]!=reference['config'][section]:raise ValueError('Native protocol changed: '+section)
    streams={};active={};writes=deque();audits={};attempt_index={r['id']:r for r in manifest['attempts']}
    digests={r.get('content_digest') for r in manifest['episodes'] if r.get('content_digest')}
    tmp=Path('outputs/visuotactile/tmp')/args.run_id;tmp.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();previous_success=len(manifest['episodes']);last_publish=0;steps=0
    writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='episode_writer')
    auditors=ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn'),initializer=audit_worker_init)
    metadata.update(collection_device=args.device,writer_compression='lossless_lzf',writer_chunk_rows=8)

    def identity(i,ep):return f'{args.run_id}_e{i:02d}_p{ep:06d}'

    def publish(state='collecting',force=False):
        nonlocal last_publish
        if not force and time.monotonic()-last_publish<5:return
        last_publish=time.monotonic()
        write_json(manifest_path,manifest)
        count=len(manifest['episodes']);elapsed=time.monotonic()-started
        rate=(count-previous_success)/elapsed*60 if elapsed else 0
        outcomes=Counter(r.get('outcome','unknown') for r in manifest['attempts'])
        write_json(run/'progress.json',dict(state=state,pid=os.getpid(),run_id=args.run_id,
            updated_epoch=time.time(),deadline_epoch=args.deadline_epoch,successful_episodes=count,
            remaining=max(0,args.target_successes-count),target=args.target_successes,control_steps=steps,
            num_envs=args.num_envs,device=args.device,active_episodes=len(active),pending_audits=len(audits),
            pending_writer_batches=len(writes),committed_successes_per_minute=rate,
            eta_seconds=(args.target_successes-count)/rate*60 if rate else None,
            run_elapsed_s=elapsed,attempt_outcomes=dict(outcomes),
            historical_successes_not_in_current_manifest=sum(r.get('outcome')=='success' and r['id'] not in {x['id'] for x in manifest['episodes']} for r in manifest['attempts']),
            data_bytes=sum(r['files']['data.h5']['bytes'] for r in manifest['episodes']),
            free_disk_bytes=shutil.disk_usage(root).free,last_episode=manifest['episodes'][-1]['directory'] if count else None))
        from .initialization_report import manifest_statistics
        manifest_statistics(manifest,run/'statistics')

    def consume_resets():
        for event in env.reset_events:
            i=event['env_index'];name=identity(i,event['episode_id'])
            if not event['initial_geometry']['valid_box_initialization']:
                row=dict(id=name,seed=args.seed,run_id=args.run_id,env_index=i,episode_id=event['episode_id'],
                    origin_w=env.scene.env_origins[i].cpu().tolist(),
                    outcome='invalid_initialization',episode_started=False,initial=event['initial'],
                    reason=event['initial_geometry'],initial_offset=event['initial_offset'],global_tick=event['tick'])
                manifest['attempts'].append(row);attempt_index[name]=row
        env.reset_events.clear()

    def write_batch(records):
        completed=[]
        for row,entry,m,result in records:
            name=entry['id']
            if name not in streams:streams[name]=CompactEpisodeStream(tmp/name,m,entry['episode'])
            streams[name].append(row)
            if result is not None:
                path=streams.pop(name).finish(result)
                completed.append((entry,path))
        return completed

    def reap_writes(wait=False):
        while writes and (wait or writes[0].done()):
            for entry,path in writes.popleft().result():
                audits[auditors.submit(finalize_stream,path,reference)]=(entry,path)
            if not wait:continue
            break

    def reap_audits(wait=False):
        for future in list(audits):
            if not future.done() and not wait:continue
            entry,path=audits.pop(future)
            try: info=future.result()
            except Exception as exc:
                attempt_index[entry['id']].update(outcome='audit_error',error=repr(exc),retained_path=str(path))
                publish('audit_error',True)
                raise
            outcome=info['summary']['outcome']
            attempt_index[entry['id']].update(outcome=outcome,summary=info['summary'],initial=info['initial'],
                control_steps=info['length'],effective_delay_counts=info['effective_delay_counts'],
                full_cycle_seconds=time.time()-entry['started_epoch'])
            if outcome=='success':
                if info['content_digest'] in digests:raise ValueError('Duplicate action/state trajectory; not counted')
                if len(manifest['episodes'])>=args.target_successes:
                    attempt_index[entry['id']]['outcome']='surplus_success_not_counted'
                    write_json(run/(entry['id']+'_surplus.json'),info)
                else:
                    dest=root/'episodes'/entry['id'];dest.parent.mkdir(parents=True,exist_ok=True)
                    path.rename(dest)
                    record=dict(entry,**info,directory='episodes/'+entry['id'],outcome='success',valid=True,
                        parameter_updates=0,nominal_predictions_available=False)
                    manifest['episodes'].append(record);digests.add(info['content_digest'])
                    print(json.dumps(dict(committed=entry['id'],successes=len(manifest['episodes']),
                        steps=info['length'],bytes=info['files']['data.h5']['bytes'])),flush=True)
            publish(force=True)
            if outcome!='success':
                # Known, fully audited failed/censored payload. Its reason and
                # initial/perturbation statistics are already durable above.
                shutil.rmtree(path)
            if wait:break

    try:
        consume_resets();publish(force=True)
        while len(manifest['episodes'])<args.target_successes:
            reap_writes();reap_audits()
            if (run/'STOP').exists() or time.time()>=args.deadline_epoch-45:break
            if shutil.disk_usage(root).free<12_000_000_000:raise RuntimeError('Disk reserve below 12 GB')
            while len(audits)>=max(4,args.num_envs*2):reap_audits(True)
            while len(writes)>=4:reap_writes(True)
            if len(manifest['episodes'])+len(audits)>=args.target_successes and not active:
                reap_audits(True);continue
            packet=env.research_packet
            for i in range(args.num_envs):
                if i not in active:
                    in_flight=sum(a.get('outcome')=='awaiting_audit' for a in manifest['attempts'])
                    if len(manifest['episodes'])+in_flight+len(active)>=args.target_successes:continue
                    event=env.initial_by_env[i];ep=int(packet['episode_id'][i]);name=identity(i,ep)
                    if ep!=event['episode_id'] or int(packet['step_id'][i])!=0:
                        # Nonrecorded envs near the target keep running until a
                        # real new reset; never start recording a mid-episode suffix.
                        continue
                    entry=dict(id=name,seed=args.seed,pair_id=name,condition='persistent_native_v4',
                        stage='production',offset_xy=[0.,0.],wrench_frame_mode='world_hold',
                        run_id=args.run_id,env_index=i,episode_id=ep,episode=f'env{i:03d}_ep{ep:06d}',
                        started_epoch=time.time(),origin_w=env.scene.env_origins[i].cpu().tolist())
                    m=episode_metadata(metadata,event,packet,i,args)
                    m.update(condition=entry['condition'],pair_id=name)
                    active[i]=(entry,m)
                    row=dict(entry,outcome='running',episode_started=True,initial=event['initial'])
                    manifest['attempts'].append(row);attempt_index[name]=row
            rng_before_inference=torch.random.get_rng_state().clone()
            with torch.no_grad():
                actor=player.obs_to_torch(teacher_obs['policy'])
                if player.intr_reward_coef_embd is not None:actor=torch.cat([actor,player.intr_reward_coef_embd],dim=1)
                action=player.get_action(actor,is_deterministic=True);actor=actor.clone()
            rng_before_step=torch.random.get_rng_state().clone()
            teacher_obs,reward,terminated,truncated,_=env.step(action)
            done=terminated|truncated;tr=env.last_transition
            tr['torch_rng_before_inference']=rng_before_inference[None].expand(args.num_envs,-1)
            tr['torch_rng_before_step']=rng_before_step[None].expand(args.num_envs,-1)
            tr['expert_actor_input_diagnostic']=actor
            tr['provenance']=dict(expert_computed=torch.ones_like(done),expert_command=action.clone(),
                nominal_computed=torch.zeros_like(done),correction_computed=torch.zeros_like(done),
                correction_executed=torch.zeros_like(done),selected_source=torch.ones_like(done,dtype=torch.int8))
            tr['observation_after']=post_observation(env.research_packet,done)
            batch=flatten(numpy_tree({'obs':packet,'transition':tr}))
            records=[]
            for i,(entry,m) in list(active.items()):
                row={k:v[i] for k,v in batch.items()}
                row.update(terminated=np.asarray(bool(terminated[i])),truncated=np.asarray(bool(truncated[i])))
                result=None
                if bool(done[i]):
                    diag=tr['diagnostics_after']
                    success=bool(diag['successes'][i]>=diag['max_goals'][i])
                    if env.cfg.precise_assembly.enable_retract:
                        success &= bool(diag['retract_succeeded'][i] or diag['is_random_goal'][i])
                    result=dict(steps=int(row['obs/step_id'])+1,started_episodes=1,completed_episodes=1,
                        successful_episodes=int(success),externally_censored=False,num_envs=1,
                        actual_parallel_envs=args.num_envs,trial_status='completed',source_env_index=i)
                    attempt_index[entry['id']]['outcome']='awaiting_audit';del active[i]
                records.append((row,entry,m,result))
            if records:writes.append(writer.submit(write_batch,records))
            if player.is_rnn and done.any():
                for state in player.states:state[:,done,:]=0
            consume_resets();steps+=1;publish()
        while writes:reap_writes(True)
        while audits:reap_audits(True)
        # A deliberate graceful stop is a known censor, not corrupt data.
        # Finish its real last post-frame, audit, persist the reason, then
        # delete the explicitly classified non-success payload.
        def close_partial():
            finished=[]
            for name,stream in streams.items():
                path=stream.finish(dict(steps=stream.length,started_episodes=1,completed_episodes=0,
                    successful_episodes=0,externally_censored=True,trial_status='external_budget_censored'))
                finished.append((name,path))
            streams.clear();return finished
        for name,path in writer.submit(close_partial).result():
            info=auditors.submit(finalize_stream,path,reference).result()
            if info['summary']['outcome']!='censored':raise ValueError('Unexpected graceful-stop outcome')
            attempt_index[name].update(outcome='external_censored',summary=info['summary'],initial=info['initial'],
                control_steps=info['length'],reason='Graceful target/deadline/STOP boundary, never a task failure')
            publish(force=True);shutil.rmtree(path)
        active.clear()
        publish('complete' if len(manifest['episodes'])>=args.target_successes else 'stopped',True)
    except BaseException:
        publish('collector_error',True)
        raise
    finally:
        writer.shutdown(wait=True);auditors.shutdown(wait=True)


def process_alive(pid):
    try:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        return b'visuotactile.collect_success_batch' in cmd and b'supervise' in cmd
    except (OSError,TypeError):return False


def recover_pending(root, run):
    """Complete files survive worker exit; only re-audited successes publish.

    Unknown/partial files are retained. A restart uses a new run seed/ID and
    never treats an existing filename as proof of successful completion.
    """
    from .finalize_collection import finalize_stream
    root=Path(root);manifest_path=root/'manifest.json'
    m=json.loads(manifest_path.read_text());reference=json.loads((root/'reference_metadata.json').read_text())
    present={r['id'] for r in m['episodes']};reports=[]
    for row in m['attempts']:
        if row['id'] in present or row.get('outcome') not in ('awaiting_audit','running','audit_error'):
            continue
        source=Path('outputs/visuotactile/tmp')/row['run_id']/row['id']
        published=root/'episodes'/row['id']
        if not source.exists() and published.exists():source=published
        if (source/'data.h5').exists():
            try:info=finalize_stream(source,reference)
            except Exception as exc:
                row.update(outcome='audit_error',error=repr(exc),retained_path=str(source))
                reports.append(dict(id=row['id'],error=repr(exc)));continue
            outcome=info['summary']['outcome']
            row.update(outcome=outcome,summary=info['summary'],initial=info['initial'],
                       control_steps=info['length'],effective_delay_counts=info['effective_delay_counts'])
            if outcome=='success':
                destination=root/'episodes'/row['id']
                if source!=destination:source.rename(destination)
                m['episodes'].append(dict(row,**info,directory='episodes/'+row['id'],valid=True,
                    parameter_updates=0,nominal_predictions_available=False,recovered_after_worker_exit=True))
                present.add(row['id'])
            reports.append(dict(id=row['id'],outcome=outcome))
            write_json(manifest_path,m)
            if outcome!='success':shutil.rmtree(source)
        else:
            row.update(outcome='external_censor_partial',retained_path=str(source),
                       reason='Worker exited before independent episode finalization; not a task failure')
    write_json(manifest_path,m)
    write_json(Path(run)/'recovery.json',dict(epoch=time.time(),reports=reports,successes=len(m['episodes'])))
    return reports


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('start','supervise','status','stop'))
    p.add_argument('--output',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--num-envs',type=int,default=4);p.add_argument('--threads',type=int,default=4)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--target-successes',type=int,default=400)
    p.add_argument('--deadline-epoch',type=float,default=DEADLINE)
    p.add_argument('--seed-start',type=int,default=931000)
    args=p.parse_args();run=DEFAULT_RUN;run.mkdir(parents=True,exist_ok=True)
    owner=run/'owner.json';old=json.loads(owner.read_text()) if owner.exists() else {}
    if args.command=='status':
        result=dict(supervisor=old,supervisor_alive=process_alive(old.get('pid')))
        for key in ('progress','resources','status'):
            path=run/(key+'.json');result[key]=json.loads(path.read_text()) if path.exists() else None
        if (args.output/'manifest.json').exists():
            m=json.loads((args.output/'manifest.json').read_text());result['manifest_successes']=len(m['episodes'])
            result['attempt_outcomes']=dict(Counter(a.get('outcome') for a in m['attempts']))
        print(json.dumps(result,indent=2));return
    if args.command=='stop':
        (run/'STOP').touch();print('Requested graceful stop for this collection only; check status.');return
    if args.command=='start':
        if process_alive(old.get('pid')):raise SystemExit('This collection supervisor is already alive')
        (run/'STOP').unlink(missing_ok=True)
        command=[sys.executable,'-m','visuotactile.collect_success_batch','supervise',*sys.argv[2:]]
        with (run/'supervisor.log').open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(owner,dict(pid=child.pid,started_epoch=time.time(),command=command))
        print(json.dumps(dict(pid=child.pid,log=str(run/'supervisor.log'),manifest=str(args.output/'manifest.json'))));return
    from .collection_resources import Supervisor
    import fcntl
    lock=(run/'supervisor.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (args.output/'manifest.json').exists():raise ValueError('Initialize and verify inherited manifest first')
    if sha256(CHECKPOINT)!=CHECKPOINT_SHA:raise ValueError('Released expert changed')
    supervisor=Supervisor(run,args.deadline_epoch,disk_reserve_bytes=12_000_000_000,average_only=True)
    recover_pending(args.output,run)
    session_file=run/'sessions.json';sessions=json.loads(session_file.read_text()) if session_file.exists() else []
    last_success=len(json.loads((args.output/'manifest.json').read_text())['episodes'])
    failures_without_progress=0
    try:
        for _ in range(20):
            m=json.loads((args.output/'manifest.json').read_text())
            if len(m['episodes'])>=args.target_successes or (run/'STOP').exists():break
            supervisor.check()
            index=len(sessions);name=f'r{args.seed_start+index}_{int(time.time())}'
            output=Path('outputs/visuotactile/run')/name
            log=run/(name+'.log')
            cmd=['env','MALLOC_ARENA_MAX=2',f'RESEARCH_THREADS={args.threads}',
                'RESEARCH_CPUS=20-23','bash','visuotactile/run_cpu.sh','-m','visuotactile.collect',
                '--output',str(output),'--production-root',str(args.output),'--run-id',name,
                '--env-spacing','8.0','--num-envs',str(args.num_envs),'--threads',str(args.threads),'--device',args.device,
                '--target-successes',str(args.target_successes),'--deadline-epoch',str(args.deadline_epoch),
                '--seed',str(args.seed_start+index),'--camera-rig','wrist_bracket_close',
                '--image-size','448','252','--render-mode','quality','--wrench-frame-mode','world_hold','--disable-extension-watch']
            session=dict(id=name,seed=args.seed_start+index,command=cmd,started_epoch=time.time(),log=str(log))
            sessions.append(session);write_json(session_file,sessions)
            write_json(run/'status.json',dict(state='running',pid=os.getpid(),session=name,log=str(log)))
            supervisor_error=None
            try:
                rc=supervisor.run(cmd,log,timeout=max(1,args.deadline_epoch-time.time()),reserve=2_000_000_000,
                                  stop_file=run/'STOP',heartbeat=run/'progress.json')
            except RuntimeError as exc:
                if 'heartbeat stalled' not in str(exc):raise
                supervisor_error=str(exc);rc=-9
            session.update(returncode=rc,finished_epoch=time.time());write_json(session_file,sessions)
            m=json.loads((args.output/'manifest.json').read_text());count=len(m['episodes'])
            if count>=args.target_successes or (run/'STOP').exists():break
            text=log.read_text(errors='replace')[-20000:]
            session['tail_error']=text[-4000:]
            write_json(session_file,sessions)
            m['attempts'].append(dict(id=name+'_worker_exit',seed=args.seed_start+index,
                run_id=name,outcome='startup_error' if not any(a.get('run_id')==name for a in m['attempts']) else 'worker_error',
                returncode=rc,log=str(log),error=text[-4000:],episode_started=False))
            write_json(args.output/'manifest.json',m)
            failures_without_progress=failures_without_progress+1 if count==last_success else 0
            last_success=count
            # Only known transient native startup failures get finite automatic
            # recovery. Python/data errors need a code fix, not blind restarts.
            transient=supervisor_error is not None or any(k in text for k in ('LLVM ERROR: out of memory','FileThread','inotify_add_watch','std::system_error'))
            if not transient or failures_without_progress>=3:
                raise RuntimeError(f'Worker exit {rc}; inspect {log}; finite recovery exhausted or nontransient error')
            recover_pending(args.output,run)
            supervisor.wait(15)
        count=len(json.loads((args.output/'manifest.json').read_text())['episodes'])
        write_json(run/'status.json',dict(state='complete' if count>=args.target_successes else 'stopped',
            pid=os.getpid(),successes=count,finished_epoch=time.time()))
    except BaseException as exc:
        write_json(run/'status.json',dict(state='needs_recovery',error=repr(exc),pid=os.getpid(),finished_epoch=time.time()))
        raise


if __name__=='__main__':main()
