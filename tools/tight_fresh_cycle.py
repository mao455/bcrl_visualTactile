"""Execute one explicitly chosen finite DAgger cycle and preserve stage receipts.

This runner does not choose scientific follow-ups or declare learning success.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read
from tools.tight_fresh_eval_recovery import wait_owned, stop_owned


def parallel_admission(row, now):
    """Joint launch headroom, in addition to each worker's ordinary reserves."""
    age=now-row['utc_epoch']
    free=row.get('gpu_free_mib')
    return dict(admitted=0<=age<=8 and isinstance(free,(int,float)) and free>=22*1024
        and row['ram_available_bytes']>=24*1024**3,
        observed_gpu_free_mib=row.get('gpu_free_mib'),observed_ram_available_bytes=row['ram_available_bytes'],
        telemetry_age_seconds=age,required_free_gpu_mib=22*1024,required_available_ram_bytes=24*1024**3,
        scope='Conservative joint startup admission; per-worker 4GiB VRAM/10GiB RAM checks continue; not a hard allocation quota')


def wait_parallel_admission(resource_path, receipt_path, *, deadline, stop_paths=()):
    """Wait at most 30 seconds for a post-stage sample without extending the session.

    A fresh sample must be newer than this admission attempt, so a just-finished
    GPU stage cannot be rejected solely on its last still-allocated sample.
    """
    began=time.time(); started=time.monotonic(); end=min(deadline,began+30)
    samples=[]; latest={}; stop_paths=tuple(Path(path) for path in stop_paths)
    def save(reason, admitted=False):
        receipt=dict(latest,admitted=admitted,reason=reason,started_utc=began,
            ended_utc=time.time() if reason!='waiting' else None,deadline=deadline,
            admission_deadline=end,max_wait_seconds=30,wait_seconds=time.monotonic()-started,
            samples=samples,required_free_gpu_mib=22*1024,required_available_ram_bytes=24*1024**3,
            semantics='Require a sample newer than admission start and age 0–8s; thresholds unchanged; no deadline reset')
        atomic_json(receipt_path,receipt)
        return receipt
    while True:
        now=time.time()
        if any(path.exists() for path in stop_paths):
            save('requested_stop')
            raise InterruptedError('Fresh cycle STOP during joint resource admission')
        if now>=deadline:
            save('original_session_deadline')
            raise TimeoutError('Original fresh session deadline during joint resource admission')
        if time.monotonic()-started>=30 or now>=end:
            return save('wait_exhausted')
        try:
            row=read(resource_path);latest=parallel_admission(row,now)
            latest.update(observed_utc=row['utc_epoch'],sample_after_admission_start=row['utc_epoch']>began)
            latest['admitted']=latest['admitted'] and latest['sample_after_admission_start']
            latest['gate_reason']=('admitted' if latest['admitted'] else
                'waiting_for_post_admission_sample' if not latest['sample_after_admission_start'] else
                'stale_or_future_sample' if not 0<=latest['telemetry_age_seconds']<=8 else
                'insufficient_joint_headroom')
        except (OSError,ValueError,KeyError,TypeError) as error:
            latest=dict(admitted=False,gate_reason='telemetry_unavailable',error=repr(error))
        samples.append(dict(latest,checked_utc=now))
        if latest['admitted']:
            return save('admitted',True)
        save('waiting')
        time.sleep(min(.25,max(.001,end-time.time()),max(.001,30-(time.monotonic()-started))))


def paired_frontends(execute, output, epochs, *, offline_anchor_batch=None):
    """Join both independent learners before admitting any dependent stage.

    A normal failure preserves the other learner's completed work. Each child
    retains the same absolute deadline and durable per-stage receipts.
    """
    from concurrent.futures import ThreadPoolExecutor
    errors={};began=time.time()
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='fresh-dp') as pool:
        jobs={role:pool.submit(execute,'frontend_'+role,
            ['tools.tight_fresh_train','train','--root',output,'--role',role,'--epochs',epochs]+
            (['--offline-anchor-batch',offline_anchor_batch] if offline_anchor_batch is not None else []),
            output/role/'training/finished.json',update_status=False) for role in ('A','B')}
        for role,future in jobs.items():
            try:future.result()
            except Exception as error:errors[role]=repr(error)
    return dict(started=began,ended=time.time(),roles=['A','B'],errors=errors,
        complete=not errors,scope='Two independent DP processes only; unchanged per-role data, RNG and epoch budgets; dependent stages wait for both')


def backend_chain(execute, output, role, epochs, *, continue_rear=False, update_status=True, stop_paths=(),
        offline_anchor_batch=None):
    """The same per-role commands and dependencies in serial and parallel modes."""
    rear=output/role/'backend'
    options=dict(update_status=update_status)
    if stop_paths:options['stop_paths']=stop_paths
    execute('backend_prepare_'+role,['tools.tight_fresh_backend','--root',output,'--role',role,'--epochs',epochs]+
        (['--continue-rear'] if continue_rear else [])+
        (['--offline-anchor-batch',offline_anchor_batch] if offline_anchor_batch is not None else []),rear/'prepared.json',**options)
    execute('backend_cache_'+role,['tools.tight_finish','residual_cache','--root',rear],rear/'cache_complete.json',**options)
    execute('backend_train_'+role,['visuotactile.anchor_continue','residual-train','--root',rear/role,
        '--job','image','--output',rear/role/'training'],rear/role/'training/finished.json',**options)
    execute('backend_offline_'+role,['visuotactile.anchor_continue','residual-offline','--root',rear/role,
        '--job','image','--output',rear/role/'offline'],rear/role/'offline/result.json',**options)


def paired_backends(execute, output, epochs, *, continue_rear=False, stop_paths=(),offline_anchor_batch=None):
    """Join independent complete backend chains; one failure does not cancel its peer."""
    from concurrent.futures import ThreadPoolExecutor
    errors={};outcomes={};began=time.time()
    def role_chain(role):
        start=time.time()
        try:
            backend_chain(execute,output,role,epochs,continue_rear=continue_rear,
                update_status=False,stop_paths=stop_paths,offline_anchor_batch=offline_anchor_batch)
        except Exception as error:
            return dict(complete=False,error=repr(error),started=start,ended=time.time(),seconds=time.time()-start)
        return dict(complete=True,started=start,ended=time.time(),seconds=time.time()-start)
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='fresh-backend') as pool:
        jobs={role:pool.submit(role_chain,role) for role in ('A','B')}
        for role,future in jobs.items():
            outcomes[role]=future.result()
            if not outcomes[role]['complete']:errors[role]=outcomes[role]['error']
    return dict(started=began,ended=time.time(),seconds=time.time()-began,roles=['A','B'],errors=errors,
        outcomes=outcomes,complete=not errors,
        scope='Per-role prepare/cache/train/offline commands unchanged; two frontend completions required; both chains join before development; actual overlap is not a speed claim')


def wait_backend_owned(process, deadline, stop_paths):
    """Keep the ordinary absolute bound and process-group cleanup, adding cycle STOP."""
    try:
        while process.poll() is None:
            if any(Path(path).exists() for path in stop_paths):return 'requested_stop'
            now=time.time()
            if now>=deadline:return 'original_session_deadline'
            time.sleep(min(.5,max(.001,deadline-now)))
    finally:
        if process.poll() is None:stop_owned(process)
    return None


def configure_frontend_sampling(output, repeat):
    """Bind optional latest-episode reuse before either learner starts."""
    from visuotactile.latest_takeover_sampling import SAMPLING_VERSION,sampling_plan
    spec=dict(version=SAMPLING_VERSION,latest_repeat=repeat)
    pending={};populations={}
    for role in ('A','B'):
        directory=output/role
        if (directory/'training').exists():raise ValueError('Cannot change a started frontend sampling contract')
        data=read(directory/'state_data/data.json')
        validated,multiplicities=sampling_plan(data,spec)
        prior=read(directory/'continuation.json')
        if 'sampling_contract' in prior:raise ValueError('Preserve the registered frontend sampling contract')
        pending[role]=dict(prior,sampling_contract=validated)
        populations[role]=dict(data_sha256=sha256(directory/'state_data/data.json'),
            latest_ids=data['aggregation']['new_training_ids'],episode_multiplicities=multiplicities)
    if (populations['A']['latest_ids']!=populations['B']['latest_ids']
            or populations['A']['episode_multiplicities']!=populations['B']['episode_multiplicities']):
        raise ValueError('A/B sampling populations differ')
    for role,value in pending.items():atomic_json(output/role/'continuation.json',value)
    receipt=dict(contract=spec,roles=populations,backend_latest_repeat=1,
        scope='Frontend visits only; each replay recomputes complete actual history, retains legal expert masks; backend keeps one visit per episode per epoch; reuse is not new data')
    atomic_json(output/'frontend_sampling.json',receipt)
    return receipt


def configure_episode_io(output, config):
    """Register identical bounded frozen-feature staging before either trainer."""
    from visuotactile.episode_io import normalize_io
    validated=normalize_io(config)
    if validated is None:return None
    for role in ('A','B'):
        if (output/role/'training').exists() or (output/role/'episode_io.json').exists():
            raise ValueError('Cannot replace started or registered episode I/O')
    for role in ('A','B'):atomic_json(output/role/'episode_io.json',validated)
    return validated


def run(root, collection, output, epochs, rear_epochs, count, seed, *, continue_rear=False, reuse_index=None,
        parallel_frontends=False,latest_repeat=None,evaluation_lanes=1,execution_backend=None,episode_io=None,vision_microbatch=2,
        paired_episode_io=False,vision_tail_profile=None,inactive_render_profile=None,payload_audit_memo=False,
        parallel_backends=False,offline_anchor_batch=None,backend_offline_anchor_batch=None):
    from visuotactile.models.native_vision import validate_vision_tail_profile
    tail=validate_vision_tail_profile(vision_tail_profile)
    from tools.tight_fresh_batch_run import validate_inactive_render_request
    inactive=validate_inactive_render_request(inactive_render_profile,execution_backend)
    if type(payload_audit_memo) is not bool:raise ValueError('Payload audit memo must be an explicit boolean')
    if type(parallel_backends) is not bool:raise ValueError('Parallel backends must be an explicit boolean')
    offline_planning=None
    if offline_anchor_batch is not None:
        from visuotactile.offline_planning import profile
        offline_planning=profile(offline_anchor_batch)
    backend_offline_planning=None
    if backend_offline_anchor_batch is not None:
        from visuotactile.offline_planning import profile
        backend_offline_planning=profile(backend_offline_anchor_batch)
    if paired_episode_io and not parallel_frontends:
        raise ValueError('Paired episode I/O requires two parallel independent frontends')
    if type(evaluation_lanes) is not int or evaluation_lanes<1:
        raise ValueError('Positive evaluation concurrency required')
    if type(vision_microbatch) is not int or vision_microbatch<1:raise ValueError('Positive vision microbatch required')
    if evaluation_lanes==1 and vision_microbatch!=2:
        raise ValueError('Serial evaluation retains vision microbatch2; choose batch evaluation to apply another size')
    if evaluation_lanes==1 and tail is not None:
        raise ValueError('Exact vision tail profile requires explicit batch evaluation')
    if evaluation_lanes==1 and inactive is not None:
        raise ValueError('Inactive render products require explicit batch evaluation')
    ledger=output.with_name(output.name+'_stages')
    if ledger.exists() or output.exists():raise FileExistsError('Preserve previous cycle identity')
    if not read(collection/'result.json').get('software_complete'):raise ValueError('Collection not complete')
    if not read(collection/'correction_entries.json'):raise ValueError('No qualified successful expert suffixes')
    session=read(root/'session.json');deadline=session['deadline']
    if parallel_frontends and session.get('resource_policy')!='full_gpu_v1':
        raise ValueError('Parallel frontends require the explicit full-GPU resource contract')
    if parallel_backends and session.get('resource_policy')!='full_gpu_v1':
        raise ValueError('Parallel backends require the explicit full-GPU resource contract')
    sources={str(p):sha256(p) for folder in ['tools','visuotactile'] for p in sorted(Path(folder).rglob('*.py'))}
    ledger.mkdir(parents=True)
    registration=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,source_root=str(root),collection=str(collection),output=str(output),
        collection_entries_sha256=sha256(collection/'correction_entries.json'),
        frontend_epochs=epochs,backend_epochs=rear_epochs,backend_continuation=continue_rear,development_count_per_role=count,seed=seed,
        parallel_frontends=parallel_frontends,
        parallel_admission_contract=dict(version='fresh_stage_sample_wait_v1',max_wait_seconds=30,
            sample_newer_than_admission=True,minimum_free_gpu_mib=22*1024,minimum_available_ram_bytes=24*1024**3)
            if parallel_frontends else None,
        paired_episode_io=paired_episode_io,
        frontend_latest_repeat=latest_repeat,backend_latest_repeat=1,
        development_startup_seconds=300 if evaluation_lanes>1 else 180,
        evaluation_lanes=evaluation_lanes,execution_backend=execution_backend,episode_io=episode_io,vision_microbatch=vision_microbatch,
        started=time.time(),deadline=deadline,pid=os.getpid(),test_used=False)
    if tail is not None:registration['vision_tail_profile']=tail
    if inactive is not None:registration['inactive_render_profile']=inactive
    if payload_audit_memo:registration['payload_audit_memo']=True
    if offline_planning is not None:registration['frontend_offline_planning']=offline_planning
    if backend_offline_planning is not None:registration['backend_offline_planning']=backend_offline_planning
    if parallel_backends:
        registration['parallel_backends']=True
        registration['parallel_backend_contract']=dict(version='independent_backend_chains_v1',
            depends_on=['frontend_A','frontend_B'],role_stages=['prepare','cache','train','offline'],
            join_before='development_prepare',ordinary_role_failure='preserve_peer_chain',
            admission=dict(version='fresh_stage_sample_wait_v1',max_wait_seconds=30,
                sample_newer_than_admission=True,minimum_free_gpu_mib=22*1024,minimum_available_ram_bytes=24*1024**3),
            fallback='serial',deadline=deadline,stop_paths=[str(p/'STOP') for p in (root,output,ledger)])
    if reuse_index is not None:
        if not read(reuse_index).get('complete'):raise ValueError('Reusable frozen vision cache is incomplete')
        registration['reuse_visual_index']=dict(path=str(reuse_index),sha256=sha256(reuse_index))
    atomic_json(ledger/'registration.json',registration)
    stages=[]
    def execute(name,args,completion,*,update_status=True,stop_paths=()):
        if any(Path(path).exists() for path in stop_paths):raise InterruptedError('Fresh cycle STOP before '+name)
        if time.time()>=deadline:raise TimeoutError('Original fresh session deadline exhausted')
        if any(sha256(p)!=s for p,s in sources.items()):raise ValueError('Active cycle source drift')
        command=[sys.executable,'-m']+list(map(str,args))
        if update_status:atomic_json(ledger/'status.json',dict(stage=name,command=command,started=time.time()))
        start=time.time()
        with (ledger/(name+'.log')).open('x') as stream:
            proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            try:
                proc_stat=Path('/proc',str(proc.pid),'stat').read_text().rsplit(')',1)[1].split()
                atomic_json(ledger/(name+'_worker.json'),dict(stage=name,command=command,pid=proc.pid,
                    proc_start_ticks=int(proc_stat[19]),cwd=str(Path.cwd()),
                    started=start,deadline=deadline,completion=str(completion)))
                if stop_paths:
                    reason=wait_backend_owned(proc,deadline,stop_paths)
                else:
                    reason=wait_owned(proc,deadline,
                        startup_marker=completion.parent/'initialization.json' if name=='development' else None,
                        startup_seconds=registration['development_startup_seconds'] if name=='development' else None)
            finally:
                if proc.poll() is None:stop_owned(proc)
        receipt=dict(stage=name,command=command,returncode=proc.returncode,started=start,seconds=time.time()-start,
            stop_reason=reason,
            completion=str(completion),completion_sha256=sha256(completion) if completion.exists() else None,utc=time.time())
        atomic_json(ledger/(name+'.json'),receipt);stages.append(receipt)
        if reason or proc.returncode or not completion.exists():raise RuntimeError('Stage failed: '+name)
        proof=read(completion)
        if any(proof.get(key) is False for key in ('complete','software_complete','target_reached')):
            raise RuntimeError('Stage receipt is incomplete: '+name)
        print(json.dumps(receipt),flush=True)
    try:
        execute('aggregate',['tools.tight_fresh_dagger','aggregate','--root',root,'--collection',collection,'--output',output],output/'prepared.json')
        if episode_io is not None:
            atomic_json(ledger/'episode_io.json',configure_episode_io(output,episode_io))
        if latest_repeat is not None:
            sampling=configure_frontend_sampling(output,latest_repeat)
            atomic_json(ledger/'frontend_sampling.json',sampling)
        execute('visual_cache',['tools.tight_fresh_dagger','cache','--root',root,'--output',output]+
            (['--payload-audit-memo'] if payload_audit_memo else [])+
            (['--reuse-index',reuse_index] if reuse_index is not None else []),output/'B/cache_index.json')
        admitted=False
        if parallel_frontends:
            atomic_json(ledger/'status.json',dict(stage='parallel_admission',started=time.time(),deadline=deadline))
            admission=wait_parallel_admission(root/'shared_resources.json',ledger/'parallel_admission.json',
                deadline=deadline,stop_paths=(root/'STOP',output/'STOP',ledger/'STOP'))
            admission['frontend_mode']=('parallel' if admission['admitted'] else
                'paired_rejected_no_serial_fallback' if paired_episode_io else 'serial_fallback')
            atomic_json(ledger/'parallel_admission.json',admission)
            admitted=admission['admitted']
        if paired_episode_io:
            if not admitted:
                raise RuntimeError('Paired episode I/O lacks joint resource admission; no serial fallback')
            from visuotactile.paired_episode_io import configure_pairing
            atomic_json(ledger/'paired_episode_io.json',configure_pairing(output,epochs))
        if admitted:
            atomic_json(ledger/'status.json',dict(stage='parallel_frontends',roles=['A','B'],started=time.time()))
            paired=paired_frontends(execute,output,epochs,**(
                dict(offline_anchor_batch=offline_anchor_batch) if offline_anchor_batch is not None else {}))
            atomic_json(ledger/'parallel_frontends.json',paired)
            if not paired['complete']:raise RuntimeError('Frontend stage failed: '+str(paired['errors']))
        else:
            for role in ['A','B']:
                execute('frontend_'+role,['tools.tight_fresh_train','train','--root',output,'--role',role,'--epochs',epochs]+
                    (['--offline-anchor-batch',offline_anchor_batch] if offline_anchor_batch is not None else []),output/role/'training/finished.json')
        backends_admitted=False
        backend_stop_paths=(root/'STOP',output/'STOP',ledger/'STOP') if parallel_backends else ()
        if parallel_backends:
            atomic_json(ledger/'status.json',dict(stage='parallel_backend_admission',started=time.time(),deadline=deadline))
            admission=wait_parallel_admission(root/'shared_resources.json',ledger/'parallel_backend_admission.json',
                deadline=deadline,stop_paths=backend_stop_paths)
            admission['backend_mode']='parallel' if admission['admitted'] else 'serial_fallback'
            atomic_json(ledger/'parallel_backend_admission.json',admission)
            backends_admitted=admission['admitted']
        if backends_admitted:
            atomic_json(ledger/'status.json',dict(stage='parallel_backends',roles=['A','B'],started=time.time()))
            paired=paired_backends(execute,output,rear_epochs,continue_rear=continue_rear,stop_paths=backend_stop_paths,
                offline_anchor_batch=backend_offline_anchor_batch)
            atomic_json(ledger/'parallel_backends.json',paired)
            if not paired['complete']:raise RuntimeError('Backend stage failed: '+str(paired['errors']))
        else:
            for role in ['A','B']:
                backend_chain(execute,output,role,rear_epochs,continue_rear=continue_rear,stop_paths=backend_stop_paths,
                    offline_anchor_batch=backend_offline_anchor_batch)
        if any(path.exists() for path in backend_stop_paths):raise InterruptedError('Fresh cycle STOP before development')
        development=output/'development'
        if evaluation_lanes>1:
            arguments=['tools.tight_fresh_batch_run','prepare','--root',output,'--output',development,
                '--count',count,'--seed',seed,'--lanes',evaluation_lanes,'--vision-microbatch',vision_microbatch]
            if tail is not None:arguments+=['--vision-exact-tail']
            if inactive is not None:arguments+=['--inactive-render-products']
            if execution_backend is not None:
                backend_path=ledger/'execution_backend.json';atomic_json(backend_path,execution_backend)
                arguments+=['--backend-json',backend_path]
            execute('development_prepare',arguments,development/'plan.json')
            execute('development',['tools.tight_fresh_batch_run','run','--output',development],development/'result.json')
        else:
            execute('development_prepare',['tools.tight_fresh_rollouts','prepare','--root',output,'--output',development,'--count',count,'--seed',seed],development/'plan.json')
            execute('development',['tools.tight_fresh_rollouts','run','--root',output,'--output',development],development/'result.json')
        atomic_json(ledger/'finished.json',dict(software_complete=True,stages=stages,utc=time.time(),
            research_conclusion='Pending actual Agent review of paired errors, exposure and complete autonomous outcomes'))
        atomic_json(ledger/'status.json',dict(stage='complete',software_complete=True,utc=time.time()))
    except BaseException as error:
        atomic_json(ledger/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),stages=stages,utc=time.time()))
        atomic_json(ledger/'status.json',dict(stage='failed',error=repr(error),utc=time.time()))
        raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--rear-epochs',type=int,default=5);p.add_argument('--count',type=int,default=4);p.add_argument('--seed',type=int,default=28140000)
    p.add_argument('--continue-rear',action='store_true')
    p.add_argument('--reuse-index',type=Path)
    p.add_argument('--parallel-frontends',action='store_true')
    p.add_argument('--offline-anchor-batch',type=int,help='Explicit frontend offline metrics DDIM batch only; does not enable backend cache batching')
    p.add_argument('--backend-offline-anchor-batch',type=int,help='Separate opt-in backend cache DDIM batch; requires its own cache acceptance')
    p.add_argument('--parallel-backends',action='store_true',
        help='Opt-in independent A/B backend chains after both frontends finish and fresh joint admission; serial fallback')
    p.add_argument('--paired-episode-io',action='store_true',
        help='Opt-in same-deck A/B I/O rendezvous; requires parallel frontends and shared frozen shards')
    p.add_argument('--latest-repeat',type=int,choices=[1,2,4])
    p.add_argument('--evaluation-lanes',type=int,default=1)
    p.add_argument('--backend-json',type=Path)
    p.add_argument('--prefetch-mib',type=int,help='Opt-in frozen CPU episode staging, at most 1024 MiB per trainer')
    p.add_argument('--vision-microbatch',type=int,default=2)
    p.add_argument('--vision-exact-tail',action='store_true',help='Explicit evaluation-only DINO tail-shape candidate')
    p.add_argument('--inactive-render-products',action='store_true',help='Explicit evaluation-only CPU-standard inactive render candidate')
    p.add_argument('--payload-audit-memo',action='store_true',help='Explicit frozen payload SHA memo for the cache stage only')
    a=p.parse_args()
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    inactive=None
    if a.inactive_render_products:
        from visuotactile.inactive_render_products import PROFILE
        inactive=PROFILE
    if not 1<=a.epochs<=100 or not 1<=a.rear_epochs<=100 or a.count<1 or a.evaluation_lanes<1:p.error('Positive finite stage exposure required')
    if a.prefetch_mib is not None and not 1<=a.prefetch_mib<=1024:p.error('Prefetch budget must be 1–1024 MiB')
    if a.vision_microbatch<1:p.error('Positive vision microbatch required')
    if a.paired_episode_io and not a.parallel_frontends:p.error('Paired episode I/O requires --parallel-frontends')
    run(a.root.resolve(),a.collection.resolve(),a.output.resolve(),a.epochs,a.rear_epochs,a.count,a.seed,
        continue_rear=a.continue_rear,reuse_index=a.reuse_index.resolve() if a.reuse_index else None,
        parallel_frontends=a.parallel_frontends,paired_episode_io=a.paired_episode_io,latest_repeat=a.latest_repeat,
        evaluation_lanes=a.evaluation_lanes,execution_backend=read(a.backend_json) if a.backend_json else None,
        episode_io=dict(version='frozen_episode_prefetch_v1',max_bytes=a.prefetch_mib<<20,
            minimum_available_bytes=10<<30) if a.prefetch_mib is not None else None,vision_microbatch=a.vision_microbatch,
        vision_tail_profile=EXACT_TAIL_PROFILE if a.vision_exact_tail else None,
        inactive_render_profile=inactive,payload_audit_memo=a.payload_audit_memo,parallel_backends=a.parallel_backends,
        offline_anchor_batch=a.offline_anchor_batch,backend_offline_anchor_batch=a.backend_offline_anchor_batch)


if __name__=='__main__':main()
