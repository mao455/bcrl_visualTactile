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


def paired_frontends(execute, output, epochs):
    """Join both independent learners before admitting any dependent stage.

    A normal failure preserves the other learner's completed work. Each child
    retains the same absolute deadline and durable per-stage receipts.
    """
    from concurrent.futures import ThreadPoolExecutor
    errors={};began=time.time()
    with ThreadPoolExecutor(max_workers=2,thread_name_prefix='fresh-dp') as pool:
        jobs={role:pool.submit(execute,'frontend_'+role,
            ['tools.tight_fresh_train','train','--root',output,'--role',role,'--epochs',epochs],
            output/role/'training/finished.json',update_status=False) for role in ('A','B')}
        for role,future in jobs.items():
            try:future.result()
            except Exception as error:errors[role]=repr(error)
    return dict(started=began,ended=time.time(),roles=['A','B'],errors=errors,
        complete=not errors,scope='Two independent DP processes only; unchanged per-role data, RNG and epoch budgets; dependent stages wait for both')


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
        parallel_frontends=False,latest_repeat=None,evaluation_lanes=1,execution_backend=None,episode_io=None,vision_microbatch=2):
    if type(evaluation_lanes) is not int or evaluation_lanes<1:
        raise ValueError('Positive evaluation concurrency required')
    if type(vision_microbatch) is not int or vision_microbatch<1:raise ValueError('Positive vision microbatch required')
    ledger=output.with_name(output.name+'_stages')
    if ledger.exists() or output.exists():raise FileExistsError('Preserve previous cycle identity')
    if not read(collection/'result.json').get('software_complete'):raise ValueError('Collection not complete')
    if not read(collection/'correction_entries.json'):raise ValueError('No qualified successful expert suffixes')
    session=read(root/'session.json');deadline=session['deadline']
    if parallel_frontends and session.get('resource_policy')!='full_gpu_v1':
        raise ValueError('Parallel frontends require the explicit full-GPU resource contract')
    sources={str(p):sha256(p) for folder in ['tools','visuotactile'] for p in sorted(Path(folder).rglob('*.py'))}
    ledger.mkdir(parents=True)
    registration=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,source_root=str(root),collection=str(collection),output=str(output),
        collection_entries_sha256=sha256(collection/'correction_entries.json'),
        frontend_epochs=epochs,backend_epochs=rear_epochs,backend_continuation=continue_rear,development_count_per_role=count,seed=seed,
        parallel_frontends=parallel_frontends,
        frontend_latest_repeat=latest_repeat,backend_latest_repeat=1,
        development_startup_seconds=300 if evaluation_lanes>1 else 180,
        evaluation_lanes=evaluation_lanes,execution_backend=execution_backend,episode_io=episode_io,vision_microbatch=vision_microbatch,
        started=time.time(),deadline=deadline,pid=os.getpid(),test_used=False)
    if reuse_index is not None:
        if not read(reuse_index).get('complete'):raise ValueError('Reusable frozen vision cache is incomplete')
        registration['reuse_visual_index']=dict(path=str(reuse_index),sha256=sha256(reuse_index))
    atomic_json(ledger/'registration.json',registration)
    stages=[]
    def execute(name,args,completion,*,update_status=True):
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
            (['--reuse-index',reuse_index] if reuse_index is not None else []),output/'B/cache_index.json')
        admitted=False
        if parallel_frontends:
            admission=parallel_admission(read(root/'shared_resources.json'),time.time())
            atomic_json(ledger/'parallel_admission.json',admission)
            admitted=admission['admitted']
        if admitted:
            atomic_json(ledger/'status.json',dict(stage='parallel_frontends',roles=['A','B'],started=time.time()))
            paired=paired_frontends(execute,output,epochs)
            atomic_json(ledger/'parallel_frontends.json',paired)
            if not paired['complete']:raise RuntimeError('Frontend stage failed: '+str(paired['errors']))
        else:
            for role in ['A','B']:
                execute('frontend_'+role,['tools.tight_fresh_train','train','--root',output,'--role',role,'--epochs',epochs],output/role/'training/finished.json')
        for role in ['A','B']:
            rear=output/role/'backend'
            execute('backend_prepare_'+role,['tools.tight_fresh_backend','--root',output,'--role',role,'--epochs',rear_epochs]+(['--continue-rear'] if continue_rear else []),rear/'prepared.json')
            execute('backend_cache_'+role,['tools.tight_finish','residual_cache','--root',rear],rear/'cache_complete.json')
            execute('backend_train_'+role,['visuotactile.anchor_continue','residual-train','--root',rear/role,'--job','image','--output',rear/role/'training'],rear/role/'training/finished.json')
            execute('backend_offline_'+role,['visuotactile.anchor_continue','residual-offline','--root',rear/role,'--job','image','--output',rear/role/'offline'],rear/role/'offline/result.json')
        development=output/'development'
        if evaluation_lanes>1:
            arguments=['tools.tight_fresh_batch_run','prepare','--root',output,'--output',development,
                '--count',count,'--seed',seed,'--lanes',evaluation_lanes,'--vision-microbatch',vision_microbatch]
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
    p.add_argument('--latest-repeat',type=int,choices=[1,2,4])
    p.add_argument('--evaluation-lanes',type=int,default=1)
    p.add_argument('--backend-json',type=Path)
    p.add_argument('--prefetch-mib',type=int,help='Opt-in frozen CPU episode staging, at most 1024 MiB per trainer')
    p.add_argument('--vision-microbatch',type=int,default=2)
    a=p.parse_args()
    if not 1<=a.epochs<=100 or not 1<=a.rear_epochs<=100 or a.count<1 or a.evaluation_lanes<1:p.error('Positive finite stage exposure required')
    if a.prefetch_mib is not None and not 1<=a.prefetch_mib<=1024:p.error('Prefetch budget must be 1–1024 MiB')
    if a.vision_microbatch<1:p.error('Positive vision microbatch required')
    run(a.root.resolve(),a.collection.resolve(),a.output.resolve(),a.epochs,a.rear_epochs,a.count,a.seed,
        continue_rear=a.continue_rear,reuse_index=a.reuse_index.resolve() if a.reuse_index else None,
        parallel_frontends=a.parallel_frontends,latest_repeat=a.latest_repeat,
        evaluation_lanes=a.evaluation_lanes,execution_backend=read(a.backend_json) if a.backend_json else None,
        episode_io=dict(version='frozen_episode_prefetch_v1',max_bytes=a.prefetch_mib<<20,
            minimum_available_bytes=10<<30) if a.prefetch_mib is not None else None,vision_microbatch=a.vision_microbatch)


if __name__=='__main__':main()
