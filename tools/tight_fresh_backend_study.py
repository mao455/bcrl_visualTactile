"""Run an explicitly chosen paired rear-loss study with unchanged DP/data/caches.

This script executes a finite registered comparison; it makes no research decision.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from tools.tight_fresh_backend import prepare as prepare_backend
from tools.tight_fresh_eval_recovery import wait_owned
from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.fresh_recovery import FRESH_RECOVERY_WEIGHTING_VERSION
from visuotactile.training_resources import atomic_json


def prepare(source, output, epochs, ticks, weight, count, seed):
    if output.exists():raise FileExistsError('Preserve prior study identity')
    if not 1<=epochs<=5 or not 1<=ticks<=48 or not 1<weight<=10 or not 1<=count<=8:
        raise ValueError('Require a finite small paired continuation')
    session=read(source/'session.json')
    if time.time()>=session['deadline']:raise TimeoutError('Original session expired')
    development=source/'development'
    if not read(development/'result.json').get('software_complete'):
        raise ValueError('Review the source complete autonomous cycle first')
    plan=read(development/'plan.json');pairs={role:completed_pair(source,role) for role in ('A','B')}
    if plan['pairs']!=pairs:raise ValueError('Source autonomous evidence uses a different full pair')
    rows=read(development/'results.json')
    if (len(rows)!=len(plan['cases']) or any(not row['result'].get('legal') or not row['result'].get('software_complete') for row in rows)):
        raise ValueError('Source autonomous population is incomplete')
    protected={}
    for role,pair in pairs.items():
        for key in ('frontend','backend'):
            protected[pair[key]]=pair[key+'_sha256']
        for relative in ('state_data/data.json','state_data/statistics.pt','cache_index.json',
                         f'backend/{role}/nominal/index.json',f'backend/{role}/nominal/scales.json'):
            path=source/role/relative;protected[str(path)]=sha256(path)
        index=read(source/role/'backend'/role/'nominal/index.json')
        if (index['base_sha256']!=pair['frontend_sha256']
                or index['data_sha256']!=sha256(source/role/'state_data/data.json')):
            raise ValueError('Source DP cache does not match frozen pair/data')
    output.mkdir(parents=True)
    for arm,loss_weight in (('control',1.),('weighted',weight)):
        root=output/arm;root.mkdir()
        atomic_json(root/'session.json',session)
        (root/'shared_resources.json').symlink_to(source/'shared_resources.json')
        for role,pair in pairs.items():
            directory=root/role;directory.mkdir()
            for name in ('state_data','training','cache_index.json'):
                (directory/name).symlink_to(source/role/name)
            atomic_json(directory/'continuation.json',dict(path=pair['frontend'],sha256=pair['frontend_sha256'],
                statistics_sha256=sha256(source/role/'state_data/statistics.pt'),
                previous_backend=pair['backend'],previous_backend_sha256=pair['backend_sha256'],
                initialization='Frozen current DP; selected current rear parameters with a fresh optimizer'))
            prepare_backend(root,role,epochs,continue_rear=True,fresh_recovery_weighting=dict(
                version=FRESH_RECOVERY_WEIGHTING_VERSION,ticks=ticks,backend_weight=loss_weight))
            nominal=directory/'backend'/role/'nominal';nominal.rmdir()
            original=(source/role/'backend'/role/'nominal').resolve()
            nominal.symlink_to(original,target_is_directory=True)
            atomic_json(directory/'backend/cache_complete.json',dict(complete=True,recomputed=False,
                unchanged_frontend_sha256=pair['frontend_sha256'],unchanged_data_sha256=sha256(directory/'state_data/data.json'),
                reused_nominal_directory=str(original),index_sha256=sha256(original/'index.json'),
                reason='Identical DP parameters, data and normalization; each file is revalidated by residual_setup'))
    atomic_json(output/'registration.json',dict(source_root=str(source),source_pairs=pairs,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={str(p):sha256(p) for folder in ('tools','visuotactile') for p in sorted(Path(folder).rglob('*.py'))},
        protected=protected,epochs=epochs,first_legal_recovery_ticks=ticks,control_weight=1.,weighted_weight=weight,
        count_per_role=count,development_seed=seed,started=time.time(),deadline=session['deadline'],
        source_autonomous_results_sha256=sha256(development/'results.json'),
        method='Same selected complete pair, same episode schedule and real exposure, new optimizers; frozen DP and exact caches. Only legal onset loss weight differs. Fixed validation selects each rear. Complete autonomous cases retain all original task criteria.',
        test_used=False))
    atomic_json(output/'status.json',dict(stage='prepared',utc=time.time()))


def run(output):
    config=read(output/'registration.json')
    if (output/'run_started.json').exists():raise FileExistsError('No duplicate study worker')
    if time.time()>=config['deadline']:raise TimeoutError('Original session expired')
    atomic_json(output/'run_started.json',dict(pid=os.getpid(),utc=time.time()))
    ledger=output/'stages';ledger.mkdir();receipts=[]
    def execute(name,args,completion,*,startup=False):
        for path,digest in {**config['protected'],**config['source_sha256']}.items():
            if sha256(path)!=digest:raise ValueError('Frozen study dependency changed: '+path)
        if time.time()>=config['deadline']:raise TimeoutError('Original session expired')
        command=[sys.executable,'-m',*map(str,args)]
        atomic_json(output/'status.json',dict(stage=name,command=command,utc=time.time()))
        began=time.time()
        with (ledger/(name+'.log')).open('x') as log:
            worker=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
            reason=wait_owned(worker,config['deadline'],startup_marker=completion.parent/'initialization.json' if startup else None,
                startup_seconds=180 if startup else None)
        receipt=dict(stage=name,returncode=worker.returncode,stop_reason=reason,seconds=time.time()-began,
            command=command,completion=str(completion),completion_sha256=sha256(completion) if completion.exists() else None)
        atomic_json(ledger/(name+'.json'),receipt);receipts.append(receipt)
        if reason or worker.returncode or not completion.exists():raise RuntimeError('Study stage failed: '+name)
        proof=read(completion)
        if any(proof.get(key) is False for key in ('complete','software_complete','target_reached')):
            raise RuntimeError('Study stage incomplete: '+name)
        print(json.dumps(receipt),flush=True)
    try:
        for arm in ('control','weighted'):
            root=output/arm
            for role in ('A','B'):
                rear=root/role/'backend'/role
                for stage,completion in [('residual-train',rear/'training/finished.json'),('residual-offline',rear/'offline/result.json')]:
                    execute(arm+'_'+role+'_'+stage,['visuotactile.anchor_continue',stage,'--root',rear,
                        '--job','image','--output',completion.parent],completion)
            development=root/'development'
            execute(arm+'_prepare',['tools.tight_fresh_rollouts','prepare','--root',root,'--output',development,
                '--count',config['count_per_role'],'--seed',config['development_seed']],development/'plan.json')
            execute(arm+'_development',['tools.tight_fresh_rollouts','run','--root',root,'--output',development],
                development/'result.json',startup=True)
        atomic_json(output/'finished.json',dict(software_complete=True,stages=receipts,utc=time.time(),
            research_conclusion='Pending actual Agent review; no automatic next experiment'))
        atomic_json(output/'status.json',dict(stage='complete',utc=time.time()))
    except BaseException as error:
        atomic_json(output/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),stages=receipts,utc=time.time()))
        atomic_json(output/'status.json',dict(stage='failed',error=repr(error),utc=time.time()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','run']);p.add_argument('--source',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--ticks',type=int,default=12);p.add_argument('--weight',type=float,default=3.)
    p.add_argument('--count',type=int,default=4);p.add_argument('--seed',type=int,default=28140000);a=p.parse_args()
    if a.stage=='prepare':
        if a.source is None:p.error('Preparation requires the reviewed complete source cycle')
        prepare(a.source.resolve(),a.output.resolve(),a.epochs,a.ticks,a.weight,a.count,a.seed)
    else:run(a.output.resolve())
