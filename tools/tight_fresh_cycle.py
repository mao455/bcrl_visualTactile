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


def run(root, collection, output, epochs, rear_epochs, count, seed, *, continue_rear=False, reuse_index=None):
    ledger=output.with_name(output.name+'_stages')
    if ledger.exists() or output.exists():raise FileExistsError('Preserve previous cycle identity')
    if not read(collection/'result.json').get('software_complete'):raise ValueError('Collection not complete')
    if not read(collection/'correction_entries.json'):raise ValueError('No qualified successful expert suffixes')
    session=read(root/'session.json');deadline=session['deadline']
    sources={str(p):sha256(p) for folder in ['tools','visuotactile'] for p in sorted(Path(folder).rglob('*.py'))}
    ledger.mkdir(parents=True)
    registration=dict(source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=sources,source_root=str(root),collection=str(collection),output=str(output),
        collection_entries_sha256=sha256(collection/'correction_entries.json'),
        frontend_epochs=epochs,backend_epochs=rear_epochs,backend_continuation=continue_rear,development_count_per_role=count,seed=seed,
        started=time.time(),deadline=deadline,pid=os.getpid(),test_used=False)
    if reuse_index is not None:
        if not read(reuse_index).get('complete'):raise ValueError('Reusable frozen vision cache is incomplete')
        registration['reuse_visual_index']=dict(path=str(reuse_index),sha256=sha256(reuse_index))
    atomic_json(ledger/'registration.json',registration)
    stages=[]
    def execute(name,args,completion):
        if time.time()>=deadline:raise TimeoutError('Original fresh session deadline exhausted')
        if any(sha256(p)!=s for p,s in sources.items()):raise ValueError('Active cycle source drift')
        command=[sys.executable,'-m']+list(map(str,args))
        atomic_json(ledger/'status.json',dict(stage=name,command=command,started=time.time()))
        start=time.time()
        with (ledger/(name+'.log')).open('x') as stream:
            proc=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                timeout=max(1,deadline-time.time()),env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        receipt=dict(stage=name,command=command,returncode=proc.returncode,seconds=time.time()-start,
            completion=str(completion),completion_sha256=sha256(completion) if completion.exists() else None,utc=time.time())
        atomic_json(ledger/(name+'.json'),receipt);stages.append(receipt)
        if proc.returncode or not completion.exists():raise RuntimeError('Stage failed: '+name)
        proof=read(completion)
        if any(proof.get(key) is False for key in ('complete','software_complete','target_reached')):
            raise RuntimeError('Stage receipt is incomplete: '+name)
        print(json.dumps(receipt),flush=True)
    try:
        execute('aggregate',['tools.tight_fresh_dagger','aggregate','--root',root,'--collection',collection,'--output',output],output/'prepared.json')
        execute('visual_cache',['tools.tight_fresh_dagger','cache','--root',root,'--output',output]+
            (['--reuse-index',reuse_index] if reuse_index is not None else []),output/'B/cache_index.json')
        for role in ['A','B']:
            execute('frontend_'+role,['tools.tight_fresh_train','train','--root',output,'--role',role,'--epochs',epochs],output/role/'training/finished.json')
        for role in ['A','B']:
            rear=output/role/'backend'
            execute('backend_prepare_'+role,['tools.tight_fresh_backend','--root',output,'--role',role,'--epochs',rear_epochs]+(['--continue-rear'] if continue_rear else []),rear/'prepared.json')
            execute('backend_cache_'+role,['tools.tight_finish','residual_cache','--root',rear],rear/'cache_complete.json')
            execute('backend_train_'+role,['visuotactile.anchor_continue','residual-train','--root',rear/role,'--job','image','--output',rear/role/'training'],rear/role/'training/finished.json')
            execute('backend_offline_'+role,['visuotactile.anchor_continue','residual-offline','--root',rear/role,'--job','image','--output',rear/role/'offline'],rear/role/'offline/result.json')
        development=output/'development'
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
    a=p.parse_args()
    if not 1<=a.epochs<=100 or not 1<=a.rear_epochs<=100 or not 1<=a.count<=32:p.error('Positive finite stage exposure required')
    run(a.root.resolve(),a.collection.resolve(),a.output.resolve(),a.epochs,a.rear_epochs,a.count,a.seed,
        continue_rear=a.continue_rear,reuse_index=a.reuse_index.resolve() if a.reuse_index else None)


if __name__=='__main__':main()
