"""Ordinary bounded owner for cache continuation, training and behavior review."""
import argparse,json,time,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.tight_fresh_collect_cycle_owner import run_stage
from visuotactile.training_resources import atomic_json

TREE=Path(__file__).resolve().parents[2];BUNDLE=Path(__file__).resolve().parent
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
ROOT=BASE/'tight_fresh_20261115_10';OLD=BASE/'tight_fresh_20261115_09'
read=lambda p:json.loads(p.read_text())
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args();ROOT=args.root.resolve()
    ledger=ROOT/'continuation';ledger.mkdir();out=ROOT/'round_15';collection=OLD/'new_family_15'
    deadline=read(ROOT/'session.json')['deadline']-60;python=str(TREE/'.venv_isaacsim/bin/python')
    context=dict(ledger=ledger,paths=dict(worker_tree=TREE,collection=collection,output=out))
    stops=(ROOT/'STOP',ledger/'STOP',out/'STOP');rows=[];complete=False;error=None
    cmd=[python,'-m','tools.tight_fresh_cycle','--root',str(ROOT),'--collection',str(collection),'--output',str(out),
        '--prepared-from',str(OLD/'round_15'),'--epochs','14','--rear-epochs','12','--count','8','--seed','28500000',
        '--latest-repeat','4','--backend-latest-repeat','4','--evaluation-lanes','16','--vision-microbatch','8',
        '--parallel-frontends','--parallel-backends','--full-trajectory-batch','--continue-rear','--payload-audit-memo',
        '--reuse-index',str(BASE/'tight_fresh_20261115_04/round_14/A/cache_index.json'),
        '--offline-anchor-batch','32','--backend-offline-anchor-batch','32','--queued-gpu-evaluation',
        '--grasp-evaluation-ticks','300','--backend-json',str(BUNDLE/'evaluation_backend.json'),
        '--training-sampling-json',str(BUNDLE/'training_sampling.json'),'--frozen-visual-io-json',str(BUNDLE/'frozen_visual_io.json')]
    atomic_json(ledger/'owner_started.json',dict(utc=time.time(),deadline=deadline,command=cmd,collection_repeated=False))
    try:
        row=run_stage('cycle',cmd,context,deadline,stops);rows.append(row)
        if not row['complete']:raise RuntimeError('Continuation cycle incomplete; preserve all stage receipts')
        cmd=[python,'-m','tools.tight_fresh_grasp_review','--root',str(out),'--collection',str(collection),
            '--output',str(out/'grasp_review'),'--configuration',str(BUNDLE/'grasp_review.json')]
        row=run_stage('grasp_review',cmd,context,deadline,stops);rows.append(row)
        complete=row['complete']
    except BaseException as caught:error=repr(caught)
    finally:
        atomic_json(ledger/'owner_finished.json',dict(software_complete=complete,ended=time.time(),stages=rows,error=error))
        (ROOT/'STOP_MONITOR').touch(exist_ok=True)
    if not complete:raise RuntimeError(error or 'Grasp review incomplete')
