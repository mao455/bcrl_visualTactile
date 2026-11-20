"""Recover closed R2 captures without repeating any physical opportunity."""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time

PROJECT=Path(__file__).resolve().parents[3]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))
from tools import tight_adaptive_dagger as experiment
from tools.tight_frozen_frontend import read,write,sources
from visuotactile.compact import sha256

ROOT=experiment.ROOT
PUBLIC=experiment.PUBLIC
OLD=ROOT/'round02/collection02'
OUTPUT=ROOT/'round02/collection02_closed_recovery01'


def census(plan):
    completed=[];interrupted=[];unstarted=[]
    for case in plan['cases']:
        folder=OLD/case['name']
        if (folder/'result.json').exists() and (folder/'capture_complete.json').exists():
            result=read(folder/'result.json')
            assert result['software_complete'] and result['legal']
            assert result['residual_ticks']==result['steps']
            completed.append(case)
        elif (folder/'started.json').exists():interrupted.append(case['name'])
        else:unstarted.append(case['name'])
    assert len(completed)+len(interrupted)+len(unstarted)==len(plan['cases'])
    return completed,interrupted,unstarted


def recover():
    from tools.tight_fresh_batch_run import admit_capture
    from visuotactile.state_benchmark import DaggerCaptureAuditMemo
    assert (ROOT/'round2_continuation_failure.json').exists()
    assert not OUTPUT.exists() and not (ROOT/'round02/learning').exists()
    assert time.time()<read(ROOT/'session.json')['deadline']
    old=read(OLD/'plan.json');completed,interrupted,unstarted=census(old)
    assert len(completed)==5 and len(interrupted)==2 and len(unstarted)==1
    assert old['source_sha256']==sources()
    accepted=read(ROOT/'round01/learning/exposure_acceptance.json')
    assert accepted['passed']
    selected={x['model']:x['selected_sha256'] for x in accepted['components']}
    for role in 'AB':
        pair=old['pairs'][role]
        assert pair['frontend_sha256']==selected[role+'-DP']
        assert pair['backend_sha256']==selected[role+'-Transformer']
    OUTPUT.mkdir();bindings={}
    for case in completed:
        origin=OLD/case['name'];destination=OUTPUT/case['name'];destination.mkdir()
        for p in origin.iterdir():
            assert p.name not in ['result_before_admission.json','capture_audit.json']
            if p.is_file() and p.suffix=='.json':shutil.copy2(p,destination/p.name)
            else:(destination/p.name).symlink_to(p,target_is_directory=p.is_dir())
        for name in ['result.json','numeric.h5','trace.npz','capture_complete.json']:
            bindings[case['name']+'/'+name]=sha256(origin/name)
    plan=copy.deepcopy(old);plan['cases']=completed
    plan.update(closed_execution_source=dict(root=str(OLD),files=bindings),
        recovery_scope='Five already closed opportunities; CPU admission only; two interrupted prefixes excluded and preserved; one unstarted case deferred')
    write(OUTPUT/'plan.json',plan);write(OUTPUT/'session.json',read(OLD/'session.json'))
    entries={};rows=[];memo=DaggerCaptureAuditMemo()
    for case in completed:
        result=read(OUTPUT/case['name']/'result.json')
        admit_capture(OUTPUT,case,result,plan['pairs'][case['role']],plan,entries,audit_memo=memo)
        rows.append(dict(role=case['role'],case=case['name'],result=result,
            timing=dict(interpretation='Original completed physics; CPU-only recovered admission')))
        write(OUTPUT/'results.json',rows);write(OUTPUT/'capture_audit_memo.json',memo.statistics())
    assert all(sha256(OLD/name)==digest for name,digest in bindings.items())
    assert len(entries)==5
    write(OUTPUT/'result.json',dict(software_complete=True,cases=5,accepted_training_trajectories=5,
        recovered_closed_execution=True,new_physical_opportunities=0,utc=time.time()))
    strategy=read(PUBLIC/'round02/strategy.json')
    remaining=[]
    for item in strategy['schedule'][3:]:
        for family,stages in zip(item['families'],item['stages']):
            remaining.append(dict(families=[family],stages=[stages]))
    batches=[str(ROOT/'round02'/name) for name in ['collection00','collection01',OUTPUT.name]]
    document=dict(version='round2_ram_recovery_v1',utc=time.time(),original_failure=read(ROOT/'round2_continuation_failure.json'),
        original_deadline=read(ROOT/'session.json')['deadline'],verified_batches=batches,
        verified_batch_result_sha256={p:sha256(Path(p)/'result.json') for p in batches},
        recovered_closed_cases=5,interrupted_cases_preserved=interrupted,unstarted_case_deferred=unstarted,
        original_files=bindings,remaining_schedule=remaining,maximum_cases_per_new_process=2,
        source_strategy_sha256=sha256(PUBLIC/'round02/strategy.json'),new_optimization_updates=0,
        physical_opportunities_repeated=0,resource_guards_unchanged=True,
        admission_policy='Resume already-spent collection budget; existing per-batch storage headroom and minimum20 total/8 each stop remain; no new40GiB initial gate',
        limitation='Interrupted physics not resumable; excluded prefixes preserved; subsequent fresh physical batches retain source models and native task, no exact hidden RNG replay claim')
    write(ROOT/'round02/ram_recovery01.json',document);write(PUBLIC/'round02/ram_recovery01.json',document)
    print(json.dumps(dict(passed=True,recovered=5,new_physical=0,verified_batches=batches,remaining_two_case_batches=len(remaining))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
    if args.apply:recover()
    else:
        a,b,c=census(read(OLD/'plan.json'))
        print(json.dumps(dict(completed=[x['name'] for x in a],interrupted=b,unstarted=c)))
