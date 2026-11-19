"""Continue only after the completed first-round review evidence exists."""
import collections
import argparse
import copy
import importlib.util
import os
from pathlib import Path
import shutil
import time

from tools import tight_adaptive_dagger as experiment
from tools.tight_frozen_frontend import read,write,sources
from tools.tight_fresh_rollouts import completed_pair
from visuotactile.adaptive_takeover import profile
from visuotactile.compact import sha256

ROOT=experiment.ROOT
PUBLIC=experiment.PUBLIC
SOURCE=ROOT/'round01/learning'


def load(name):
    spec=importlib.util.spec_from_file_location(name,PUBLIC/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def wait_review():
    while not (ROOT/'round1_review_ready.json').exists():
        if (ROOT/'round1_boundary_failure.json').exists():
            raise RuntimeError('First-round diagnosis failed; retain evidence and repair explicitly')
        if time.time()>=read(ROOT/'session.json')['deadline']:raise TimeoutError('Original session deadline')
        time.sleep(30)
    assert read(ROOT/'round1_review_ready.json')['final_test_opened'] is False


def storage_admission():
    # Includes R2 raw captures, new immutable caches, four models/logs, evals
    # and the original twelve-GiB reserve. No historical asset is retired here.
    minimum=40
    while shutil.disk_usage(ROOT).free<minimum*2**30:
        write(ROOT/'round2_storage_wait.json',dict(free_gib=shutil.disk_usage(ROOT).free/2**30,
            required_gib=minimum,utc=time.time(),reason='Await storage headroom; do not delete protected or historical assets',
            deadline=read(ROOT/'session.json')['deadline']))
        if time.time()>=read(ROOT/'session.json')['deadline']:raise TimeoutError('Storage headroom unavailable within finite session')
        time.sleep(30)


def collect(strategy):
    destination=ROOT/'round02';destination.mkdir(exist_ok=True)
    entries={};rows=[];cases=[];batches=[]
    attempted=collections.Counter();rescued=collections.Counter()
    for index,item in enumerate(strategy['schedule']):
        # Reserve future frozen inputs and model/log/evaluation storage before
        # spending another physical batch; the twelve-GiB hard guard remains.
        required_gib=21+.5*len(entries)+3
        free_gib=shutil.disk_usage(ROOT).free/2**30
        if free_gib<required_gib:
            population=collections.Counter(e['source_group'] for e in entries.values())
            if len(entries)>=20 and min(population.get(r,0) for r in 'AB')>=8:
                write(destination/'collection_budget_stop.json',dict(accepted=len(entries),population=dict(population),
                    free_gib=free_gib,required_for_next_batch_gib=required_gib,utc=time.time(),
                    reason='End collection at verified two-student coverage; retain headroom for full exposure training'))
                break
            while shutil.disk_usage(ROOT).free/2**30<required_gib:
                write(destination/'storage_wait.json',dict(accepted=len(entries),required_gib=required_gib,
                    free_gib=shutil.disk_usage(ROOT).free/2**30,utc=time.time()))
                if time.time()>=read(ROOT/'session.json')['deadline']:raise TimeoutError('Round2 capture storage admission')
                time.sleep(30)
        output=destination/f'collection{index:02d}'
        assert not output.exists(),'Preserve prior attempted collection identity'
        families=item['families'];chosen=copy.deepcopy(item['stages'])
        for per_role in chosen:
            for role,stage in per_role.items():
                if attempted[role,stage]>=4 and rescued[role,stage]==0:
                    alternatives=('transport','contact','instability')
                    per_role[role]=max(alternatives,key=lambda s:(rescued[role,s],-attempted[role,s]))
        plan=experiment.prepare_collection(SOURCE,output,families,[v['A'] for v in chosen],20+index)
        from visuotactile.state_benchmark import dagger_configuration
        for case in plan['cases']:
            seed=case['spec']['seed'];stage=chosen[families.index(seed)][case['role']]
            case['name']=f"r{20+index:02d}_{case['role']}_{case['spec']['source_identity']}_{stage}"
            case['spec']['identity']=case['name'];case['dagger']['trigger']=profile(stage)
            case['adaptive_stage']=stage
            dagger_configuration(case['dagger'],case['spec'],read(Path(case['source_data'])))
            attempted[case['role'],stage]+=1
        plan.update(reason='Round1 new complete-student failure diagnosis; role-specific stage assignment',
            round2_strategy_sha256=sha256(PUBLIC/'round02/strategy.json'),source_sha256=sources())
        write(output/'plan.json',plan)
        experiment.execute('dagger',f'r2_collection{index:02d}',[
            dict(entry='dagger',args=['--output',str(output)])],simulation=True)
        assert read(output/'result.json')['software_complete']
        new=read(output/'correction_entries.json');assert not set(new)&set(entries)
        for name in new:
            case=next(c for c in plan['cases'] if c['name']==name)
            rescued[case['role'],case['adaptive_stage']]+=1
        entries.update(new);rows.extend(read(output/'results.json'));cases.extend(plan['cases']);batches.append(str(output))
        experiment.collection_report(destination,rows,entries,cases,batches)
        population=collections.Counter(e['source_group'] for e in entries.values())
        if len(entries)>=strategy['target_accepted'] and min(population.get(r,0) for r in 'AB')>=12:break
    population=collections.Counter(e['source_group'] for e in entries.values())
    assert len(entries)>=20 and min(population.get(r,0) for r in 'AB')>=8,'Insufficient two-student coverage; diagnose before further collection'
    combined=destination/'collection_combined';combined.mkdir()
    merged=copy.deepcopy(plan);merged['cases']=cases;merged['source_sha256']=sources()
    for batch in map(Path,batches):
        for case in read(batch/'plan.json')['cases']:
            (combined/case['name']).symlink_to(batch/case['name'],target_is_directory=True)
    write(combined/'session.json',read(ROOT/'session.json'));write(combined/'plan.json',merged)
    write(combined/'correction_entries.json',entries);write(combined/'results.json',rows)
    write(combined/'result.json',dict(software_complete=True,cases=len(rows),accepted_training_trajectories=len(entries)))
    load('inspect_records').inspect([combined],PUBLIC/'round02/combined_analysis')
    load('shadow_diagnosis').analyze(2)
    load('captured_parent_fit').export(2)
    return combined


def evaluate(parent,current,round_index,heldout=False):
    root,waves=experiment.evaluation_plans(parent,current,round_index,heldout=heldout)
    for wave in waves:
        plan=read(wave/'plan.json')
        for case in plan['cases']:case['video']=True
        plan['video_policy']='All actual lanes retained; select representative records without repeating outcomes'
        write(wave/'plan.json',plan)
    experiment.execute('evaluate','final_test' if heldout else f'r{round_index}_evaluate',[
        dict(entry='adaptive_heldout' if heldout else 'evaluate',args=['--output',str(w)]) for w in waves],simulation=True)
    report=experiment.evaluation_report(root,round_index,heldout=heldout)
    load('execution_audit').audit(waves,PUBLIC/('final_test' if heldout else f'round{round_index:02d}/evaluation')/'execution_audit')
    return report


def run(*,continue_existing=False):
    receipt=ROOT/'round2_continuation_started.json'
    if continue_existing:
        assert receipt.exists() and not (ROOT/'round1_review_ready.json').exists()
        assert not (ROOT/'round02').exists()
        assert read(receipt)['original_deadline']==read(ROOT/'session.json')['deadline']
    else:
        if receipt.exists():raise FileExistsError('Retain prior continuation attempt')
        write(receipt,dict(pid=os.getpid(),utc=time.time(),original_deadline=read(ROOT/'session.json')['deadline'],
            purpose='Await actual R1 diagnosis; adapt two-student training-only collection; train/audit/evaluate R2; freeze then open new test',
            model_promotion=False,source_changes=False))
    wait_review()
    strategy=load('round2_strategy').derive()
    write(ROOT/'progress.json',dict(round=2,stage='storage_admission',utc=time.time()))
    storage_admission()
    write(ROOT/'progress.json',dict(round=2,stage='collect',utc=time.time()))
    combined=collect(strategy)
    write(ROOT/'progress.json',dict(round=2,stage='train',utc=time.time()))
    target=experiment.prepare_learning(SOURCE,combined,2)
    experiment.train(target,2)
    for role in 'AB':
        load('audit_backend_masks').audit(target,role,PUBLIC/f'round02/{role}_backend_raw_mask_acceptance.json')
    load('exposure_phases').export(2)
    write(ROOT/'progress.json',dict(round=2,stage='evaluate',utc=time.time()))
    evaluate(SOURCE,target,2)
    load('analyze_round').forks(2)
    write(ROOT/'progress.json',dict(round=2,stage='offline',utc=time.time()))
    load('analyze_round').offline(target,PUBLIC/'round02/offline')
    write(ROOT/'round02/complete.json',dict(collection=str(combined),learning=str(target),evaluation=True,utc=time.time()))
    pairs={f'{r}_parent':completed_pair(experiment.BASE,r) for r in 'AB'}
    pairs.update({f'{r}_new':completed_pair(target,r) for r in 'AB'})
    write(ROOT/'final_freeze.json',dict(pairs=pairs,seeds=read(experiment.CONFIG/'experiment.json')['final_test_seeds'],utc=time.time(),
        selection='Own validation-eligible round2 complete pairs, frozen before opening any new final-test outcome'))
    write(ROOT/'progress.json',dict(round=2,stage='final_test',utc=time.time()))
    evaluate(experiment.BASE,target,2,heldout=True)
    write(ROOT/'cycle_complete.json',dict(rounds=2,final_test_complete=True,utc=time.time(),automatic_promotion=False))
    experiment.close_report()
    write(ROOT/'progress.json',dict(round=2,stage='complete',utc=time.time()))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continue-existing',action='store_true');args=parser.parse_args()
    try:run(continue_existing=args.continue_existing)
    except BaseException as error:
        write(ROOT/'round2_continuation_failure.json',dict(error=repr(error),utc=time.time()))
        raise
