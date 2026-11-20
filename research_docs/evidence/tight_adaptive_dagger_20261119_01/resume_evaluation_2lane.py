"""Retire only an idle coordinator and split unexecuted waves into paired lanes."""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

PROJECT=Path(__file__).resolve().parents[3]
PUBLIC=Path(__file__).resolve().parent
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')
EVALUATION=ROOT/'round01/evaluation'
RECEIPT=ROOT/'evaluation_2lane_handover.json'
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))


def read(path): return json.loads(path.read_text())


def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(data,stream,indent=2,ensure_ascii=False)
        stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)


def split(plan):
    cases=plan['cases']
    if all(c['role'].endswith(('_parent','_new')) for c in cases):
        groups=[(role,[c for c in cases if c['role'].startswith(role+'_')]) for role in 'AB']
        for role,selected in groups:
            assert {c['role'] for c in selected}=={role+'_parent',role+'_new'}
    else:
        assert {c['role'] for c in cases}=={'A_72','B_72'}
        groups=[(str(seed),[c for c in cases if c['spec']['seed']==seed])
                for seed in sorted({c['spec']['seed'] for c in cases})]
    result=[]
    for label,selected in groups:
        assert len(selected)==2 and selected[0]['spec']==selected[1]['spec']
        p=copy.deepcopy(plan);p['cases']=copy.deepcopy(selected)
        p['pairs']={c['role']:copy.deepcopy(plan['pairs'][c['role']]) for c in selected}
        p['batch']['lanes']=2
        p['evaluation_scope']=dict(protocol='Same-wave role-specific parent/new pair; historical72 A/B in separate same-seed waves',
            test_used=False,hold_metrics='30/60/120 are diagnostic only',hidden_state_replay_claim=False,
            limitation='A and B parent/new comparisons occur in separate physical waves; do not claim four-policy same-wave replay')
        result.append((label,p))
    assert sorted(json.dumps(c,sort_keys=True) for _,p in result for c in p['cases'])==sorted(json.dumps(c,sort_keys=True) for c in cases)
    return result


def check():
    original=EVALUATION/'registration_4lane_unexecuted.json'
    registration=read(original if original.exists() else EVALUATION/'registration.json');count=0;waves=0;expected=[]
    for folder in registration['waves']:
        plan=read(Path(folder)/'plan.json')
        expected.extend(plan['cases'])
        assert not plan['test_used'] and len(plan['cases'])==4
        for _,p in split(plan):
            assert p['batch']['lanes']==len(p['cases'])==2
            assert p['source_sha256']==plan['source_sha256']
            assert all(p['pairs'][key]==plan['pairs'][key] for key in p['pairs'])
            count+=len(p['cases']);waves+=1
    assert count==144 and waves==72
    if original.exists():
        actual=[]
        for folder in read(EVALUATION/'registration.json')['waves']:
            plan=read(Path(folder)/'plan.json')
            assert plan['batch']['lanes']==len(plan['cases'])==2
            assert plan['cases'][0]['spec']==plan['cases'][1]['spec']
            actual.extend(plan['cases'])
        assert sorted(json.dumps(c,sort_keys=True) for c in actual)==sorted(json.dumps(c,sort_keys=True) for c in expected)
    return dict(original_waves=len(registration['waves']),paired_waves=waves,cases=count,
        cases_and_specs_exact=True,pair_weights_exact=True,source_hashes_exact=True,
        native_batch_seed_unchanged=True,applied_case_set_exact=original.exists(),utc=time.time())


def handover(pid):
    import psutil
    assert not RECEIPT.exists()
    verified=check();old=psutil.Process(pid)
    assert old.cwd()==str(PROJECT)
    assert old.cmdline()[-2:]==['research_docs/evidence/tight_adaptive_dagger_20261119_01/round1_verified_boundary.py','--continue-existing']
    assert read(ROOT/'progress.json')['stage']=='evaluate'
    assert not old.children(recursive=True), 'A worker may have started; preserve it'
    assert not (ROOT/'owners/r1_evaluate/driver.log').exists()
    old.send_signal(signal.SIGSTOP)
    try:
        until=time.monotonic()+2
        while old.status()!=psutil.STATUS_STOPPED and time.monotonic()<until:time.sleep(.01)
        assert old.status()==psutil.STATUS_STOPPED
        assert not old.children(recursive=True) and not (ROOT/'owners/r1_evaluate/driver.log').exists()
        original=(EVALUATION/'registration.json').read_bytes();registration=json.loads(original)
        assert not (EVALUATION/'registration_4lane_unexecuted.json').exists()
        waves=[]
        for folder in registration['waves']:
            source=Path(folder)
            assert {p.name for p in source.iterdir()}=={'plan.json','session.json'}, 'Actual wave evidence exists'
            plan=read(source/'plan.json')
            for label,new in split(plan):
                destination=EVALUATION/f'paired2_{source.name}_{label}'
                destination.mkdir()
                new['schedule_adaptation']=dict(original_plan=str(source/'plan.json'),
                    original_plan_sha256=hashlib.sha256((source/'plan.json').read_bytes()).hexdigest(),
                    reason='34GiB four-lane RAM admission unavailable; use existing28GiB two-lane admission, retain10GiB run guard',
                    zero_prior_physics=True)
                write(destination/'plan.json',new)
                write(destination/'session.json',read(source/'session.json'))
                waves.append(str(destination))
        (EVALUATION/'registration_4lane_unexecuted.json').write_bytes(original)
        registration.update(waves=waves,scheduling_revision='paired_two_lane_resource_recovery_v1',
            original_registration_sha256=hashlib.sha256(original).hexdigest(),
            old_plans_preserved=True,case_count=144,original_deadline=read(ROOT/'session.json')['deadline'])
        write(EVALUATION/'registration.json',registration)
        write(RECEIPT,dict(prior_pid=pid,prior_created=old.create_time(),prior_rss_bytes=old.memory_info().rss,
            utc=time.time(),reason='Release completed CPU audit allocations and schedule all144 unchanged cases in same-seed paired two-lane waves',
            verification=verified,original_deadline=read(ROOT/'session.json')['deadline'],
            optimizer_updates_repeated=0,physical_opportunities_repeated=0))
    except BaseException:
        old.send_signal(signal.SIGCONT)
        raise
    old.send_signal(signal.SIGTERM);old.send_signal(signal.SIGCONT);old.wait(timeout=5)
    environment=os.environ.copy();environment.update(RESEARCH_CPUS='8-23',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONPATH=os.pathsep.join([str(PROJECT),str(PROJECT/'rl_games')]))
    with (ROOT/'evaluation_2lane_controller.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--continue-existing'],cwd=PROJECT,
            stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
    write(ROOT/'evaluation_2lane_controller.json',dict(pid=child.pid,created=psutil.Process(child.pid).create_time(),utc=time.time()))
    print(json.dumps(dict(pid=child.pid,verification=verified)))


def load(name):
    spec=importlib.util.spec_from_file_location(name,PUBLIC/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def run():
    from tools import tight_adaptive_dagger as experiment
    assert RECEIPT.exists() and read(ROOT/'round01/learning/exposure_acceptance.json')['passed']
    registration=read(EVALUATION/'registration.json')
    assert registration['scheduling_revision']=='paired_two_lane_resource_recovery_v1'
    stages=[dict(entry='evaluate',args=['--output',p]) for p in registration['waves']]
    experiment.execute('evaluate','r1_evaluate_2lane01',stages,simulation=True)
    result=experiment.evaluation_report(EVALUATION,1)
    write(ROOT/'round01/complete.json',dict(collection=str(ROOT/'round01/collection_combined'),
        learning=str(ROOT/'round01/learning'),evaluation=True,utc=time.time()))
    write(ROOT/'progress.json',dict(round=1,stage='round1_diagnosis',utc=time.time()))
    analysis=load('analyze_round');analysis.forks(1)
    load('execution_audit').audit([Path(p) for p in registration['waves']],PUBLIC/'round01/evaluation/execution_audit')
    analysis.offline(ROOT/'round01/learning',PUBLIC/'round01/offline')
    write(ROOT/'progress.json',dict(round=1,stage='round1_review',utc=time.time()))
    write(ROOT/'round1_review_ready.json',dict(utc=time.time(),cases=len(result['rows']),
        training='Four selected models independently passed committed-step exposure acceptance and backend raw-mask checks',
        evaluation='Role-specific parent/new same-wave native paired checks; A and B in separate waves',final_test_opened=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--handover-pid',type=int)
    p.add_argument('--check',action='store_true');p.add_argument('--import-check',action='store_true');p.add_argument('--continue-existing',action='store_true');args=p.parse_args()
    if args.check:print(json.dumps(check()))
    elif args.import_check:
        from tools import tight_adaptive_dagger as experiment
        assert experiment.PROJECT==PROJECT
        print(json.dumps(dict(project=str(PROJECT),direct_script_import_passed=True)))
    elif args.handover_pid:handover(args.handover_pid)
    elif args.continue_existing:
        try:run()
        except BaseException as error:
            write(ROOT/'round1_boundary_failure.json',dict(error=repr(error),utc=time.time(),coordinator='two_lane_resource_recovery'))
            raise
    else:p.error('Explicit check, handover or continue mode required')
