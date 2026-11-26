"""Original-machine diagnostic dispatch over existing native rollout consumers.

A finite config is the only case scheduler. This module adds no policy/control
loop: batches use tight_evaluate, expert controls use state_benchmark.trial, and
real takeovers use the existing collector. Historical sessions are never resumed.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'outputs/visuotactile/run/tight_machine_review_20261126_01'
AUDIT=PROJECT/'outputs/diagnostics/tight_machine_review_20261126_01'
OLD=PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01'
PAIRED=PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
CONFIG=PROJECT/'configs/experiments/machine_review_20261126'


def read(path):return json.loads(Path(path).read_text())


def sources():
    return {str(p.relative_to(PROJECT)):sha256(p) for folder in ('tools','visuotactile')
            for p in sorted((PROJECT/folder).rglob('*.py'))}


def session_at(path, *, duration=8*3600):
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    session=read(OLD/'session.json');now=time.time()
    session.update(version='original_machine_review_v1',started_utc=now,deadline=now+duration,
        collection_deadline=now+duration,training_deadline=now+duration,resource_root=str(path),
        owned_worker_roots=[str(path)],resource_policy='full_gpu_v1',reserve_bytes=16*2**30,
        authorization='Original-machine diagnosis, diagnostic positive controls and real takeover branches; conditional fitting only after rescue. Original64 denominator immutable.')
    session['resources'].update(min_gpu_free_mib=4096,min_ram_gib=10,min_disk_gib=16)
    atomic_json(path/'session.json',session)
    return session


def job(owner,stages,name,operation='evaluate'):
    value=dict(version='tight_project_job_v1',operation=operation,resource_root=str(owner),stages=stages)
    atomic_json(CONFIG/(name+'.json'),value)
    return value


def controls():
    from scripts.tight_layout_20261116.pilot import reset_configuration
    from visuotactile.narrow_data import direct_configuration
    assert not ROOT.exists(),'Preserve attempted identity'
    session=session_at(ROOT)
    historical=read(OLD/'registration.json');parents=read(PAIRED/'parent_pairs.json')
    pairs={r:historical['pairs'][r] for r in ('A_130_50','B_130_50')}
    pairs.update(A_parent=parents['A'],B_parent=parents['B'])
    stages=[];hashes={};template=read(OLD/'wave06/plan.json')
    seed=28851006
    for repeat in range(2):
        for displaced in (False,True):
            name=f'positive_r{repeat}_'+('layout10' if displaced else 'unshifted')
            out=ROOT/name;out.mkdir()
            reset=(reset_configuration(session['domain'],seed,.1) if displaced else direct_configuration(session['domain'],seed))
            cases=[]
            for role in pairs:
                spec=dict(identity=f'{role}_{name}',family=f'diagnostic_only_{seed}',seed=seed,event_seed=seed+100000,
                    split='diagnostic_positive_control',cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',
                    narrow_reset=reset,layout=session['layout'],diagnostic_only=True,original_test_reuse=True)
                cases.append(dict(name=spec['identity'],role=role,spec=spec,video=False))
            cases=cases[repeat:]+cases[:repeat]
            plan=copy.deepcopy(template)
            plan.pop('holdout_contract',None)
            plan.update(kind='autonomous',resource_root=str(ROOT),pairs=pairs,cases=cases,test_used=False,
                cohort='diagnostic_positive_control',source_sha256=sources(),source_commit='60e7b1f+original_machine_dirty_sources',
                first_plan_forensics=False,record_sampler_rng=True,diagnostic_only=True,
                evaluation_scope=dict(no_new_test=True,layout_displacement_m=.1 if displaced else 0.,repeat=repeat))
            plan['batch'].update(lanes=4,seed=seed,verification_workers=0)
            plan['execution_backend'].update(kit_tasking_threads=8,physics_threads=2)
            atomic_json(out/'session.json',session);atomic_json(out/'plan.json',plan)
            hashes[str(out)]=sha256(out/'plan.json')
            stages.append(dict(entry='machine_review',args=['worker','--output',str(out),'--mode','autonomous']))
    job(ROOT,stages,'positive_controls')
    # Original expert control uses the existing single-lane native expert runner.
    owner=ROOT/'expert_controls';session_at(owner)
    spec=historical['specifications'][6]
    cases=[]
    for repeat in range(2):
        for displaced in (False,True):
            s=copy.deepcopy(spec);name=f'expert_r{repeat}_'+('layout10' if displaced else 'unshifted')
            s.update(identity=name,split='diagnostic_positive_control',cohort='fresh',diagnostic_only=True)
            s['narrow_reset']=(reset_configuration(session['domain'],seed,.1) if displaced else direct_configuration(session['domain'],seed))
            cases.append(dict(name=name,spec=s))
    atomic_json(owner/'plan.json',dict(mode='expert',cases=cases,source_sha256=sources()))
    job(owner,[dict(entry='machine_review',args=['worker','--output',str(owner),'--mode','expert'])],'expert_controls')
    atomic_json(ROOT/'registration.json',dict(registered_utc=time.time(),source_sha256=sources(),
        pairs=pairs,wave_plan_sha256=hashes,positive_seed=seed,paired_cases=16,expert_cases=4,
        tests_read_for_rule_selection=False,original64_changed=False,optimizer_updates=0,
        hypothesis='paired new cohorts used direct_configuration without the historical actual100mm peg displacement',
        attachment='No patch/zip/python attachment found in exposed attachment directory; original-machine source is authoritative'))
    print(ROOT)


def watchdog(output):
    """Bound our own process if native initialization/cleanup stops progressing."""
    import threading,faulthandler
    stream=(output/'watchdog_tracebacks.log').open('x')
    began=time.time()
    def watch():
        while True:
            time.sleep(5)
            files=[*output.glob('batch_progress.json'),*output.glob('*/progress.json'),*output.glob('*/result.json')]
            newest=max([began]+[p.stat().st_mtime for p in files if p.exists()])
            initialized=(output/'initialization.json').exists()
            bound=300 if initialized else 600
            if time.time()-newest>bound:
                atomic_json(output/'watchdog_failure.json',dict(utc=time.time(),pid=os.getpid(),
                    initialized=initialized,last_progress_utc=newest,limit_s=bound,reason='diagnostic worker no progress'))
                faulthandler.dump_traceback(file=stream,all_threads=True);stream.flush();os._exit(124)
    threading.Thread(target=watch,daemon=True).start()


def worker(output,mode):
    output=Path(output).resolve();plan=read(output/'plan.json')
    for p,digest in plan['source_sha256'].items():
        if sha256(p)!=digest:raise ValueError('Diagnostic source changed '+p)
    watchdog(output)
    if mode=='autonomous':
        from visuotactile.paired_execution import install
        from tools.tight_paired_evaluate import no_expert_scope
        from tools.tight_evaluate import run
        install(output)
        with no_expert_scope(output,plan['batch']['lanes']):run(output)
    elif mode=='expert':
        from tools.tight_state import environment
        from visuotactile.state_benchmark import trial
        from visuotactile.position_control import ExpertPositionBridge
        from visuotactile.native_evaluation import exit_completed_kit_worker
        session=read(output/'session.json')
        app,env,player,lease=environment(output,evaluation=True)
        atomic_json(output/'initialization.json',dict(utc=time.time(),pid=os.getpid()))
        env.speed_config=dict(ExpertPositionBridge.speed_scheme)
        rows=[]
        for case in plan['cases']:
            row=trial(env,lease,output/case['name'],case['spec'],{},player,session['expert_checkpoint'],
                expert_speed=ExpertPositionBridge.speed_scheme,video=False,capture_dataset=False)
            rows.append(dict(case=case['name'],result=row));atomic_json(output/'results.json',rows)
        atomic_json(output/'result.json',dict(software_complete=True,utc=time.time()))
        exit_completed_kit_worker(output)
    else:raise ValueError('Unregistered worker mode')


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':
        from tools import tight_project
        tight_project.ENTRIES['machine_review']=('evaluate','tools.tight_machine_review',[])
        sys.argv=[sys.argv[0],*sys.argv[2:]];tight_project.main();return
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare-controls','worker'])
    p.add_argument('--output',type=Path);p.add_argument('--mode',choices=['autonomous','expert'])
    a=p.parse_args()
    if a.stage=='prepare-controls':controls()
    else:
        try:worker(a.output,a.mode)
        except BaseException as e:
            atomic_json(a.output/'worker_failure.json',dict(error=repr(e),utc=time.time(),traceback=traceback.format_exc()))
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)

if __name__=='__main__':main()
