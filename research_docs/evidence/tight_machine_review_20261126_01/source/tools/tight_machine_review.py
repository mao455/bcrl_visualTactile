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
    elif mode=='takeover':
        takeover_worker(output,plan)
    elif mode=='history':
        history_worker(output,plan)
    else:raise ValueError('Unregistered worker mode')


def prepare_pilot():
    """Two historical development families, real expert versus continued student."""
    from visuotactile.adaptive_takeover import profile
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.state_benchmark import CAPTURE_AUDIT_MEMO_PROFILE
    from scripts.tight_layout_20261116.pilot import reset_configuration
    owner=ROOT/'never_gate_pilot02'; session=session_at(owner,duration=7200)
    out=owner/'rollouts';out.mkdir();atomic_json(out/'session.json',session)
    template=read(PAIRED/'collection/plan.json')
    hist=read(OLD/'registration.json')['pairs'];cases=[]
    for seed in (28826003,28826006):
        for observe in (True,False):
            name=f'B_entry_{seed}_'+('continue' if observe else 'expert')
            spec=dict(identity=name,family=f'cycle_dev_{seed}',seed=seed,event_seed=seed+100000,
                split='diagnostic_development',cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',
                narrow_reset=reset_configuration(session['domain'],seed,.1),layout=session['layout'],
                takeover_tick=10**9,diagnostic_only=True,source_identity=f'B_entry_{seed}')
            cases.append(dict(name=name,role='B',spec=spec,video=False,
                source_data=str(PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01/B/state_data/data.json'),
                dagger=dict(version=ExpertPositionBridge.version,expert_speed=ExpertPositionBridge.speed_scheme,
                    diagnostic_takeover=True,trigger=profile('alignment',never_entered_only=True,observe_only=observe))))
    template.update(kind='takeover',resource_root=str(owner),pairs={'B':hist['B_entry']},cases=cases,
        stage_takeover_contract='original_machine_never_gate_diagnostic_v1',round_index=0,
        source_root=str(PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01'),
        source_data_sha256={'B':sha256(cases[0]['source_data'])},source_session_sha256=sha256(owner/'session.json'),
        source_sha256=sources(),first_plan_forensics=False,test_used=False,record_sampler_rng=True,
        diagnostic_only=True,selection='Isolated rescue and fixed-history check only; no main-pool or model promotion',
        history_rule='expert queried from tick0 on its own actual observation and delay history; student prefix mask0')
    template.pop('training_expansion_registry',None)
    template['batch'].update(lanes=4,seed=28826003,verification_workers=0,capture_audit_memo=dict(CAPTURE_AUDIT_MEMO_PROFILE))
    template['execution_backend'].update(kit_tasking_threads=8,physics_threads=2)
    atomic_json(out/'plan.json',template)
    atomic_json(owner/'registration.json',dict(utc=time.time(),families=[28826003,28826006],opportunities=4,
        terminal64_used_for_selection=False,same_state_branch=False,
        pairing='Same initial/reset policy noise; physical state equality at handoff is measured, not assumed',
        rule=profile('alignment',never_entered_only=True),plan_sha256=sha256(out/'plan.json')))
    job(owner,[dict(entry='machine_review',args=['worker','--output',str(out),'--mode','takeover'])],'never_gate_pilot')
    print(out)


def takeover_worker(output,plan):
    from tools import tight_fresh_batch_run as runner
    from tools.tight_fresh_report import audit_trigger_case
    from visuotactile.takeover_trigger import validate_trigger
    from visuotactile.paired_execution import install
    def validate(p):
        assert p['stage_takeover_contract']=='original_machine_never_gate_diagnostic_v1' and p['diagnostic_only'] and not p['test_used']
        assert len(p['cases']) in (2,4)
        seeds={c['spec']['seed'] for c in p['cases']}
        assert len(seeds) in (1,2) and seeds <= {28817004,28817008,28826000,28826003,28826006,28850001,28850003,28911002,28911003}
        assert all(c['spec']['split'] in ('diagnostic_development','diagnostic_train') for c in p['cases'])
        for c in p['cases']:
            assert c['spec']['takeover_tick']==10**9 and c['dagger']['trigger']['never_entered_only'] is True
            validate_trigger(c['dagger']['trigger'],10**9)
    def classify(directory,case,result,p):
        if not result.get('legal'):return 'diagnostic_stage_audit.json','invalid_initialization',None
        audit=audit_trigger_case(directory,result)
        return 'diagnostic_stage_audit.json',audit['category'],audit
    def admit(out,case,result,pair,frozen,entries,**kwargs):
        from visuotactile.state_benchmark import audit_dagger_capture
        result['accepted_for_training']=False
        if result.get('legal'):
            audit=audit_dagger_capture(out/case['name'],pair,**({'audit_memo':kwargs['audit_memo']} if kwargs.get('audit_memo') is not None else {}))
            atomic_json(out/case['name']/'capture_audit.json',audit)
            if not audit['passed']:raise ValueError('Diagnostic handoff/action-chain audit failed')
            stage=audit_trigger_case(out/case['name'],result)
            atomic_json(out/case['name']/'diagnostic_stage_audit.json',stage)
        result['diagnostic_main_pool_admission_blocked']=True
        atomic_json(out/case['name']/'result.json',result)
        atomic_json(out/'correction_entries.json',entries)
    validate(plan);install(output)
    runner.validate_takeover_plan=validate;runner.classify_capture=classify;runner.admit_capture=admit
    runner.run(output)


def history_worker(output, plan):
    # Reuse the existing real-history GPU consumer and exact feedback/RNG audits.
    from scripts.tight_entry_final24_20261122 import entry_history
    from types import SimpleNamespace
    entry_history.AUDIT=output
    entry_history.NEW=Path(plan.get('diagnostic_data_root',PAIRED/'learning'))
    def summarize(rows, original, began):
        import numpy as np
        summary=[]
        for role in 'AB':
            for tag in ('parent','new'):
                for first in (True,False):
                    subset=[r for r in rows if r['role']==role and r['policy']==tag and r['first_legal']==first]
                    if subset:
                        summary.append(dict(role=role,policy=tag,first_legal=first,slots=len(subset),
                            final_arm_rad=float(np.mean([r['final_command']['arm_rmse_rad'] for r in subset])),
                            final_hand_rad=float(np.mean([r['final_command']['hand_rmse_rad'] for r in subset]))))
        return dict(complete=True,rows=summary,wall_s=time.time()-began,optimizer_updates=0,physics_steps=0,
            interpretation=plan.get('interpretation','Fixed real-history fit check; no independent autonomous recovery estimate'))
    entry_history.summarize=summarize
    entry_history.run(SimpleNamespace(output='inference',population='all150',identities='',limit=0,
        device='cuda:0',roles='AB',tags='parent,new'))
    if plan.get('replay_source_policy',False):
        entry_history.run(SimpleNamespace(output='inference_source',population='all150',identities='',limit=0,
            device='cuda:0',roles='B',tags='source'))
    atomic_json(output/'result.json',dict(software_complete=True,utc=time.time(),optimizer_updates=0,physics_steps=0))


def main():
    if len(sys.argv)>1 and sys.argv[1]=='project':
        from tools import tight_project
        tight_project.ENTRIES['machine_review']=('evaluate','tools.tight_machine_review',[])
        sys.argv=[sys.argv[0],*sys.argv[2:]];tight_project.main();return
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare-controls','prepare-pilot','worker'])
    p.add_argument('--output',type=Path);p.add_argument('--mode',choices=['autonomous','expert','takeover','history'])
    a=p.parse_args()
    if a.stage=='prepare-controls':controls()
    elif a.stage=='prepare-pilot':prepare_pilot()
    else:
        try:worker(a.output,a.mode)
        except BaseException as e:
            atomic_json(a.output/'worker_failure.json',dict(error=repr(e),utc=time.time(),traceback=traceback.format_exc()))
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)

if __name__=='__main__':main()
