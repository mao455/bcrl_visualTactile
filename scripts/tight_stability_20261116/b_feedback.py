"""Adaptive finite B rear6/22 closed-history comparison after common-input probe."""
import copy,json,os,subprocess,sys,time
from pathlib import Path
from scripts.tight_stability_20261116.analyze_existing import TREE,OLD,OUT,read
from scripts.tight_stability_20261116.run import ROOT
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

def prepare():
    import torch
    from visuotactile.state_benchmark import dagger_policy_identity
    assert read(ROOT/'complete.json')['complete'] and read(OUT/'common_student.json')['complete']
    d=ROOT/'b_feedback';assert not d.exists();d.mkdir()
    original=read(OLD/'evaluation_extended_dp_rear8/plan.json')
    oldcases=[c for c in original['cases'] if c['role']=='B' and c.get('diagnostic_cohort','fixed_development')=='fixed_development']
    newcases=[c for c in read(ROOT/'candidates/plan.json')['cases'] if c['role']=='B'][:4]
    assert len(oldcases)==len(newcases)==4
    cases=[];pairs={}
    for label,file in [('B_short','best_through_epoch8.pt'),('B_long','best.pt')]:
        path=OLD/'B/backend/B/training'/file;s=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        pairs[label]=dagger_policy_identity(s,str(path.resolve()))
    for c in oldcases+newcases:
        for label in pairs:
            item=copy.deepcopy(c);item.update(name=label+'_'+c['spec']['identity'],role=label,grasp_evaluation_ticks=300,video=False)
            item['diagnostic_cohort']='previous_fixed_development' if c in oldcases else 'already_observed_new_development'
            cases.append(item)
    plan=copy.deepcopy(read(ROOT/'candidates/plan.json'))
    registration=dict(utc=time.time(),attempts=16,deadline=read(ROOT/'session.json')['deadline'],
        reason='New B short-policy common-input replay gives only1.4% first48 degradation under rear22; old separate closed trajectories differed~2x. Need both driving histories at same registered families to isolate feedback amplification.',
        scope='Adaptive mechanistic diagnostic, not independent new performance validation; 4 old fixed + first4 already observed new development per rear; all full DP+Transformer,300controls; no DAgger/training, no outcome replacement',
        roles='B_short and B_long are aliases for two B command-supervised complete pairs, not the A/B learning definitions',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    plan.update(cases=cases,pairs=pairs,utc=time.time(),evaluation_scope=registration,
        source_commit=registration['source_commit'],source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools','scripts/tight_stability_20261116'] for p in sorted(Path(folder).rglob('*.py'))})
    plan['batch']['seed']=28500000
    atomic_json(d/'plan.json',plan);atomic_json(d/'session.json',read(ROOT/'session.json'));atomic_json(d/'registration.json',registration)
    atomic_json(OUT/'b_feedback_registration.json',registration)

def run():
    d=ROOT/'b_feedback';assert not (d/'run_started.json').exists()
    assert (ROOT/'STOP_MONITOR').exists() and (ROOT/'monitor_exit.json').exists()
    for name in ['monitor.json','monitor_exit.json']:(ROOT/name).rename(ROOT/(name+'.before_b_feedback'))
    (ROOT/'STOP_MONITOR').rename(ROOT/'STOP_MONITOR.before_b_feedback')
    with (d/'monitor.log').open('x') as ml,(d/'worker.log').open('x') as log:
        monitor=subprocess.Popen([sys.executable,'-m','visuotactile.state_resources','--root',str(ROOT)],cwd=TREE,stdout=ml,stderr=subprocess.STDOUT)
        child=None
        try:
            start=time.time()
            while read(ROOT/'shared_resources.json')['utc_epoch']<start:
                if monitor.poll() is not None or time.time()-start>15:raise RuntimeError('Monitor startup')
                time.sleep(.25)
            child=subprocess.Popen([sys.executable,'-m','scripts.tight_stability_20261116.capture_eval','--root',str(ROOT),'--output',str(d)],
                cwd=TREE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'))
            last=None;changed=time.time();state=dict(pid=child.pid,started=time.time())
            while child.poll() is None:
                progress=max(p.stat().st_mtime for p in d.rglob('*.json'))
                if progress!=last:changed=time.time();last=progress
                state.update(heartbeat=time.time(),last_progress=progress);atomic_json(d/'owner.json',state)
                if time.time()>read(ROOT/'session.json')['deadline'] or time.time()-changed>600 or (ROOT/'STOP').exists():raise RuntimeError('Finite deadline/stall/STOP')
                time.sleep(3)
            state.update(ended=time.time(),exit_code=child.returncode);atomic_json(d/'owner.json',state)
            assert child.returncode==0 and read(d/'input_capture_complete.json')['complete']
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,15)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait()
            (ROOT/'STOP_MONITOR').touch();monitor.wait(timeout=15)
if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='run':run()
    else:raise ValueError('prepare or run')
