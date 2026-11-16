"""Two finite evaluation groups; no data admission, optimizer or old owner."""
import copy,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from scripts.tight_stability_20261116.analyze_existing import TREE,OLD,OUT,read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

ROOT=TREE/'outputs/visuotactile/run/tight_stability_20261116_01'

def prepare():
    import torch
    from visuotactile.state_benchmark import dagger_policy_identity
    from visuotactile.narrow_data import direct_configuration
    assert not ROOT.exists();assert shutil.disk_usage(TREE).free>16*2**30
    ROOT.mkdir();now=time.time();session=read(OLD/'session.json')
    session.update(started_utc=now,deadline=now+3*3600,resource_root=str(ROOT.resolve()),
        training_deadline=now+3*3600,offline_deadline=now+3*3600,cache_deadline=now+3*3600,
        owned_worker_roots=[str(ROOT.resolve())],authorization='Bounded grasp stability diagnosis; no training, DAgger or sampling change',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=TREE,text=True).strip())
    atomic_json(ROOT/'session.json',session)
    pairsets={}
    for group in ['candidates','fallbacks']:
        pairsets[group]={}
        for role in 'AB':
            if group=='fallbacks':path=Path(read(OLD/role/'continuation.json')['previous_backend'])
            else:path=OLD/role/'backend'/role/'training'/('best.pt' if role=='A' else 'best_through_epoch8.pt')
            saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            pairsets[group][role]=dagger_policy_identity(saved,str(path.resolve()))
    registration=dict(utc=now,deadline=session['deadline'],groups=pairsets,
        seeds=list(range(28500100,28500108)),cohort='New development families, no training/validation/test use',
        attempts=32,lanes=16,grasp_horizon=480,full_task_seeds=[28500100],
        primary='Ever hold60/120/180 and final60 at common first480 controls; native lift separate',
        invalid_starts='Retain and exclude from legal denominator; no replacement',
        full_execution='First registered seed per role/group runs to native termination, not grasp cutoff',
        capture='Read-only exact rear input stream for independent same-input checkpoint replay; no cached trainable hidden state',
        rationale='Existing A evidence: two post-hold contact-loss/drop events at phases4 and10, one two-to-one finger event atphase0; B same-state expert error worsens before contact. Compare retained candidates to safe fallbacks before any more training.',
        source_commit=session['source_commit'],old_completed_run_untouched=str(OLD.resolve()))
    for group,pairs in pairsets.items():
        d=ROOT/group;d.mkdir();plan=copy.deepcopy(read(OLD/'evaluation_extended_best/plan.json'))
        plan.update(pairs=pairs,cases=[],resource_root=str(ROOT.resolve()),utc=now,source_commit=session['source_commit'],
            source_sha256={str(p):sha256(p) for folder in ['visuotactile','tools','scripts/tight_stability_20261116'] for p in sorted(Path(folder).rglob('*.py'))})
        plan['batch']['seed']=28500100
        for seed in registration['seeds']:
            for role in 'AB':
                spec=dict(identity=f'stability_development_{seed}',family=f'fresh_development_{seed}',split='development',
                    seed=seed,event_seed=seed+100000,cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=direct_configuration(session['domain'],seed))
                plan['cases'].append(dict(name=role+'_'+spec['identity'],role=role,spec=spec,video=False,
                    grasp_evaluation_ticks=None if seed in registration['full_task_seeds'] else 480,diagnostic_cohort='new_development'))
        plan['evaluation_scope']=registration;plan.pop('training_rechecks',None)
        atomic_json(d/'plan.json',plan);atomic_json(d/'session.json',session)
    atomic_json(ROOT/'registration.json',registration);OUT.mkdir(parents=True,exist_ok=True);atomic_json(OUT/'registration.json',registration)

def execute():
    for group in ['candidates','fallbacks']:
        d=ROOT/group;assert not (d/'run_started.json').exists(),'No implicit restart of prior simulation'
        with (ROOT/(group+'.log')).open('x') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.tight_stability_20261116.capture_eval','--output',str(d)],
                cwd=TREE,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1'),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state=dict(group=group,pid=child.pid,started=time.time());last=None;changed=time.time()
            while child.poll() is None:
                progress=max([p.stat().st_mtime for p in d.rglob('*.json')]+[log.name and Path(log.name).stat().st_mtime])
                if last!=progress:changed=time.time();last=progress
                state.update(heartbeat=time.time(),last_progress=progress);atomic_json(ROOT/'state.json',state)
                if time.time()>read(ROOT/'session.json')['deadline'] or time.time()-changed>600 or (ROOT/'STOP').exists():
                    os.killpg(child.pid,15)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,9);child.wait()
                    raise RuntimeError('Deadline/STOP/stall; no automatic bad-task restart')
                time.sleep(5)
            state.update(exit_code=child.returncode,ended=time.time());atomic_json(ROOT/(group+'_finished.json'),state)
            assert child.returncode==0 and read(d/'result.json')['software_complete'] and read(d/'input_capture_complete.json')['complete']
    atomic_json(ROOT/'complete.json',dict(complete=True,utc=time.time(),collection=0,optimizer_updates=0))

if __name__=='__main__':
    try:
        if sys.argv[1]=='prepare':prepare()
        elif sys.argv[1]=='run':execute()
        else:raise ValueError('prepare or run')
    except BaseException:
        if ROOT.exists():atomic_json(ROOT/'failure.json',dict(utc=time.time(),error=traceback.format_exc()))
        raise
