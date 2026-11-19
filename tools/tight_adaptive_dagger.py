"""Finite two-round experiment glue. All physics/training use existing workers."""
import argparse
import collections
import copy
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from tools.tight_frozen_frontend import PROJECT, read, sources, write
from tools.tight_fresh_rollouts import completed_pair
from visuotactile.compact import sha256

CANONICAL = Path('/home/CNS2025331827/docs/bc+rl_RSS')
BASE = CANONICAL/'outputs/visuotactile/run/tight_exposure50_20261119_01'
ROOT = CANONICAL/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01'
PUBLIC = PROJECT/'research_docs/evidence/tight_adaptive_dagger_20261119_01'
CONFIG = PROJECT/'configs/experiments/adaptive_dagger'


def session_at(path, session):
    path.mkdir(parents=True, exist_ok=True)
    s=copy.deepcopy(session);s.update(resource_root=str(path),owned_worker_roots=[str(ROOT)])
    write(path/'session.json',s)
    return s


def resource_wait(path, *, simulation=False, lanes=4):
    import psutil
    s=read(ROOT/'session.json')
    while True:
        free=shutil.disk_usage(ROOT).free/2**30;ram=psutil.virtual_memory().available/2**30
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        vram=int(gpu.strip().splitlines()[0])
        ready=free>=s['resources']['min_disk_gib']+4 and ram >= ((28 if lanes<=2 else 34) if simulation else 18) and vram >= (16000 if simulation else 11000)
        write(path/'resource_admission.json',dict(ready=ready,disk_gib=free,ram_gib=ram,vram_free_mib=vram,simulation=simulation,utc=time.time()))
        if ready:return
        if time.time()>=s['deadline']:raise TimeoutError('New experiment resource deadline')
        time.sleep(30)


def job(operation, name, stages):
    path=CONFIG/(name+'.json');owner=ROOT/'owners'/name
    session_at(owner,read(ROOT/'session.json'))
    write(owner/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
    write(path,dict(version='tight_project_job_v1',operation=operation,resource_root=str(owner),stages=stages))
    return path,owner


def execute(operation, name, stages, *, simulation=False):
    path,owner=job(operation,name,stages)
    lanes = max((read(Path(stage['args'][stage['args'].index('--output')+1])/'plan.json')['batch']['lanes'] for stage in stages), default=4) if simulation else 0
    resource_wait(owner,simulation=simulation,lanes=lanes)
    telemetry=ROOT/'shared_resources.json'
    if telemetry.is_symlink():telemetry.unlink()
    elif telemetry.exists():raise ValueError('Unexpected root telemetry file')
    telemetry.symlink_to(owner/'shared_resources.json')
    with (owner/'driver.log').open('x') as log:
        subprocess.run([sys.executable,'-m','tools.tight_project',operation,'--job',str(path)],cwd=PROJECT,
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    return owner


def initialize():
    if ROOT.exists():raise FileExistsError('Do not overwrite attempted experiment')
    ROOT.mkdir(parents=True);PUBLIC.mkdir(parents=True);CONFIG.mkdir(parents=True)
    session=read(BASE/'session.json');now=time.time()
    session.update(version='adaptive_dagger_two_rounds_v1',started_utc=now,deadline=now+48*3600,
        resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        authorization='Two adaptive complete-student DAgger rounds; all four selected components >=50 newly committed exposures per legal tick/trajectory; native task unchanged')
    for k in ('collection_deadline','training_deadline','cache_deadline','offline_deadline'):session[k]=session['deadline']
    session_at(ROOT,session)
    protected={}
    for role in 'AB':
        pair=completed_pair(BASE,role)
        for key in ('frontend','backend'):protected[pair[key]]=pair[key+'_sha256']
        data=read(BASE/role/'state_data/data.json')
        for name in ('data.json','statistics.pt','numeric.pt'):
            p=BASE/role/'state_data'/name;protected[str(p.resolve())]=sha256(p)
        assert len(data['splits']['train'])==130 and len(data['splits']['validation'])==6
    write(PUBLIC/'protected.json',protected)
    write(CONFIG/'experiment.json',dict(version='adaptive_dagger_two_rounds_v1',root=str(ROOT),base=str(BASE),
        source_worktree=str(PROJECT),epochs=60,rounds=2,target_new_trajectories_per_round=[24,40],
        maximum_opportunities_per_round=56,minimum_per_student=10,lanes=4,
        training_seeds=[28817000+4*i for i in range(8)],development_seeds=list(range(28826000,28826008)),
        final_test_seeds=list(range(28839000,28839008)),
        final_test_policy='Unopened until both rounds and final validation selection freeze; no learning/selection after opening',
        dp_lr=1.5e-5,backend_lr=1e-4,
        optimizer='Fresh AdamW/DP EMA from each own parent; per-run restores full optimizer/RNG/exposure',
        storage_policy='No historical deletion; old raw and frozen caches linked; new round data cumulative',
        resources='12GiB disk,10GiB RAM,4GiB VRAM reserve; simulation admission28GiB RAM/16GiB VRAM free',session=session))
    write(PUBLIC/'registration.json',read(CONFIG/'experiment.json'))
    diagnose()


def diagnose():
    import h5py
    import numpy as np
    from visuotactile.adaptive_takeover import profile
    from visuotactile.takeover_trigger import make_trigger,observe_transition
    rows=[];forks=[]
    for wave in sorted((BASE/'evaluation').glob('wave*')):
        plan=read(wave/'plan.json');lookup={c['role']:c for c in plan['cases']}
        for role in 'AB':
            new=lookup[role+'130_50'];old=lookup[role+'72'];folder=wave/new['name'];result=read(folder/'result.json')
            with h5py.File(folder/'numeric.h5') as f:
                t=f['episode/transition'];screens={}
                for stage in ('contact','transport','alignment','instability','post_subgoal'):
                    trigger=make_trigger(profile(stage),10**9)
                    for tick in range(result['steps']):
                        observe_transition(trigger,tick,t,index=tick)
                        if trigger.trigger_completed_tick is not None:break
                    screens[stage]=trigger.receipt()
                row=dict(role=role,seed=new['spec']['seed'],cohort=new.get('diagnostic_cohort'),
                         success=result['success'],steps=result['steps'],screens=screens)
                rows.append(row)
                oldresult=read(wave/old['name']/'result.json')
                if oldresult['success'] and not result['success']:
                    with h5py.File(wave/old['name']/'numeric.h5') as g:
                        ot=g['episode/transition'];n=min(len(t['target_after_rad']),len(ot['target_after_rad']))
                        def first(x):
                            z=np.flatnonzero(x);return int(z[0]) if len(z) else None
                        command=np.abs(t['target_after_rad'][:n]-ot['target_after_rad'][:n]).max(-1)
                        newpose=t['diagnostics_after/object_pose_w'][:n,:3]-t['diagnostics_after/base_pose_w'][:n,:3]
                        oldpose=ot['diagnostics_after/object_pose_w'][:n,:3]-ot['diagnostics_after/base_pose_w'][:n,:3]
                        pose=np.linalg.norm(newpose-oldpose,axis=-1)
                        forks.append(dict(role=role,seed=new['spec']['seed'],cohort=new.get('diagnostic_cohort'),
                            first_command_difference_tick=first(command>1e-6),first_object_divergence_1mm_tick=first(pose>.001),
                            old_success=True,new_success=False,source_wave=str(wave),
                            limitation='Paired observed starts; no counterfactual expert labels or exact hidden-state replay'))
    write(PUBLIC/'baseline_diagnosis.json',dict(rows=rows,lost_success_forks=forks,
        selection='Thresholds fixed before new data; future schedule tuning uses training families and development diagnostics; development/test never training captures'))
    print(json.dumps(dict(baseline_cases=len(rows),forks=forks,training_screen_hits={s:sum(r['screens'][s]['trigger_completed_tick'] is not None for r in rows if r['cohort']=='training') for s in ('contact','transport','alignment','instability','post_subgoal')})),flush=True)


def prepare_collection(source, output, families, stages, round_index):
    from tools.tight_fresh_batch_run import prepare
    from visuotactile.adaptive_takeover import profile
    from visuotactile.state_benchmark import dagger_configuration
    template=read(BASE/'evaluation/wave00/plan.json')
    ids=['layout10_train_'+str(seed) for seed in families]
    prepare(source,output,lanes=min(2,len(ids)*2),identities=ids,delays=[0]*len(ids),
            round_index=round_index,seed=28840000+round_index*100+families[0]%100,
            session_root=ROOT,execution_backend=template['execution_backend'],
            writer_bytes=128<<20,vision_microbatch=1,verification_workers=0,capture_audit_memo=True,queued_gpu=True)
    plan=read(output/'plan.json');plan['stage_takeover_contract']='layout_multistage_only_v1'
    for c in plan['cases']:
        index=ids.index(c['spec']['source_identity']);stage=stages[index]
        old=c['name'];c['name']=old.replace('contact_height_only_d0',stage)
        c['spec'].update(identity=c['name'],layout=read(ROOT/'session.json')['layout']);c['dagger']['trigger']=profile(stage)
        c['video']=index==0
        dagger_configuration(c['dagger'],c['spec'],read(Path(c['source_data'])))
        c['adaptive_stage']=stage
    plan.update(source_sha256=sources(),reason='Stage-specific recovery screens on completed full-student histories; no time fallback',
                source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    write(output/'plan.json',plan)
    return plan


def collect(source, round_index, schedule):
    destination=ROOT/f'round{round_index:02d}'
    destination.mkdir(exist_ok=True);batches=[];entries={};allrows=[];allcases=[]
    for index,(families,stages) in enumerate(schedule):
        output=destination/f'collection{index:02d}'
        suffix=''
        if output.exists():
            attempted=[output]+sorted(destination.glob(output.name+'_recovery*'))
            assert round_index==1 and index==0
            assert all(not (p/'result.json').exists() for p in attempted), 'Completed capture needs explicit resume, never retry'
            assert all(not list(p.glob('*/result.json')) for p in attempted), 'Retain completed cases before resuming'
            failed=attempted[-1];suffix=f'_recovery{len(attempted):02d}'
            output=destination/(output.name+suffix)
            write(PUBLIC/(output.name+'_recovery.json'),dict(failed=[str(p) for p in attempted],recovery=str(output),
                reason='Prior attempts retained; reduce simultaneous capture lanes to2 and require28GiB available RAM',
                original_deadline=read(ROOT/'session.json')['deadline'],prior_physical_prefixes_preserved=True,
                previous_partial_prefixes_counted_as_training=False))
        plan=prepare_collection(source,output,families,stages,round_index*10+index)
        execute('dagger',f'r{round_index}_collection{index:02d}{suffix}',[dict(entry='dagger',args=['--output',str(output)])],simulation=True)
        result=read(output/'result.json');assert result['software_complete']
        new=read(output/'correction_entries.json');assert not set(new)&set(entries)
        entries.update(new);allrows.extend(read(output/'results.json'));allcases.extend(plan['cases']);batches.append(str(output))
        collection_report(destination,allrows,entries,allcases,batches)
        counts=collections.Counter(e['source_group'] for e in entries.values())
        if len(entries)>=28 and min(counts.get(r,0) for r in 'AB')>=12:break
        attempted=collections.Counter(c['adaptive_stage'] for c in allcases)
        recovered=collections.Counter(next(c['adaptive_stage'] for c in allcases if c['name']==name) for name in entries)
        bad={stage for stage,n in attempted.items() if n>=4 and recovered[stage]==0}
        for future in range(index+1,len(schedule)):
            families0,stages0=schedule[future]
            schedule[future]=(families0,['transport' if stage in bad else stage for stage in stages0])
    combined=destination/'collection_combined';combined.mkdir()
    combined_plan=copy.deepcopy(plan);combined_plan['cases']=allcases;combined_plan['source_sha256']=sources()
    for output in map(Path,batches):
        for case in read(output/'plan.json')['cases']:(combined/case['name']).symlink_to(output/case['name'],target_is_directory=True)
    write(combined/'session.json',read(ROOT/'session.json'));write(combined/'plan.json',combined_plan)
    write(combined/'correction_entries.json',entries);write(combined/'results.json',allrows)
    write(combined/'result.json',dict(software_complete=True,cases=len(allrows),accepted_training_trajectories=len(entries)))
    counts=collections.Counter(e['source_group'] for e in entries.values())
    if len(entries)<12 or min(counts.get(r,0) for r in 'AB')<4:raise ValueError('Insufficient valid recoveries; inspect/adapt schedule before training')
    return combined


def collection_report(destination,rows,entries,cases,batches):
    lookup={c['name']:c for c in cases};table=[]
    for role in 'AB':
        rr=[r for r in rows if r['role']==role]
        table.append(dict(role=role,attempts=len(rr),accepted=sum(r['case'] in entries for r in rr),
            expert_takeovers=sum(r['result'].get('dagger',{}).get('recovery_coverage',{}).get('switch_tick') is not None for r in rr),
            expert_success=sum(bool(r['result'].get('dagger',{}).get('expert_takeover_success')) for r in rr),
            legal_ticks=sum(r['result'].get('dagger',{}).get('expert_valid_ticks',0) for r in rr if r['case'] in entries),
            stage_attempts=dict(collections.Counter(lookup[r['case']]['adaptive_stage'] for r in rr)),
            stage_accepted=dict(collections.Counter(lookup[r['case']]['adaptive_stage'] for r in rr if r['case'] in entries))))
    public=PUBLIC/destination.name;public.mkdir(exist_ok=True)
    write(public/'collection_summary.json',dict(table=table,batches=batches,rows=rows))
    write(public/'collection_manifest.json',entries)
    print(json.dumps(table),flush=True)


def prepare_learning(source, collection, round_index):
    from tools.tight_fresh_dagger import prepare_aggregate
    from visuotactile.trajectory_training import profile,budget
    from visuotactile.bounded_extension import exposure50_profile,split_audit
    from visuotactile.latest_takeover_sampling import sampling_plan
    from visuotactile.effective_exposure import profile as exposure_profile
    target=ROOT/f'round{round_index:02d}'/'learning'
    prepare_aggregate(source,collection,target)
    s=copy.deepcopy(read(ROOT/'session.json'));s.update(resource_root=str(target),phase_dagger=True,fixed56_long=False,matched_update_diagnostic=False)
    write(target/'session.json',s)
    (target/'shared_resources.json').unlink();(target/'shared_resources.json').symlink_to(ROOT/'shared_resources.json')
    replay=dict(version='original_demonstration_replay_v1',original_repeat=3)
    write(target/'backend_episode_sampling.json',replay)
    reg=dict(profile=profile(),roles={})
    for role in 'AB':
        child=target/role;write(child/'session.json',s)
        (child/'shared_resources.json').unlink();(child/'shared_resources.json').symlink_to(target/'shared_resources.json')
        d=read(child/'state_data/data.json');cont=read(child/'continuation.json')
        cont.update(sampling_contract=replay,learning_rate=1.5e-5)
        write(child/'continuation.json',cont)
        _,mult=sampling_plan(d,replay)
        reg['roles'][role]=dict(**budget(sum(mult.values()),60),data_sha256=sha256(child/'state_data/data.json'),episode_multiplicities=mult)
        assert set(read(source/role/'state_data/data.json')['splits']['train'])<=set(d['splits']['train'])
        assert sha256(child/'state_data/statistics.pt')==sha256(source/role/'state_data/statistics.pt')
        split_audit(d)
    for name in ('training_sampling.json','cuda_allocator_policy.json','frozen_visual_io.json','frozen_visual_hotset.json'):
        shutil.copyfile(BASE/name,target/name)
    write(target/'effective_exposure.json',exposure_profile())
    write(target/'bounded_extension.json',dict(policy=exposure50_profile()))
    write(target/'full_trajectory_batch.json',reg)
    write(PUBLIC/f'round{round_index:02d}'/'learning_registration.json',dict(source=str(source),target=str(target),session=s,training=reg,
        fresh_exposure=True,minimum_tick_steps=50,minimum_visits=50,ancestors_counted=False,
        selection='Validation minimum among actual selected checkpoint exposure>=50; 60 complete decks',
        dp_lr=1.5e-5,backend_lr=1e-4,old_statistics_unchanged=True))
    return target


def train(target, round_index):
    execute('prepare',f'r{round_index}_visual_cache',[dict(entry='visual_cache',args=['--root',str(target),'--output',str(target),'--payload-audit-memo'])])
    stages=[]
    for role in 'AB':
        stages.append(dict(entry='frontend',args=['--root',str(target),'--role',role,'--epochs','60','--full-trajectory-batch','--offline-anchor-batch','64']))
    for role in 'AB':
        backend=target/role/'backend'
        stages.extend([
            dict(entry='backend_prepare',args=['--root',str(target),'--role',role,'--epochs','60','--continue-rear','--offline-anchor-batch','64','--frozen-visual-io-json',str(target/'frozen_visual_io.json')]),
            dict(entry='backend_cache',args=['--root',str(backend)]),
            dict(entry='backend_train',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'training')]),
            dict(entry='backend_offline',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'offline')])])
    execute('train',f'r{round_index}_train',stages)
    audit_training(target,round_index)


def audit_training(target,round_index):
    import importlib.util
    spec=importlib.util.spec_from_file_location('exposure_export',PROJECT/'research_docs/evidence/tight_exposure50_20261119_01/analyze.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    public=PUBLIC/f'round{round_index:02d}'
    module.paths=lambda: (read(CONFIG/'experiment.json'),target,BASE,BASE,public)
    module.training()
    rows=read(public/'training_completion.json')['components']
    assert len(rows)==4 and all(r['eligible'] and r['minimum_visits']>=50 and r['minimum_tick_steps']>=50 and r['receipt_replay_exact'] for r in rows)
    write(target/'exposure_acceptance.json',dict(passed=True,components=rows,utc=time.time()))
    return rows


def evaluation_plans(parent,current,round_index, *, heldout=False):
    from scripts.tight_layout_20261116.pilot import reset_configuration
    c=read(CONFIG/'experiment.json');s=read(ROOT/'session.json')
    old=read(BASE/'evaluation/wave00/plan.json');specs={}
    for wave in (BASE/'evaluation').glob('wave*'):
        for case in read(wave/'plan.json')['cases']:specs[case['spec']['seed']]=copy.deepcopy(case['spec'])
    pairs={f'{r}_parent':completed_pair(parent,r) for r in 'AB'}
    pairs.update({f'{r}_new':completed_pair(current,r) for r in 'AB'})
    root=ROOT/('final_test' if heldout else f'round{round_index:02d}/evaluation');root.mkdir()
    seeds=c['final_test_seeds'] if heldout else [x for pair in zip(c['training_seeds'],c['development_seeds']) for x in pair]+list(range(28838000+500*(round_index-1),28838008+500*(round_index-1)))
    waves=[]
    for index,seed in enumerate(seeds):
        wave=root/f'wave{index:02d}';wave.mkdir();p=copy.deepcopy(old)
        if heldout or seed not in specs:
            spec=dict(identity=f'adaptive_final_{seed}',family=f'adaptive_final_{seed}',split='test' if heldout else 'development',seed=seed,event_seed=seed+100000,
                      cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',narrow_reset=reset_configuration(s['domain'],seed,.1),layout=s['layout'])
        else:spec=specs[seed]
        specs[seed]=copy.deepcopy(spec)
        names=list(pairs);names=names[index%4:]+names[:index%4]
        p.update(pairs=pairs,cases=[dict(name=name+'_'+spec['identity'],role=name,spec=copy.deepcopy(spec),video=seed in (28817004,28826000,28839000)) for name in names],
                 source_sha256=sources(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 resource_root=str(ROOT),test_used=heldout)
        p['batch'].update(lanes=4,vision_microbatch=2)
        p['evaluation_scope']=dict(protocol='Within-wave complete parent/new A/B; native success unchanged',test_used=heldout,
                                  hold_metrics='30/60/120 are diagnostic only',hidden_state_replay_claim=False)
        if heldout:p['holdout_contract']='layout_final_test_v1'
        write(wave/'session.json',s);write(wave/'plan.json',p);waves.append(wave)
    if not heldout:
        historical=CANONICAL/'outputs/visuotactile/run/tight_long_20261117_01/round05/learning'
        pp={f'{r}_72':completed_pair(historical,r) for r in 'AB'}
        for index in range(0,len(seeds),2):
            wave=root/f'history{index//2:02d}';wave.mkdir();p=copy.deepcopy(old)
            p.update(pairs=pp,cases=[],source_sha256=sources(),resource_root=str(ROOT),test_used=False)
            for seed in seeds[index:index+2]:
                spec=specs[seed]
                for name in pp:p['cases'].append(dict(name=name+'_'+spec['identity'],role=name,spec=copy.deepcopy(spec),video=False))
            p['batch'].update(lanes=4,vision_microbatch=2)
            write(wave/'session.json',s);write(wave/'plan.json',p);waves.append(wave)
    write(root/'registration.json',dict(parent=str(parent),current=str(current),pairs=pairs,seeds=seeds,test_used=heldout,waves=[str(p) for p in waves],utc=time.time()))
    return root,waves


def evaluate(parent,current,round_index, *, heldout=False):
    assert read(current/'exposure_acceptance.json')['passed']
    root,waves=evaluation_plans(parent,current,round_index,heldout=heldout)
    stages=[dict(entry='adaptive_heldout' if heldout else 'evaluate',args=['--output',str(w)]) for w in waves]
    execute('evaluate','final_test' if heldout else f'r{round_index}_evaluate',stages,simulation=True)
    return evaluation_report(root,round_index,heldout=heldout)


def evaluation_report(root,round_index, *, heldout=False):
    import h5py
    import numpy as np
    from scripts.tight_layout_20261116.phase_report import summarize
    from visuotactile.grasp_curriculum import holding_metrics
    c=read(CONFIG/'experiment.json');public=PUBLIC/('final_test' if heldout else f'round{round_index:02d}/evaluation');public.mkdir(parents=True,exist_ok=True)
    rows=[]
    for folder in read(root/'registration.json')['waves']:
        wave=Path(folder);out=public/wave.name;out.mkdir(exist_ok=True)
        summarize(wave,out/'summary.json')
        check=read(wave/'paired_initial_check.json');assert check['passed']
        for row in read(out/'summary.json')['rows']:
            assert row['legal'] and row['software_complete']
            row['cohort']='test' if heldout else 'training' if row['seed'] in c['training_seeds'] else 'development' if row['seed'] in c['development_seeds'] else 'new_development'
            with h5py.File(wave/row['case']/'numeric.h5') as f:
                t=f['episode/transition'];row['hold30']=holding_metrics(t['diagnostics_after/object_pose_w'][:],t['object_contact_force_w'][:],window=30)
                command=t['target_after_rad'][:];position=t['diagnostics_after/joint_position_rad'][:]
                assert np.isfinite(command).all() and np.isfinite(position).all()
                tracking=command-position;delta=np.diff(command,axis=0)
                row['execution_diagnostics']=dict(clipped_ticks=int(t['position_clip_mask'][:].reshape(len(command),-1).any(-1).sum()),
                    tracking_arm_rmse_rad=float(np.sqrt(np.mean(tracking[:,:7]**2))),tracking_hand_rmse_rad=float(np.sqrt(np.mean(tracking[:,7:]**2))),
                    command_delta_max_rad=float(abs(delta).max()) if len(delta) else 0.)
            row['failure_stage']='success' if row['success'] else 'after_subgoal' if row['native_subgoals'] else 'after_lift' if row['native_lift'] else 'before_lift'
            rows.append(row)
        for name in ('plan.json','result.json','paired_initial_check.json','videos.json'):shutil.copyfile(wave/name,out/name)
    counts=[];changes=[]
    for cohort in sorted({r['cohort'] for r in rows}):
        for model in sorted({r['role'] for r in rows}):
            rr=[r for r in rows if r['cohort']==cohort and r['role']==model]
            if not rr:continue
            counts.append(dict(cohort=cohort,model=model,n=len(rr),success=sum(r['success'] for r in rr),
                first_subgoal=sum(r['native_subgoals']>0 for r in rr),later_subgoal=sum(r['native_subgoals']>=2 for r in rr),
                native_lift=sum(bool(r['native_lift']) for r in rr),**{f'hold{w}':sum(r[f'hold{w}']['sustained'] for r in rr) for w in (30,60,120)},
                failure_stages=dict(collections.Counter(r['failure_stage'] for r in rr))))
    for role in 'AB':
        for seed in read(root/'registration.json')['seeds']:
            old=next(r for r in rows if r['role']==role+'_parent' and r['seed']==seed)
            new=next(r for r in rows if r['role']==role+'_new' and r['seed']==seed)
            changes.append(dict(role=role,seed=seed,cohort=new['cohort'],old_success=old['success'],new_success=new['success'],
                                retained=bool(old['success'] and new['success']),lost=bool(old['success'] and not new['success']),gained=bool(not old['success'] and new['success'])))
    write(public/'per_case.json',rows);write(public/'counts.json',counts);write(public/'paired_changes.json',changes)
    for name,data in [('per_case',rows),('counts',counts),('paired_changes',changes)]:
        with (public/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    write(root/'analysis_complete.json',dict(cases=len(rows),counts=counts,utc=time.time(),test_used=heldout))
    print(json.dumps(counts),flush=True)
    return dict(rows=rows,counts=counts,changes=changes)


def heldout_worker(output):
    freeze=read(ROOT/'final_freeze.json');plan=read(output/'plan.json')
    assert plan['test_used'] is True and plan['holdout_contract']=='layout_final_test_v1'
    assert {c['spec']['seed'] for c in plan['cases']}<=set(freeze['seeds'])
    assert plan['pairs']==freeze['pairs']
    from visuotactile.paired_execution import install
    from tools.tight_evaluate import run
    install(output);run(output,allow_heldout=True)


def schedule(round_index,previous=None):
    # Separate real training families per opportunity; never development/test resets.
    seeds=[28817004]+[28817000+i for i in range(30) if i!=4]
    stages=['contact','transport','alignment','instability']
    if round_index==2:
        summary=read(PUBLIC/'round01/collection_summary.json')['table']
        totals=collections.Counter();accepted=collections.Counter()
        for r in summary:totals.update(r['stage_attempts']);accepted.update(r['stage_accepted'])
        useful=[s for s in stages if accepted[s]>0]
        stages=useful if useful else ['contact','transport']
        # Keep post-subgoal evidence if round1 native progress actually reaches it.
        if previous and any(c['model'].endswith('_new') and c['first_subgoal']>=2 for c in previous['counts']):stages=stages+['post_subgoal']
    result=[]
    for offset in range(0,28,4):
        families=seeds[offset:offset+4];result.append((families,[stages[(offset+j)%len(stages)] for j in range(len(families))]))
    return result


def run_cycle(*,recover_prephysics=False,recover_resource=False):
    attempt='cycle'
    if (ROOT/'cycle_started.json').exists():
        if not (recover_prephysics or recover_resource):raise FileExistsError('Preserve attempted cycle; explicit recovery required')
        assert (ROOT/'cycle_failure.json').exists()
        if not recover_resource:assert not list((ROOT/'round01').rglob('numeric.h5'))
        else:
            assert (ROOT/'cycle_recovery01_failure.json').exists()
            assert not list((ROOT/'round01').glob('collection*/*/result.json'))
        assert read(ROOT/'progress.json')==read(ROOT/'progress.json') and read(ROOT/'progress.json')['stage']=='collect'
        attempt='cycle_recovery02' if recover_resource else 'cycle_recovery01'
        assert not (ROOT/(attempt+'_started.json')).exists()
    write(ROOT/(attempt+'_started.json'),dict(pid=os.getpid(),utc=time.time(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source_sha256=sources()))
    source=BASE;previous=None
    try:
        for round_index in (1,2):
            write(ROOT/'progress.json',dict(round=round_index,stage='collect',utc=time.time()))
            combined=collect(source,round_index,schedule(round_index,previous))
            write(ROOT/'progress.json',dict(round=round_index,stage='train',utc=time.time()))
            target=prepare_learning(source,combined,round_index);train(target,round_index)
            write(ROOT/'progress.json',dict(round=round_index,stage='evaluate',utc=time.time()))
            previous=evaluate(source,target,round_index)
            write(ROOT/f'round{round_index:02d}'/'complete.json',dict(collection=str(combined),learning=str(target),evaluation=True,utc=time.time()))
            source=target
        pairs={f'{r}_parent':completed_pair(BASE,r) for r in 'AB'}
        pairs.update({f'{r}_new':completed_pair(source,r) for r in 'AB'})
        write(ROOT/'final_freeze.json',dict(pairs=pairs,seeds=read(CONFIG/'experiment.json')['final_test_seeds'],utc=time.time(),selection='Own round2 validation-eligible pairs; freeze before final test'))
        evaluate(BASE,source,2,heldout=True)
        write(ROOT/'cycle_complete.json',dict(rounds=2,final_test_complete=True,utc=time.time(),automatic_promotion=False))
        close_report()
    except BaseException as error:
        write(ROOT/(attempt+'_failure.json'),dict(error=repr(error),utc=time.time()));raise


def close_report():
    protected=read(PUBLIC/'protected.json');assert all(sha256(p)==v for p,v in protected.items())
    write(PUBLIC/'closure.json',dict(protected_sha_passed=True,files=len(protected),complete=read(ROOT/'cycle_complete.json'),
                                   free_disk_gib=shutil.disk_usage(ROOT).free/2**30,source_sha256=sources(),utc=time.time()))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['initialize','diagnose','run','recover-prephysics','recover-resource','test-worker','audit'])
    p.add_argument('--output',type=Path);p.add_argument('--round',type=int);a=p.parse_args()
    if a.mode=='initialize':initialize()
    elif a.mode=='diagnose':diagnose()
    elif a.mode=='run':run_cycle()
    elif a.mode=='recover-prephysics':run_cycle(recover_prephysics=True)
    elif a.mode=='recover-resource':run_cycle(recover_resource=True)
    elif a.mode=='audit':audit_training(ROOT/f'round{a.round:02d}'/'learning',a.round)
    else:heldout_worker(a.output)

if __name__=='__main__':main()
