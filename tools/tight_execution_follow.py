"""Bounded full-policy follow-up; learning, physical checks and publishing are separate."""
import argparse,copy,json,os,shutil,subprocess,time,traceback
from pathlib import Path
import torch
from tools.runtime import RUN,read,write,sha,kill_owned,supervised_owner,proc
from tools.tight_execution_learning import GROUPS,PARENT,HISTORY,run_child,publish_after_parent


def state(root,stage,**kw):
    value=dict(state=kw.pop('state','RUNNING'),stage=stage,utc=time.time(),pid=os.getpid(),**kw)
    write(root/'follow_status.json',value);write(root.parent/'status.json',value)


def prepare(root):
    from tools.tight_finish import residual_prepare
    from tools.tight_dagger import share_monitor,reuse_nominal_cache
    selection=read(root/'adaptive_selection.json');spec=selection['chosen'];reg=read(root/'registration.json')
    candidates=[p for p in sorted(PARENT.glob('round_[0-9][0-9]')) if (p/'round_finished.json').exists() and all(
        (p/g/'backend/evaluation_complete.json').exists() and read(p/g/'backend/evaluation_complete.json').get('complete') and
        (p/g/'backend'/g/'training/exposure_endpoint.json').exists() for g in GROUPS)]
    if not candidates:raise ValueError('No complete original pair/cache snapshot')
    source=candidates[-1];models={};populations={};shared=None
    for group in GROUPS:
        old=source/group/'backend';checkpoint=old/group/'training/best.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        cfg=saved['configuration'];data=read(old/'state_data/data.json');index=read(old/group/'nominal/index.json')
        if set(index['episodes'])&set(data['splits']['test']):raise ValueError('Test cache forbidden')
        identity=(data['snapshot_id'],data['splits'])
        if shared is None:shared=identity
        elif identity!=shared:raise ValueError('A/B source pool mismatch')
        if sha(cfg['base_checkpoint'])!=cfg['base_sha256']:raise ValueError('Formal frontend SHA drift')
        models[group]=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),dp_checkpoint=cfg['base_checkpoint'],
            dp_sha256=cfg['base_sha256'],command_statistics_sha256=cfg['residual_statistics_sha256'],source_backend=str(old))
        populations[group]=read(old/group/'nominal/scales.json')['control_tick_population']
    rates=[v['optimization_wall_s']/v['supervised_ticks'] for rows in selection['results'].values() for v in rows.values()]
    available=reg['compute_deadline']-time.time()-21600
    choices=[e for e in [6,3,1] if 2*max(rates)*sum(populations.values())*e<available]
    if not choices:raise TimeoutError('Insufficient measured budget for paired learning plus physical follow-up')
    epochs=choices[0]
    write(root/'formal_budget.json',dict(epochs_each=epochs,populations=populations,seconds_per_tick=rates,
        conservative_estimate_s=2*max(rates)*sum(populations.values())*epochs,
        physical_and_recovery_reserve_s=21600,registered=time.time(),absolute_deadline=reg['absolute_deadline']))
    write(root/'formal_sources.json',dict(parent_round=str(source),models=models,snapshot_id=shared[0],
        short_probe_round='round_01',same_probe_pair=source.name=='round_01',test22_reads=0))
    for group in GROUPS:
        item=models[group];old=Path(item['source_backend']);configuration=copy.deepcopy(read(old/'configuration.json'));v=configuration['residual_pair']
        v.update(loss_contract=spec['loss_contract'],history_contract=HISTORY,output_contract=spec['output_contract'],
            learning_rate=spec['lr'],maximum_control_tick_epochs=epochs,minimum_control_tick_epochs=epochs,
            comparison_control_tick_epochs=[epochs],validation_epoch_interval=1,initialization_seed=2026111301,videos_per_group=1,
            continuation_from=dict(v['continuation_from'],path=item['checkpoint'],sha256=item['checkpoint_sha256']),
            dagger_evaluation=dict(initial_trials=10,max_trials=10,expand_if_success=1,expand_if_subgoal=1,expand_if_native_lift=2))
        v.pop('residual_scale_adaptation',None)
        dest=root/'formal'/group/'backend';residual_prepare(dest,configuration=configuration);share_monitor(root,dest)
        reuse_nominal_cache(old/group/'nominal',dest/group/'nominal',v['bases'][group],dest/'state_data',item['command_statistics_sha256'])
        shutil.copy2(old/group/'nominal/scales.json',dest/group/'nominal/scales.json');shutil.copy2(PARENT/'evaluation_plan.json',dest/'evaluation_plan.json')
        for p in [dest/'session.json',dest/group/'session.json']:
            session=read(p);session.update(resource_root=str(root),matrix_owner=str(root),matrix_deadline=reg['compute_deadline'],
                started=reg['started_utc'],deadline=reg['absolute_deadline'],training_deadline=reg['compute_deadline']-21600,
                offline_deadline=reg['compute_deadline']-18000,evaluation_deadline=reg['compute_deadline']-14400,
                source_head=read(root/'follow_launch.json')['source_head'],cwd=str(Path.cwd()))
            write(p,session)
        write(dest/'cache_complete.json',dict(complete=True,exact_reuse=True,source=str(old/group/'nominal'),new_frontend_updates=0))
    return models


def select_safe_candidate(directory):
    """Keep old weights immutable; deploy validation-selected adapted candidate.

    Endpoint and initialization comparisons remain visible. Update-zero is not
    a trained candidate when output semantics changed.
    """
    saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    if saved['updates']>0:return dict(selection='validation-selected actual executed error',checkpoint=sha(directory/'best.pt'),updates=saved['updates'])
    last=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    if last['updates']<1:raise ValueError('No actual full-mixture updates')
    shutil.copy2(directory/'best.pt',directory/'update0_reference.pt')
    shutil.copy2(directory/'last.pt',directory/'best.pt')
    value=dict(selection='trained exposure endpoint for diagnostic evaluation; update0 validated better, candidate not adopted',
        updates=last['updates'],checkpoint=sha(directory/'best.pt'),regression=True)
    write(directory/'diagnostic_selection.json',value)
    if (directory/'selected.json').exists():shutil.copy2(directory/'selected.json',directory/'update0_selected.json')
    write(directory/'selected.json',dict(value,update=last['updates'],initialization_candidate=False,checkpoint_sha256=value['checkpoint']))
    return value


def physical_diagnosis(root,base_root=None,report_name="physical_diagnosis.json"):
    import numpy as np
    rows=[]
    for group in GROUPS:
        base=(base_root or root/'formal')/group/'backend'/group
        for p in sorted((base/'evaluation').glob('*/trace.npz')):
            z=np.load(p);n=len(z['command']);force=np.linalg.norm(z['object_contact'],axis=-1).reshape(n,-1).max(-1)
            contact=force>.1;lift=z['lifted'].reshape(n,-1).any(-1)
            first=lambda v:int(np.flatnonzero(v)[0]) if np.any(v) else None
            lost=contact[:-1]&~contact[1:];drop=lift[:-1]&~lift[1:]
            rows.append(dict(group=group,identity=p.parent.name,ticks=n,first_contact=first(contact),first_lift=first(lift),
                first_contact_loss=first(lost),first_lift_loss=first(drop),source=str(p),sha256=sha(p),
                semantics='observable native events, not hidden-state paired counterfactual'))
    write(root/report_name,dict(rows=rows,test22=False))
    # Focus takeover just before the observed first loss/lift instability, or
    # precontact approach if no contact/lift. Keep native delay-clean alignment.
    events=[next((r[k] for k in ('first_lift_loss','first_contact_loss','first_contact') if r[k] is not None),94) for r in rows]
    center=int(np.median(events)) if events else 94
    ticks=sorted({max(22,min(298,((center+offset)//12)*12-2)) for offset in [-24,0,24]})
    return ticks


def checkpoint_pair(root,group):
    source=Path(read(root/'formal_sources.json')['parent_round']);pair=read(source/'round_finished.json')['groups'][group]['selection']['selected_pair']
    pair=dict(pair);p=root/'formal'/group/'backend'/group/'training/best.pt';saved=torch.load(p,map_location='cpu',weights_only=False)
    pair.update(backend=str(p),backend_sha256=sha(p),backend_updates=saved['updates'],evaluation_root=str(p.parents[1]/'evaluation'),
        output_contract=saved['configuration'].get('output_contract'),loss_contract=saved['configuration'].get('loss_contract'),history_contract=HISTORY)
    return pair


def targeted_recovery(root,ticks,attempt=1):
    """One bounded targeted full-student DAgger round, both modules updated."""
    from tools.tight_dagger import (takeover_plan,assert_paired_plans,share_monitor,Stages,
        prepare_frontend_stages,train_frontend_stage,prepare_backend,select_exposure_endpoint,phase_deadlines)
    from tools.tight_finish import residual_prepare
    from visuotactile.narrow_data import ingest_dagger_pool,freeze_dagger_snapshot
    import fcntl
    reg=read(root/'registration.json');remaining=reg['compute_deadline']-time.time()
    if remaining<5*3600:
        write(root/f'recovery_{attempt}_not_admitted.json',dict(reason='Remaining absolute budget below measured two-stage recovery reserve',remaining_s=remaining,required_s=18000));return
    out=root/f'targeted_{attempt:02d}';out.mkdir();run_id=root.parent.name+f'_targeted_{attempt:02d}';cfg=copy.deepcopy(read(PARENT/'configuration.json'))
    if attempt==1:pairs={g:checkpoint_pair(root,g) for g in GROUPS}
    else:
        from tools.tight_dagger import candidate_pair
        pairs={g:candidate_pair(root/f'targeted_{attempt-1:02d}/round_01'/g) for g in GROUPS}
        cfg['pool_root']=str(root/f'targeted_{attempt-1:02d}/pool')
    source_pool=Path(cfg['pool_root']).resolve();pool=out/'pool';pool.mkdir()
    # Copy one locked metadata snapshot; raw records remain hash-bound originals.
    with (source_pool/'pool.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_SH);shutil.copy2(source_pool/'manifest.json',pool/'manifest.json')
    cfg.update(pool_root=str(pool),rounds=1,first_round=1,maximum_hours=remaining/3600)
    cfg['frontend'].update(maximum_anchor_epochs=2,minimum_anchor_epochs=2,validation_anchor_epochs=1,maximum_hours=1.5)
    cfg['backend'].update(maximum_control_tick_epochs=2,minimum_control_tick_epochs=2,validation_epoch_interval=1,maximum_hours=2)
    cfg['collection'].update(attempts_per_group=12,takeover_ticks=ticks,plan_seed=2026111307+attempt,seed_start=34113000+attempt*1000,
        event_seed_start=35113000+attempt*1000,development_indices=[1,7],maximum_hours=.75)
    cfg['evaluation'].update(attempts_per_group=10,maximum_attempts_per_group=10,
        rollback='Independent trained candidates; validation-selected best, trained endpoint diagnostic if update0 wins; originals protected')
    cfg['matched_protocol'].update(version='execution_repair_targeted_v1',rounds=1,attempts_per_group=12,
        frontend_anchor_epochs=2,backend_tick_epochs=2,maximum_hours=remaining/3600,
        endpoint_selection='Frontend complete common exposure; rear validation best with trained endpoint diagnostic fallback')
    cfg.pop('round_schedule',None)
    cfg['stopping']='One bounded targeted stage per identity; at most two repair stages, original repair deadline unchanged'
    if cfg.get('performance'):cfg['performance']['first_frontend_round']=1
    write(out/'configuration.json',cfg);shutil.copy2(PARENT/'evaluation_plan.json',out/'evaluation_plan.json')
    session=copy.deepcopy(read(root/'formal/QG01/backend/session.json'))
    session.update(source_head=read(root/'follow_launch.json')['source_head'],compute_deadline=reg['compute_deadline'],
        deadline=reg['absolute_deadline'],resource_root=str(root),owned_worker_roots=[str(out)],dagger=cfg)
    session.pop('matrix_owner',None);write(out/'session.json',session);share_monitor(root,out)
    write(out/'registration.json',dict(parent_run=str(root.parent),reason='Complete student failed; target observed contact/lift transitions',
        takeover_ticks=ticks,attempts_each=12,rounds=1,logical_repair_round=attempt,frontend_anchor_epochs=2,backend_tick_epochs=2,
        started=time.time(),compute_deadline=reg['compute_deadline'],absolute_deadline=reg['absolute_deadline'],test22=False))
    files=subprocess.check_output(['git','ls-files','*.py','visuotactile/run_cpu.sh','configs/tight_state.json'],text=True).splitlines()
    write(out/'source_freeze.json',{p:sha(p) for p in files})
    families=read(Path(read(Path(cfg['family_source']))['parent'])/'frozen_families.json');plans={}
    for group in GROUPS:
        policy=dict(cfg['collection'],round_index=1,source_group=group)
        plans[group]=takeover_plan(read(Path(pairs[group]['data_root'])/'data.json'),families,policy)
    write(out/'paired_opportunities.json',assert_paired_plans(plans));directory=out/'round_01';directory.mkdir()
    for group in GROUPS:
        job=directory/group;job.mkdir();local=copy.deepcopy(cfg);local['collection'].update(round_index=1,source_group=group)
        write(job/'configuration.json',local);write(job/'initial_pair.json',pairs[group]);write(job/'collection_plan.json',plans[group])
        write(job/'session.json',dict(session,source_root=pairs[group]['data_root'],collection_deadline=min(reg['compute_deadline']-14400,time.time()+2700)))
        write(job/'collector_registration.json',{n:sha(job/n) for n in ['configuration.json','initial_pair.json','collection_plan.json']});share_monitor(root,job)
        state(root,'targeted_full_student_collection',group=group)
        run_child(root,f'targeted{attempt}_'+group+'_collect',['-m','tools.tight_finish','dagger_collect','--root',str(job)],min(reg['compute_deadline']-14400,time.time()+2800))
        receipt=ingest_dagger_pool(pool,job,run_id=run_id,source_group=group,round_index=1,core_data_root=pairs[group]['data_root'])
        write(job/'ingestion.json',receipt)
        if not receipt['integrity_passed']:raise ValueError('Targeted legal correction audit failed')
    frozen=freeze_dagger_snapshot(pool,directory/'snapshot.json',run_id=run_id,round_index=1)
    current=[e for e in frozen['entries'].values() if e['cohort']=='train' and e['pool_source']['run_id']==run_id]
    if sum(e['exposure']['supervised_anchors'] for e in current)<4:
        write(out/'insufficient_legal_supervision.json',dict(reason='Targeted opportunities yielded fewer than four legal DP anchors; no counterfactual labels or automatic replacements'));return
    stages=Stages(out);selection=read(root/'adaptive_selection.json')['chosen'];reuse=None
    try:
        for group in GROUPS:
            state(root,'targeted_frontend_and_cache',group=group)
            front=prepare_frontend_stages(out,group,1,stages,directory/'snapshot.json',reuse)
            train_frontend_stage(out,group,1,stages,front);reuse=front/'cache_index.json'
            backend=prepare_backend(out,group,1)
            # Preparation has no parameter updates. Bind new output/loss/history
            # before cache/trainer launch; no hot changes to a running child.
            for p in [backend/'configuration.json',backend/group/'configuration.json',backend/'session.json',backend/group/'session.json']:
                v=read(p);v['residual_pair'].update(loss_contract=selection['loss_contract'],history_contract=HISTORY,
                    output_contract=selection['output_contract'],learning_rate=selection['lr'])
                write(p,v)
            phase_deadlines(out,backend,cache_deadline=3600)
            run_child(root,f'targeted{attempt}_'+group+'_cache',['-m','tools.tight_finish','residual_cache','--root',str(backend)],min(reg['compute_deadline']-7200,time.time()+3600))
            phase_deadlines(out,backend,training_deadline=min(7200,reg['compute_deadline']-time.time()-3600))
            run_child(root,f'targeted{attempt}_'+group+'_train',['-m','visuotactile.anchor_continue','residual-train','--root',str(backend/group)],reg['compute_deadline']-3600)
            select_safe_candidate(backend/group/'training')
            phase_deadlines(out,backend,offline_deadline=min(1800,reg['compute_deadline']-time.time()-1800))
            run_child(root,f'targeted{attempt}_'+group+'_offline',['-m','visuotactile.anchor_continue','residual-offline','--root',str(backend/group)],reg['compute_deadline']-1800)
            phase_deadlines(out,backend,evaluation_deadline=min(3600,reg['compute_deadline']-time.time()))
            run_child(root,f'targeted{attempt}_'+group+'_eval',['-m','tools.tight_finish','residual_evaluate','--root',str(backend)],reg['compute_deadline'])
        exposure={g:dict(frontend=read(directory/g/'dp/training/finished.json'),
            backend=read(directory/g/'backend'/g/'training/finished.json')) for g in GROUPS}
        differences={kind:{k:[exposure[g][kind].get(k) for g in GROUPS] for k in keys
            if exposure['QG01'][kind].get(k)!=exposure['G01'][kind].get(k)}
            for kind,keys in [('frontend',['updates','supervised_target_ticks','target_reached']),('backend',['updates','supervised_ticks','target_reached'])]}
        write(out/'results.json',dict(groups={g:read(directory/g/'backend'/g/'evaluation_summary.json') for g in GROUPS},
            evaluation_receipts={g:read(directory/g/'backend/evaluation_complete.json') for g in GROUPS},
            exposure=exposure,exposure_differences=differences,strict_controlled_comparison=not any(differences.values()),test22=False))
    finally:stages.close()


def export(root):
    """Small evidence and existing documentation; no raw trajectories or models."""
    target=Path('research_docs/evidence/tight_execution_learning_20261113/adaptive');target.mkdir(parents=True,exist_ok=True)
    names=['registration.json','capacity.json','probe_gates.json','adaptive_probe_plan.json','adaptive_probe_budget.json',
        'tail800_comparison.json','next_hypothesis.json','adaptive_selection.json','adaptive_probe_complete.json',
        'autonomy_authorization.json','follow_launch.json','follow_status.json','formal_sources.json','formal_budget.json','formal_exposure_comparison.json','formal_results.json',
        'physical_diagnosis.json','recovery_1_not_admitted.json','recovery_2_not_admitted.json','follow_exit.json','final_results.json']
    for name in names:
        p=root/name
        if p.exists():
            if p.is_symlink() or p.stat().st_size>2*1024**2:raise ValueError('Oversized public evidence')
            shutil.copy2(p,target/name)
    initial=read(root/'probe_results.json')
    write(target/'initial_probe_exposure.json',{g:{c:{k:v for k,v in row.items() if k!='metrics'} for c,row in candidates.items()} for g,candidates in initial['groups'].items()})
    index=[]
    for stage,base in [('formal',root/'formal'),('targeted_01',root/'targeted_01/round_01'),('targeted_02',root/'targeted_02/round_01')]:
        for group in GROUPS:
            model=base/group/'backend'/group
            for relative in ['training/finished.json','training/validation.json','training/curves.png','training/diagnostic_selection.json',
                'offline/stage_errors.csv','offline/stage_errors.png','evaluation_summary.json']:
                p=model/relative
                if p.is_file() and p.stat().st_size<2*1024**2:
                    name=stage+'_'+group+'_'+relative.replace('/','_')
                    if p.suffix=='.csv':(target/name).write_text(p.read_text())
                    else:shutil.copy2(p,target/name)
            for p in (model/'evaluation').glob('**/*'):
                if p.is_file() and p.suffix in {'.npz','.mp4'}:index.append(dict(stage=stage,group=group,path=str(p),sha256=sha(p),bytes=p.stat().st_size))
    write(target/'large_artifact_index.json',dict(artifacts=index,raw_files_remain_local=True))
    status=read(root/'follow_status.json') if (root/'follow_status.json').exists() else read(root/'status.json')
    body=('<!-- EXECUTION_LEARNING_STATUS_START -->\n## 2026-11-13 自主执行学习修复：实际状态\n\n'
        f"状态 **{status['state']} / {status['stage']}**；原绝对截止 **2026-11-14 00:05:15 UTC+8** 不变。\n"
        '首个旧/FY各200更新已完成，FY的A/B dev执行误差比1.139/1.181，未通过原门槛。固定DP输出域下界占A归一化错误33.46%，B为0；据此比较保持共同执行器的线性尾部提案。\n'
        '配对延长、必要优化调整、完整混合数据后端训练、完整DP+Transformer开发评估分层记录；若闭环失败且剩余预算允许，最多两轮、每轮12机会/组的定向DAgger同时更新DP、缓存和后端。初始门槛失败不结束整个修复，也不改写旧结果。\n'
        'GPU工作与Git发布分别调度；未完成阶段不能称为成功。test22隔离、源数据和旧checkpoint保留。\n'
        '[方法与限制](DAGGER_METHOD.md#execution-learning-repair) · [阶段状态](evidence/tight_execution_learning_20261113/adaptive/follow_status.json) · [配对计划](evidence/tight_execution_learning_20261113/adaptive/adaptive_probe_plan.json) · [表达下界](evidence/tight_execution_learning_20261113/adaptive/capacity.json)。\n'
        '<!-- EXECUTION_LEARNING_STATUS_END -->\n\n')
    for name in ['STATE_DP_RESULTS.md','SESSION_HANDOFF.md','README.md']:
        p=Path('research_docs')/name;old=p.read_text();start=old.find('<!-- EXECUTION_LEARNING_STATUS_START -->');end=old.find('<!-- EXECUTION_LEARNING_STATUS_END -->')
        if start>=0 and end>=start:old=old[:start]+old[end+len('<!-- EXECUTION_LEARNING_STATUS_END -->'):].lstrip('\n')
        p.write_text(body+old)


def merge_parent_status(base,ours,theirs):
    """Resolve only our managed status block; preserve the parent's full report."""
    start='<!-- EXECUTION_LEARNING_STATUS_START -->';end='<!-- EXECUTION_LEARNING_STATUS_END -->'
    def split(value):
        a=value.find(start);b=value.find(end)
        if a<0 and b<0:return '',value
        if a<0 or b<a:raise ValueError('Malformed managed publication block')
        return value[a:b+len(end)],(value[:a]+value[b+len(end):]).strip()
    block,remaining=split(ours);_,original=split(base);_,incoming=split(theirs)
    if not block or remaining.strip()!=original.strip():
        raise ValueError('Concurrent documentation change outside managed status; preserve merge evidence')
    return block+'\n\n'+incoming+'\n'


def publish_sources(root):
    try:publish_after_parent(root)
    except subprocess.CalledProcessError:
        def git(*args):return subprocess.check_output(['git',*args],text=True,stderr=subprocess.STDOUT,timeout=90)
        conflicts=git('diff','--name-only','--diff-filter=U').splitlines()
        if conflicts!=['research_docs/STATE_DP_RESULTS.md']:raise
        path=conflicts[0]
        merged=merge_parent_status(git('show',':1:'+path),git('show',':2:'+path),git('show',':3:'+path))
        Path(path).write_text(merged);git('add','--',path);git('commit','--no-edit')
        git('push','origin','HEAD:feature/tight-state-foundation')
        write(root/'source_publication.json',dict(head=git('rev-parse','HEAD').strip(),utc=time.time(),
            resolution='Preserved incoming parent report and our independent managed status; no code conflict'))


def publish(root):
    reg=read(root/'registration.json');published=False
    while time.time()<reg['absolute_deadline']-90:
        old=read(PARENT/'terminal_publication_status.json')
        if old['state']=='COMPLETE':
            publish_sources(root);published=True;break
        if old['state']=='FAILED':
            write(root/'publication_blocked.json',dict(reason='Old terminal publisher failed; preserve Git guard',detail=old));return
        time.sleep(10)
    if not published:return
    write(root/'publisher_source_complete.json',dict(utc=time.time()))
    # Report at completed experimental boundaries, never per-update pushes.
    previous=None
    while time.time()<reg['absolute_deadline']-45:
        status=read(root/'follow_status.json') if (root/'follow_status.json').exists() else {}
        terminal=(root/'follow_exit.json').exists();boundary='final' if terminal else 'formal' if (root/'formal_results.json').exists() else 'probe' if (root/'adaptive_selection.json').exists() else None
        if boundary and boundary!=previous:
            export(root)
            paths=['research_docs/evidence/tight_execution_learning_20261113/adaptive','research_docs/STATE_DP_RESULTS.md','research_docs/SESSION_HANDOFF.md','research_docs/README.md']
            subprocess.run(['git','add','--',*paths],check=True)
            check=subprocess.run(['git','diff','--cached','--quiet'])
            if check.returncode==1:subprocess.run(['git','commit','-m','Record autonomous execution repair '+boundary+' results'],check=True)
            elif check.returncode!=0:raise RuntimeError('Publication index read failed')
            subprocess.run(['git','push','origin','HEAD:feature/tight-state-foundation'],check=True,timeout=90)
            write(root/'publication.json',dict(boundary=boundary,utc=time.time(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()));previous=boundary
        if terminal:return
        time.sleep(15)


@supervised_owner
def run(root):
    reg=read(root/'registration.json');monitor=None;publisher=None
    if (root/'follow_launch.json').exists():raise ValueError('No implicit full-policy follower restart')
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():raise ValueError('Reviewed committed source required')
    write(root/'follow_launch.json',dict(pid=os.getpid(),utc=time.time(),source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),deadline=reg['absolute_deadline']))
    try:
        # Wait only for the local optimization dependency, not Git publication.
        state(root,'awaiting_adaptive_probe_dependency')
        while not (root/'adaptive_probe_exit.json').exists():
            if time.time()>reg['compute_deadline']-21600:raise TimeoutError('Physical/training reserve exhausted')
            time.sleep(5)
        if not (root/'adaptive_probe_complete.json').exists():raise RuntimeError('Adaptive worker failed; retain log for technical recovery')
        shutil.copy2(root/'source_freeze.json',root/'initial_probe_source_freeze.json')
        files=subprocess.check_output(['git','ls-files','*.py','visuotactile/run_cpu.sh','configs/tight_state.json'],text=True).splitlines()
        write(root/'source_freeze.json',{str(Path(p).resolve()):sha(p) for p in files})
        with (root/'publisher.log').open('a') as log:
            publisher=subprocess.Popen(RUN+['-m','tools.tight_execution_follow','publish','--root',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        (root/'STOP_MONITOR').unlink(missing_ok=True)
        with (root/'follow_monitor.log').open('a') as log:
            monitor=subprocess.Popen(RUN+['-m','visuotactile.state_resources','--root',str(root)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        began=time.time()
        while not (root/'shared_resources.json').exists() or time.time()-read(root/'shared_resources.json')['utc_epoch']>3:
            if monitor.poll() is not None or time.time()-began>30:raise RuntimeError('Fresh memory monitor required')
            time.sleep(.2)
        state(root,'formal_data_and_budget_binding');prepare(root)
        for group in GROUPS:
            dest=root/'formal'/group/'backend';state(root,'formal_backend_training',group=group)
            run_child(root,group+'_formal_train',['-m','visuotactile.anchor_continue','residual-train','--root',str(dest/group)],reg['compute_deadline']-21600)
            choice=select_safe_candidate(dest/group/'training');write(dest/'candidate_selection.json',choice)
            run_child(root,group+'_formal_offline',['-m','visuotactile.anchor_continue','residual-offline','--root',str(dest/group)],reg['compute_deadline']-18000)
        exposure={g:read(root/'formal'/g/'backend'/g/'training/finished.json') for g in GROUPS}
        differences={k:[exposure[g].get(k) for g in GROUPS] for k in ['updates','supervised_ticks','target_reached']
            if exposure['QG01'].get(k)!=exposure['G01'].get(k)}
        write(root/'formal_exposure_comparison.json',dict(differences=differences,
            strict_controlled_comparison=not differences,all_targets_reached=all(e.get('target_reached') for e in exposure.values())))
        for group in GROUPS:
            state(root,'complete_student_development',group=group)
            run_child(root,group+'_formal_eval',['-m','tools.tight_finish','residual_evaluate','--root',str(root/'formal'/group/'backend')],reg['compute_deadline']-14400)
        results={g:read(root/'formal'/g/'backend'/g/'evaluation_summary.json') for g in GROUPS}
        write(root/'formal_results.json',dict(groups=results,evaluation_receipts={g:read(root/'formal'/g/'backend/evaluation_complete.json') for g in GROUPS},exposure=exposure,exposure_differences=differences,strict_controlled_comparison=not differences,test22=False,protected_original_models_replaced=False))
        ticks=physical_diagnosis(root)
        recovery={};latest=results
        for attempt in (1,2):
            if all(r['successes']>0 for r in latest.values()):break
            state(root,'targeted_recovery_admission',logical_round=attempt);targeted_recovery(root,ticks,attempt)
            result=root/f'targeted_{attempt:02d}/results.json'
            if not result.exists():
                evidence=[root/f'recovery_{attempt}_not_admitted.json',root/f'targeted_{attempt:02d}/insufficient_legal_supervision.json']
                recovery[str(attempt)]=dict(completed=False,reasons=[read(p) for p in evidence if p.exists()]);break
            recovery[str(attempt)]=read(result);latest=recovery[str(attempt)]['groups']
            ticks=physical_diagnosis(root,root/f'targeted_{attempt:02d}/round_01',f'physical_diagnosis_recovery_{attempt}.json')
        write(root/'final_results.json',dict(formal=results,targeted=recovery,
            test22=False,original_deadline_preserved=True,original_models_preserved=True,
            success_not_guaranteed=True,maximum_registered_repair_rounds=2))
        state(root,'finite_learning_and_physical_stages_resolved',state='COMPLETE')
    except BaseException as error:
        state(root,'technical_or_budget_stop',state='FAILED',error=repr(error),traceback=traceback.format_exc());raise
    finally:
        (root/'STOP_MONITOR').touch()
        if monitor is not None:
            try:monitor.wait(timeout=5)
            except subprocess.TimeoutExpired:kill_owned(monitor)
        active=read(root/'active_stage.json') if (root/'active_stage.json').exists() else {}
        worker=proc(active['pid']) if active else None
        worker_exited=worker is None or worker.get('start_ticks')!=active.get('process',{}).get('start_ticks')
        write(root/'follow_exit.json',dict(utc=time.time(),owned_direct_stage_exited=worker_exited,
            gpu_cleanup_scope='run_child direct worker plus targeted Stages.close; no assertion about unrelated GPU jobs',
            monitor_exited=monitor is None or monitor.poll() is not None,
            publisher_pid=publisher.pid if publisher is not None else None))
        # Publisher has its own remaining bounded wait; it does not occupy GPU.
        if publisher is not None:
            try:publisher.wait(timeout=max(1,reg['absolute_deadline']-time.time()-20))
            except subprocess.TimeoutExpired:kill_owned(publisher)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','publish','export']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(2)
    {'run':run,'publish':publish,'export':export}[a.command](a.root.resolve())
