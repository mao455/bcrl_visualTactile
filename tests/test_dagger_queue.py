import json
from pathlib import Path
import pytest
from tools import tight_dagger as d


def config():
    return json.loads((Path(__file__).parents[1]/'configs/tight_state.json').read_text())['dagger']


def test_round_decisions_are_finite_and_phase_aligned():
    cfg=config()
    cfg.pop('scope_revision',None)  # Retain the original queue's decision contract.
    for previous in ({},{'g':{'collection':{'attempts':32,'accepted':4}}},
        {'g':{'collection':{'attempts':32,'accepted':28},'selection':{'candidate_metrics':{'lifts':2}}}},
        {'g':{'collection':{'attempts':32,'accepted':28}}}):
        out=d.round_decision(cfg,previous,18000)
        assert out['attempts_per_group']==32
        assert all((k+2)%12==0 for k in out['takeover_ticks'])
        assert len(out['clean_ticks'])==8 and not out['zero_student_success_is_stop']
    assert d.round_decision(cfg,{},15000)['attempts_per_group']==24


def test_complete_strategy_schedule_reserves_two_module_updates_and_original_time():
    cfg=config()
    first=d.round_decision(cfg,{},13*3600,2)
    assert first['active_groups']==['QG01','G01']
    assert first['attempts_by_group']=={'QG01':96,'G01':48}
    assert first['reserved_compute_seconds']>=3*cfg['planning']['pair_stage_reserve_seconds']
    later=d.round_decision(cfg,{},4*3600,3)
    assert later['active_groups']==['QG01'] and later['not_scheduled']==['G01']
    assert 16<=later['attempts_by_group']['QG01']<96
    assert later['attempts_by_group']['QG01']*cfg['planning']['seconds_per_attempt']+later['reserved_compute_seconds']<=4*3600
    with pytest.raises(ValueError,match='remaining budget'):
        d.round_decision(cfg,{},2*3600,3)
    assert cfg['groups']['QG01']['supervision']=='q_reached'
    assert cfg['groups']['G01']['supervision']=='u_expert'
    assert {v['modality'] for v in cfg['groups'].values()}=={'G00'}
    assert cfg['frontend']['maximum_updates']==800 and cfg['frontend']['minimum_updates']==400


def test_scope_revision_archives_receipts_and_preserves_deadline(tmp_path,monkeypatch):
    from tools.runtime import write,read,sha
    from tools import tight_finish
    import time
    code=tmp_path/'code';root=tmp_path/'existing';root.mkdir();code.mkdir()
    cfg=config();cfg['predecessor']='existing-predecessor'
    write(code/'configs/tight_state.json',{'dagger':cfg})
    session=dict(source_head='old-source',started=12345.,deadline=time.time()+10000,
        compute_deadline=time.time()+9400,owned_worker_roots=[str(root)])
    write(root/'session.json',session)
    write(root/'launch.json',dict(pid=123,start_ticks=456,source_head='old-source',
        session_sha256=sha(root/'session.json'),started=session['started'],deadline=session['deadline']))
    write(root/'configuration.json',{'predecessor':cfg['predecessor']})
    write(root/'registration.json',{'predecessor':'/completed/old'})
    write(root/'status.json',{'state':'STOPPED','owned_workers_exited':True})
    write(root/'worker_processes.json',[]);write(root/'source_freeze.json',{})
    write(root/'scope_revision_request.json',{'reason':'explicit user scope change'})
    old_digest=sha(root/'session.json')
    monkeypatch.setattr(d,'CODE',code);monkeypatch.setattr(d,'proc',lambda pid:None)
    monkeypatch.setattr(tight_finish,'matrix_worker_absence',lambda rows:(True,[]))
    def git(*args):
        if args[0]=='status':return ''
        if args[0]=='rev-parse':return 'reviewed-new-source'
        if args[0]=='ls-files':return 'configs/tight_state.json'
        raise AssertionError(args)
    monkeypatch.setattr(d,'git',git)
    monkeypatch.setattr(d,'validate_selected_pair',lambda spec,parent,group:dict(strategy_role=spec['strategy_role'],modality=spec['modality']))
    d.revise(root)
    current=read(root/'session.json')
    assert all(current[k]==session[k] for k in ('started','deadline','compute_deadline'))
    assert current['source_head']=='reviewed-new-source'
    assert sha(root/'revisions/complete_strategy_ab_v1/session.json')==old_digest
    assert read(root/'launch.json')['session_sha256']==sha(root/'session.json')
    assert read(root/'registration.json')['configuration_sha256']==sha(root/'configuration.json')
    assert read(root/'scope_revision.json')['deadline_reset'] is False
    assert read(root/'source_freeze.json')=={'configs/tight_state.json':sha(code/'configs/tight_state.json')}
    with pytest.raises(ValueError,match='already recorded'):d.revise(root)


def test_takeover_can_exceed_family_count_without_heldout_or_id_collision():
    data={'splits':{'train':['a','b']},'entries':{'a':{'family':'fa'},'b':{'family':'fb'}}}
    families=[{'family':'fa','split':'train'},{'family':'fb','split':'train'},{'family':'held','split':'test'}]
    policy=dict(config()['collection'],attempts_per_group=32,source_group='G11',round_index=2)
    rows=d.takeover_plan(data,families,policy)
    assert len(rows)==len({r['identity'] for r in rows})==32
    assert {r['family'] for r in rows}=={'fa','fb'}
    assert sum(r['pool_partition']=='correction_dev' for r in rows)==4
    assert all('r02_G11_' in r['identity'] for r in rows)


def test_pair_rollback_never_mixes_front_and_rear():
    old={'frontend':'oldfront','backend':'oldrear'};new={'frontend':'newfront','backend':'newrear'}
    rows={str(i):{'success':False,'native_subgoals':0,'native_lift':False,'reasons':{'timeout':True}} for i in range(4)}
    assert d.paired_policy_selection(old,new,rows,rows,True)['selected_pair']==new
    assert d.paired_policy_selection(old,new,rows,rows,False)['selected_pair']==old
    worse={k:dict(v,reasons={'fall':True}) for k,v in rows.items()}
    assert d.paired_policy_selection(old,new,rows,worse,True)['selected_pair']==old
    assert d.paired_policy_selection(old,new,rows,{'0':rows['0']},True)['selected_pair']==old


def test_completed_stage_reuse_does_not_need_live_process_or_new_deadline(tmp_path,monkeypatch):
    from tools.runtime import write,sha
    write(tmp_path/'configuration.json',{'version':'successful_takeover_pool_v1'})
    write(tmp_path/'session.json',{'source_head':'abc','deadline':0})
    write(tmp_path/'source_freeze.json',{})
    done=tmp_path/'finished.json';write(done,{'complete':True})
    write(tmp_path/'stages'/'s.json',{'source_head':'abc','args':['same'], 'completion':str(done),'completion_sha256':sha(done)})
    result=d.Stages(tmp_path).execute('s',['same'],1,done)
    assert result['source_head']=='abc'
    with pytest.raises(ValueError):d.Stages(tmp_path).execute('s',['changed'],1,done)


def test_partial_collection_receipt_is_not_completion(tmp_path):
    from tools.runtime import write
    write(tmp_path/'configuration.json',{'version':'successful_takeover_pool_v1'})
    write(tmp_path/'session.json',{'source_head':'abc','deadline':0})
    write(tmp_path/'source_freeze.json',{})
    done=tmp_path/'finished.json';write(done,{'complete':False})
    with pytest.raises(TimeoutError):d.Stages(tmp_path).execute('s',['same'],1,done)
    assert not (tmp_path/'stages/s.json').exists()


def test_incomplete_preparation_only_removes_reproducible_metadata(tmp_path):
    folder=tmp_path/'partial';folder.mkdir();(folder/'configuration.json').write_text('{}')
    source=tmp_path/'protected';source.mkdir();(source/'data.h5').write_bytes(b'raw')
    (folder/'state_data').symlink_to(source,target_is_directory=True)
    removed=d.discard_incomplete_metadata(folder,{'configuration.json','state_data'})
    assert len(removed)==2 and (source/'data.h5').read_bytes()==b'raw'
    folder.mkdir();(folder/'best.pt').write_bytes(b'preserve checkpoint')
    with pytest.raises(ValueError):d.discard_incomplete_metadata(folder,{'configuration.json'})
    assert (folder/'best.pt').exists()


def test_pool_resume_never_repeats_started_physical_attempt(tmp_path,monkeypatch):
    from test_dagger_collection import fixture
    from tools.runtime import write,read,sha
    root,calls,loaded=fixture(tmp_path,monkeypatch,['success','success','success'])
    cfg=read(root/'configuration.json');cfg['version']='successful_takeover_pool_v1';write(root/'configuration.json',cfg)
    write(root/'collector_registration.json',{p:sha(root/p) for p in ('initial_pair.json','configuration.json','collection_plan.json')})
    (root/'attempts.jsonl').write_text(json.dumps(dict(identity='attempt_0',index=0,seed=100,takeover_tick=60))+'\n')
    d.collect(root)
    assert calls==['attempt_1','attempt_2']
    rows=read(root/'collection_results.json')
    assert len(rows)==3 and rows[0]['error']=='interrupted_started_attempt'
    assert read(root/'collection_finished.json')['complete']
    d.collect(root)
    assert calls==['attempt_1','attempt_2']


def candidate_fixture(root,group='QG01',*,expanded=False,success_index=None):
    """Small real JSON/hash evidence; neural tensors are tested elsewhere."""
    from tools.runtime import write,sha
    plan=[dict(identity=f'case_{i}',spec=dict(identity=f'case_{i}',seed=100+i,split='new',cohort='fresh')) for i in range(10)]
    write(root/'evaluation_plan.json',plan)
    job=root/'round_02'/group;back=job/'backend';out=back/group
    for directory in (job/'dp/training',out/'training'):
        directory.mkdir(parents=True,exist_ok=True);(directory/'best.pt').write_bytes(str(directory).encode())
        write(directory/'finished.json',dict(new_optimizer_updates=800,disposable=False))
        write(directory/'selected.json',dict(checkpoint_sha256=sha(directory/'best.pt'),selection_score=.9))
    pair=dict(frontend=str(job/'dp/training/best.pt'),frontend_sha256=sha(job/'dp/training/best.pt'),
        backend=str(out/'training/best.pt'),backend_sha256=sha(out/'training/best.pt'),statistics_sha256='stats',
        data_root='/preserved/core',modality='G00',supervision='q_reached' if group=='QG01' else 'u_expert',
        strategy_role='A' if group=='QG01' else 'B',geometry=group=='QG01',evaluation_root=str(out/'evaluation'))
    write(job/'initial_pair.json',dict(pair,frontend='oldfront',frontend_sha256='oldfront',backend='oldrear',
        backend_sha256='oldrear',evaluation_root=str(root/'old_evaluation'/group)))
    write(job/'collection_finished.json',dict(complete=True,attempts=48,accepted_episodes=32))
    write(back/'evaluation_plan.json',plan)
    rules=dict(initial_trials=4,max_trials=10,expand_if_success=1,expand_if_subgoal=1,expand_if_native_lift=2)
    frozen=dict(checkpoints={group:pair['backend_sha256']},plan_sha256=sha(root/'evaluation_plan.json'),
        expert_takeover=False,rules=rules)
    write(back/'dagger_evaluation_freeze.json',frozen)
    n=10 if expanded else 4
    for i,case in enumerate(plan[:n]):
        folder=out/'evaluation'/case['identity'];folder.mkdir(parents=True)
        result=dict(case['spec'],software_complete=True,legal=True,success=i==success_index,
            ever_complete_success=i==success_index,reasons=dict(max_successes=i==success_index),
            residual_version='target_residual_transformer60_v1',residual_ticks=601,
            checkpoint_sha256=pair['backend_sha256'],native_lift=False,native_subgoals=0)
        write(folder/'result.json',result);(folder/'trace.npz').write_bytes(b'durable numeric trace')
        write(folder/'evaluation_receipt.json',dict(binding=dict(identity=case['identity'],spec=case['spec'],
            checkpoint_sha256=pair['backend_sha256'],plan_sha256=frozen['plan_sha256']),
            result_sha256=sha(folder/'result.json'),trace_sha256=sha(folder/'trace.npz'),video_sha256=None))
    write(back/'evaluation_complete.json',dict(complete=True,groups=[group],software_failures=0))
    write(out/'evaluation_summary.json',dict(complete=True,expert_takeover=False,formal_test22_used=False,
        software_failures=0,attempts=n,completed=n,valid=n,successes=int(success_index is not None),
        expansion=dict(expanded=expanded)))
    return pair


def test_conditional_counts_all_candidate_cases_not_only_old_shared_four(tmp_path,monkeypatch):
    pair=candidate_fixture(tmp_path,expanded=True,success_index=9)
    monkeypatch.setattr(d,'candidate_pair',lambda job:pair)
    monkeypatch.setattr(d,'validate_selected_pair',lambda spec,parent,group:spec)
    value=d.conditional_candidate_evidence(tmp_path,'QG01',2)
    assert value['valid']==10 and value['successes']==1 and value['unrun']==0
    assert value['candidate_pair']==pair


@pytest.mark.parametrize('damage',['incomplete','software_failure','few_legal','old_checkpoint','missing_receipt','native_disagreement','bare_dp'])
def test_conditional_incomplete_or_wrong_evidence_is_never_zero_success(tmp_path,monkeypatch,damage):
    from tools.runtime import read,write,sha
    pair=candidate_fixture(tmp_path);back=tmp_path/'round_02/QG01/backend';out=back/'QG01'
    monkeypatch.setattr(d,'candidate_pair',lambda job:pair)
    monkeypatch.setattr(d,'validate_selected_pair',lambda spec,parent,group:spec)
    if damage in ('incomplete','software_failure','few_legal'):
        path=out/'evaluation_summary.json';v=read(path)
        v[{'incomplete':'complete','software_failure':'software_failures','few_legal':'valid'}[damage]]={'incomplete':False,'software_failure':1,'few_legal':3}[damage]
        write(path,v)
    elif damage=='old_checkpoint':
        path=back/'dagger_evaluation_freeze.json';v=read(path);v['checkpoints']['QG01']='old';write(path,v)
    elif damage=='missing_receipt':(out/'evaluation/case_3/evaluation_receipt.json').unlink()
    else:
        path=out/'evaluation/case_0/result.json';v=read(path)
        if damage=='native_disagreement':v['reasons']['max_successes']=True
        else:v['residual_ticks']=0
        write(path,v);receipt=path.parent/'evaluation_receipt.json';v=read(receipt);v['result_sha256']=sha(path);write(receipt,v)
    with pytest.raises((ValueError,FileNotFoundError)):d.conditional_candidate_evidence(tmp_path,'QG01',2)


def extension_fixture(tmp_path,monkeypatch,*,success=False):
    import time
    from tools.runtime import write,read,sha
    from tools import tight_finish
    code=tmp_path/'code';root=tmp_path/'existing';root.mkdir();code.mkdir()
    cfg=config();write(code/'configs/tight_state.json',{'dagger':cfg})
    session=dict(source_head='old-source',started=12345.,deadline=time.time()+10000,
        compute_deadline=time.time()+9400,owned_worker_roots=[str(root)],dagger=cfg)
    write(root/'configuration.json',cfg);write(root/'session.json',session)
    write(root/'launch.json',dict(pid=123,start_ticks=456,source_head='old-source',
        session_sha256=sha(root/'session.json'),started=session['started'],deadline=session['deadline']))
    write(root/'registration.json',{'predecessor':'/completed/old'})
    write(root/'status.json',dict(state='STOPPED',owned_workers_exited=True))
    write(root/'worker_processes.json',[]);write(root/'source_freeze.json',{})
    write(root/'rounds.json',{'round_02':{'state':'RUNNING','groups':{}}})
    write(root/'conditional_extension_request.json',dict(version='conditional_zero_success_v1',extension_seconds=21600,
        maximum_extra_rounds=1,prior_round=2,next_round=3,groups={'QG01':'A','G01':'B'},
        prior_deadline=session['deadline'],prior_compute_deadline=session['compute_deadline']))
    pairs={g:candidate_fixture(root,g,success_index=0 if success and g=='G01' else None) for g in cfg['groups']}
    write(root/'round_02/snapshot.json',{'snapshot_id':'shared-snapshot'})
    write(root/'round_02/G01/dp/cache_index.json',{'raw_frozen_dino':'reusable'})
    monkeypatch.setattr(d,'CODE',code);monkeypatch.setattr(d,'proc',lambda pid:None)
    monkeypatch.setattr(tight_finish,'matrix_worker_absence',lambda rows:(True,[]))
    monkeypatch.setattr(d,'candidate_pair',lambda job:pairs[job.name])
    monkeypatch.setattr(d,'validate_selected_pair',lambda spec,parent,group:spec)
    def git(*args):
        if args[0]=='status':return ''
        if args[0]=='rev-parse':return 'reviewed-new-source'
        if args[0]=='ls-files':return 'configs/tight_state.json'
        raise AssertionError(args)
    monkeypatch.setattr(d,'git',git)
    return root,session,pairs


def test_conditional_extension_is_one_round_from_new_pairs_and_absolute_old_deadline(tmp_path,monkeypatch):
    from tools.runtime import read,sha
    root,old,pairs=extension_fixture(tmp_path,monkeypatch)
    # The old owner can mkdir the next round before observing its STOP.
    (root/'round_03').mkdir()
    old_session_sha=sha(root/'session.json');receipt=d.conditional_extend(root)
    assert receipt['triggered'] and receipt['evaluation_ready'] and receipt['initial_pairs']==pairs
    assert receipt['deadline']==old['deadline']+21600 and receipt['compute_deadline']==old['compute_deadline']+21600
    assert receipt['started']==old['started'] and not receipt['deadline_reset']
    assert sha(root/'revisions/conditional_zero_success_v1/session.json')==old_session_sha
    cfg=read(root/'configuration.json');assert cfg['first_round']==cfg['rounds']==3
    assert cfg['scope_revision']=='complete_strategy_ab_v1'
    assert cfg['round_schedule']['3']['active_groups']==['QG01','G01']
    assert cfg['round_schedule']['3']['attempts']=={'QG01':96,'G01':96}
    assert all(cfg['groups'][g]['frontend']==pairs[g]['frontend'] and cfg['groups'][g]['backend']==pairs[g]['backend'] for g in pairs)
    assert read(root/'status.json')['state']=='STOPPED'
    # CPU recovery preserves the old ranking while next collection uses candidates.
    prior=read(root/'round_02/round_finished.json')
    assert prior['recovered_after_evaluation_boundary'] and prior['additional_optimizer_updates']==0
    assert all(v['selection']['selected']=='previous_complete_pair' for v in prior['groups'].values())
    assert all(v['selection']['additional_evaluation_trials']==0 for v in prior['groups'].values())
    assert d.previous_round_groups(root,cfg)==prior['groups']
    assert read(root/'launch.json')['session_sha256']==sha(root/'session.json')
    with pytest.raises(ValueError,match='already recorded'):d.conditional_extend(root)


def test_success_ends_old_a_only_queue_without_extending_or_retraining(tmp_path,monkeypatch):
    from tools.runtime import read
    root,old,_=extension_fixture(tmp_path,monkeypatch,success=True)
    receipt=d.conditional_extend(root)
    assert receipt['evaluation_ready'] and not receipt['triggered'] and receipt['extension_seconds']==0
    assert receipt['deadline']==old['deadline'] and receipt['initial_pairs']=={}
    assert read(root/'status.json')['state']=='COMPLETE'
    cfg=read(root/'configuration.json');assert cfg['first_round']==cfg['rounds']==2
    assert cfg['round_schedule'].keys()=={'2'}
    with pytest.raises(ValueError,match='not admitted'):d.resume(root)


def test_conditional_recovery_preserves_existing_selection_bytes(tmp_path,monkeypatch):
    root,_,_=extension_fixture(tmp_path,monkeypatch)
    evidence={g:d.conditional_candidate_evidence(root,g,2) for g in ('QG01','G01')}
    d.recover_completed_round(root,2,evidence)
    existing=(root/'round_02/QG01/selection.json').read_bytes()
    (root/'round_02/G01/selection.json').unlink();(root/'round_02/round_finished.json').unlink()
    d.conditional_extend(root)
    assert (root/'round_02/QG01/selection.json').read_bytes()==existing


def test_missing_evaluation_ends_condition_as_incomplete_without_extension(tmp_path,monkeypatch):
    from tools.runtime import read
    root,old,_=extension_fixture(tmp_path,monkeypatch)
    (root/'round_02/G01/backend/evaluation_complete.json').unlink()
    receipt=d.conditional_extend(root)
    assert not receipt['evaluation_ready'] and not receipt['triggered'] and receipt['deadline']==old['deadline']
    assert 'G01' in receipt['errors'] and read(root/'status.json')['state']=='PARTIAL'
    assert not (root/'round_02/round_finished.json').exists()
    with pytest.raises(ValueError,match='not admitted'):d.resume(root)


@pytest.mark.parametrize('damage',['started_future','deadline_changed','live_owner','session_changed'])
def test_conditional_migration_rejects_unsafe_boundary_without_mutation(tmp_path,monkeypatch,damage):
    from tools.runtime import read,write,sha
    root,old,_=extension_fixture(tmp_path,monkeypatch)
    if damage=='started_future':write(root/'round_03/decision.json',{'registered':'old work'})
    elif damage=='deadline_changed':
        path=root/'conditional_extension_request.json';v=read(path);v['prior_deadline']+=1;write(path,v)
    elif damage=='live_owner':monkeypatch.setattr(d,'proc',lambda pid:dict(start_ticks=456))
    else:
        path=root/'session.json';v=read(path);v['deadline']+=1;write(path,v)
    digest=sha(root/'session.json')
    with pytest.raises(ValueError):d.conditional_extend(root)
    assert sha(root/'session.json')==digest and not (root/'revisions/conditional_zero_success_v1').exists()


def test_conditional_attempt_budget_is_equal_and_reserves_both_updates():
    cfg=config();cfg['conditional_extension']={'triggered':True}
    cfg['round_schedule']={'3':dict(active_groups=['QG01','G01'],attempts={'QG01':96,'G01':96})}
    cfg['planning']['pair_stage_reserve_seconds']=9600
    full=d.round_decision(cfg,{},11*3600,3)
    assert full['attempts_by_group']=={'QG01':96,'G01':96}
    limited=d.round_decision(cfg,{},8*3600,3)
    counts=limited['attempts_by_group'];assert 16<=counts['QG01']==counts['G01']<96
    assert sum(counts.values())*100+limited['reserved_compute_seconds']+cfg['planning']['transition_reserve_seconds']<=8*3600
    with pytest.raises(ValueError,match='remaining budget'):d.round_decision(cfg,{},6*3600,3)
