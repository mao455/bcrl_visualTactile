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
