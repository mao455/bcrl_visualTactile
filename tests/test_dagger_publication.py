"""CPU publication of incomplete receipts must never manufacture completion."""
import gzip
import json

import pytest

from tools import tight_diagnostics as diagnostics


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


@pytest.fixture
def publication(tmp_path,monkeypatch):
    code=tmp_path/'code';code.mkdir();root=tmp_path/'run';root.mkdir()
    monkeypatch.setattr(diagnostics,'CODE',code)
    document=code/'research_docs/STATE_DP_RESULTS.md';document.parent.mkdir()
    document.write_text('# Existing results\n\nKeep the earlier failed gate unchanged.\n')
    write(root/'configuration.json',dict(groups={'G11':{'supervision':'u_expert'},'QG01':{'supervision':'q_reached'}},
        collection={'attempts_per_group':12}))
    write(root/'collection_plan.json',[dict(identity=f'c{i}',seed=100+i) for i in range(12)])
    write(root/'evaluation_plan.json',[dict(identity=f'e{i}',spec={'seed':200+i}) for i in range(10)])
    write(root/'status.json',dict(state='PARTIAL',groups={'G11':'FAILED','QG01':'INSUFFICIENT_DP_CORRECTIONS'},
        owned_workers_exited=True,utc=120.,test22_used=False,no_next_round=True))
    write(root/'launch.json',dict(started=100.,source_head='frozen-source'))
    write(root/'owned_cleanup.json',{'verified':True})
    return root,code/'research_docs/evidence/tight_dagger_20261111_01',document


def test_empty_failed_round_exports_all_slots_and_preserves_other_document(publication):
    root,out,document=publication
    summary=diagnostics.dagger_publish(root,out)
    assert not summary['verified_round_complete'] and not summary['inference_or_training_performed']
    for row in summary['groups'].values():
        assert row['collection']['planned']==12 and row['collection']['unstarted']==12
        assert row['autonomous_development']['planned']==10 and row['autonomous_development']['unstarted']==10
        assert row['autonomous_development']['successes'] is None
        assert row['stages']['frontend']['state']=='unmeasured' and row['stages']['frontend']['updates'] is None
    original=document.read_text()
    assert original.startswith('# Existing results\n\nKeep the earlier failed gate unchanged.')
    assert '未测 / 0 / 10' in original
    assert not (out/'training_curves.png').exists()
    diagnostics.dagger_publish(root,out)
    assert document.read_text()==original
    assert document.read_text().count('<!-- tight-dagger-results:start -->')==1


def test_partial_training_collection_success_not_autonomous_success(publication):
    root,out,document=publication;job=root/'round_01/G11'
    write(job/'collection_results.json',[dict(identity='c0',accepted=True,label_contract_checked=True,
        exposure={'supervised_ticks':30,'supervised_anchors':2},result=dict(legal=True,software_complete=True,success=True,
            dagger=dict(expert_takeover_success=True,student_autonomous_success=False,takeover_triggered=True,expert_valid_ticks=30)))])
    (job/'attempts.jsonl').write_text(json.dumps({'identity':'c0'})+'\n'+json.dumps({'identity':'c1'})+'\n')
    dp=job/'dp/training'
    write(dp/'progress.json',dict(update=8,anchor_equivalent_epochs=.2))
    write(dp/'configuration.json',dict(continuation_audit={'source_updates':1000},supervision='u_expert'))
    (dp/'training.jsonl').write_text(json.dumps(dict(update=8,anchor_equivalent_epochs=.2,loss=.3,
        components={'base':.28,'hand_geometry':.02}))+'\n{partial tail')
    (dp/'validation.jsonl').write_text(json.dumps(dict(update=8,anchor_equivalent_epochs=.2,
        metrics={'selection_score':.4,'arm_rmse':.01,'hand_rmse':.02},train8={'selection_score':.3}))+'\n')
    back=job/'backend/G11';write(back/'training/finished.json',dict(updates=3,new_optimizer_updates=3,
        control_tick_epochs=1.,supervised_ticks=30,supervised_tick_population=30,target_reached=False,stop_reason='fixed_wall_budget'))
    (back/'training/best.pt').write_bytes(b'opaque checkpoint; publisher must never deserialize')
    digest=diagnostics.sha(back/'training/best.pt')
    write(back/'evaluation/e0/result.json',dict(identity='e0',seed=200,legal=True,software_complete=True,success=False,
        checkpoint_sha256=digest,reasons={'max_successes':False,'timeout':True},steps=601))
    write(back/'evaluation/e1/started.json',{'seed':201})
    write(back/'evaluation/e2/result.json',dict(identity='e2',seed=202,legal=False,software_complete=False,success=False))
    write(back/'evaluation/e3/result.json',dict(identity='e3',seed=203,legal=True,software_complete=True,success=True,
        checkpoint_sha256='stale',reasons={'max_successes':True}))
    telemetry=root/'resources/shared.jsonl';telemetry.parent.mkdir()
    telemetry.write_text('\n'.join(json.dumps(dict(utc_epoch=t,gpu_util_percent=g,gpu_free_mib=10000,ram_available_bytes=30*1024**3))
        for t,g in ((101,20),(102,40),(115,60),(116,80)))+'\n')
    summary=diagnostics.dagger_publish(root,out);group=summary['groups']['G11']
    assert group['collection']['accepted_episodes']==1 and group['collection']['expert_takeover_successes']==1
    assert group['collection']['attempted']==2 and group['collection']['software_incomplete']==1
    assert group['autonomous_development']['successes']==0 and group['autonomous_development']['valid']==1
    assert group['autonomous_development']['software_incomplete']==1 and group['autonomous_development']['rejected']==1
    assert group['autonomous_development']['unverified_model_or_seed']==1 and group['autonomous_development']['unstarted']==6
    assert group['stages']['frontend']['state']=='partial' and group['stages']['backend']['state']=='stopped_before_ceiling'
    assert not summary['verified_round_complete']
    assert summary['resources']['time_weighted_whole_card_percent']==40.
    assert summary['resources']['gap_seconds_above8']==13.
    assert summary['resources']['leading_uncovered_s']==1. and summary['resources']['trailing_uncovered_s']==4.
    assert any(r['problem']=='incomplete_or_invalid_jsonl_line' for r in summary['publication_issues'])
    with gzip.open(out/'G11_frontend_every_update.csv.gz','rt') as stream:assert 'components.hand_geometry' in stream.read()
    assert (out/'training_curves.png').is_file() and not list(out.glob('*.pt'))
    assert 'Keep the earlier failed gate unchanged.' in document.read_text()
    # An optimistic owner label cannot supply missing stage/trial receipts.
    status=json.loads((root/'status.json').read_text());status.update(state='COMPLETE',groups={'G11':'COMPLETE','QG01':'COMPLETE'})
    write(root/'status.json',status)
    assert not diagnostics.dagger_publish(root,out)['verified_round_complete']


def test_running_owner_and_malformed_managed_block_preserve_document(publication):
    root,out,document=publication;before=document.read_text()
    write(root/'status.json',dict(state='RUNNING',owned_workers_exited=False))
    with pytest.raises(ValueError,match='stops its workers'):diagnostics.dagger_publish(root,out)
    assert document.read_text()==before
    write(root/'status.json',{'state':'FAILED'})
    document.write_text(before+'<!-- tight-dagger-results:start -->\nUser incomplete edit\n')
    before=document.read_text()
    with pytest.raises(ValueError,match='Malformed'):diagnostics.dagger_publish(root,out)
    assert document.read_text()==before


def test_pool_partial_rounds_keep_32_attempt_slots_dev_identity_and_old_result_block(publication):
    root,out,document=publication
    diagnostics.dagger_publish(root,out)
    old=document.read_text()
    pool=root/'pool'
    write(root/'configuration.json',dict(version='successful_takeover_pool_v1',rounds=3,
        groups={'G11':{},'QG01':{}},pool_root=str(pool),collection={'attempts_per_group':32}))
    write(pool/'acceptance/readback.json',dict(passed=True,pool_entries=21))
    write(root/'status.json',dict(state='RUNNING',owned_workers_exited=False,rounds={'round_01':{'state':'RUNNING'}}))
    job=root/'round_01/QG01'
    write(job/'collection_plan.json',[dict(identity=f'c{i}',pool_partition='correction_dev' if i<4 else 'train') for i in range(32)])
    write(job/'collection_results.json',[dict(identity='c0',accepted=True,label_contract_checked=True,
        exposure={'supervised_ticks':24,'supervised_anchors':2},result=dict(legal=True,software_complete=True,success=True,
            dagger=dict(expert_takeover_success=True,student_autonomous_success=False)))])
    write(job/'selection.json',dict(selected_previous=True,reason='Candidate did not improve registered development metrics'))
    write(root/'round_01/decision.json',dict(attempts_per_group=32,automatic_next_round=True))
    write(root/'round_01/snapshot.json',dict(snapshot_id='frozen',train_ids=['old'],correction_dev_ids=['dev']))
    stage=job/'dp/training'
    write(stage/'selected.json',dict(update=0,initialization_candidate=True))
    write(stage/'progress.json',dict(update=8,coverage=dict(anchor_equivalent_epochs=.2,
        strata={'core':{'anchors':32},'current_correction':{'anchors':20}})))
    (stage/'validation.jsonl').write_text(json.dumps(dict(update=0,anchor_equivalent_epochs=0,
        metrics={'selection_score':2.},correction_dev={'selection_score':3.,'initial24':{'selection_score':4.}},
        selection_components=dict(old_validation=2.,correction_dev=3.,correction_initial24=4.)))+'\n')
    output=out.parent/'tight_dagger_pool_20261111_01'
    summary=diagnostics.dagger_publish(root,output)
    assert summary['partial_snapshot'] and not summary['verified_round_complete']
    assert len(summary['rounds'])==3
    first=summary['groups']['round_01/QG01']
    assert first['collection']['planned']==32 and first['collection']['unstarted']==31
    assert first['stages']['frontend']['selection']['initialization_candidate']
    assert first['stages']['frontend']['sampling_exposure']['strata']['current_correction']['anchors']==20
    assert summary['groups']['round_02/QG01']['collection']['planned']==0  # No invented future identities.
    assert summary['groups']['round_02/QG01']['autonomous_development']['successes'] is None
    assert summary['rounds']['round_01']['snapshot']['correction_dev_ids']==['dev']
    assert summary['pool']['acceptance/readback.json']['pool_entries']==21
    text=document.read_text()
    assert text.startswith(old.rstrip())
    assert text.count('<!-- tight-dagger-results:start -->')==1
    assert text.count('<!-- tight-dagger-pool-results:start -->')==1
    with gzip.open(output/'round_01_QG01_frontend_validation.json.gz','rt') as stream:
        assert json.load(stream)[0]['correction_dev']['initial24']['selection_score']==4.
    assert (output/'sampling_exposure.csv').is_file()
    assert 'correction_initial24' in (output/'validation_curves.csv').read_text()


def test_pool_four_case_allocation_is_complete_but_six_unrun_remain(publication):
    root,out,_=publication
    write(root/'configuration.json',dict(version='successful_takeover_pool_v1',rounds=1,groups={'G11':{}},
        evaluation={'attempts_per_group':4,'maximum_attempts_per_group':10},pool_root=str(root/'pool')))
    back=root/'round_01/G11/backend';node=back/'G11'
    (node/'training').mkdir(parents=True);(node/'training/best.pt').write_bytes(b'known selected model')
    digest=diagnostics.sha(node/'training/best.pt')
    for i in range(4):
        write(node/f'evaluation/e{i}/result.json',dict(legal=True,software_complete=True,success=False,
            checkpoint_sha256=digest,seed=200+i,reasons={'max_successes':False,'timeout':True}))
    expansion=dict(expanded=False,initial_trials=4,reason='registered thresholds not reached')
    write(node/'evaluation_summary.json',dict(complete=True,expansion=expansion))
    write(back/'evaluation_complete.json',dict(complete=True,group_summary={'G11':dict(complete=True,planned=10,
        completed=4,unrun=6,software_failures=0,expansion=expansion)}))
    for name in ('controller_acceptance','takeover_timing','owner_data_acceptance','cleanup_candidates'):
        write(root/'preparation'/(name+'.json'),dict(passed=True,source='outputs/checked'))
    write(root/'cleanup.json',dict(deleted_files=8,released_allocated_bytes=1234))
    report=diagnostics.dagger_publish(root,out)
    evaluated=report['groups']['round_01/G11']['autonomous_development']
    assert evaluated['state']=='complete' and evaluated['allocated']==4
    assert evaluated['valid']==4 and evaluated['successes']==0 and evaluated['unstarted']==6
    assert not report['verified_round_complete']  # No collection or training receipt exists.
    assert json.loads((out/'cleanup.json').read_text())['deleted_files']==8
    assert json.loads((out/'controller_acceptance.json').read_text())['passed']
    (back/'evaluation_complete.json').unlink()
    assert diagnostics.dagger_publish(root,out)['groups']['round_01/G11']['autonomous_development']['state']=='partial'
