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
