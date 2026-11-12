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


def test_revised_ab_scope_retains_retired_partial_and_excludes_unscheduled_obligations(publication):
    root,out,document=publication
    write(root/'configuration.json',dict(version='successful_takeover_pool_v1',scope_revision='complete_strategy_ab_v1',
        first_round=2,rounds=3,groups={'QG01':{'strategy_role':'A'},'G01':{'strategy_role':'B'}},
        round_schedule={'2':{'active_groups':['QG01','G01']},'3':{'active_groups':['QG01']}},
        evaluation={'attempts_per_group':4,'maximum_attempts_per_group':10},pool_root=str(root/'pool')))
    write(root/'status.json',dict(state='COMPLETE',owned_workers_exited=True,rounds={
        'round_01':{'state':'STOPPED','groups':{'G11':{'state':'COLLECTED'}}}}))
    write(root/'round_01/G11/dp/training/progress.json',dict(update=754,anchor_equivalent_epochs=7.5294))
    write(root/'round_01/G11/dp/training/selected.json',dict(update=201))
    write(root/'complete_policy_acceptance.json',dict(passed=True,bare_dp_problem_found=False))
    write(root/'recovery_coverage.json',dict(independent_takeover_starts=12))
    for index,groups in ((2,['QG01','G01']),(3,['QG01'])):
        round_root=root/f'round_{index:02d}'
        write(round_root/'round_finished.json',dict(state='RESOLVED',groups={g:{'state':'COMPLETE'} for g in groups}))
        write(round_root/'decision.json',dict(active_groups=groups))
        write(round_root/'snapshot.json',dict(snapshot_id=f'pool-{index}'))
        for group in groups:
            job=round_root/group
            write(job/'collection_finished.json',dict(complete=True,sufficient=True))
            write(job/'collection_gate.json',dict(passed=True))
            write(job/'complete_policy_acceptance.json',dict(passed=True,backend_required=True))
            write(job/'recovery_coverage.json',dict(independent_takeover_starts=4,recovery_prefix_ticks=48))
            (job/'recovery_coverage.csv').write_text('phase,count\ninitial_contact,4\n')
            node=job/'backend'/group
            for training in (job/'dp/training',node/'training'):
                write(training/'finished.json',dict(updates=400,disposable=False,target_reached=True))
                (training/'best.pt').write_bytes(b'selected opaque model')
            digest=diagnostics.sha(node/'training/best.pt')
            for i in range(4):
                write(node/f'evaluation/e{i}/result.json',dict(legal=True,software_complete=True,success=False,
                    checkpoint_sha256=digest,seed=200+i,reasons={'max_successes':False,'timeout':True}))
            expansion=dict(expanded=False,initial_trials=4,reason='registered thresholds not reached')
            write(node/'evaluation_summary.json',dict(complete=True,expansion=expansion))
            write(job/'backend/evaluation_complete.json',dict(complete=True,group_summary={group:dict(
                complete=True,planned=10,completed=4,unrun=6,software_failures=0,expansion=expansion)}))
    report=diagnostics.dagger_publish(root,out)
    assert report['verified_round_complete']
    assert report['groups']['round_01/G11']['participation']=='historical'
    assert report['groups']['round_01/G11']['stages']['frontend']['updates']==754
    assert report['groups']['round_01/G11']['stages']['frontend']['selection']['update']==201
    assert 'round_01/G01' not in report['groups']  # Never invent a historical B job.
    unscheduled=report['groups']['round_03/G01']
    assert unscheduled['role']=='B' and unscheduled['owner_state']=='NOT_SCHEDULED'
    assert unscheduled['autonomous_development']['state']=='not_scheduled'
    assert unscheduled['autonomous_development']['planned']==0
    assert unscheduled['autonomous_development']['successes'] is None
    assert report['completion_scope']['required_groups']==['round_02/G01','round_02/QG01','round_03/QG01']
    assert json.loads((out/'round_02_QG01_complete_policy_acceptance.json').read_text())['passed']
    assert 'initial_contact,4' in (out/'round_02_QG01_recovery_coverage.csv').read_text()
    assert json.loads((out/'recovery_coverage.json').read_text())['independent_takeover_starts']==12
    assert 'round_03,G01,autonomous_development_after_dagger' not in (out/'closed_loop.csv').read_text()
    assert 'not_scheduled' in document.read_text()
    # New-scope missing evidence must still prevent an optimistic completion claim.
    (root/'round_03/QG01/backend/evaluation_complete.json').unlink()
    assert not diagnostics.dagger_publish(root,out)['verified_round_complete']


@pytest.fixture
def conditional_publication(publication):
    root,out,document=publication
    write(root/'configuration.json',dict(version='successful_takeover_pool_v1',scope_revision='complete_strategy_ab_v1',
        first_round=2,rounds=3,groups={'QG01':{'strategy_role':'A'},'G01':{'strategy_role':'B'}},
        round_schedule={'2':{'active_groups':['QG01','G01']},'3':{'active_groups':['QG01','G01']}},
        evaluation={'attempts_per_group':4,'maximum_attempts_per_group':10},pool_root=str(root/'pool')))
    write(root/'conditional_extension_request.json',dict(version='conditional_zero_success_v1',prior_round=2,
        next_round=3,groups={'QG01':'A','G01':'B'},extension_seconds=21600,max_extra_rounds=1,
        prior_deadline=100000,prior_compute_deadline=99400))
    return root,out,document


def test_conditional_request_is_not_a_zero_success_result_or_started_round(conditional_publication):
    root,out,document=conditional_publication
    write(root/'conditional_extension_status.json',dict(state='WAITING_CURRENT_EVALUATIONS',started=False))
    write(root/'conditional_extension_service.json',dict(service='ordinary-watcher.service',pid=123))
    summary=diagnostics.dagger_publish(root,out)
    condition=summary['conditional_extension']
    assert condition['requested'] and not condition['evaluation_ready']
    assert condition['triggered'] is None and not condition['continuation_started']
    assert not condition['collection_startup_verified'] and not summary['verified_round_complete']
    assert summary['completion_scope']['required_groups']==['round_02/G01','round_02/QG01']
    for group in ('G01','QG01'):
        row=summary['groups']['round_03/'+group]
        assert row['participation']=='conditional_pending'
        assert row['autonomous_development']['successes'] is None
        assert row['autonomous_development']['planned']==0
    assert json.loads((out/'conditional_extension_request.json').read_text())['max_extra_rounds']==1
    assert json.loads((out/'conditional_extension_service.json').read_text())['pid']==123
    assert '第3轮B未调度' not in document.read_text()
    assert '请求不等于触发' in document.read_text()


def test_conditional_decision_launch_and_actual_startup_remain_distinct(conditional_publication):
    root,out,_=conditional_publication
    decision=dict(revision='conditional_zero_success_v1',triggered=True,evaluation_ready=True,
        reason='Both complete legal candidate evaluations have zero native success',extension_seconds=21600,
        prior_deadline=100000,deadline=121600,initial_pairs={'G01':{'frontend_sha256':'new-b'},'QG01':{'frontend_sha256':'new-a'}})
    write(root/'conditional_extension.json',decision)
    write(root/'conditional_extension_decision.json',dict(triggered=True,evaluation_ready=True))
    report=diagnostics.dagger_publish(root,out)
    condition=report['conditional_extension']
    assert condition['triggered'] and condition['evaluation_ready'] and not condition['continuation_started']
    assert report['completion_scope']['required_groups']==[
        'round_02/G01','round_02/QG01','round_03/G01','round_03/QG01']
    assert not report['verified_round_complete']  # A trigger does not supply any training receipt.
    assert json.loads((out/'conditional_extension.json').read_text())==decision
    write(root/'conditional_extension_status.json',dict(state='CONTINUATION_RUNNING',started=False,owner_service_started=True))
    condition=diagnostics.dagger_publish(root,out)['conditional_extension']
    assert condition['continuation_started'] and not condition['collection_startup_verified']
    write(root/'conditional_extension_status.json',dict(state='STARTUP_VERIFIED',started=True))
    condition=diagnostics.dagger_publish(root,out)['conditional_extension']
    assert condition['collection_startup_verified']


@pytest.mark.parametrize('evaluation_ready,participation',[(False,'conditional_pending'),(True,'not_scheduled')])
def test_not_triggered_requires_completed_evaluation_to_resolve_future_round(conditional_publication,evaluation_ready,participation):
    root,out,_=conditional_publication
    write(root/'conditional_extension.json',dict(triggered=False,evaluation_ready=evaluation_ready,
        reason='Incomplete evaluation' if not evaluation_ready else 'At least one native success',extension_seconds=0))
    write(root/'conditional_extension_status.json',dict(state='FAILED' if not evaluation_ready else 'NOT_TRIGGERED',started=False))
    summary=diagnostics.dagger_publish(root,out)
    assert summary['conditional_extension']['triggered'] is False
    assert not summary['conditional_extension']['continuation_started']
    for group in ('G01','QG01'):
        row=summary['groups']['round_03/'+group]
        assert row['participation']==participation and not row['current_scope_required']
        assert row['autonomous_development']['successes'] is None
    assert 'round_03,G01,autonomous_development_after_dagger' not in (out/'closed_loop.csv').read_text()


def test_matched_protocol_keeps_phase0_separate_and_preserves_old_pool_block(publication):
    root,out,document=publication
    old='<!-- tight-dagger-pool-results:start -->\nHistorical A/B unequal rounds remain.\n<!-- tight-dagger-pool-results:end -->\n'
    document.write_text(document.read_text()+old)
    write(root/'configuration.json',dict(version='successful_takeover_pool_v1',scope_revision='matched_control_ab_v1',
        first_round=1,rounds=2,groups={'QG01':{'strategy_role':'A'},'G01':{'strategy_role':'B'}},
        round_schedule={'1':{'active_groups':['QG01','G01']},'2':{'active_groups':['QG01','G01']}},
        matched_protocol={'version':'matched_control_ab_v1'},pool_root=str(root/'pool')))
    write(root/'phase0/finished.json',dict(complete=True,kind='rear_control_scale_adaptation_not_dagger_round',
        groups={'QG01':{'pair':{'frontend_sha256':'oldfront','backend_sha256':'adaptedrear'}}}))
    write(root/'phase0/QG01/backend/QG01/training/finished.json',dict(updates=10,new_optimizer_updates=10,control_tick_epochs=6))
    report=diagnostics.dagger_publish(root,out)
    assert report['phase0']['reported_complete'] and not report['phase0']['complete']
    assert not report['phase0']['counts_as_dagger_round']
    assert report['phase0']['groups']['QG01']['frontend_updates']==0
    assert report['phase0']['groups']['QG01']['backend_updates']==10
    assert len(report['method_rounds'])==4 and all(r['counts_as_dagger_round'] for r in report['method_rounds'])
    assert not report['verified_round_complete']
    assert all(v['stages']['frontend']['updates'] is None for v in report['groups'].values())
    text=document.read_text();assert old in text and 'tight-dagger-matched-results:start' in text
    assert 'A/B已训练残差幅度分别Arm0.2/Hand0.9与全0.1' not in text
    assert (out/'phase0.json').is_file() and (out/'method_rounds.csv').is_file()


@pytest.fixture
def terminal_follow(tmp_path, monkeypatch):
    """Real local Git/bare remote, with only process/time/publication substituted."""
    import subprocess
    from tools import runtime, tight_finish
    code=tmp_path/'reviewed';code.mkdir()
    deploy=tmp_path/'frozen';deploy.mkdir();(deploy/'worker.py').write_text('frozen runtime\n')
    remote=tmp_path/'origin.git'
    def command(*args, cwd=code):
        return subprocess.run(list(args),cwd=cwd,check=True,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()
    command('git','init','--bare',str(remote),cwd=tmp_path)
    command('git','init','--initial-branch=feature/tight-state-foundation')
    command('git','config','user.name','Publication Test');command('git','config','user.email','test@example.invalid')
    doc=code/'research_docs/STATE_DP_RESULTS.md';doc.parent.mkdir();doc.write_text('Reviewed diagnosis\n')
    command('git','add','.');command('git','commit','-m','Reviewed code and report')
    command('git','remote','add','origin',str(remote));command('git','push','origin','HEAD:refs/heads/feature/tight-state-foundation')
    prepared=command('git','rev-parse','HEAD')
    root=tmp_path/'run';root.mkdir();clock=[1000.];calls=[]
    write(root/'session.json',dict(deadline=1200.,source_head='frozen-head'))
    write(root/'source_freeze.json',{'worker.py':diagnostics.sha(deploy/'worker.py')})
    write(root/'status.json',dict(state='COMPLETE',owned_workers_exited=True))
    write(root/'worker_processes.json',[dict(pid=8,start_ticks=80)])
    request=root/'publication_request.json'
    write(request,dict(version='dagger_terminal_publication_follow_v1',owner=dict(pid=7,start_ticks=70),
        deployment_tree=str(deploy),prepared_source=prepared,target='refs/heads/feature/tight-state-foundation',
        deadline=1200.,deployment_source_head='frozen-head',source_freeze_sha256=diagnostics.sha(root/'source_freeze.json')))
    monkeypatch.setattr(diagnostics,'CODE',code)
    # Patch only the publisher's reference. subprocess timeout polling and other
    # imported libraries must keep their real process-global time module.
    from types import SimpleNamespace
    monkeypatch.setattr(diagnostics,'time',SimpleNamespace(time=lambda:clock[0],
        sleep=lambda seconds:clock.__setitem__(0,clock[0]+seconds)))
    monkeypatch.setattr(runtime,'proc',lambda pid:None)
    monkeypatch.setattr(tight_finish,'matrix_worker_absence',lambda rows:(True,rows))
    def publish(run):
        calls.append('publish');assert run==root
        doc.write_text(doc.read_text()+'Terminal numerical readback\n')
        output=code/'research_docs/evidence'/root.name
        write(output/'summary.json',dict(state='COMPLETE',successes=0))
        # The real exporter emits both scalar CSV and nested validation JSON gzip.
        with gzip.open(output/'round_01_QG01_backend_validation.json.gz','wt') as stream:
            json.dump([dict(update=32,validation=dict(arm_rmse_rad=.01,hand_rmse_rad=.02))],stream)
        return dict(verified_round_complete=True)
    monkeypatch.setattr(diagnostics,'dagger_publish',publish)
    return dict(code=code,deploy=deploy,root=root,request=request,clock=clock,calls=calls,
        command=command,prepared=prepared,doc=doc,runtime=runtime,tight_finish=tight_finish)


def test_terminal_publication_waits_for_owner_and_workers_without_stopping_them(terminal_follow,monkeypatch):
    s=terminal_follow;start_sha=diagnostics.sha(s['root']/'session.json')
    monkeypatch.setattr(s['runtime'],'proc',lambda pid:dict(start_ticks=70,state='R') if s['clock'][0]<1010 else None)
    monkeypatch.setattr(s['tight_finish'],'matrix_worker_absence',lambda rows:(s['clock'][0]>=1015,rows))
    receipt=diagnostics.dagger_publication_follow(s['root'],s['request'])
    assert receipt['complete'] and receipt['pushed'] and s['clock'][0]>=1015
    assert s['calls']==['publish'] and receipt['experiment_mutations']==0
    assert not (s['root']/'STOP').exists() and diagnostics.sha(s['root']/'session.json')==start_sha
    assert receipt['head']==s['command']('git','ls-remote','origin','refs/heads/feature/tight-state-foundation').split()[0]
    assert all(name.startswith('research_docs/') for name in receipt['files'])
    compressed='research_docs/evidence/run/round_01_QG01_backend_validation.json.gz'
    assert compressed in receipt['files']
    with gzip.open(s['code']/compressed,'rt') as stream:
        assert json.load(stream)[0]['validation']['arm_rmse_rad']==.01


def test_terminal_publication_admits_only_scoped_remote_report_advance(terminal_follow):
    s=terminal_follow;git=s['command']
    s['doc'].write_text(s['doc'].read_text()+'Owner terminal report\n')
    git('git','add','.');git('git','commit','-m','Scoped owner report');remote=git('git','rev-parse','HEAD')
    git('git','push','origin','HEAD:refs/heads/feature/tight-state-foundation')
    git('git','reset','--hard',s['prepared'])
    receipt=diagnostics.dagger_publication_follow(s['root'],s['request'])
    assert receipt['previous_remote']==remote and 'Owner terminal report' in s['doc'].read_text()
    assert s['calls']==['publish']


def test_terminal_publication_rejects_unrelated_remote_code_before_readback(terminal_follow):
    s=terminal_follow;git=s['command'];(s['code']/'unreviewed.py').write_text('changed\n')
    git('git','add','.');git('git','commit','-m','Unrelated code');git('git','push','origin','HEAD:refs/heads/feature/tight-state-foundation')
    git('git','reset','--hard',s['prepared'])
    with pytest.raises(ValueError,match='Unrelated remote'):
        diagnostics.dagger_publication_follow(s['root'],s['request'])
    assert not s['calls'] and git('git','rev-parse','HEAD')==s['prepared']
    assert not (s['root']/'STOP').exists()


def test_terminal_publication_stops_waiting_at_original_deadline_plus_600(terminal_follow,monkeypatch):
    s=terminal_follow;s['clock'][0]=1799.
    monkeypatch.setattr(s['runtime'],'proc',lambda pid:dict(start_ticks=70,state='R'))
    with pytest.raises(TimeoutError,match='600-second'):
        diagnostics.dagger_publication_follow(s['root'],s['request'])
    assert not s['calls'] and not (s['root']/'STOP').exists()
    assert s['clock'][0]<=1800.1


@pytest.mark.parametrize('change',['deployment','dirty_reviewed_tree','freeze_registration'])
def test_terminal_publication_rejects_source_drift_without_touching_run(terminal_follow,change):
    s=terminal_follow
    if change=='deployment':(s['deploy']/'worker.py').write_text('hot changed\n')
    elif change=='dirty_reviewed_tree':(s['code']/'unexpected.txt').write_text('unreviewed\n')
    else:write(s['root']/'source_freeze.json',{})
    before=diagnostics.sha(s['root']/'session.json')
    with pytest.raises(ValueError):diagnostics.dagger_publication_follow(s['root'],s['request'])
    assert not s['calls'] and diagnostics.sha(s['root']/'session.json')==before
    assert not (s['root']/'STOP').exists()


@pytest.mark.parametrize('sizes,accepted',[
    ({'summary.json':49},True),
    ({'summary.json':51},False),
    ({'other.json':26},False),
    ({'model.pt':1},False),
    ({'summary.json':49,'a.csv':24,'b.csv':24,'c.csv':4},False),
])
def test_terminal_publication_has_larger_summary_only_and_keeps_other_limits(terminal_follow,monkeypatch,sizes,accepted):
    """Exercise size admission without storing/pushing artificial huge fixtures."""
    import os
    from pathlib import Path
    s=terminal_follow;output=s['code']/'research_docs/evidence'/s['root'].name
    publish=diagnostics.dagger_publish;original_stat=Path.stat
    def generate(root):
        result=publish(root)
        for name in sizes:
            path=output/name
            if not path.exists():path.write_text('0\n')
        return result
    def measured_stat(path,*args,**kwargs):
        result=original_stat(path,*args,**kwargs)
        if path.parent==output and path.name in sizes:
            fields=list(result);fields[6]=sizes[path.name]*1024**2
            return os.stat_result(fields)
        return result
    monkeypatch.setattr(diagnostics,'dagger_publish',generate)
    monkeypatch.setattr(Path,'stat',measured_stat)
    if accepted:
        assert diagnostics.dagger_publication_follow(s['root'],s['request'])['pushed']
    else:
        with pytest.raises(ValueError,match='oversized|100 MiB'):
            diagnostics.dagger_publication_follow(s['root'],s['request'])
        assert s['command']('git','rev-parse','HEAD')==s['prepared']
