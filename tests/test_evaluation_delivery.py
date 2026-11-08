import pytest
from tools.evaluation_delivery import summarize,report_hash,ALLOWED


def summary(native_fresh=0):
    trials=[]
    for cohort,n in [('train',90),('validation',10),('fresh',100)]:
        for i in range(n):
            success=cohort=='fresh' and i<native_fresh
            trials.append(dict(trial=dict(cohort=cohort,success=success,native_subgoals=2 if success else 0,
                contact=dict(object_contact=True,stable_grasp_diagnostic=False),reasons=dict(timeout=not success)),
                success_audit=dict(terminal_geometry=dict(ever_satisfied=cohort=='fresh' and i<10))))
    return dict(trials=trials,valid=200,status=dict(state='completed'),
        groups=[dict(group=c,valid=n,success=native_fresh if c=='fresh' else 0)
            for c,n in [('historical',100),('train',90),('validation',10),('fresh',100)]],
        software_missing=0,smoke_software_missing=2,acceptance_failed=0,shutdown_faults=0,
        checkpoint=dict(sha256='protected'),source_commit='frozen')


def test_terminal_geometry_is_not_promoted_to_native_success():
    result=summarize(summary())
    assert '未达到' in result['decision']
    assert result['observed_failure_stages']['fresh']['terminal_geometry_without_ordered_success']==10
    assert result['failure_flags']['fresh']['timeout']==100
    assert result['smoke_software_missing']==2


def test_image_recommendation_uses_completed_fresh_native_results():
    assert '可建议进入Image' in summarize(summary(10))['decision']
    value=summary();value['valid']=199;value['trials'].pop()
    with pytest.raises(AssertionError):summarize(value)


def test_publication_guards_distinguish_generated_sections_from_human_edits():
    a='# Report\nHuman-reviewed text.\n<!-- RANDOM_EVALUATION_START -->\n0/0\n<!-- RANDOM_EVALUATION_END -->\n'
    b=a.replace('0/0','10/100')
    assert report_hash(a)==report_hash(b)
    assert report_hash(a)!=report_hash(a.replace('Human-reviewed','Changed'))
    assert all(p.startswith('research_docs/') and p.endswith(('.md','.json','.png')) for p in ALLOWED)


def test_local_finalization_preserves_reviewed_report_and_real_denominators(tmp_path,monkeypatch):
    import json
    import tools.evaluation_delivery as delivery
    report=tmp_path/'research_docs/STATE_DP_RESULTS.md'
    report.parent.mkdir();report.write_text('# Reviewed native success audit\n')
    root=tmp_path/'run';root.mkdir()
    (root/'summary.json').write_text(json.dumps(summary()))
    monkeypatch.setattr(delivery,'CODE',tmp_path);monkeypatch.setattr(delivery,'REPORT',report)
    result=delivery.finish(root,dict(publish=False,outside_report_sha256=delivery.report_hash(report.read_text())))
    assert result['state']=='completed_local' and result['result']['valid']==200
    assert report.read_text().startswith('# Reviewed native success audit\n')
    assert '终点条件成立但顺序未完成: 10' in report.read_text()
    assert json.loads((tmp_path/'research_docs/evidence/state_dp/random_delivery.json').read_text())['formal_software_missing']==0


def test_palm_evaluation_reuses_training_plateau_without_relaxation():
    from tools.tight_videos import plateau_evidence
    rules=dict(minimum_anchor_epochs=200,plateau_validations=4,plateau_relative_improvement=.01)
    def check(scores,epochs=200):
        return plateau_evidence([dict(metrics=dict(selection_score=s)) for s in scores],
            dict(anchor_equivalent_epochs=epochs),rules)
    assert check([.21,.209,.2085,.2082])['passed']
    assert not check([.21,.209,.2085,.2082],199.9)['passed']
    assert not check([.2240595,.2189849,.2157132,.213184])['passed']
    assert not check([.21,.209,.2085])['passed']
    assert not check([.21,.209,float('nan'),.2082])['passed']
    assert not check([.21,.209,.2085,float('inf')])['passed']
    assert not check([.1,.1,.3,.3,.3])['passed']


def test_palm_evaluation_counts_reject_interrupted_and_unstarted_denominators(tmp_path):
    from tools.runtime import write
    from tools.tight_videos import evaluation_counts
    write(tmp_path/'plan.json',[dict(identity=str(i)) for i in range(5)])
    for i in range(4):write(tmp_path/str(i)/'started.json',{})
    for i,legal,success in [(0,True,False),(1,True,True),(2,False,False)]:
        write(tmp_path/str(i)/'result.json',dict(software_complete=True,legal=legal,success=success,
            native_lift=legal,native_subgoals=1 if success else 0))
    write(tmp_path/'0/video_verified.json',{})
    assert evaluation_counts(tmp_path)==dict(planned=5,physical_attempts=4,completed=3,valid=2,
        successes=1,initial_rejections=1,incomplete_attempts=1,unstarted=1,success_rate=.5,
        native_lift=2,first_subgoal=1,videos_verified=1)


def test_palm_dependency_never_loads_checkpoint_before_original_gate(tmp_path):
    import json
    from tools.runtime import write
    from tools.tight_videos import await_training
    parent=tmp_path/'parent';root=tmp_path/'evaluation';root.mkdir()
    write(parent/'status.json',dict(state='BUDGET_STOP',stage='trained_and_offline_evaluated',owned_workers_exited=True))
    write(parent/'active_owner.json',{})
    write(parent/'training/progress.json',dict(update=100,anchor_equivalent_epochs=199))
    rows=[dict(metrics=dict(selection_score=s)) for s in [.21,.209,.2085,.2082]]
    (parent/'training/validation.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    rules=dict(minimum_anchor_epochs=200,plateau_validations=4,plateau_relative_improvement=.01)
    assert not await_training(root,dict(source_root=str(parent),wait_deadline=float('inf'),after_training=rules))
    assert json.loads((root/'status.json').read_text())['state']=='SKIPPED'
    assert not (root/'selected_checkpoint.json').exists()


def test_palm_dependency_requires_successful_parent(tmp_path):
    from tools.runtime import write
    from tools.tight_videos import await_training
    parent=tmp_path/'parent';root=tmp_path/'evaluation';root.mkdir()
    write(parent/'status.json',dict(state='FAILED',stage='offline'))
    write(parent/'training/progress.json',dict(update=100,anchor_equivalent_epochs=201))
    with pytest.raises(RuntimeError,match='predecessor failed'):
        await_training(root,dict(source_root=str(parent),wait_deadline=float('inf')))


def test_palm_dependency_pins_only_final_offline_selected_weights(tmp_path):
    import json
    from tools.runtime import write,sha,read
    from tools.tight_videos import await_training
    parent=tmp_path/'parent';root=tmp_path/'evaluation';root.mkdir()
    write(root/'inputs.json',{})
    write(parent/'status.json',dict(state='COMPLETE',stage='trained_and_offline_evaluated',owned_workers_exited=True))
    write(parent/'active_owner.json',{})
    write(parent/'training/progress.json',dict(update=20000,anchor_equivalent_epochs=236))
    write(parent/'training/finished.json',dict(updates=20000))
    write(parent/'training/configuration.json',{})
    (parent/'training/best.pt').write_bytes(b'opaque fixture; helper never deserializes checkpoint')
    digest=sha(parent/'training/best.pt')
    write(parent/'offline/result.json',dict(complete=True,episodes=38,selected_update=19500,checkpoint_sha256=digest))
    rows=[dict(metrics=dict(selection_score=s)) for s in [.21,.209,.2085,.2082]]
    (parent/'training/validation.jsonl').write_text('\n'.join(map(json.dumps,rows)))
    rules=dict(minimum_anchor_epochs=200,plateau_validations=4,plateau_relative_improvement=.01)
    cfg=dict(source_root=str(parent),checkpoint='training/best.pt',wait_deadline=float('inf'),after_training=rules)
    assert await_training(root,cfg)
    assert read(root/'selected_checkpoint.json')['updates']==19500
    assert read(root/'selected_checkpoint.json')['sha256']==digest
    (parent/'training/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):await_training(root,cfg)


def test_palm_waits_for_actual_owner_exit_even_after_terminal_status(tmp_path,monkeypatch):
    import os
    import tools.tight_videos as videos
    from tools.runtime import write,proc
    parent=tmp_path/'parent';root=tmp_path/'evaluation';root.mkdir()
    write(parent/'status.json',dict(state='COMPLETE',stage='trained_and_offline_evaluated',owned_workers_exited=True))
    write(parent/'active_owner.json',dict(owner=proc(os.getpid())))
    write(parent/'training/progress.json',dict(update=20000,anchor_equivalent_epochs=236))
    def waiting(seconds):
        assert seconds==30
        raise RuntimeError('CPU owner remains waiting')
    monkeypatch.setattr(videos.time,'sleep',waiting)
    with pytest.raises(RuntimeError,match='remains waiting'):
        videos.await_training(root,dict(source_root=str(parent),wait_deadline=float('inf')))
    assert not (root/'selected_checkpoint.json').exists()
