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
