import pytest

from tools.narrow_repair_pipeline import select_speed, cohort_results, capture_classification, pending_production


def sample(family, alpha, success=True, duration=6., speed=.6):
    return dict(family=family,alpha=alpha,success=success,visual_passed=True,duration_s=duration,
        motion=dict(phases={'contact_low_object':dict(seconds=duration/2,arm_actual_speed_rms=speed)}))


def test_real_slowing_chooses_tested_continuous_interval():
    rows=[sample(i,1.,duration=3.,speed=1.) for i in range(3)]
    rows += [sample(i,a) for i in range(3) for a in (.55,.60,.65)]
    result=select_speed(rows)
    assert result['passed'] and result['selected']['interval'] in ([.55,.60],[.60,.65])


def test_waiting_alone_or_failed_pilot_does_not_admit_production():
    rows=[sample(i,1.,duration=3.,speed=1.) for i in range(3)]
    rows += [sample(i,a,speed=1.) for i in range(3) for a in (.55,.60,.65)]
    assert not select_speed(rows)['passed']
    for r in rows:
        if r['alpha']<1:r['motion']['phases']['contact_low_object']['arm_actual_speed_rms']=.5
        if r['family']==0 and r['alpha']<1:r['success']=False
    assert not select_speed(rows)['passed']


def test_preaction_software_failure_is_not_legal_task_denominator():
    row=dict(split='train',legal=True,success=False,software_complete=False)
    result=cohort_results([row])['train']
    assert result['attempts']==1 and result['valid']==0 and result['software_failures']==1


def test_fallen_object_out_of_view_is_a_task_failure_not_a_broken_camera():
    result=dict(identity='fallen',software_complete=True,legal=True,success=False)
    visual=dict(passed=True,sampled_target_projection=[dict(views=[dict(in_frame=[False,True]),dict(in_frame=[True,True])])])
    c=capture_classification(result,visual)
    assert c['task_failure'] and not c['successful_demonstration'] and not c['fixed_target_visible']
    with pytest.raises(RuntimeError,match='Successful capture'):
        capture_classification(dict(result,success=True),visual)
    with pytest.raises(RuntimeError,match='RGB-D content'):
        capture_classification(result,dict(visual,passed=False))


def test_pending_suffix_never_repeats_or_replaces_failed_attempt():
    jobs=[dict(identity=x) for x in ('a','b','c')]
    results=[dict(identity='pilot'),dict(identity='a',success=True),dict(identity='b',success=False)]
    assert pending_production(jobs,results)==[dict(identity='c')]
    for bad in ([dict(identity='b')],[dict(identity='a'),dict(identity='a')],[dict(identity='a'),dict(identity='c')]):
        with pytest.raises(ValueError):pending_production(jobs,bad)
