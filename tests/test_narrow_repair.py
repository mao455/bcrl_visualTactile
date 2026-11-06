from tools.narrow_repair_pipeline import select_speed, cohort_results


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
