"""Retained historical decision/compatibility helpers for existing controls.

Collection, training and publication owners are retired; use tools.tight_finish.
Full historical implementation remains in Git b85738e.
"""
from collections import Counter


def select_speed(rows):
    """Adjacent tested endpoints, real slowing, all three starts successful."""
    import numpy as np
    native={r['family']:r for r in rows if r['alpha']==1.}
    candidates=[]
    for low,high in ((.55,.60),(.60,.65)):
        chosen=[r for r in rows if r['alpha'] in (low,high)]
        if len(chosen)!=6 or not all(r['success'] and r['visual_passed'] for r in chosen):continue
        ratios=[r['duration_s']/native[r['family']]['duration_s'] for r in chosen]
        # Contact motion is from actual measured velocity, not timestamps or a
        # trailing idle interval. Proxy phases are disclosed in each report.
        speed_ratios=[];contact_ratios=[]
        for r in chosen:
            for phase in ('contact_low_object','native_alignment','second_insertion'):
                a=r['motion']['phases'].get(phase);b=native[r['family']]['motion']['phases'].get(phase)
                if a and b and b['arm_actual_speed_rms']>1e-4:
                    speed_ratios.append(a['arm_actual_speed_rms']/b['arm_actual_speed_rms'])
                    contact_ratios.append(a['seconds']/b['seconds'])
        if not all(np.isfinite(r) and 1.5<=r<=3. for r in ratios) or not speed_ratios or np.median(speed_ratios)>.95:continue
        candidates.append(dict(interval=[low,high],successes=6,attempts=6,
            median_duration_s=float(np.median([r['duration_s'] for r in chosen])),
            median_duration_ratio=float(np.median(ratios)),median_contact_arm_speed_ratio=float(np.median(speed_ratios)),
            median_contact_duration_ratio=float(np.median(contact_ratios)),
            scope='Exploratory 3-family paired screen, not a population reliability estimate'))
    if not candidates:return dict(passed=False,reason='No adjacent alpha endpoints show reliable real slowing in this bounded pilot')
    chosen=min(candidates,key=lambda c:abs(c['median_duration_s']-6.))
    return dict(passed=True,selected=chosen,candidates=candidates,policy='One stable alpha per episode, continuous uniform between tested endpoints')


def cohort_results(rows):
    from visuotactile.narrow_decision import failure_stage
    return {split:dict(attempts=len(rr),valid=sum(r['legal'] and r['software_complete'] for r in rr),successes=sum(r['success'] for r in rr),
        software_failures=sum(not r['software_complete'] for r in rr),
        initial_rejections=sum(not r['legal'] for r in rr),failures=dict(Counter(failure_stage(r) for r in rr if r['software_complete'] and not r['success'])))
        for split in ('train','unseen') for rr in [[r for r in rows if r['split']==split]]}


def capture_classification(result, visual):
    """Sensor correctness and natural task outcome are separate gates.

    A falling object may leave a camera. Keep that failed trajectory as failure
    evidence, never as a successful demonstration, without calling it a broken
    sensor. Successful demonstrations retain the same target-visibility gate.
    """
    if not result['software_complete'] or not result['legal']:
        raise RuntimeError('Incomplete capture or initial rejection requires review')
    if not visual['passed']:
        raise RuntimeError('RGB-D content gate failed: '+result['identity'])
    visible=all(all(v['in_frame']) for sample in visual['sampled_target_projection'] for v in sample['views'][:2])
    if result['success'] and not visible:
        raise RuntimeError('Successful capture target region left fixed frame: '+result['identity'])
    return dict(task_failure=not result['success'],fixed_target_visible=visible,
        successful_demonstration=result['success'],sensor_content_passed=True)


def pending_production(jobs, results):
    """Continue exactly the unexecuted suffix, including failures in the prefix."""
    planned=[j['identity'] for j in jobs]
    done=[r['identity'] for r in results if r['identity'] in set(planned)]
    if done!=planned[:len(done)]:raise ValueError('Production results are not the frozen plan prefix')
    return jobs[len(done):]
