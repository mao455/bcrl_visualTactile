"""Small reproducible public receipts from actual fresh takeover artifacts."""
import argparse
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read


def audit_trigger_case(directory, result):
    """Recompute scheduling solely from saved completed student transitions."""
    import h5py
    import numpy as np
    from visuotactile.takeover_trigger import make_trigger, CONTACT_HEIGHT_VERSION
    dagger=result.get('dagger',{})
    config=dagger.get('configuration',{})
    if config.get('trigger') is None: return None
    trigger=make_trigger(config['trigger'], config['takeover_tick'])
    with h5py.File(directory/'numeric.h5') as f:
        t=f['episode/transition'];source=t['dagger/control_source'][:]
        for tick in np.flatnonzero(source==0):
            trigger.observe(int(tick),t['diagnostics_after/object_pose_w'][tick],
                t['diagnostics_after/palm_body_pose_w'][tick],t['object_contact_force_w'][tick])
    if trigger.receipt()!=dagger['trigger_receipt']:
        raise ValueError('Recorded causal takeover trigger cannot be reproduced')
    actual=dagger['recovery_coverage']['switch_tick']
    if actual is not None and actual!=trigger.scheduled_tick:
        raise ValueError('Actual authority switch differs from causal schedule')
    if config['trigger']['version'] == CONTACT_HEIGHT_VERSION:
        category=('ended_before_takeover' if actual is None else
            'student_contact_height_then_takeover' if trigger.reason=='completed_student_contact_height_window_plus_delay' else
            'fallback_before_requested_delay' if trigger.first_contact_height_completed_tick is not None else
            'fallback_without_contact_height')
    else:
        category=('ended_before_takeover' if actual is None else
            'student_stable_then_takeover' if trigger.reason=='completed_student_stable_window_plus_delay' else
            'fallback_before_requested_delay' if trigger.first_stable_completed_tick is not None else
            'fallback_without_stable_grasp')
    return dict(passed=True,category=category,receipt=trigger.receipt(),actual_switch_tick=actual,
        numeric_sha256=sha256(directory/'numeric.h5'),student_input=False)


def takeover(collection,output):
    result=read(collection/'result.json');plan=read(collection/'plan.json')
    if not result.get('software_complete'):raise ValueError('Collection is incomplete')
    rows=read(collection/'results.json');entries=read(collection/'correction_entries.json')
    planned={c['name'] for c in plan['cases']}
    if len(rows)!=len(planned) or {x['case'] for x in rows}!=planned:raise ValueError('Opportunity accounting differs')
    cases=[]
    for row in rows:
        x=row['result'];accepted=bool(x.get('accepted_for_training'))
        if accepted!=(row['case'] in entries):raise ValueError('Retained subset differs')
        if x.get('legal') and not read(collection/row['case']/'capture_audit.json')['passed']:
            raise ValueError('Capture audit failed')
        trigger=audit_trigger_case(collection/row['case'],x)
        if trigger is not None: atomic_json(collection/row['case']/'trigger_audit.json',trigger)
        dagger=x.get('dagger',{})
        cases.append(dict(case=row['case'],role=row['role'],
            takeover_tick=dagger.get('recovery_coverage',{}).get('switch_tick'),trigger=trigger,
            expert_takeover_success=dagger.get('expert_takeover_success',False),
            legal=x.get('legal'),software_complete=x.get('software_complete'),success=x.get('success'),
            accepted=accepted,steps=x.get('steps'),reasons=x.get('reasons'),native_subgoals=x.get('native_subgoals'),
            contact=x.get('contact'),actual_exposure=x.get('actual_exposure'),
            frontend_sha256=plan['pairs'][row['role']]['frontend_sha256'],
            backend_sha256=plan['pairs'][row['role']]['backend_sha256'],
            capture_sha256=entries[row['case']]['source_sha256'] if accepted else None))
    retained=[c for c in cases if c['accepted']]
    counts={role:dict(opportunities=sum(c['role']==role for c in cases),
        legal=sum(c['role']==role and bool(c['legal']) for c in cases),
        expert_successes=sum(c['role']==role and bool(c['expert_takeover_success']) for c in cases),
        trigger_categories={kind:sum(c['role']==role and c['trigger'] is not None and c['trigger']['category']==kind for c in cases)
            for kind in ['student_stable_then_takeover','student_contact_height_then_takeover',
                'fallback_without_stable_grasp','fallback_without_contact_height',
                'fallback_before_requested_delay','ended_before_takeover']},
        accepted=sum(c['role']==role for c in retained)) for role in ['A','B']}
    exposure={key:sum(c['actual_exposure'][key] for c in retained)
        for key in ['history_ticks','supervised_ticks','supervised_anchors','supervised_window_targets']}
    atomic_json(output,dict(source_commit=plan['source_commit'],plan_sha256=sha256(collection/'plan.json'),
        results_sha256=sha256(collection/'results.json'),entries_sha256=sha256(collection/'correction_entries.json'),
        counts=counts,accepted=len(retained),retained_exposure=exposure,cases=cases,test_used=False,
        meaning='Actual expert rescues after complete student prefixes; not autonomous student successes. All failed attempts retained locally.',
        learning_exposure='Source population only; optimizer exposure is recorded by the following trainers.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();takeover(a.collection,a.output)

if __name__=='__main__':main()
