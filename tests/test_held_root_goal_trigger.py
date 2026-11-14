"""Current rigid-root proximity is causal collection metadata, never task success."""
import copy
import time

import h5py
import numpy as np
import pytest

from visuotactile.takeover_trigger import (HELD_ROOT_GOAL_VERSION,NEVER_FALLBACK,
    make_trigger,observe_transition,validate_trigger)
from visuotactile.training_expansion import schedule_trigger,classify_case,validate_collection_plan
from test_held_near_goal_trigger import observation,stacked
from test_training_expansion import setup_registry


def config():return schedule_trigger('held_root60')


def root_observation(tick,*,root_distance=.1,**kwargs):
    value=observation(tick,**kwargs)
    # World rigid roots, not COM or sensor/body-local positions. Include a wrong
    # pre-step object position to detect accidental use of diagnostics_before.
    value['diagnostics_before']['object_pose_w']=np.array([10.,0.,.5,1.,0.,0.,0.])
    for group in ('diagnostics_before','diagnostics_after'):
        value[group]['goal_pose_w'][0]+=root_distance
        value[group]['goal_pose_w'][2]=value['diagnostics_after']['object_pose_w'][2]
    return value


def advance(trigger,tick,**kwargs):
    return observe_transition(trigger,tick,stacked([root_observation(tick,**kwargs)],tensor=True))


def test_rotating_held_root_near_state_triggers_despite_wrong_keypoints_and_angle():
    candidate=make_trigger(config(),NEVER_FALLBACK)
    original=make_trigger(schedule_trigger('near60'),NEVER_FALLBACK)
    for tick in range(80):
        value=stacked([root_observation(tick,root_distance=.060001,distance=.2)],tensor=True)
        assert observe_transition(candidate,tick,value) is None
        assert observe_transition(original,tick,value) is None
    assert candidate.first_contact_height_completed_tick==60
    value=stacked([root_observation(80,root_distance=.06,distance=.2)],tensor=True)
    assert observe_transition(candidate,80,value)==81
    assert observe_transition(original,80,value) is None
    receipt=candidate.receipt();boundary=receipt['root_goal_at_trigger']
    assert boundary['object_goal_root_distance_m']==.06
    assert boundary['native_keypoint_max_distance_m']==.2
    assert boundary['object_goal_quaternion_angle_rad']>1.
    assert receipt['orientation_threshold'] is None
    assert receipt['root_proximity_is_native_near_or_alignment'] is False
    assert receipt['student_inputs_changed'] is False and receipt['native_success_changed'] is False
    assert receipt['first_held_root_near_goal_window_end_completed_tick']==81
    with pytest.raises(ValueError,match='exactly once'):advance(candidate,81,root_distance=0.)


@pytest.mark.parametrize('bad',['contact_first','contact_second','height','goal_pose','goal_stage'])
def test_root_proximity_cannot_bypass_hold_or_current_target_consistency(bad):
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(59):assert advance(trigger,tick) is None
    row=root_observation(59,root_distance=.01,goal_changed=bad=='goal_pose',stage_after=int(bad=='goal_stage'),
        height=.519 if bad=='height' else .53)
    if bad.startswith('contact'):row['object_contact_force_w'][0 if bad=='contact_first' else 1,1]=0
    assert observe_transition(trigger,59,stacked([row],tensor=True)) is None
    if bad in ('contact_first','contact_second','height'):
        for tick in range(60,71):assert advance(trigger,tick,root_distance=.01) is None
        assert advance(trigger,71,root_distance=.01)==72
    else:
        assert trigger.receipt()['target_change_transitions']==1
        assert advance(trigger,60,root_distance=.01,stage_before=1,stage_after=1)==61


def test_no_reference_or_far_root_never_uses_future_information_or_timeout_fallback():
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(601):assert advance(trigger,tick,root_distance=.060001,distance=0.) is None
    assert trigger.receipt()['first_held_root_near_goal_window_end_completed_tick'] is None
    assert trigger.scheduled_tick==NEVER_FALLBACK
    assert trigger.receipt()['completed_student_transitions_observed']==601


@pytest.mark.parametrize('changed,fallback',[
    (dict(delay_ticks=1),NEVER_FALLBACK),(dict(max_root_distance_m=.08),NEVER_FALLBACK),
    (dict(max_root_distance_m=True),NEVER_FALLBACK),(dict(max_root_distance_m=float('nan')),NEVER_FALLBACK),
    (dict(max_quaternion_angle_rad=.3),NEVER_FALLBACK),({},360)])
def test_strict_registered_profile_cannot_relax_distance_or_add_orientation_filter(changed,fallback):
    with pytest.raises(ValueError,match='60mm'):validate_trigger(dict(config(),**changed),fallback)


def test_two_lanes_match_singleton_and_closed_hdf5_replay_without_cross_lane_state(tmp_path):
    from tools.tight_fresh_report import audit_trigger_case
    from visuotactile.batched_controller import lane_tree
    triggers=[make_trigger(config(),NEVER_FALLBACK) for _ in range(2)]
    single=[make_trigger(config(),NEVER_FALLBACK) for _ in range(2)]
    switches=[None,None];recorded=[[],[]]
    for tick in range(72):
        rows=[root_observation(tick,root_distance=.01 if tick>=60+5*lane else .1,distance=.2) for lane in range(2)]
        transition=stacked(rows,tensor=True)
        for lane in range(2):
            recorded[lane].append(rows[lane])
            if switches[lane] is None:
                actual=observe_transition(triggers[lane],tick,lane_tree(transition,lane,2))
                assert actual==observe_transition(single[lane],tick,stacked([rows[lane]],tensor=True))
                switches[lane]=actual
    assert switches==[61,66]
    for lane in range(2):
        assert triggers[lane].receipt()==single[lane].receipt()
        directory=tmp_path/f'lane{lane}';directory.mkdir()
        tree=stacked(recorded[lane]);tree['dagger']=dict(control_source=(np.arange(72)>=switches[lane]).astype(int))
        def save(group,values):
            for key,value in values.items():
                if isinstance(value,dict):save(group.create_group(key),value)
                else:group[key]=value
        with h5py.File(directory/'numeric.h5','w') as stream:save(stream.create_group('episode/transition'),tree)
        result=dict(legal=True,software_complete=True,dagger=dict(configuration=dict(trigger=config(),takeover_tick=NEVER_FALLBACK),
            trigger_receipt=triggers[lane].receipt(),recovery_coverage=dict(switch_tick=switches[lane]),
            expert_ticks=72-switches[lane],expert_valid_ticks=72-switches[lane]-3,
            expert_takeover_success=True,student_autonomous_success=False))
        audit=audit_trigger_case(directory,result)
        assert audit['category']=='student_held_root_near_goal_then_takeover'
        case=dict(expansion_schedule='held_root60',spec=dict(takeover_tick=NEVER_FALLBACK),dagger=dict(trigger=config()))
        assert classify_case(case,result,audit)=='expert_rescue_after_held_root_near_goal'
        from tools.tight_fresh_batch_run import classify_capture
        from visuotactile.training_expansion import COLLECTION_CONTRACT
        audit_name,category,formal=classify_capture(directory,case,result,dict(stage_takeover_contract=COLLECTION_CONTRACT))
        assert audit_name=='training_expansion_audit.json' and formal==audit
        assert category=='expert_rescue_after_held_root_near_goal'
        result['dagger']['expert_takeover_success']=False
        assert classify_case(case,result,audit)=='expert_failed_after_held_root_near_goal'
        with h5py.File(directory/'numeric.h5','a') as stream:
            stream['episode/transition/diagnostics_after/object_pose_w'][switches[lane]-1,0]=5.
        with pytest.raises(ValueError,match='cannot be reproduced'):audit_trigger_case(directory,result)


def test_terminal_qualification_without_another_action_is_never_an_expert_recovery():
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(60):
        switch=advance(trigger,tick,root_distance=.01 if tick==59 else .1)
    assert switch==60
    audit=dict(receipt=trigger.receipt(),actual_switch_tick=None,category='ended_before_takeover')
    result=dict(legal=True,software_complete=True,dagger=dict(expert_ticks=0,expert_valid_ticks=0,
        expert_takeover_success=False,student_autonomous_success=False))
    case=dict(expansion_schedule='held_root60')
    assert classify_case(case,result,audit)=='ended_after_held_root_near_goal_before_scheduled_takeover'


def test_expansion_registers_both_root60_and_original_near60_before_any_run(setup_registry,monkeypatch):
    from tools.tight_fresh_expansion import prepare
    from test_training_expansion import write
    setup=setup_registry
    write(setup['source']/'session.json',dict(deadline=time.time()+1000,domain=setup['domain'],expert_sha256='e'))
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair',lambda root,role:dict(role=role))
    output=setup['root']/'root_prepared'
    plan=prepare(setup['source'],output,session_root=setup['source'],registry_id='root_fixture',seeds=[6000,6001],
        partition_manifest=setup['partition'],frozen_families=setup['families'],schedules=['near60','held_root60','fixed40'],
        existing_identities=['old'],lanes=8,event_seed=8000,round_index=10)
    validate_collection_plan(plan)
    assert len([c for c in plan['cases'] if c['expansion_schedule']=='held_root60'])==2
    for case in plan['cases']:
        if case['expansion_schedule']=='near60':assert case['dagger']['trigger']==schedule_trigger('near60')
        elif case['expansion_schedule']=='held_root60':assert case['dagger']['trigger']==config()
    for field in ('threshold','fallback'):
        changed=copy.deepcopy(plan);case=next(c for c in changed['cases'] if c['expansion_schedule']=='held_root60')
        if field=='threshold':case['dagger']['trigger']['max_root_distance_m']=.08
        else:case['spec']['takeover_tick']=100
        with pytest.raises(ValueError,match='trigger or acquired a fallback'):validate_collection_plan(changed)
    assert not (output/'run_started.json').exists()
