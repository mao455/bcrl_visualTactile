"""Current-goal semantics, causal holding windows and real HDF5 replay."""
import copy
import time

import h5py
import numpy as np
import pytest
import torch

from visuotactile.takeover_trigger import (HELD_NEAR_GOAL_VERSION,CONTACT_HEIGHT_VERSION,
    NEVER_FALLBACK,make_trigger,observe_transition,validate_trigger)
from visuotactile.training_expansion import schedule_trigger,classify_case,validate_collection_plan


def config():return schedule_trigger('near60')


def observation(tick,*,distance=.2,stage_before=0,stage_after=0,goal_changed=False,contact=True,height=.53):
    angle=tick*.06
    pose=np.array([0.,0.,.5 if tick<48 else height,np.cos(angle/2),0.,0.,np.sin(angle/2)])
    palm=np.array([0.,0.,.5,1.,0.,0.,0.]);force=np.zeros((2,5,3));force[:,:2,2]=.2
    if not contact:force[1,1]=0
    goal=np.array([0.,0.,.6,1.,0.,0.,0.]);after=goal.copy()
    if goal_changed:after[0]+=.001
    return dict(diagnostics_after=dict(object_pose_w=pose,palm_body_pose_w=palm,goal_pose_w=after,successes=np.array(stage_after)),
        diagnostics_before=dict(goal_pose_w=goal,successes=np.array(stage_before)),
        native_keypoints_max_dist=np.array(distance),object_contact_force_w=force)


def stacked(rows,*,tensor=False):
    def stack(items):
        if isinstance(items[0],dict):return {key:stack([item[key] for item in items]) for key in items[0]}
        value=np.stack(items)
        return torch.from_numpy(value) if tensor else value
    return stack(rows)


def advance(trigger,tick,**kwargs):
    return observe_transition(trigger,tick,stacked([observation(tick,**kwargs)],tensor=True))


def test_holding_rotating_far_from_goal_waits_for_current_near_state_and_next_action():
    near=make_trigger(config(),NEVER_FALLBACK)
    early=make_trigger(dict(version=CONTACT_HEIGHT_VERSION,delay_ticks=0),NEVER_FALLBACK)
    for tick in range(80):
        assert advance(near,tick,distance=.060001)==None
        if tick<60:
            result=advance(early,tick)
            assert result==(60 if tick==59 else None)
    assert near.first_contact_height_completed_tick==60
    assert advance(near,80,distance=.06)==81
    receipt=near.receipt()
    assert receipt['near_goal_at_trigger']['native_keypoint_max_distance_m']==.06
    assert receipt['first_held_near_goal_window_end_completed_tick']==81
    assert receipt['future_information_used'] is False and receipt['native_success_changed'] is False
    with pytest.raises(ValueError,match='exactly once'):advance(near,81,distance=.01)


@pytest.mark.parametrize('bad', ['contact','height','goal_pose','goal_stage'])
def test_near_distance_cannot_bypass_complete_hold_or_current_goal(bad):
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(59):assert advance(trigger,tick) is None
    options=dict(distance=.01)
    options.update({'contact':dict(contact=False),'height':dict(height=.519),
        'goal_pose':dict(goal_changed=True),'goal_stage':dict(stage_after=1)}[bad])
    assert advance(trigger,59,**options) is None
    if bad in ('contact','height'):
        for tick in range(60,71):assert advance(trigger,tick,distance=.01) is None
        assert advance(trigger,71,distance=.01)==72
    else:
        assert trigger.receipt()['target_change_transitions']==1
        assert advance(trigger,60,distance=.01,stage_before=1,stage_after=1)==61
        assert trigger.receipt()['near_goal_at_trigger']['active_goal_index']==1


def test_no_near_state_means_no_takeover_even_after_original_timeout_and_bad_config_rejects():
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(601):assert advance(trigger,tick,distance=.061) is None
    assert trigger.scheduled_tick==NEVER_FALLBACK
    assert trigger.receipt()['first_held_near_goal_window_end_completed_tick'] is None
    for changed,fallback in [(dict(config(),delay_ticks=12),NEVER_FALLBACK),
        (dict(config(),max_keypoint_distance_m=.08),NEVER_FALLBACK),(config(),360),
        (dict(config(),max_keypoint_distance_m=float('nan')),NEVER_FALLBACK)]:
        with pytest.raises(ValueError,match='60mm'):validate_trigger(changed,fallback)


def test_per_lane_current_transition_matches_saved_hdf5_replay_and_rejects_goal_tamper(tmp_path):
    from tools.tight_fresh_report import audit_trigger_case
    from visuotactile.batched_controller import lane_tree
    triggers=[make_trigger(config(),NEVER_FALLBACK) for _ in range(2)]
    observations=[[],[]];switches=[None,None]
    for tick in range(72):
        rows=[observation(tick,distance=.05 if tick>=60+lane*5 else .2) for lane in range(2)]
        transition=stacked(rows,tensor=True)
        for lane in range(2):
            observations[lane].append(rows[lane])
            if switches[lane] is None:
                switches[lane]=observe_transition(triggers[lane],tick,lane_tree(transition,lane,2))
    assert switches==[61,66]
    for lane in range(2):
        directory=tmp_path/f'lane{lane}';directory.mkdir()
        values=stacked(observations[lane]);values['dagger']=dict(control_source=(np.arange(72)>=switches[lane]).astype(int))
        def save(group,tree):
            for key,value in tree.items():
                if isinstance(value,dict):save(group.create_group(key),value)
                else:group[key]=value
        with h5py.File(directory/'numeric.h5','w') as stream:save(stream.create_group('episode/transition'),values)
        result=dict(legal=True,software_complete=True,dagger=dict(configuration=dict(trigger=config(),takeover_tick=NEVER_FALLBACK),
            trigger_receipt=triggers[lane].receipt(),recovery_coverage=dict(switch_tick=switches[lane]),
            expert_ticks=72-switches[lane],expert_valid_ticks=72-switches[lane]-3,
            expert_takeover_success=True,student_autonomous_success=False))
        audit=audit_trigger_case(directory,result)
        assert audit['category']=='student_held_near_goal_then_takeover'
        case=dict(expansion_schedule='near60',spec=dict(takeover_tick=NEVER_FALLBACK))
        assert classify_case(case,result,audit)=='expert_rescue_after_held_near_goal'
        result['dagger']['expert_takeover_success']=False
        assert classify_case(case,result,audit)=='expert_failed_after_held_near_goal'
        with h5py.File(directory/'numeric.h5','a') as stream:
            stream['episode/transition/diagnostics_after/goal_pose_w'][switches[lane]-1,0]=1.
        with pytest.raises(ValueError,match='cannot be reproduced'):audit_trigger_case(directory,result)


def test_expansion_prepare_registers_near60_and_rejects_threshold_or_fallback_drift(setup_registry,monkeypatch):
    from tools.tight_fresh_expansion import prepare
    from test_training_expansion import write
    setup=setup_registry
    write(setup['source']/'session.json',dict(deadline=time.time()+1000,domain=setup['domain'],expert_sha256='e'))
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair',lambda root,role:dict(role=role))
    output=setup['root']/'near_prepared'
    plan=prepare(setup['source'],output,session_root=setup['source'],registry_id='near_fixture',seeds=[6000,6001],
        partition_manifest=setup['partition'],frozen_families=setup['families'],schedules=['fixed40','near60','contact0'],
        existing_identities=['old'],lanes=8,event_seed=8000,round_index=8,
        verification_workers=2,verification_max_pending=4)
    validate_collection_plan(plan)
    cases=[case for case in plan['cases'] if case['expansion_schedule']=='near60']
    assert len(cases)==2 and all(case['dagger']['trigger']==config() for case in cases)
    assert plan['batch']['verification_workers']==2 and plan['batch']['verification_max_pending']==4
    for field,value in [('threshold',.08),('fallback',360)]:
        changed=copy.deepcopy(plan);case=next(c for c in changed['cases'] if c['expansion_schedule']=='near60')
        if field=='threshold':case['dagger']['trigger']['max_keypoint_distance_m']=value
        else:case['spec']['takeover_tick']=value
        with pytest.raises(ValueError,match='trigger or acquired a fallback'):validate_collection_plan(changed)
    assert not (output/'run_started.json').exists()


# This imports a fixture only; its factory opens partition metadata, not held-out trajectories.
from test_training_expansion import setup_registry
