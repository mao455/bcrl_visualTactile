"""Current held-window admission, real authority timing, and saved-data replay."""
import copy
import json
import time

import h5py
import numpy as np
import pytest
import torch

from visuotactile.takeover_trigger import HELD_RECHECK_VERSION,NEVER_FALLBACK,make_trigger,observe_transition,validate_trigger
from visuotactile.training_expansion import schedule_trigger,classify_case,validate_collection_plan
from test_held_near_goal_trigger import observation,stacked,advance
from test_training_expansion import setup_registry,write


def config(delay=60):return schedule_trigger('held'+str(delay))


@pytest.mark.parametrize('delay',[0,60,120])
def test_elapsed_time_never_schedules_early_and_current_rotating_window_switches_next_action(delay):
    trigger=make_trigger(config(delay),NEVER_FALLBACK)
    for tick in range(60+delay):
        # No distance field exists: this rule cannot accidentally use proximity.
        row=observation(tick);del row['native_keypoints_max_dist']
        switch=observe_transition(trigger,tick,stacked([row],tensor=True))
        if tick<59+delay:
            assert switch is None and trigger.scheduled_tick==NEVER_FALLBACK
        else:assert switch==60+delay
    receipt=trigger.receipt()
    assert receipt['first_contact_height_window_end_completed_tick']==60
    assert receipt['earliest_allowed_completed_tick']==60+delay
    assert receipt['first_rechecked_contact_height_window_end_completed_tick']==60+delay
    assert receipt['latest_contact_height_window'] is True
    assert receipt['no_expert_fallback'] and not receipt['native_success_changed']
    with pytest.raises(ValueError,match='exactly once'):advance(trigger,60+delay)


@pytest.mark.parametrize('defect',['contact','height'])
def test_loss_at_due_time_requires_twelve_fresh_qualifying_transitions(defect):
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(132):
        options=({'contact':False} if defect=='contact' else {'height':.519}) if tick==119 else {}
        switch=advance(trigger,tick,**options)
        assert switch==(132 if tick==131 else None)
        if tick==119:
            assert trigger.scheduled_tick==NEVER_FALLBACK
            assert trigger.receipt()['reason']=='minimum_delay_elapsed_without_current_contact_height_window'
    receipt=trigger.receipt()
    assert receipt['earliest_allowed_completed_tick']==120 and receipt['actual_scheduled_takeover_tick']==132
    assert receipt['nonheld_windows_after_first']==12


def test_intervening_contact_loss_does_not_reset_elapsed_time_or_require_rigid_grasp():
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(120):assert advance(trigger,tick,contact=tick!=70)==(120 if tick==119 else None)
    assert trigger.receipt()['nonheld_windows_after_first']==12
    assert trigger.first_contact_height_completed_tick==60


def test_native_goal_transition_is_recorded_but_is_not_a_distance_gate_or_success():
    trigger=make_trigger(config(),NEVER_FALLBACK)
    for tick in range(120):
        options=dict(stage_before=0 if tick<=90 else 1,stage_after=0 if tick<90 else 1,goal_changed=tick==90)
        if tick==119:options.update(stage_before=1,stage_after=2,goal_changed=True)
        assert advance(trigger,tick,distance=float('nan'),**options)==(120 if tick==119 else None)
    receipt=trigger.receipt()
    assert receipt['first_contact_height_goal']['index_after']==0
    assert receipt['latest_goal']['index_before']==1 and receipt['latest_goal']['index_after']==2
    assert receipt['target_change_transitions']==2 and receipt['cross_goal_wait_allowed']
    assert receipt['native_success_changed'] is False and receipt['future_information_used'] is False


@pytest.mark.parametrize('bad',[dict(version=HELD_RECHECK_VERSION,min_delay_ticks=12),
    dict(version=HELD_RECHECK_VERSION,min_delay_ticks=True),dict(version=HELD_RECHECK_VERSION,min_delay_ticks=60.),
    dict(version=HELD_RECHECK_VERSION,delay_ticks=60),dict(config(),max_keypoint_distance_m=.06)])
def test_explicit_version_rejects_ambiguous_or_mutable_contract(bad):
    with pytest.raises(ValueError,match='minimum0/60/120'):validate_trigger(bad,NEVER_FALLBACK)


def test_no_fallback_and_no_duplicate_skip_or_nonfinite_observation():
    with pytest.raises(ValueError,match='no fallback'):make_trigger(config(),601)
    trigger=make_trigger(config(),NEVER_FALLBACK)
    with pytest.raises(ValueError,match='exactly once'):advance(trigger,1)
    advance(trigger,0)
    with pytest.raises(ValueError,match='exactly once'):advance(trigger,0)
    with pytest.raises(ValueError,match='Finite'):advance(trigger,1,height=float('nan'),stage_after=float('nan'))
    assert len(trigger.poses)==1


def write_capture(directory,rows,trigger,switch,*,success=False):
    directory.mkdir()
    values=stacked(rows);values['dagger']=dict(control_source=(np.arange(len(rows))>=switch).astype(int)
        if switch is not None else np.zeros(len(rows),int))
    def save(group,tree):
        for key,value in tree.items():
            if isinstance(value,dict):save(group.create_group(key),value)
            else:group[key]=value
    with h5py.File(directory/'numeric.h5','w') as stream:save(stream.create_group('episode/transition'),values)
    count=0 if switch is None else len(rows)-switch
    return dict(legal=True,software_complete=True,reasons=dict(timeout=not success,dropped=False,fall=False),
        dagger=dict(configuration=dict(trigger=trigger.configuration,takeover_tick=NEVER_FALLBACK),
            trigger_receipt=trigger.receipt(),recovery_coverage=dict(switch_tick=switch),
            expert_ticks=count,expert_valid_ticks=max(0,count-3),expert_takeover_success=bool(success and switch is not None),
            student_autonomous_success=bool(success and switch is None)))


@pytest.mark.parametrize('ending',['never_held','before_minimum','without_recheck','same_tick_terminal','true_student_success'])
def test_native_end_never_creates_fallback_or_conflates_goal_success_and_takeover(tmp_path,ending):
    from tools.tight_fresh_report import audit_trigger_case
    n=110 if ending=='before_minimum' else 120 if ending in ('same_tick_terminal','true_student_success') else 601
    trigger=make_trigger(config(),NEVER_FALLBACK);rows=[]
    for tick in range(n):
        contact=False if ending=='never_held' or ending=='without_recheck' and tick>=90 else True
        rows.append(observation(tick,contact=contact,stage_before=int(tick>80),stage_after=int(tick>=80)))
        observe_transition(trigger,tick,stacked([rows[-1]]))
    result=write_capture(tmp_path/'case',rows,trigger,None,success=ending=='true_student_success')
    audit=audit_trigger_case(tmp_path/'case',result)
    actual=classify_case(dict(expansion_schedule='held60'),result,audit)
    expected=dict(never_held='ended_without_student_contact_height_window',
        before_minimum='ended_after_first_held_before_minimum_delay',
        without_recheck='ended_after_minimum_delay_without_current_held_window',
        same_tick_terminal='ended_after_held_recheck_before_scheduled_takeover',
        true_student_success='student_success_without_takeover')
    assert actual==expected[ending] and result['dagger']['expert_ticks']==0
    assert audit['actual_switch_tick'] is None and 'dropped' not in actual


def test_shared_singleton_batch_and_saved_audit_agree_without_reading_expert_future(tmp_path):
    from tools.tight_fresh_report import audit_trigger_case
    from visuotactile.batched_controller import lane_tree
    singles=[make_trigger(config(delay),NEVER_FALLBACK) for delay in (60,120)]
    batches=[make_trigger(config(delay),NEVER_FALLBACK) for delay in (60,120)]
    observations=[[],[]];switches=[None,None]
    for tick in range(190):
        rows=[observation(tick,contact=not(lane==0 and 119<=tick<124)) for lane in range(2)]
        transition=stacked(rows,tensor=True)
        for lane in range(2):
            observations[lane].append(rows[lane])
            if switches[lane] is None:
                a=observe_transition(singles[lane],tick,stacked([rows[lane]],tensor=True))
                b=observe_transition(batches[lane],tick,lane_tree(transition,lane,2))
                assert a==b and singles[lane].receipt()==batches[lane].receipt()
                switches[lane]=a
    assert switches==[136,180]
    for lane in range(2):
        directory=tmp_path/f'lane{lane}';result=write_capture(directory,observations[lane],batches[lane],switches[lane],success=True)
        audit=audit_trigger_case(directory,result)
        assert audit['category']=='student_held_recheck_then_takeover'
        case=dict(expansion_schedule='held'+str((60,120)[lane]))
        assert classify_case(case,result,audit)=='expert_rescue_after_held_recheck'
        result['dagger']['expert_takeover_success']=False
        assert classify_case(case,result,audit)=='expert_failed_after_held_recheck'
        with h5py.File(directory/'numeric.h5','a') as stream:
            stream['episode/transition/diagnostics_after/object_pose_w'][switches[lane]:]=float('nan')
        replay=audit_trigger_case(directory,result)
        assert replay['numeric_sha256']!=audit['numeric_sha256']
        assert {k:v for k,v in replay.items() if k!='numeric_sha256'}=={k:v for k,v in audit.items() if k!='numeric_sha256'}
        with h5py.File(directory/'numeric.h5','a') as stream:
            stream['episode/transition/object_contact_force_w'][switches[lane]-1,1,1]=0
        with pytest.raises(ValueError,match='cannot be reproduced'):audit_trigger_case(directory,result)


@pytest.mark.parametrize('delays',[(0,60),(60,120)])
def test_real_serial_and_batch_bridges_switch_only_on_next_action_and_mask_queue_flush(delays):
    from test_batched_position_bridge import environment,serial_environment,native_pipeline
    from visuotactile.batched_position_bridge import BatchedExpertPositionBridge
    from visuotactile.position_control import ExpertPositionBridge
    env=environment(2);single_env=[serial_environment(env,i) for i in range(2)]
    batch=BatchedExpertPositionBridge(env,[NEVER_FALLBACK]*2,ExpertPositionBridge.speed_scheme,pipeline=native_pipeline())
    singles=[ExpertPositionBridge(e,NEVER_FALLBACK,ExpertPositionBridge.speed_scheme,pipeline=native_pipeline()) for e in single_env]
    triggers=[make_trigger(config(delay),NEVER_FALLBACK) for delay in delays]
    switches=[60+delay for delay in delays];active=torch.ones(2,dtype=torch.bool);draw=torch.zeros(2,dtype=torch.long)
    for tick in range(max(switches)+7):
        raw=torch.zeros(2,29);student=torch.full((2,29),.01)
        expected=[]
        for lane in range(2):
            single_env[lane]._prev_targets=env._prev_targets[lane:lane+1].clone()
            single_env[lane]._cur_targets=env._cur_targets[lane:lane+1].clone()
            singles[lane].submit(raw[lane:lane+1],raw[lane:lane+1],tick)
            expected.append(singles[lane].select(single_env[lane],student[lane:lane+1],draw[lane:lane+1]))
        batch.submit(raw,raw,tick,active);actual=batch.select(env,student,draw)
        assert torch.equal(actual,torch.cat(expected))
        for lane in range(2):
            assert bool(batch.trace['control_source'][lane])==(tick>=switches[lane])
            queue=int(batch.trace['expert_queue_length'][lane])
            assert bool(batch.trace['expert_label_valid'][lane])==(tick>=switches[lane]+queue-1)
            if tick<switches[lane]:
                event=advance(triggers[lane],tick)
                if event is not None:
                    assert event==tick+1==switches[lane]
                    batch.schedule_takeover(lane,event);singles[lane].schedule_takeover(event)
        env._prev_targets=actual[:,env._perm_canon_to_lab].clone();env._cur_targets=env._prev_targets.clone()


@pytest.mark.parametrize('schedules',[['held60','held120'],['held0','held120']])
def test_expansion_registration_pins_minimum_delay_and_never_fallback(setup_registry,monkeypatch,schedules):
    from tools.tight_fresh_expansion import prepare
    setup=setup_registry
    write(setup['source']/'session.json',dict(deadline=time.time()+1000,domain=setup['domain'],expert_sha256='e'))
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair',lambda root,role:dict(role=role))
    output=setup['root']/'held_prepared'
    plan=prepare(setup['source'],output,session_root=setup['source'],registry_id='held_fixture',seeds=[6000,6001],
        partition_manifest=setup['partition'],frozen_families=setup['families'],schedules=schedules,
        lanes=8,event_seed=8000,round_index=9)
    validate_collection_plan(plan)
    assert len(plan['cases'])==4 and {case['expansion_schedule'] for case in plan['cases']}==set(schedules)
    for case in plan['cases']:
        assert case['spec']['takeover_tick']==NEVER_FALLBACK
        assert case['dagger']['trigger']==schedule_trigger(case['expansion_schedule'])
    for mutation in ('delay','fallback','version'):
        changed=copy.deepcopy(plan);case=changed['cases'][0]
        if mutation=='delay':case['dagger']['trigger']['min_delay_ticks']=120
        elif mutation=='fallback':case['spec']['takeover_tick']=601
        else:case['dagger']['trigger']=dict(version='contact_height_then_delay_v1',delay_ticks=60)
        with pytest.raises(ValueError,match='trigger or acquired a fallback'):validate_collection_plan(changed)
    assert not (output/'run_started.json').exists()


@pytest.mark.parametrize('schedule,switch',[('held0',60),('held60',120),('held120',180)])
def test_completed_collection_review_uses_rechecked_boundary_not_first_held(setup_registry,monkeypatch,schedule,switch):
    from test_batch_collection_review import completed_mixed,numeric
    from tools import tight_batch_collection_review as review
    from tools.tight_fresh_report import audit_trigger_case
    from visuotactile.compact import sha256
    root=completed_mixed(setup_registry,monkeypatch)
    plan=json.loads((root/'plan.json').read_text());case=next(c for c in plan['cases'] if c['name']=='near')
    case['expansion_schedule']=schedule;case['dagger']['trigger']=schedule_trigger(schedule)
    with h5py.File(root/'near/numeric.h5') as stream:metadata=json.loads(stream.attrs['metadata_json'])
    result=numeric(root/'near',switch,length=switch+20,metadata=metadata,trigger=case['dagger']['trigger'])
    write(root/'near/result.json',result)
    rows=json.loads((root/'results.json').read_text());next(r for r in rows if r['case']=='near')['result']=result
    write(root/'results.json',rows);write(root/'plan.json',plan)
    for item in plan['cases']:
        path=root/item['name']/'training_expansion_audit.json';proof=json.loads(path.read_text())
        proof['source_plan_sha256']=sha256(root/'plan.json')
        if item['name']=='near':
            proof.update(expansion_schedule=schedule,category='expert_rescue_after_held_recheck',
                causal_trigger=audit_trigger_case(root/'near',result))
        write(path,proof)
    report=review.review(root);row=next(r for r in report['cases'] if r['case']=='near')
    assert row['first_qualifying_condition_completed_tick']==switch
    assert row['boundary']['completed_student_tick']==switch and row['boundary']['held_window'] is True
    assert row['category']=='expert_rescue_after_held_recheck' and row['causal_condition_met'] is True
    assert report['summary']['by_schedule'][schedule]['takeovers']==1
    from tools.tight_collection_summary import held_recheck
    summary=held_recheck(case,result,boundary=row['boundary'])
    assert summary['category']=='actual_rechecked_takeover'
    assert summary['wait_after_first_held_ticks']==switch-60


def test_held0_terminal_after_first_window_does_not_create_expert_action(tmp_path):
    from tools.tight_fresh_report import audit_trigger_case
    from tools.tight_collection_summary import held_recheck
    trigger=make_trigger(config(0),NEVER_FALLBACK)
    rows=[observation(tick) for tick in range(60)]
    for tick,row in enumerate(rows):
        assert observe_transition(trigger,tick,stacked([row]))==(60 if tick==59 else None)
    result=write_capture(tmp_path/'case',rows,trigger,None)
    audit=audit_trigger_case(tmp_path/'case',result)
    case=dict(expansion_schedule='held0',dagger=dict(trigger=config(0)))
    assert classify_case(case,result,audit)=='ended_after_held_recheck_before_scheduled_takeover'
    assert held_recheck(case,result)['category']=='rechecked_without_next_action'
    assert result['dagger']['expert_ticks']==0 and not result['dagger']['expert_takeover_success']
