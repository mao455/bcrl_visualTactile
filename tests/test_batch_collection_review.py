"""Completed mixed opportunities keep causal denominators and real label masks."""
import json

import h5py
import numpy as np
import pytest

from test_fresh_stage_audit import fields
from test_held_near_goal_trigger import observation, stacked
from test_training_expansion import setup_registry, write, entry_for
from tools import tight_batch_collection_review as review
from tools import tight_fresh_stage_audit as stage
from tools.tight_fresh_report import audit_trigger_case
from visuotactile.takeover_trigger import make_trigger, observe_transition, NEVER_FALLBACK
from visuotactile.training_expansion import COLLECTION_CONTRACT, schedule_trigger


def save(group, tree):
    for key, value in tree.items():
        if isinstance(value, dict):
            save(group.create_group(key), value)
        else:
            group[key] = value


def numeric(directory, switch, *, length=80, metadata=None, trigger=None):
    observations = [observation(i, distance=.05 if i>=60 else .2) for i in range(length)]
    values = stacked(observations)
    after = values['diagnostics_after']
    after.update(joint_position_rad=np.zeros((length,29)), lifted_object=np.arange(length)>=50)
    for key in ('object_pose_w','palm_body_pose_w','goal_pose_w','successes'):
        values['diagnostics_before'][key] = np.concatenate((after[key][:1],after[key][:-1]))
    values.update(target_after_rad=np.zeros((length,29)),
        native_success_diagnostic=dict(near_goal=np.zeros(length,bool)),
        native_alignment_condition=np.zeros(length,bool),
        time_before_s=np.arange(length)/60,time_after_s=np.arange(1,length+1)/60,
        step_id=np.arange(length),episode_id=np.zeros(length,int))
    values['dagger'] = fields(switch=switch,length=length)
    values['dagger'].update(u_expert=values['target_after_rad'],q_reached=after['joint_position_rad'])
    with h5py.File(directory/'numeric.h5','w') as stream:
        stream.attrs['metadata_json'] = json.dumps(metadata or {})
        save(stream.create_group('episode/transition'), values)
        stream['episode/obs/time_s'] = values['time_before_s']
    alignment = stage.label_alignment(values['dagger'])
    receipt = None
    if trigger is not None:
        replay = make_trigger(trigger,NEVER_FALLBACK)
        for tick in range(switch if switch is not None else length):
            observe_transition(replay,tick,values,index=tick)
        receipt = replay.receipt()
    return dict(software_complete=True, legal=True, success=True, accepted_for_training=False,
        steps=length,residual_ticks=length,residual_forward_ticks=length,
        physics_hz=120,simulation_step_hz=60,plan_hz=5,
        complete_student_policy=dict(frontend_sha256='front',backend_sha256='rear'),
        reasons=['fixture'],dataset_capture=dict(sha256='fixture_capture'),
        dagger=dict(configuration=dict(trigger=trigger,takeover_tick=NEVER_FALLBACK if trigger else switch),
            trigger_receipt=receipt,expert_ticks=alignment['expert_ticks'],
            expert_valid_ticks=alignment['legal_expert_ticks'],expert_takeover_success=True,
            student_autonomous_success=False,recovery_coverage=dict(switch_tick=switch,
                population=dict(first_valid_expert_tick=alignment['first_legal_label_action_index']))))


@pytest.mark.parametrize('length',[44,80])
def test_fixed40_does_not_use_expert_frames_to_qualify_student_height(tmp_path,length):
    result=numeric(tmp_path,40,length=length)
    labels,phase=stage.audit_numeric(tmp_path/'numeric.h5',result)
    assert labels['student_history_ticks']==40
    assert phase['student_settled_reference_available'] is False
    assert phase['student_window_counts']['contact_and_elevation'] is None
    assert phase['student_max_later_rise_over_settled_m'] is None
    assert phase['settled_reference_authority']==('includes_expert_ticks' if length>=48 else 'unavailable')


def completed_mixed(setup,monkeypatch):
    root=setup['root']/'collection';root.mkdir()
    # Admission validation itself has separate full registry tests. Here isolate
    # completed-case auditing while keeping real source/capture binding checks.
    monkeypatch.setattr('visuotactile.training_expansion.validate_collection_plan',lambda plan:None)
    cases=[];results=[];proofs=[]
    for name,schedule,switch,index,legal in [('fixed','fixed40',40,0,True),
            ('near','near60',61,1,True),('invalid','near60',None,1,False)]:
        role='A';entry=entry_for(setup,index)
        spec=dict(entry,identity=name,takeover_tick=int(schedule[5:]) if schedule.startswith('fixed') else NEVER_FALLBACK)
        trigger=schedule_trigger(schedule)
        case=dict(name=name,role=role,spec=spec,source_data=str(setup['manifests'][role]),
            expansion_schedule=schedule,dagger=dict(trigger=trigger))
        cases.append(case);directory=root/name;directory.mkdir()
        registered=setup['registry']['families'][entry['family']]
        metadata=dict(training_expansion=setup['binding'],initial_family=entry['family'],
            source_identity=registered['source_identity'],dataset_split='train',
            initial_configuration_sha256=registered['configuration_sha256'])
        result=numeric(directory,switch,metadata=metadata,trigger=trigger) if legal else dict(
            software_complete=True,legal=False,success=False,accepted_for_training=False)
        if not legal:(directory/'numeric.h5').write_bytes(b'invalid must not be opened')
        write(directory/'result.json',result);write(directory/'capture_audit.json',dict(passed=True))
        if legal:write(directory/'capture_complete.json',dict(dataset_capture=result['dataset_capture']))
        results.append(dict(case=name,role=role,result=result))
        receipt=audit_trigger_case(directory,result) if legal and trigger else None
        category=('expert_rescue_after_fixed_takeover' if schedule=='fixed40' else
            'expert_rescue_after_held_near_goal' if legal else 'invalid_initialization')
        proofs.append((directory,dict(category=category,causal_trigger=receipt,
            expansion_schedule=schedule,training_expansion=setup['binding'])))
    plan=dict(cases=cases,pairs=dict(A=dict(frontend_sha256='front',backend_sha256='rear')),
        stage_takeover_contract=COLLECTION_CONTRACT,training_expansion_registry=setup['binding'],
        source_commit='fixture',source_sha256={})
    write(root/'plan.json',plan)
    for directory,proof in proofs:
        write(directory/'training_expansion_audit.json',dict(proof,source_plan_sha256=stage.sha256(root/'plan.json')))
    write(root/'results.json',results);write(root/'correction_entries.json',{})
    write(root/'result.json',dict(software_complete=True))
    write(root/'owner_finished.json',dict(software_complete=True,returncode=0))
    write(root/'owner_started.json',dict(cwd=str(root)))
    return root


def test_mixed_review_separates_fixed_causal_legal_recovered_and_retained(setup_registry,monkeypatch):
    root=completed_mixed(setup_registry,monkeypatch)
    result=review.review(root);summary=result['summary']
    assert summary['attempts']==3 and summary['legal_initializations']==2
    assert summary['triggered']==summary['fixed_takeovers']==1
    assert summary['causal_condition_met']==1
    assert summary['causal_opportunities']['attempts']==2
    assert summary['causal_opportunities']['legal_initializations']==1
    assert summary['recovered']==2 and summary['accepted']==0
    assert summary['all_valid_expert_ticks']==55 and summary['accepted_valid_expert_ticks']==0
    assert len(summary['attempted_new_families'])==2 and summary['accepted_new_families']==[]
    assert result['cases'][0]['boundary']['held_window'] is None
    assert result['cases'][0]['coverage_ticks']['student']['completed_contact_height_window'] is None
    assert result['cases'][0]['student_longest_two_finger_contact_ticks']==40
    assert result['cases'][1]['boundary']['held_window'] is True
    assert result['cases'][1]['boundary']['native_keypoint_distance_m']==.05
    near=result['cases'][1]
    assert near['first_qualifying_condition_completed_tick']==61
    assert near['student_closest_original_goal']['completed_tick']==61
    assert result['test_trajectories_opened']==0


def test_qualifying_window_is_separate_from_reaching_scheduled_takeover(setup_registry,monkeypatch):
    root=completed_mixed(setup_registry,monkeypatch)
    rows=review.review(root)['cases']
    # A completed window can schedule a later action beyond natural termination.
    # These two counters must not collapse even when the trigger itself is valid.
    rows[1]['boundary']=None
    summary=review.summarize(rows)
    assert summary['causal_condition_met']==1 and summary['triggered']==0
    assert summary['fixed_takeovers']==1


def test_accepted_expansion_entry_is_bound_to_actual_capture_policy_and_registry(setup_registry,monkeypatch):
    root=completed_mixed(setup_registry,monkeypatch)
    rows=json.loads((root/'results.json').read_text());rows[0]['result']['accepted_for_training']=True
    write(root/'results.json',rows);write(root/'fixed/result.json',rows[0]['result'])
    entry=dict(entry_for(setup_registry),takeover_tick=40,source_sha256='fixture_capture',
        episode='fixed',frontend_sha256='front',backend_sha256='rear')
    write(root/'correction_entries.json',dict(fixed=entry))
    result=review.review(root)
    assert result['summary']['accepted']==1
    assert result['summary']['accepted_valid_expert_ticks']==38
    assert result['summary']['unretained_valid_expert_ticks']==17
    assert result['summary']['accepted_new_families']==[entry['family']]
    entry['source_sha256']='another_capture'
    write(root/'correction_entries.json',dict(fixed=entry))
    with pytest.raises(ValueError,match='capture/policy binding'):review.review(root)


@pytest.mark.parametrize('fault',['capture_binding','source','near_current_goal','result_index'])
def test_mixed_review_rejects_provenance_or_causal_corruption(setup_registry,monkeypatch,fault):
    root=completed_mixed(setup_registry,monkeypatch)
    if fault=='capture_binding':
        with h5py.File(root/'fixed/numeric.h5','r+') as stream:
            metadata=json.loads(stream.attrs['metadata_json']);metadata['dataset_split']='test'
            stream.attrs['metadata_json']=json.dumps(metadata)
    elif fault=='source':
        with setup_registry['manifests']['A'].open('a') as stream:stream.write('\n')
    elif fault=='near_current_goal':
        with h5py.File(root/'near/numeric.h5','r+') as stream:
            stream['episode/transition/diagnostics_after/goal_pose_w'][60,0]=.1
    else:
        result=json.loads((root/'near/result.json').read_text());result['steps']=79
        write(root/'near/result.json',result)
    with pytest.raises(ValueError):review.review(root)
