import copy
import json

import h5py
import numpy as np
import pytest
import torch

from tools import tight_fresh_anchor_study_report as report
from tools.tight_fresh_study_onset import command_motion_sum, aggregate_command_motion
from visuotactile.residual_output import (MATCHED_FINAL_HEAD_RESET,
    REACHABLE_CENTERED_OUTPUT, REACHABLE_PREVIOUS_TARGET_OUTPUT)


def receipts():
    return [dict(update=1,epoch=1,supervised_ticks=130,contact_ticks=40,history_ticks=180,
        window_chunks=2,initial_recovery_supervised_ticks=0,remaining_supervised_ticks=0,
        fixed_denominator=128,group_tail=False,loss_weight_mass=130.)]


def test_same_optimizer_exposure_and_unweighted_mass_required():
    a=receipts();b=copy.deepcopy(a)
    assert report.check_equal_exposure(a,b)['history_ticks']==180
    b[0]['loss_weight_mass']=140
    with pytest.raises(ValueError,match='unweighted'):report.check_equal_exposure(a,b)
    b=copy.deepcopy(a);b[0]['history_ticks']=179
    with pytest.raises(ValueError,match='schedule or exposure'):report.check_equal_exposure(a,b)


def configuration_pair(study):
    result={}
    for arm in report.ARMS:
        output=REACHABLE_CENTERED_OUTPUT if arm=='control' else REACHABLE_PREVIOUS_TARGET_OUTPUT
        audit=dict(continuation_head_reset=MATCHED_FINAL_HEAD_RESET,
            reset_parameter_names=['policy.2.weight','policy.2.bias'],
            output_contract_before=REACHABLE_CENTERED_OUTPUT,output_contract_after=output,
            output_contract_changed=arm=='previous',requires_training_before_deployment=True,
            optimizer_state_restored=False,new_optimizer_updates=0,
            statistics_and_delta_buffers_unchanged=True,command_normalizer_unchanged=True,scale_adaptation=None)
        config=dict(continuation_from={'sha256':'source_rear'},base_sha256='source_dp',
            output_contract=output,continuation_head_reset=MATCHED_FINAL_HEAD_RESET,
            continuation_audit=audit,source_root=str(study/arm/'A'),learning_rate=1e-4)
        result[arm]=dict(configuration=config,residual_output_version=output['version'],updates=10,
            optimizer=dict(state={0:{'step':torch.tensor(10)}},param_groups=[{'lr':1e-4}]))
    return result


def test_anchor_configuration_only_permits_declared_output_coordinate_change(tmp_path):
    saved=configuration_pair(tmp_path)
    registration={'source_pairs':{'A':{'backend_sha256':'source_rear','frontend_sha256':'source_dp'}}}
    result=report.check_configuration(tmp_path,'A',saved,registration)
    assert result['configurations_match_except_anchor_and_paths']
    changed=copy.deepcopy(saved);changed['previous']['configuration']['learning_rate']=2e-4
    with pytest.raises(ValueError,match='beyond output anchor'):report.check_configuration(tmp_path,'A',changed,registration)
    changed=copy.deepcopy(saved);changed['control']['configuration']['continuation_audit']['reset_parameter_names']=['policy.2.bias']
    with pytest.raises(ValueError,match='final-head reset'):report.check_configuration(tmp_path,'A',changed,registration)
    changed=copy.deepcopy(saved);changed['previous']['optimizer']['state'][0]['step']=torch.tensor(9)
    with pytest.raises(ValueError,match='Adam step'):report.check_configuration(tmp_path,'A',changed,registration)


def test_command_motion_distinguishes_holding_and_reverse_motion_from_correct_direction():
    previous=torch.zeros(3,29);target=previous.clone();target[:2]=.02
    target[1,0]=0.;target[0,1]=1e-7
    mask=torch.tensor([True,True,False])
    holding=command_motion_sum(previous,target,previous,mask)
    result=aggregate_command_motion([holding])
    assert result['parts']['hand']['predicted_command_delta_rms_rad']==0
    assert result['parts']['hand']['same_direction_fraction']==0
    assert result['parts']['hand']['hold_previous_target_rmse_rad']==pytest.approx(.02)
    prediction=target.clone();prediction[:,7:]*=-1
    motion=aggregate_command_motion([command_motion_sum(prediction,target,previous,mask)])
    assert motion['parts']['arm']['same_direction_fraction']==1
    assert motion['parts']['hand']['same_direction_fraction']==0
    assert motion['parts']['arm']['effective_gt_components']==12


def test_motion_summaries_pool_actual_components_and_zero_denominators():
    previous=torch.zeros(2,29);target=previous.clone();target[0]=.01;target[1]=.02
    prediction=target.clone();prediction[1]=0
    first=command_motion_sum(prediction,target,previous,torch.tensor([True,False]))
    second=command_motion_sum(prediction,target,previous,torch.tensor([False,True]))
    pooled=aggregate_command_motion([first,second])
    whole=aggregate_command_motion([command_motion_sum(prediction,target,previous,torch.tensor([True,True]))])
    assert pooled==whole
    assert pooled['parts']['hand']['same_direction_fraction']==.5
    empty=aggregate_command_motion([command_motion_sum(prediction,target,previous,torch.tensor([False,False]))])
    assert empty['parts']['arm']['same_direction_fraction'] is None
    assert empty['parts']['hand']['hold_previous_target_rmse_rad'] is None


def test_interleaved_autonomous_keeps_invalid_initializations_in_opportunity_count(tmp_path,monkeypatch):
    from tools.tight_fresh_anchor_study import development_cases
    from visuotactile import narrow_data
    monkeypatch.setattr(narrow_data,'direct_configuration',lambda domain,seed:dict(domain=domain,seed=seed))
    def write(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value))
    pairs={arm+'_'+role:{'backend':str(tmp_path/arm/role/'best.pt')} for arm in report.ARMS for role in ('A','B')}
    monkeypatch.setattr(report,'completed_pair',lambda root,role:pairs[root.name+'_'+role])
    cases=development_cases({'domain':'fixed'},1,9);rows=[]
    for expected in cases:
        key=expected['role'];spec=expected['spec'];name=expected['name']
        result=dict(software_complete=True,legal=False,**spec)
        rows.append(dict(case=name,role=key,result=result,timing={'total_seconds':1.}))
        write(tmp_path/'development'/name/'result.json',result)
        write(tmp_path/'development'/name/'started.json',dict(spec=spec,checkpoint=pairs[key]['backend']))
    plan=dict(kind='autonomous',test_used=False,pairs=pairs,cases=cases,study='matched_rear_output_anchor_study_v1')
    write(tmp_path/'development/plan.json',plan);write(tmp_path/'development/results.json',rows)
    write(tmp_path/'development/result.json',dict(software_complete=True,cases=4))
    write(tmp_path/'development/run_started.json',dict(plan_sha256=report.sha256(tmp_path/'development/plan.json')))
    registration=dict(version=plan['study'],count_per_role=1,development_seed=9,session_binding={'domain':{'domain':'fixed'}})
    result=report.autonomous(tmp_path,registration)
    assert result['totals']['previous']['B']['opportunities']==1
    assert result['totals']['previous']['B']['legal']==0
    assert result['totals']['previous']['B']['invalid_initializations']==1
    plan['cases'][0]['spec']={'seed':10};write(tmp_path/'development/plan.json',plan)
    write(tmp_path/'development/run_started.json',dict(plan_sha256=report.sha256(tmp_path/'development/plan.json')))
    with pytest.raises(ValueError,match='registered seed/domain/count'):report.autonomous(tmp_path,registration)


def test_registered_opportunities_reject_common_change_across_all_four_cells(monkeypatch):
    from tools.tight_fresh_anchor_study import development_cases
    from visuotactile import narrow_data
    monkeypatch.setattr(narrow_data,'direct_configuration',lambda domain,seed:dict(seed=seed))
    registration=dict(count_per_role=2,development_seed=17,session_binding={'domain':{}})
    plan=dict(cases=development_cases({},2,17));report.check_registered_opportunities(plan,registration)
    plan['cases']=development_cases({},2,18)
    with pytest.raises(ValueError,match='registered seed/domain/count'):report.check_registered_opportunities(plan,registration)


def test_stage_snapshot_uses_goal_before_subgoal_change_and_does_not_invent_turning_pass(tmp_path):
    n=60;pose=np.zeros((n,7),np.float32);pose[:,3]=1;pose[48:,2]=.1
    goal=pose.copy();palm=np.zeros_like(pose);palm[:,3]=1
    force=np.zeros((n,2,5,3),np.float32);force[:,:,:2,2]=1
    count=np.zeros(n);count[-1]=1
    with h5py.File(tmp_path/'numeric.h5','w') as stream:
        t=stream.create_group('episode/transition');before=t.create_group('diagnostics_before');after=t.create_group('diagnostics_after')
        after['object_pose_w']=pose;after['palm_body_pose_w']=palm;before['goal_pose_w']=goal
        after_goal=goal.copy();after_goal[-1,0]=1.;after['goal_pose_w']=after_goal
        before['successes']=np.zeros(n);after['successes']=count;after['lifted_object']=np.arange(n)>=48;t['object_contact_force_w']=force
        t.create_group('native_success_diagnostic')['near_goal']=np.ones(n,bool)
        t['native_alignment_condition']=count.astype(bool);t['native_keypoints_max_dist']=np.zeros(n)
    phases=dict(steps=n,numeric_sha256=report.sha256(tmp_path/'numeric.h5'))
    result=report.phase_snapshots(tmp_path,phases)
    assert result['first_native_lift']['completed_tick']==49
    assert result['first_stable_hold']['completed_tick']==60
    assert result['first_native_subgoal']['completed_tick']==60
    assert result['first_native_subgoal']['goal_root_distance_m']==0.
    assert result['first_native_subgoal']['goal_index_before']==0
    assert result['first_native_subgoal']['subgoals_after']==1
    assert result['first_goal_after_hold_best_orientation']['completed_tick']==60
    assert 'orientation_pass' not in result


def test_source_reference_keeps_registered_earlier_cohort_separate_and_checks_reset_order(tmp_path,monkeypatch):
    from tools import tight_fresh_followup_session
    directory=tmp_path/'round_05/development';directory.mkdir(parents=True)
    specs=[dict(seed=17),dict(seed=18)]
    plan=dict(cases=[dict(name=role+str(i),role=role,spec=spec) for role in ('A','B') for i,spec in enumerate(specs)])
    rows=[dict(case=row['name'],role=row['role'],timing={'total_seconds':2.}) for row in plan['cases']]
    (directory/'plan.json').write_text(json.dumps(plan));(directory/'results.json').write_text(json.dumps(rows))
    audit={'cases':4}
    registration=dict(source_pairs={'A':{},'B':{}},source_evidence=dict(root=str(directory.parent),audit=audit,
        plan_sha256=report.sha256(directory/'plan.json'),results_sha256=report.sha256(directory/'results.json')))
    monkeypatch.setattr(tight_fresh_followup_session,'evaluated_pairs',lambda *args:audit)
    monkeypatch.setattr(report,'phase_case',lambda path:dict(case=path.name,legal=True,success=False,
        native_lift=True,settled_height_stable_grasp=True,native_alignment_first_tick=None,
        native_near_goal_first_tick=None,native_first_subgoal_tick=None,steps=600,authority={'expert_ticks':0}))
    monkeypatch.setattr(report,'phase_snapshots',lambda *args:{'first_stable_hold':{'completed_tick':120}})
    result=report.source_autonomous(registration,specs)
    assert result['root']==str(directory.parent)
    assert result['totals']['A']['opportunities']==2 and result['totals']['A']['steps']==1200
    assert len(result['cases'])==4 and {row['arm'] for row in result['cases']}=={'source'}
    assert result['totals']['B']['seconds']==4
    with pytest.raises(ValueError,match='reset opportunities differ'):
        report.source_autonomous(registration,list(reversed(specs)))


def test_history_accounting_keeps_zero_supervision_forward_ticks_outside_optimizer_receipts():
    from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT
    mask=torch.zeros(100,dtype=torch.bool);mask[64:]=True
    rng=torch.Generator().manual_seed(9);torch.randperm(1,generator=rng)
    saved=dict(train_ids=['case'],configuration=dict(maximum_control_tick_epochs=1,microbatch=1,
        initialization_seed=9,tick_window_contract=TICK_WINDOW_CONTRACT,history_contract={'version':'current_weight_prefix_v1'}),episode_counts={'case':1},
        episode_order=['case'],schedule_rng=rng.get_state(),control_ticks=100,supervised_ticks=36)
    receipts=[dict(update=1,epoch=1,history_ticks=36,supervised_ticks=36,window_chunks=2,fixed_denominator=128,group_tail=True)]
    result=report.audit_history_accounting({'case':mask},saved,receipts)
    assert result['chronological_history_ticks']==100
    assert result['optimizer_window_history_ticks']==36
    assert result['zero_supervision_chunk_history_ticks']==64
    assert result['derived_prefix_recompute_history_ticks']==0
    assert result['epochs'][0]['zero_supervision_chunks']==2
    receipts[0]['history_ticks']=100
    with pytest.raises(ValueError,match='receipts differ'):report.audit_history_accounting({'case':mask},saved,receipts)


def test_history_accounting_separates_repeated_prefix_compute_from_once_per_traversal_exposure():
    from visuotactile.residual_tick_window import TICK_WINDOW_CONTRACT
    mask=torch.ones(160,dtype=torch.bool);rng=torch.Generator().manual_seed(9);torch.randperm(1,generator=rng)
    saved=dict(train_ids=['case'],configuration=dict(maximum_control_tick_epochs=1,microbatch=1,
        initialization_seed=9,tick_window_contract=TICK_WINDOW_CONTRACT,history_contract={'version':'current_weight_prefix_v1'}),episode_counts={'case':1},
        episode_order=['case'],schedule_rng=rng.get_state(),control_ticks=160,supervised_ticks=160)
    receipts=[dict(update=1,epoch=1,history_ticks=128,supervised_ticks=128,window_chunks=4,fixed_denominator=128,group_tail=False),
        dict(update=2,epoch=1,history_ticks=32,supervised_ticks=32,window_chunks=1,fixed_denominator=128,group_tail=True)]
    result=report.audit_history_accounting({'case':mask},saved,receipts)
    assert result['chronological_history_ticks']==result['optimizer_window_history_ticks']==160
    assert result['derived_prefix_recompute_history_ticks']==128
    saved['schedule_rng']=torch.Generator().manual_seed(12).get_state()
    with pytest.raises(ValueError,match='shuffle order/RNG'):report.audit_history_accounting({'case':mask},saved,receipts)
