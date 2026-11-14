import copy
import json

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
    def write(path,value):path.parent.mkdir(exist_ok=True,parents=True);path.write_text(json.dumps(value))
    pairs={arm+'_'+role:{'backend':str(tmp_path/arm/role/'best.pt')} for arm in report.ARMS for role in ('A','B')}
    monkeypatch.setattr(report,'completed_pair',lambda root,role:pairs[root.name+'_'+role])
    cases=[];rows=[]
    for arm,role in [('control','A'),('previous','B'),('previous','A'),('control','B')]:
        key=arm+'_'+role;spec={'seed':9};name=key+'_seed9'
        expected=dict(name=name,role=key,study_arm=arm,student_role=role,spec=spec)
        result=dict(software_complete=True,legal=False,**spec)
        cases.append(expected);rows.append(dict(case=name,role=key,result=result,timing={'total_seconds':1.}))
        write(tmp_path/'development'/name/'result.json',result)
        write(tmp_path/'development'/name/'started.json',dict(spec=spec,checkpoint=pairs[key]['backend']))
    plan=dict(kind='autonomous',test_used=False,pairs=pairs,cases=cases,study='matched_rear_output_anchor_study_v1')
    write(tmp_path/'development/plan.json',plan);write(tmp_path/'development/results.json',rows)
    write(tmp_path/'development/result.json',dict(software_complete=True))
    registration=dict(version=plan['study'],count_per_role=1)
    result=report.autonomous(tmp_path,registration)
    assert result['totals']['previous']['B']['opportunities']==1
    assert result['totals']['previous']['B']['legal']==0
    assert result['totals']['previous']['B']['invalid_initializations']==1
    plan['cases'][0]['spec']={'seed':10};write(tmp_path/'development/plan.json',plan)
    with pytest.raises(ValueError,match='identity differs'):report.autonomous(tmp_path,registration)
