import copy

import numpy as np
import pytest
import torch

from tools.tight_fresh_motion_probe import (check_cache_alignment, check_normalization_bindings,
    check_population, check_transition_timing, compact_report, motion_sums, phase_masks,
    plan_start_observations, summarize)


def test_first_contact_bins_partition_actual_timeline_and_never_wrap_before_start():
    contact=np.zeros(40,bool);contact[7]=True
    masks=phase_masks(contact,np.full(40,'approach'))
    chosen=[masks[k] for k in ('before_first_object_contact','first_object_contact12',
        'next_object_contact12','after_first_object_contact24')]
    assert np.array_equal(np.stack(chosen).sum(0),np.ones(40))
    assert [int(x.sum()) for x in chosen]==[7,12,12,9]
    assert np.array_equal(masks['first_object_contact24'],masks['first_object_contact12']|masks['next_object_contact12'])
    none=phase_masks(np.zeros(40,bool),np.full(40,'approach'))
    assert none['before_first_object_contact'].all() and not none['first_object_contact12'].any()


def test_continuous_gain_and_cosine_distinguish_shrunken_motion_from_hold():
    previous=torch.zeros(3,2);target=torch.tensor([[.01,-.02],[.005,-.001],[0.,0.]])
    mask=torch.ones(3,dtype=torch.bool)
    half=summarize([motion_sums(target*.5,target,previous,mask)])
    assert half['regression_gain']==pytest.approx(.5)
    assert half['increment_cosine']==pytest.approx(1.)
    assert half['same_direction_fraction']==1
    assert half['error_to_hold_ratio']==pytest.approx(.5)
    hold=summarize([motion_sums(previous,target,previous,mask)])
    assert hold['regression_gain']==0 and hold['increment_cosine'] is None
    assert hold['same_direction_fraction']==0 and hold['nearzero_prediction_fraction_on_effective_gt']==1
    assert hold['error_to_hold_ratio']==1


def test_fixed_threshold_reports_component_and_energy_coverage_without_hiding_large_wrong_moves():
    previous=torch.zeros(2,2);target=torch.tensor([[.02,.00001],[.0002,.0002]])
    prediction=target.clone();prediction[0,0]*=-1
    result=summarize([motion_sums(prediction,target,previous,torch.ones(2,dtype=torch.bool))])
    assert result['effective_gt_components']==3
    assert result['effective_gt_component_fraction']==.75
    assert result['same_direction_fraction']==pytest.approx(2/3)
    assert result['same_direction_gt_motion_energy_fraction']<.001
    assert result['effective_gt_motion_energy_fraction']>.999
    assert result['regression_gain']<0


def test_original_same_sign_fraction_includes_nearzero_and_is_not_an_exclusive_partition():
    previous=torch.zeros(5,1);target=torch.full((5,1),.001)
    prediction=torch.tensor([[.00001],[-.00001],[0.],[-.001],[.001]])
    result=summarize([motion_sums(prediction,target,previous,torch.ones(5,dtype=torch.bool))])
    assert result['same_direction_fraction']==2/5
    assert result['nearzero_prediction_fraction_on_effective_gt']==3/5
    assert result['opposite_prediction_fraction_on_effective_gt']==1/5


def test_aggregation_uses_component_sums_and_handles_empty_phase():
    previous=torch.zeros(4,3);target=torch.ones(4,3)*.01;pred=target*.2
    a=motion_sums(pred,target,previous,torch.tensor([True,False,False,False]))
    b=motion_sums(pred,target,previous,torch.tensor([False,True,True,True]))
    allrows=motion_sums(pred,target,previous,torch.ones(4,dtype=torch.bool))
    assert summarize([a,b])==summarize([allrows])
    empty=summarize([motion_sums(pred,target,previous,torch.zeros(4,dtype=torch.bool))])
    assert empty['ticks']==0 and empty['regression_gain'] is None and empty['same_direction_fraction'] is None


def test_fair_planning_hold_does_not_read_observations_from_later_in_chunk():
    previous=torch.arange(25,dtype=torch.float32)[:,None]
    cursor=torch.arange(25)%12
    start=plan_start_observations(previous,cursor,12)
    assert start.flatten().tolist()==[0.]*12+[12.]*12+[24.]
    target=previous+1;prediction=target+.5
    allsteps=summarize([motion_sums(prediction,target,previous,torch.ones(25,dtype=torch.bool),start)])
    assert allsteps['hold_current_observation_rmse_rad']==1
    assert allsteps['hold_plan_start_observation_rmse_rad']>7
    assert allsteps['error_to_plan_start_hold_ratio']<allsteps['error_to_hold_ratio']
    fresh=summarize([motion_sums(prediction,target,previous,cursor==0,start)])
    assert fresh['hold_plan_start_observation_rmse_rad']==fresh['hold_current_observation_rmse_rad']
    malformed=cursor.clone();malformed[3]=0
    with pytest.raises(ValueError,match='contiguous 5Hz'):plan_start_observations(previous,malformed,12)


def test_compact_tables_preserve_original_statistics_and_document_selected_scope():
    backend=motion_sums(torch.ones(2,1),torch.ones(2,1),torch.zeros(2,1),torch.ones(2,dtype=torch.bool))
    metric=dict(split='validation',part='hand',phase='all',predictor='backend',**summarize([backend]))
    case=dict(identity='val',metrics={'backend':{'all':{'hand':summarize([backend])},'native_approach':{'hand':summarize([backend])}}})
    full=dict(roles={'A':dict(cases=[case],metrics=[metric],dp_cursor_metrics=[],
        joint_metrics=[dict(split='validation',joint='joint1',**summarize([backend])),
            dict(split='train',joint='joint1',**summarize([backend]))])},source_sha256='source')
    copied=copy.deepcopy(full);small=compact_report(full,'full-report-sha')
    packed=small['roles']['A']['metrics']
    assert dict(zip(packed['columns'],packed['rows'][0]))==metric
    assert len(small['roles']['A']['case_metrics']['rows'])==1
    assert len(small['roles']['A']['joint_metrics']['rows'])==1
    assert full==copied and small['full_report_sha256']=='full-report-sha'


def population():
    original=[f'original{i}' for i in range(30)];validation=[f'validation{i}' for i in range(6)]
    return dict(normalization_train_ids=original,splits=dict(train=original+['correction'],validation=validation,test=['untouched']),
        entries={i:{'family':i} for i in original+validation+['correction','untouched']})


def test_probe_population_is_exact_original_train2_validation6_without_test():
    data=population();before=copy.deepcopy(data)
    selected=check_population(data)
    assert selected['train']==['original0','original1'] and len(selected['validation'])==6
    assert data==before
    data['entries']['correction']['family']='validation0'
    with pytest.raises(ValueError,match='family overlaps'):check_population(data)
    data=population();data['splits']['train'][0]='correction'
    with pytest.raises(ValueError,match='original train2'):check_population(data)
    data=population();data['splits']['test'].append('validation0')
    with pytest.raises(ValueError,match='Test trajectories'):check_population(data)


@pytest.mark.parametrize('mutation,message',[
    ('duplicate_train','Unique train2 and validation6 identities'),
    ('duplicate_validation','Unique train2 and validation6 identities'),
    ('duplicate_train_family','Unique train2 initial-state families'),
    ('changed_first2','Fixed original first2')])
def test_population_rejects_repetition_and_substituted_original_training_pair(mutation,message):
    data=population()
    if mutation=='duplicate_train':data['splits']['train'][1]=data['splits']['train'][0]
    elif mutation=='duplicate_validation':data['splits']['validation'][1]=data['splits']['validation'][0]
    elif mutation=='duplicate_train_family':data['entries']['original1']['family']='original0'
    else:data['splits']['train'][:2]=['original2','original3']
    with pytest.raises(ValueError,match=message):check_population(data)


def bindings():
    return dict(base_checkpoint_sha256='base',statistics_sha256='statistics',
        residual_statistics_sha256='residual',data_sha256='data')


@pytest.mark.parametrize('mutation', [None,'base','statistics','residual_statistics','data','identity','frontend_buffer','ema_buffer','rear_buffer'])
def test_normalizer_guards_reject_rebound_checkpoint_or_changed_buffers(mutation):
    stats=dict(robot_mean=torch.zeros(177),robot_std=torch.ones(177),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    residual=copy.deepcopy(stats);residual['action_scale']*=2
    config=dict(base_sha256='base',statistics_sha256='statistics',residual_statistics_sha256='residual',
        data_sha256='data',residual_normalization_identity='command')
    front=dict(configuration=copy.deepcopy(config),model=copy.deepcopy(stats),ema=copy.deepcopy(stats))
    rear=dict(configuration=copy.deepcopy(config),model=copy.deepcopy(residual));index=copy.deepcopy(config)
    if mutation in ('base','statistics','residual_statistics','data'):rear['configuration'][mutation+'_sha256']='wrong'
    elif mutation=='identity':index['residual_normalization_identity']='wrong'
    elif mutation=='frontend_buffer':front['model']['action_offset'][0]=1
    elif mutation=='ema_buffer':front['ema']['robot_std'][0]=2
    elif mutation=='rear_buffer':rear['model']['action_scale'][0]=1
    if mutation is None:check_normalization_bindings(front,rear,index,stats,residual,bindings())
    else:
        with pytest.raises(ValueError):check_normalization_bindings(front,rear,index,stats,residual,bindings())


@pytest.mark.parametrize('mutation',[None,'before','after','rate'])
def test_recorded_observation_timing_requires_same_transition_and_exact60hz(mutation):
    actual=dict(time=torch.tensor([0.,1/60],dtype=torch.float64),
        time_before=torch.tensor([0.,1/60],dtype=torch.float64),
        time_after=torch.tensor([1/60,2/60],dtype=torch.float64))
    if mutation=='before':actual['time_before'][0]+=.001
    elif mutation=='after':actual['time_after'][0]+=.001
    elif mutation=='rate':
        for key in actual:actual[key]*=2
    if mutation is None:check_transition_timing(actual)
    else:
        with pytest.raises(ValueError,match='timestamps'):check_transition_timing(actual)


@pytest.mark.parametrize('mutation',[None,'cache_identity','time','base','q_reached','numeric_q_reached','mask','segment'])
def test_cache_guards_reject_stale_binding_shifted_label_or_illegal_supervision(mutation):
    actual=dict(time=torch.tensor([0.,1/60],dtype=torch.float64),q=torch.zeros(2,29),
        q_after=torch.ones(2,29),previous=torch.full((2,29),2.),command=torch.full((2,29),3.))
    robot=torch.zeros(2,177);robot[:,:29]=actual['q'];robot[:,58:87]=actual['previous']
    numeric=dict(robot=robot,time=actual['time'].clone(),q_reached=actual['q_after'].clone(),
        u_expert=actual['command'].clone(),command=actual['command'].clone(),
        supervision_mask=torch.ones(2,dtype=torch.bool),control_segment=torch.zeros(2,dtype=torch.int64))
    record=copy.deepcopy(numeric)
    record.update(identity='episode',nominal_supervision='q_reached',cache_identity=bindings(),
        target=actual['command'].clone(),base=actual['q_after'].clone(),plans=actual['q_after'][:,None].clone())
    if mutation=='cache_identity':record['cache_identity']['extra']='not exact'
    elif mutation=='time':record['time'][0]+=.001
    elif mutation=='base':record['base'][0,0]+=1
    elif mutation=='q_reached':record['q_reached'][0,0]+=1
    elif mutation=='numeric_q_reached':numeric['q_reached'][0,0]+=1
    elif mutation=='mask':record['supervision_mask'][0]=False
    elif mutation=='segment':record['control_segment'][0]=1
    if mutation is None:
        check_cache_alignment(record,numeric,actual,'episode','q_reached',bindings())
        record['nominal_supervision']='u_expert';record['q_reached']=None
        check_cache_alignment(record,numeric,actual,'episode','u_expert',bindings())
    else:
        with pytest.raises(ValueError):check_cache_alignment(record,numeric,actual,'episode','q_reached',bindings())
