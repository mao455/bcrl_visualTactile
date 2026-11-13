import copy

import pytest
import torch

from visuotactile.execution_learning import PROJECTION_FY_LOSS,projection_fy_elements
from visuotactile.models.native_dp import TargetResidual,load_residual
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION,project_processed_position_targets
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT,centered_proposal,validate_output_contract
from tools import tight_execution_centered as experiment
from tools.runtime import write


def test_centered_finite_inverse_and_actual_gradient():
    bounds=torch.tensor([[-.02,.02],[0.,.001],[1.,1.]],dtype=torch.double)
    base=torch.tensor([100.,-100.,0.],dtype=torch.double)
    target=torch.tensor([-.01,.0007,1.],dtype=torch.double)
    span=bounds[:,1]-bounds[:,0]
    anchor=base.clamp(bounds[:,0],bounds[:,1])
    inverse=torch.where(span>0,(target-anchor)/span.clamp_min(1e-20),0.)
    assert inverse.abs().max()<=1
    torch.testing.assert_close(centered_proposal(base,inverse,bounds,REACHABLE_CENTERED_OUTPUT),target)
    raw=torch.tensor([2.,-2.,0.],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda value:centered_proposal(base,value,bounds,REACHABLE_CENTERED_OUTPUT),(raw,))
    proposed=centered_proposal(base,raw,bounds,REACHABLE_CENTERED_OUTPUT)
    elements,_=projection_fy_elements(proposed,target,bounds,torch.tensor(1.))
    gradient=torch.autograd.grad(elements.sum(),raw)[0]
    assert gradient[0]>0 and gradient[1]<0 and gradient[2]==0


def test_centered_executor_checkpoint_and_legacy_identity():
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29)*1000,action_offset=torch.zeros(29))
    loss=dict(PROJECTION_FY_LOSS,regularizer_weight=0.)
    model=TargetResidual(stats,[.9]*29,execution_adapter=REACHABLE_POSITION_EXECUTION,loss_contract=loss,output_contract=REACHABLE_CENTERED_OUTPUT)
    raw=torch.full((1,1,29),2.,requires_grad=True)
    base=torch.full_like(raw,-.5);previous=torch.zeros_like(raw)
    limits=torch.tensor([[-1.,1.]]).expand(29,2)
    chain=model.compose_chain(base,raw,None,limits,previous=previous)
    actual=project_processed_position_targets(chain['u_joint'],previous,limits,REACHABLE_POSITION_EXECUTION)
    torch.testing.assert_close(chain['u_exec'],actual,rtol=0,atol=0)
    parts=model.objective_components(actual,previous,chain['delta'],torch.ones(1,1),u_pre=chain['u_pre'],execution_bounds=chain['execution_bounds'])
    gradient=torch.autograd.grad(parts['total'],raw)[0]
    assert torch.isfinite(gradient).all() and (gradient>0).all()
    configuration=dict(loss_contract=loss,output_contract=REACHABLE_CENTERED_OUTPUT,execution_adapter=REACHABLE_POSITION_EXECUTION)
    saved=dict(model=model.state_dict(),updates=1,residual_version=model.version,residual_loss_version=loss['version'],residual_output_version=REACHABLE_CENTERED_OUTPUT['version'],configuration=configuration)
    loaded=load_residual(saved)
    torch.testing.assert_close(loaded.compose_chain(base,raw,None,limits,previous=previous)['u_exec'],actual,rtol=0,atol=0)
    broken=copy.deepcopy(saved);broken.pop('residual_output_version')
    with pytest.raises(ValueError,match='output identity'):load_residual(broken)
    assert validate_output_contract(None) is None
    legacy=TargetResidual(stats,[.9]*29)
    old=legacy.compose_chain(base,raw,None,limits)
    torch.testing.assert_close(old['u_pre'],base+.9*raw.tanh()/1000,atol=0,rtol=0)


def test_experiment_preserves_pair_and_existing_loss_settings(tmp_path):
    configuration={'residual_pair':{'continuation_from':{'path':'original.pt','sha256':'identity'},
        'bases':{'QG01':{'checkpoint':'q_reached.pt'}},'weight_decay':1e-4,
        'recovery_prefix':{'ticks':48},'sampling_weights':{'core':.5,'historical_correction':.2,'current_correction':.3}}}
    write(tmp_path/'configuration.json',configuration)
    changed=experiment.configuration_for(tmp_path)['residual_pair']
    for key,value in configuration['residual_pair'].items():assert changed[key]==value
    assert changed['output_contract']==REACHABLE_CENTERED_OUTPUT
    assert changed['loss_contract']['regularizer_weight']==0
    assert changed['maximum_control_tick_epochs']==changed['minimum_control_tick_epochs']==1
    assert changed['learning_rate']==1e-4


def test_experiment_cli_and_immutable_deadline(monkeypatch,tmp_path):
    invoked=[]
    monkeypatch.setattr(experiment,'prepare',lambda root:invoked.append(root))
    monkeypatch.setattr('sys.argv',['centered','prepare','--root',str(tmp_path)])
    experiment.main()
    assert invoked==[tmp_path]
    assert experiment.COMPUTE_DEADLINE==1794585315.106664
    assert experiment.ABSOLUTE_DEADLINE==1794585915.106664
    with pytest.raises(ValueError,match='Unique registered'):experiment.checked_root(tmp_path)


def test_unequal_episode_padding_does_not_invalidate_execution_bounds():
    from visuotactile.native_data import residual_batch,residual_previous_targets
    def record(length):
        robot=torch.zeros(length,178);robot[:,58:87]=.5
        return dict(robot=robot,visual=torch.zeros(length,512),plans=torch.zeros(length,24,29),
            plan_valid=torch.ones(length,24,dtype=torch.bool),tactile={'dummy':torch.zeros(length*2,1)},
            tactile_valid=torch.ones(length*2,dtype=torch.bool))
    inputs=residual_batch([record(3),record(1)],0,3,'cpu')
    limits=torch.tensor([.1,1.]).expand(2,1,29,2)
    previous=residual_previous_targets(inputs,limits)
    assert torch.equal(previous[inputs['valid']],inputs['robot'][...,58:87][inputs['valid']])
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    model=TargetResidual(stats,[.9]*29,execution_adapter=REACHABLE_POSITION_EXECUTION,
        loss_contract=dict(PROJECTION_FY_LOSS,regularizer_weight=0.),output_contract=REACHABLE_CENTERED_OUTPUT)
    base=torch.full((2,3,29),.5);raw=torch.zeros_like(base,requires_grad=True)
    with pytest.raises(ValueError,match='legal'):
        model.compose_chain(base,raw,None,limits,previous=inputs['robot'][...,58:87])
    chain=model.compose_chain(base,raw,None,limits,previous=previous)
    loss=model.objective(chain['u_exec'],base,chain['delta'],inputs['valid'],
        u_pre=chain['u_pre'],execution_bounds=chain['execution_bounds'])
    assert loss==0
    assert torch.isfinite(torch.autograd.grad(loss,raw)[0]).all()
    inputs['robot'][0,0,58]=0.  # Real invalid targets must still be rejected.
    with pytest.raises(ValueError,match='legal'):
        model.compose_chain(base,raw,None,limits,previous=residual_previous_targets(inputs,limits))
