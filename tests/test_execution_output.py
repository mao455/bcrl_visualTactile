import math
import torch
from visuotactile.residual_output import REACHABLE_TAIL_OUTPUT,proposal_delta
from tools.tight_execution_capacity import error_infimum


def test_capacity_infimum_includes_actual_tolerance_rays_and_holes():
    t=lambda x:torch.tensor(x,dtype=torch.float64)
    # Interior output identity is real; lower ray disappears at exactly -tol.
    b=t([-.000001,-.0000011,3.,.5]);radius=t([0.,0.,.1,.1]);y=t([0.,0.,1.,.8])
    box=t([[0.,1.]]).expand(4,2);limits=t([[-10.,10.]]).expand(4,2)
    torch.testing.assert_close(error_infimum(b,radius,y,box,limits),t([.000001,0.,0.,.2]))


def test_tail_preserves_nonsaturated_map_and_restores_physical_gradient():
    z=torch.tensor([0.,1.,-1.,9.,-9.],dtype=torch.float64,requires_grad=True)
    a=torch.tensor([1.,2.,100.,1000.,1000.],dtype=z.dtype);s=torch.ones_like(z)*.9
    box=torch.stack((torch.zeros_like(z),torch.ones_like(z)*.02),-1)
    d=proposal_delta(z,s,a,box,REACHABLE_TAIL_OUTPUT)
    torch.testing.assert_close(d[:3],(s*z.tanh())[:3])
    grad=torch.autograd.grad((d/a).sum(),z)[0]
    torch.testing.assert_close(grad[-2:],torch.ones(2,dtype=z.dtype)*.02)
    assert ((s/a)*z.tanh())[-1].abs()<.001 and (d/a)[-1].abs()>.1


def test_tail_finite_inverse_and_true_scalar_gradcheck():
    s=torch.tensor([.9],dtype=torch.double);a=torch.tensor([1000.],dtype=torch.double)
    bounds=torch.tensor([[0.,.02]],dtype=torch.double)
    z=torch.tensor([5.],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:proposal_delta(x,s,a,bounds,REACHABLE_TAIL_OUTPUT),(z,))
    limit=math.atanh(.95);required=torch.tensor([limit+(.02-.9/1000*.95)/.02],dtype=torch.double)
    torch.testing.assert_close(proposal_delta(required,s,a,bounds,REACHABLE_TAIL_OUTPUT)/a,torch.tensor([.02],dtype=torch.double))


def test_tail_checkpoint_and_actual_model_objective():
    import pytest
    from visuotactile.models.native_dp import TargetResidual,load_residual
    from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
    from visuotactile.execution_learning import PROJECTION_FY_LOSS
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29)*1000,action_offset=torch.zeros(29))
    loss=dict(PROJECTION_FY_LOSS,regularizer_weight=0.)
    model=TargetResidual(stats,[.9]*29,execution_adapter=REACHABLE_POSITION_EXECUTION,loss_contract=loss,output_contract=REACHABLE_TAIL_OUTPUT)
    z=torch.ones(1,1,29,requires_grad=True)*3.;base=torch.full_like(z,-.05);previous=torch.zeros_like(z)
    limits=torch.tensor([[-1.,1.]]).expand(29,2)
    c=model.compose_chain(base,z,None,limits,previous=previous)
    terms=model.objective_components(c['u_exec'],previous,c['delta'],torch.ones(1,1),u_pre=c['u_pre'],execution_bounds=c['execution_bounds'])
    gradient=torch.autograd.grad(terms['total'],z)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum()>0 and terms['regularizer']==0
    cfg=dict(loss_contract=loss,output_contract=REACHABLE_TAIL_OUTPUT,execution_adapter=REACHABLE_POSITION_EXECUTION)
    saved=dict(model=model.state_dict(),updates=1,residual_version=model.version,residual_loss_version=loss['version'],residual_output_version=REACHABLE_TAIL_OUTPUT['version'],configuration=cfg)
    loaded=load_residual(saved)
    torch.testing.assert_close(loaded.compose_chain(base,z,None,limits,previous=previous)['u_exec'],c['u_exec'])
    saved.pop('residual_output_version')
    with pytest.raises(ValueError,match='output identity'):load_residual(saved)
