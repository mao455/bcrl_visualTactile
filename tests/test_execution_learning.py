import copy
import pytest
import torch
from visuotactile.execution_learning import (PROJECTION_FY_LOSS, projection_fy_elements,
    execution_expressibility, ideal_execution_bounds)
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION as ADAPTER, project_processed_position_targets
from visuotactile.models.native_dp import TargetResidual, load_residual
from test_target_residual import statistics, observations


@pytest.mark.parametrize('x,y,l,h',[(.3,.3,0,1),(3,1,0,1),(-3,0,0,1),
    (-3,1,0,1),(3,0,0,1),(3,.3,0,1),(3,1-1e-8,0,1),(-3,1e-8,0,1),
    (0,0,0,1),(1,1,0,1),(3,.2,.2,.2),(-3,.2,.2,.2)])
def test_real_autograd_matches_projection_gradient_and_finite_difference(x,y,l,h):
    x=torch.tensor([[x]],dtype=torch.float64,requires_grad=True);y=torch.tensor([[y]],dtype=torch.float64)
    b=torch.tensor([[[l,h]]],dtype=torch.float64);m=torch.ones(1)
    f=lambda z:projection_fy_elements(z,y,b,m)[0].sum()
    value=f(x);gradient=torch.autograd.grad(value,x)[0]
    torch.testing.assert_close(gradient,2*(x.detach().clamp(l,h)-y),atol=1e-12,rtol=1e-10)
    assert value>=0
    torch.autograd.gradcheck(f,(x,),eps=1e-6,atol=2e-6,rtol=1e-5)
    if float(x.detach().clamp(l,h))==float(y):assert value==0


def test_near_boundary_continuity_versus_exact_preimage_distance():
    x=torch.tensor([[100.]],dtype=torch.float64,requires_grad=True);b=torch.tensor([[[0.,1.]]])
    values=[]
    for eps in [1e-3,1e-6,1e-9,0]:
        y=torch.tensor([[1-eps]],dtype=torch.float64);v,_=projection_fy_elements(x,y,b,torch.ones(1))
        g=torch.autograd.grad(v.sum(),x)[0];values.append((v.item(),g.item()))
    assert values[-1]==(0,0) and values[-2][0]<1e-6
    assert abs(values[1][1]-2e-6)<1e-12
    assert (100-(1-1e-9))**2>9000  # exact inverse-image distance discontinuity


def test_roundoff_mask_and_unreachable_labels_are_explicit():
    x=torch.tensor([[3.],[3.]],requires_grad=True);b=torch.tensor([[[0.,1.]],[[0.,1.]]]);m=torch.tensor([1.,0.])
    y=torch.tensor([[1.+5e-7],[100.]])
    value,difference=projection_fy_elements(x,y,b,m)
    assert value.sum()==0 and 0<difference[0]<1e-6
    with pytest.raises(ValueError,match='Unreachable'):
        projection_fy_elements(x,torch.tensor([[1.01],[100.]]),b,m)
    with pytest.raises(ValueError,match='invalid'):
        projection_fy_elements(x,y,b.flip(-1),m)


def test_v1_tolerance_is_not_claimed_monotone_clip():
    p=torch.zeros(1,29,dtype=torch.float64);limits=torch.tensor([[-1.,1.]],dtype=torch.float64).expand(29,2)
    b=ideal_execution_bounds(p,limits,ADAPTER);upper=b[...,1];eps=ADAPTER['roundoff_tolerance_rad']
    inner=upper+.5*eps;outer=upper+1.5*eps
    a=project_processed_position_targets(inner,p,limits,ADAPTER)
    c=project_processed_position_targets(outer,p,limits,ADAPTER)
    assert torch.equal(a,inner) and torch.equal(c,upper) and (a>c).all()
    loss,_=projection_fy_elements(inner,inner,b,torch.ones(1))
    assert torch.equal(loss,torch.zeros_like(loss))


def test_expressibility_uses_execution_image_and_open_domain():
    base=torch.tensor([3.,.1,0.,0.,0.]);s=torch.tensor([.1,.1,2.,1.,.5]);y=torch.tensor([1.,0.,1.,1.,1.]);b=torch.tensor([[0.,1.]]).expand(5,2)
    r=execution_expressibility(base,s,torch.ones(5),y,b)
    assert r['legacy_pre_range_exceeded'][0] and r['stable'][0]
    assert r['limit_only'][1] and r['stable'][2] and r['limit_only'][3] and r['unreachable'][4]
    assert not (r['unreachable']&r['stable']).any()


def test_actual_model_compose_objective_checkpoint_and_tanh_gradient():
    model=TargetResidual(statistics(),[.2]*7+[.9]*22,execution_adapter=ADAPTER,loss_contract=PROJECTION_FY_LOSS)
    limits=torch.tensor([[-1.,1.]]).expand(29,2);previous=torch.zeros(1,29);base=torch.full((1,29),.5)
    raw=torch.zeros(1,29,requires_grad=True)
    c=model.compose_chain(base,raw,None,limits,previous=previous)
    target=c['execution_bounds'][...,0]
    values=model.objective_components(c['u_exec'],target,c['delta'],torch.ones(1),u_pre=c['u_pre'],execution_bounds=c['execution_bounds'])
    gpre,gz=torch.autograd.grad(values['fy_fit'],(c['u_pre'],raw),retain_graph=True)
    assert (gpre>0).all() and (gz>0).all()
    gold=torch.autograd.grad(values['exec_fit'],raw)[0];assert torch.equal(gold,torch.zeros_like(gold))
    saturated=torch.full((1,29),100.,requires_grad=True)
    c=model.compose_chain(base,saturated,None,limits,previous=previous)
    loss=model.objective(c['u_exec'],target,c['delta'],torch.ones(1),u_pre=c['u_pre'],execution_bounds=c['execution_bounds'])
    gp,gz=torch.autograd.grad(loss,(c['u_pre'],saturated));assert (gp>0).all() and (gz==0).all()
    payload=dict(model=model.state_dict(),updates=2,residual_version=model.version,
        residual_loss_version=PROJECTION_FY_LOSS['version'],configuration=dict(loss_contract=PROJECTION_FY_LOSS,execution_adapter=ADAPTER))
    loaded=load_residual(payload);assert loaded.loss_contract==PROJECTION_FY_LOSS
    payload['residual_loss_version']='wrong'
    with pytest.raises(ValueError,match='identity'):load_residual(payload)


def test_output_head_does_not_enter_recurrent_state():
    model=TargetResidual(statistics(),[.2]*29);other=copy.deepcopy(model)
    with torch.no_grad():other.policy[-1].bias.fill_(4)
    inputs=observations(steps=3,batch=1)
    a,sa=model(inputs);b,sb=other(inputs)
    assert not torch.equal(a,b)
    def equal(a,b):
        if isinstance(a,dict):return all(equal(a[k],b[k]) for k in a)
        return torch.equal(a,b)
    assert equal(sa,sb)


def test_nested_box_at_joint_endpoints_and_invalid_previous():
    lim=torch.tensor([[-1.,1.]],dtype=torch.float64).expand(29,2)
    for previous in [lim[:,0][None],lim[:,1][None],torch.zeros(1,29,dtype=torch.float64)]:
        b=ideal_execution_bounds(previous,lim,ADAPTER)
        assert (b[...,0]>=lim[:,0]).all() and (b[...,1]<=lim[:,1]).all()
        x=torch.linspace(-4,4,29,dtype=torch.float64)[None]
        nested=torch.maximum(torch.minimum(x.clamp(-1,1),b[...,1]),b[...,0])
        direct=torch.maximum(torch.minimum(x,b[...,1]),b[...,0]);assert torch.equal(nested,direct)
    with pytest.raises(ValueError,match='legal'):ideal_execution_bounds(torch.full((1,29),2.),lim,ADAPTER)


def test_actual_tolerance_inverse_has_missing_endpoint_band_and_joint_rays():
    from visuotactile.execution_learning import actual_execution_expressibility
    # Every proposal is just above upper but inside the identity allowance:
    # ideal clip would reach1; actual P_tol does not, so it is NOT equivalent.
    base=torch.tensor([1.+.5e-6,1.+2e-6,3.,0.],dtype=torch.float64)
    radius=torch.tensor([.1e-6,.1e-6,.1,1.],dtype=torch.float64)
    b=torch.tensor([[0.,1.]],dtype=torch.float64).expand(4,2)
    limits=torch.tensor([[-5.,5.]],dtype=torch.float64).expand(4,2)
    result=actual_execution_expressibility(base,radius,torch.ones(4),torch.ones(4),b,limits)
    assert result['unreachable'][0] and result['stable'][1] and result['stable'][2] and result['limit_only'][3]
    limits=limits.clone();limits[2]=torch.tensor([0.,1.])
    result=actual_execution_expressibility(base,radius,torch.ones(4),torch.ones(4),b,limits)
    assert result['stable'][2]  # joint-limit inverse ray
