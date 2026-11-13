"""Actual encoder tests: intervening frames must matter without future leakage."""
from types import SimpleNamespace
import pytest
import torch
from visuotactile.models.native_dp import NativePolicy, encode_segment
from visuotactile.position_control import continuous_image_contract, image_contract


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(91)
    m=NativePolicy('I-HistoryDP',24,modality='G00').eval()
    m.contract=continuous_image_contract('u_expert')
    return m


def episode(length=25):
    torch.manual_seed(42)
    patches=torch.randn(length,3,384,18,32)*.01
    depth=torch.rand(length,3,2,18,32)
    robot=torch.randn(length,178)*.01;robot[:,176]=1
    record=dict(robot=robot,time=torch.arange(length,dtype=torch.float64)/60)
    def frame(t,device):
        return dict(patches=patches[t].to(device),depth=depth[t].to(device),
            time=record['time'][t].repeat(3).to(device),valid=torch.ones(3,dtype=torch.bool,device=device))
    return SimpleNamespace(record=record,frame=frame),patches


def test_every_intervening_frame_changes_persistent_planning_history(model):
    e,p=episode()
    with torch.no_grad():
        baseline,anchors,_=encode_segment(model,e,0,25)
        original=p[1].clone();p[1].add_(.3)  # no DP planning here; beyond the latest-three read window at t12
        changed,_,_=encode_segment(model,e,0,25)
        assert torch.equal(baseline[0],changed[0])
        assert not torch.allclose(baseline[1],changed[1],atol=1e-7,rtol=0)
        p[1].copy_(original);p[13:].add_(.8)
        future,_,_=encode_segment(model,e,0,25)
        assert torch.allclose(baseline[:2],future[:2],atol=1e-7,rtol=0)
        assert anchors.tolist()==[0,12,24]


def test_full_prefix_training_equals_online_and_nonanchor_frame_is_required(model):
    e,_=episode()
    with torch.no_grad():
        offline,_,_=encode_segment(model,e,0,25)
        suffix,_,_=encode_segment(model,e,12,25)
        memory=None;online=[]
        for t in range(25):
            memory=model.observe(e.record['robot'][t],e.record['time'][t],t,e.frame(t,'cpu'),memory)
            if t%12==0:online.append(model.condition(memory,e.record['time'][t]))
        assert torch.equal(offline,torch.cat(online))
        assert torch.equal(offline[1:],suffix)
        memory=model.observe(e.record['robot'][0],e.record['time'][0],0,e.frame(0,'cpu'))
        with pytest.raises(ValueError,match='visual update'):
            model.observe(e.record['robot'][1],e.record['time'][1],1,None,memory)
    assert image_contract('I-HistoryDP',24,'G00','u_expert')!=model.contract


def test_real_gradient_reaches_intermediate_visual_input(model):
    e,p=episode(13);p.requires_grad_();model.train()
    try:
        c,_,_=encode_segment(model,e,0,13)
        grad=torch.autograd.grad(c[-1,0],p)[0]
        assert torch.isfinite(grad).all() and grad[1].abs().sum()>0
    finally:model.eval()
