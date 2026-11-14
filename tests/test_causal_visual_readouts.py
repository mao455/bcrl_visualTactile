"""Actual attention must preserve online causality and every-tick gradients."""
import pytest
import torch

from visuotactile.continuous_vision import causal_visual_readouts
from visuotactile.models.native_dp import HistoryVision


def fixture():
    torch.manual_seed(319)
    vision=HistoryVision()
    tokens=torch.randn(17,3,48,128,requires_grad=True)
    times=torch.arange(17,dtype=torch.float64)/60
    camera_times=times[:,None].expand(-1,3)-torch.tensor([0.,.001,.002])
    valid=torch.ones(17,3,dtype=torch.bool)
    valid[4,1]=False
    return vision,tokens,times,camera_times,valid


def test_all_tick_reads_match_online_with_ages_and_partial_camera_validity():
    vision,tokens,times,ct,cv=fixture()
    indices=torch.arange(17)
    batch=causal_visual_readouts(vision,tokens,times,ct,cv,indices,batch_size=4)
    serial=torch.cat([vision.read(tokens[max(0,t-2):t+1],
        times[t]-ct[max(0,t-2):t+1],cv[max(0,t-2):t+1]) for t in indices])
    torch.testing.assert_close(batch,serial,rtol=1e-5,atol=5e-7)
    legacy_indices=indices[:,None]+torch.arange(-2,1)
    selected=legacy_indices.clamp_min(0)
    from visuotactile.continuous_vision import read_batch
    legacy=read_batch(vision,tokens[selected],times[indices,None,None]-ct[selected],
        cv[selected]&(legacy_indices>=0)[...,None])
    # The default path retains the old planner's single batched call exactly.
    assert torch.equal(legacy,causal_visual_readouts(vision,tokens,times,ct,cv,indices))


def test_every_tick_has_gradient_without_crossing_future_or_last_three_boundary():
    vision,tokens,times,ct,cv=fixture()
    output=causal_visual_readouts(vision,tokens,times,ct,cv,torch.arange(17),batch_size=4)
    gradient=torch.autograd.grad(output.square().mean(),tokens,retain_graph=True)[0]
    assert torch.isfinite(gradient).all()
    assert (gradient.flatten(1).abs().sum(1)>0).all()
    last=torch.autograd.grad(output[12].square().sum(),tokens)[0]
    assert torch.count_nonzero(last[:10])==0 and torch.count_nonzero(last[13:])==0
    assert (last[10:13].flatten(1).abs().sum(1)>0).all()
    assert torch.count_nonzero(gradient[4,1])==0


def test_reject_future_camera_empty_reads_and_invalid_index():
    vision,tokens,times,ct,cv=fixture()
    future=ct.clone();future[5,0]+=1
    with pytest.raises(ValueError,match='causal'):
        causal_visual_readouts(vision,tokens,times,future,cv,torch.arange(17))
    for indices in (torch.tensor([],dtype=torch.long),torch.tensor([17]),torch.tensor([-1])):
        with pytest.raises(ValueError,match='indices'):
            causal_visual_readouts(vision,tokens,times,ct,cv,indices)
    invalid=cv.clone();invalid[:3]=False
    with pytest.raises(ValueError,match='No valid'):
        causal_visual_readouts(vision,tokens,times,ct,invalid,torch.tensor([2]))
