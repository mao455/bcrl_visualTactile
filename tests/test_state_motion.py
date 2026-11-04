import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from visuotactile.dp_objectives import arm_temporal_objective
from visuotactile.models.native_dp import NativePolicy
from visuotactile.position_control import overlap_transition
from visuotactile.state_motion import plan_metrics


def small_sampler():
    class Constant(torch.nn.Module):
        def forward(self, sample, timestep, global_cond):return torch.full_like(sample,2.)
    model=NativePolicy.__new__(NativePolicy);torch.nn.Module.__init__(model)
    model.horizon=32;model.denoiser=Constant()
    model.noise=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='sample')
    model.sampler=DDIMScheduler.from_config(model.noise.config)
    model.action_scale=torch.full((29,),2.);model.action_offset=torch.full((29,),4.)
    return model


def test_ddim_default_matches_original_and_clips_normalized_x0_only():
    model=small_sampler();condition=torch.zeros(2,512)
    generator=torch.Generator().manual_seed(31)
    sample=torch.randn((2,32,29),generator=generator)
    model.sampler.set_timesteps(16)
    for timestep in model.sampler.timesteps:
        prediction=model.denoiser(sample,timestep,global_cond=condition)
        sample=model.sampler.step(prediction,timestep,sample,eta=0.).prev_sample
    expected=(sample-model.action_offset)/model.action_scale
    actual=model.predict(condition,torch.Generator().manual_seed(31))
    assert torch.equal(actual,expected)
    for steps in (16,32,100):
        clipped=model.predict(condition,torch.Generator().manual_seed(31),inference_steps=steps,clip_sample=True)
        assert torch.equal(clipped,torch.full_like(clipped,-1.5))  # radians outside [-1,1] are legal
    assert torch.equal(model.predict(condition,torch.Generator().manual_seed(31)),expected)
    assert model.noise.config.prediction_type=='sample' and not model.noise.config.clip_sample


def test_overlap_uses_corresponding_future_time_without_shift_or_extra_delay():
    old=torch.arange(32.)[None,:,None].expand(1,32,29)
    new=torch.full_like(old,100.)
    got=overlap_transition(old,new,12,4)
    assert torch.equal(got[0,:4,0],torch.tensor([34.,56.5,78.5,100.]))
    assert torch.equal(got[:,4:],new[:,4:])
    assert overlap_transition(old,new,12,0) is new
    assert overlap_transition(None,new,0,4) is new
    with pytest.raises(ValueError):overlap_transition(old,new,11,4)


def test_temporal_loss_is_not_position_mse_and_masks_invalid_pairs_and_hand():
    gt=torch.arange(32.)[None,:,None].expand(2,32,29).clone()
    valid=torch.ones(2,32,dtype=torch.bool);valid[:,20:]=False
    pred=gt.clone();pred[:,:,:7]+=7.;pred[:,:,7:]+=99.;pred[:,20:]=-500
    assert arm_temporal_objective(pred,gt,valid)==0
    pred[:,10,:7]+=1.;pred.requires_grad_()
    loss=arm_temporal_objective(pred,gt,valid);loss.backward()
    assert loss>0 and pred.grad[:,:,:7].abs().sum()>0
    assert pred.grad[:,:,7:].abs().sum()==0 and pred.grad[:,20:].abs().sum()==0




def test_overlap_metrics_distinguish_continuous_plan_and_boundary_jump():
    plans=np.stack([np.arange(t,t+32)[:,None]*np.ones((1,29)) for t in (0,12,24)])
    result=plan_metrics(plans)
    assert result['arm_overlap_rms_rad']==0
    assert result['arm_boundary_jump_rms_rad']==result['arm_internal_jump_rms_rad']==1
    assert result['arm_acceleration_rms_rad_s2']==0
    plans[1]+=2
    assert plan_metrics(plans)['arm_overlap_rms_rad']>0
