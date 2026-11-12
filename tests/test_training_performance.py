"""RNG identity is required before changing the physical DP microbatch."""
import pytest
import torch

from visuotactile.anchor_continue import diffusion_draws,dp_performance


@pytest.mark.parametrize('batch',[1,7,64])
def test_larger_microbatch_preserves_original_noise_and_timestep_draws(batch):
    original=torch.Generator().manual_seed(2049);candidate=torch.Generator().manual_seed(2049)
    noises=[];steps=[]
    for start in range(0,batch,4):
        count=min(4,batch-start)
        noises.append(torch.randn(count,24,29,generator=original))
        steps.append(torch.randint(100,(count,),generator=original))
    noise,ticks=diffusion_draws(batch,24,candidate,'cpu')
    assert torch.equal(noise,torch.cat(noises))
    assert torch.equal(ticks,torch.cat(steps))
    assert torch.equal(original.get_state(),candidate.get_state())
    for micro in (4,8,16):
        assert torch.equal(torch.cat([noise[s:s+micro] for s in range(0,batch,micro)]),torch.cat(noises))


def test_legacy_path_remains_explicit_and_profile_cannot_change_effective_exposure():
    assert dp_performance(None,64) is None
    spec=dict(version='fp32_legacy_rng_microbatch_v1',microbatch=16,diffusion_rng_microbatch=4,raw_frame_device_cache=True)
    assert dp_performance(spec,64)==spec
    with pytest.raises(ValueError):dp_performance(spec,32)
    with pytest.raises(ValueError):dp_performance(dict(spec,diffusion_rng_microbatch=16),64)
    with pytest.raises(ValueError):dp_performance(dict(spec,amp=True),64)
