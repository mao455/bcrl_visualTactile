from types import SimpleNamespace
import torch
import pytest
from visuotactile.paired_execution import reset_registered_sampler

def base():
    return SimpleNamespace(generator=torch.Generator().manual_seed(7091),model=SimpleNamespace(contract={'sampler_seed':7091}))

def test_default_retains_persistent_queue_stream(tmp_path):
    b=base();torch.randn(27,generator=b.generator);state=b.generator.get_state().clone()
    assert reset_registered_sampler(b,{},tmp_path,identity='case') is None
    assert torch.equal(state,b.generator.get_state()) and not list(tmp_path.iterdir())

def test_explicit_paired_case_restarts_before_any_draw(tmp_path):
    b=base();torch.randn(27,generator=b.generator)
    receipt=reset_registered_sampler(b,{'sampler_episode_seed':7091},tmp_path,identity='case')
    assert torch.equal(b.generator.get_state(),base().generator.get_state())
    assert receipt['extra_random_draws']==0 and (tmp_path/'sampler_episode_reset.json').exists()

@pytest.mark.parametrize('seed',[7092,True,'7091'])
def test_invalid_seed_rejects_without_mutating_stream(tmp_path,seed):
    b=base();state=b.generator.get_state().clone()
    with pytest.raises(ValueError):reset_registered_sampler(b,{'sampler_episode_seed':seed},tmp_path,identity='case')
    assert torch.equal(state,b.generator.get_state()) and not list(tmp_path.iterdir())
