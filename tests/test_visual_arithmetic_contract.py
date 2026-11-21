from types import SimpleNamespace
import json
import pytest
import torch
from visuotactile.visual_arithmetic import profile,validate,digest
from visuotactile.sampler_audit import record


def test_contract_binds_microbatch_tail_and_precision():
    c=profile();assert c['vision_microbatch']==2 and c['tail_policy']=='duplicate_last_real_lane'
    assert validate(c)==c and digest(c)!=digest(profile(1))
    changed=dict(c,cudnn_allow_tf32=False)
    with pytest.raises(ValueError):validate(changed)


def test_episode_rng_is_actual_persistent_position_without_extra_draw(tmp_path):
    g=torch.Generator().manual_seed(7091);torch.randn(1,24,29,generator=g)
    original=g.get_state().clone()
    b=SimpleNamespace(generator=g,model=SimpleNamespace(contract={'sampler_seed':7091}),next_step=12)
    record(tmp_path,b,'start',identity='queue_episode_2',vision_arithmetic=profile())
    assert torch.equal(g.get_state(),original)
    saved=torch.load(tmp_path/'sampler_rng_start.pt',weights_only=False)
    assert torch.equal(saved['generator_state'],original)
    assert json.loads((tmp_path/'sampler_rng_start.json').read_text())['initial_seed']==7091
    with pytest.raises(FileExistsError):record(tmp_path,b,'start',identity='queue_episode_2',vision_arithmetic=profile())
