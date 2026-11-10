"""Historical EMA transfer remains a U-Net and starts a new q objective."""
import json
from pathlib import Path

import pytest
import torch

from visuotactile.compact import sha256
from visuotactile.narrow_data import actual_configuration_labels
from visuotactile.position_control import image_contract
from visuotactile.state_history import load_image_warm_start


SOURCE=Path('outputs/visuotactile/run/tight_vision_20261109_01/G00/training/best.pt')


@pytest.fixture(scope='module')
def source():
    if not SOURCE.exists():pytest.skip('Historical source stays local')
    assert sha256(SOURCE)=='4356acbcaabd95e97b1f236f853d724db5f3ad5d9effad2301fe0e1eca4ba452'
    return torch.load(SOURCE,map_location='cpu',weights_only=False)


def reached_statistics(saved):
    audit=json.loads(Path('research_docs/evidence/tight_refactor_20261110/transformer.json').read_text())['actual_configuration']
    assert audit['normalization_identity']=='q_reached_train166_affine_v1'
    return dict(robot_mean=saved['ema']['robot_mean'].clone(),robot_std=saved['ema']['robot_std'].clone(),
        action_scale=torch.tensor(audit['action_scale']),action_offset=torch.tensor(audit['action_offset']))


def test_real_ema_warm_start_preserves_parameters_and_new_target_gradient(source):
    stats=reached_statistics(source);contract=image_contract('I-HistoryDP',24,'G00','q_reached')
    rng=torch.get_rng_state().clone()
    model,audit=load_image_warm_start(source,contract,stats)
    assert torch.equal(rng,torch.get_rng_state())
    assert audit['new_modules']==[] and audit['denoiser_class']=='ConditionalUnet1D'
    assert audit['source_updates']==10132 and audit['new_optimizer_updates']==0
    assert not audit['completed_reached_policy'] and not audit['optimizer_state_restored']
    assert set(audit['normalizers_numerically_changed'])=={'action_scale','action_offset'}
    assert all(torch.equal(value,source['ema'][name]) for name,value in model.named_parameters())
    for name,value in model.named_buffers():
        assert torch.equal(value,stats[name] if name in stats else source['ema'][name])
    # Q00 and QG00 must begin with exactly the same learned tensors and buffers.
    other,_=load_image_warm_start(source,contract,stats)
    assert all(torch.equal(value,other.state_dict()[name]) for name,value in model.state_dict().items())
    del other
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
    assert len(optimizer.state)==0 and len(source['optimizer']['state'])>0
    data=json.loads((SOURCE.parents[2]/'state_data/data.json').read_text())
    identity=data['splits']['train'][0]
    assert identity not in data['splits']['test']
    reached,_=actual_configuration_labels(data['entries'][identity])
    # Real q labels and real transferred denoiser; synthetic condition isolates
    # the new clean-sample objective without regenerating visual/DP caches.
    condition=torch.zeros(1,512,requires_grad=True)
    loss=model.loss(condition,reached[:24][None],torch.ones(1,24,dtype=torch.bool),
        torch.Generator().manual_seed(71))
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(condition.grad).all() and condition.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.denoiser.parameters())
    assert sum(float(p.grad.abs().sum()) for p in model.denoiser.parameters())>0
    assert all(torch.equal(value,source['ema'][name]) for name,value in model.named_parameters())


@pytest.mark.parametrize('wrong',['G10','q_source','u_target','missing_ema','untrained'])
def test_warm_start_rejects_silent_reinterpretation(source,wrong):
    saved=dict(source);contract=image_contract('I-HistoryDP',24,'G00','q_reached')
    if wrong=='G10':saved['contract']=image_contract('I-HistoryDP',24,'G10')
    elif wrong=='q_source':saved['contract']=contract
    elif wrong=='u_target':contract=image_contract('I-HistoryDP',24,'G00','u_expert')
    elif wrong=='missing_ema':saved.pop('ema')
    else:saved['updates']=0
    with pytest.raises(ValueError):load_image_warm_start(saved,contract,reached_statistics(source))
