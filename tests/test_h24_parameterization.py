import copy
import io

import pytest
import torch
from diffusers import DDIMScheduler, DDPMScheduler

from visuotactile.dp_objectives import masked_objectives
from visuotactile.models.native_dp import NativePolicy
from visuotactile.position_control import contract
from visuotactile.state_history import PersistentState


class SmallDenoiser(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))
        self.last = None

    def forward(self, sample, timestep, global_cond):
        self.last = self.weight*sample+global_cond[:, :1, None]
        self.last.retain_grad()
        return self.last


def test_known_noise_and_ddim_clean_reconstruction():
    scheduler = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
        prediction_type='epsilon', clip_sample=False)
    clean = torch.linspace(-.7, .8, 58).reshape(2, 1, 29)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(8))
    timestep = torch.tensor([48, 48])
    alpha = scheduler.alphas_cumprod[48]
    noisy = scheduler.add_noise(clean, noise, timestep)
    assert torch.allclose(noisy, alpha.sqrt()*clean+(1-alpha).sqrt()*noise)
    sampler = DDIMScheduler.from_config(scheduler.config)
    sampler.set_timesteps(16)
    result = sampler.step(noise, 48, noisy, eta=0.)
    assert torch.allclose(result.pred_original_sample, clean, atol=1e-6)
    sample_sampler = DDIMScheduler.from_config(scheduler.config, prediction_type='sample')
    sample_sampler.set_timesteps(16)
    equivalent = sample_sampler.step(clean, 48, noisy, eta=0.)
    assert torch.allclose(result.prev_sample, equivalent.prev_sample, atol=1e-6)


@pytest.mark.parametrize('prediction_type', ['sample', 'epsilon'])
def test_actual_objective_mask_randomness_and_gradient(monkeypatch, prediction_type):
    monkeypatch.setattr('visuotactile.models.native_dp.ConditionalUnet1D', SmallDenoiser)
    model = NativePolicy('S-NativeDP', 24, prediction_type)
    command = torch.randn(4, 24, 29)
    valid = torch.ones(4, 24, dtype=torch.bool); valid[-1, 8:] = False
    condition = torch.randn(4, 512, requires_grad=True)
    generator = torch.Generator().manual_seed(9)
    expected_generator = torch.Generator().manual_seed(9)
    noise = torch.randn(command.shape, generator=expected_generator)
    timesteps = torch.randint(100, (4,), generator=expected_generator)
    clean = torch.where(valid[..., None], command, 0.)
    expected_prediction = model.denoiser.weight*model.noise.add_noise(clean, noise, timesteps)+condition[:, :1, None]
    target = noise if prediction_type == 'epsilon' else clean
    expected = masked_objectives(expected_prediction, target, valid)['B']
    loss = model.loss(condition, command, valid, generator)
    assert torch.equal(loss, expected)
    loss.backward()
    assert model.denoiser.weight.grad.abs() > 0
    assert condition.grad[:2].norm() > 0 and condition.grad[2:].norm() > 0
    assert torch.count_nonzero(model.denoiser.last.grad[~valid]) == 0
    assert len(timesteps.unique()) > 1
    assert not torch.equal(noise[0], noise[1])
    assert torch.equal(generator.get_state(), expected_generator.get_state())


def test_parameterization_changes_no_initialized_tensor(monkeypatch):
    monkeypatch.setattr('visuotactile.models.native_dp.ConditionalUnet1D', SmallDenoiser)
    torch.manual_seed(20261102); sample = PersistentState('dp', contract(24))
    torch.manual_seed(20261102); epsilon = PersistentState('dp', contract(24, True, 'epsilon'))
    assert dict(epsilon.contract, prediction_type='sample') == sample.contract
    assert sample.noise.config.prediction_type == 'sample'
    assert epsilon.noise.config.prediction_type == epsilon.sampler.config.prediction_type == 'epsilon'
    for name, value in sample.state_dict().items():
        assert torch.equal(value, epsilon.state_dict()[name])
    invalid = copy.deepcopy(epsilon.contract); invalid['horizon'] = 32
    with pytest.raises(ValueError):
        PersistentState('dp', invalid)


def test_balanced_weight_and_padding_gradient():
    prediction = torch.zeros(2, 24, 29, requires_grad=True)
    target = torch.ones_like(prediction); target[..., 7:] = 2
    valid = torch.ones(2, 24, dtype=torch.bool); valid[1, 6:] = False
    loss = masked_objectives(prediction, target, valid)['B']
    assert loss == 2.5
    loss.backward()
    assert torch.count_nonzero(prediction.grad[~valid]) == 0
    assert torch.allclose(prediction.grad[0, 0, :7], torch.full((7,), -1/(30*7)))
    assert torch.allclose(prediction.grad[0, 0, 7:], torch.full((22,), -2/(30*22)))


def test_saved_epsilon_contract_roundtrip_rejects_sample_restore(monkeypatch):
    from visuotactile.models.vendor.ema_model import EMAModel
    from visuotactile.state_full_training import training_state, restore_training
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda state: None)
    model=torch.nn.Linear(3,2);model.contract=contract(24,True,'epsilon')
    averaged=copy.deepcopy(model);ema=EMAModel(averaged,power=.75)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
    generator=torch.Generator().manual_seed(7);noise=torch.Generator().manual_seed(8)
    saved=training_state(model,averaged,optimizer,ema,generator,noise,
        updates=0,schedule_order=[2,0,1],schedule_cursor=1)
    stream=io.BytesIO();torch.save(saved,stream);stream.seek(0)
    restored=torch.load(stream,weights_only=False)
    restore_training(restored,model,averaged,optimizer,ema,generator,noise)
    assert restored['contract']['prediction_type']=='epsilon'
    assert restored['schedule_order']==[2,0,1] and restored['schedule_cursor']==1
    assert torch.equal(generator.get_state(),saved['schedule_rng'])
    model.contract=contract(24)
    with pytest.raises(ValueError,match='Checkpoint contract mismatch'):
        restore_training(restored,model,averaged,optimizer,ema,generator,noise)
