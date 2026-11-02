import copy
import pytest
import torch
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.state_full_run import should_stop
from visuotactile.state_full_training import split_ids, training_state, restore_training
from visuotactile.state_resources import policy


def test_busy_gpu_does_not_veto_memory_safe_work():
    row = dict(utc_epoch=100, gpu_free_mib=8000, ram_available_bytes=16*1024**3,
        rolling_60s_percent=99, rolling_300s_percent=99, external_gpu_processes=[dict(pid=42)])
    assert policy(row, 101) == ([], 2.)
    row['external_gpu_processes'] = []
    assert policy(row, 101) == ([], 0.)
    row['gpu_free_mib'] = 5000
    assert policy(row, 101)[0] == ['free_vram_below_6GiB']
    assert 'missing_or_stale_memory_telemetry' in policy(row, 110)[0]


def test_full_split_and_validation_stagnation():
    data = dict(splits=dict(train=list(range(320)), validation=list(range(320, 360)), test=list(range(360, 400))))
    train, validation = split_ids({}, data, False)
    assert len(train) == 320 and len(validation) == 40
    data['splits']['validation'][0] = 1
    with pytest.raises(AssertionError):
        split_ids({}, data, False)
    rows = [dict(update=update, validation=dict(selection_mse=loss))
        for update, loss in [(1000, .5), (3000, .499), (6000, .498)]]
    assert should_stop(rows)
    rows[-1]['validation']['selection_mse'] = .4
    assert not should_stop(rows)


def test_optimizer_ema_sampler_and_rng_resume_exact(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: [])
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda state: None)
    torch.manual_seed(91)
    model = torch.nn.Linear(3, 2)
    averaged = copy.deepcopy(model); ema = EMAModel(averaged, power=.75)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    generator = torch.Generator().manual_seed(4); noise = torch.Generator().manual_seed(5)
    def update():
        optimizer.zero_grad()
        values = torch.randn(7, 3, generator=noise)
        indices = torch.randperm(7, generator=generator)
        (model(values[indices])-torch.rand(7, 2)).square().mean().backward()
        optimizer.step(); ema.step(model)
    update()
    saved = training_state(model, averaged, optimizer, ema, generator, noise)
    update()
    expected = copy.deepcopy(model.state_dict()); expected_ema = copy.deepcopy(averaged.state_dict())
    restore_training(saved, model, averaged, optimizer, ema, generator, noise)
    update()
    for name in expected:
        assert torch.equal(expected[name], model.state_dict()[name])
        assert torch.equal(expected_ema[name], averaged.state_dict()[name])
    assert ema.optimization_step == 2
