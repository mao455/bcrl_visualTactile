import copy
import pytest
import torch
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.state_full_run import should_stop, recover_preaction_start
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


def test_preaction_recovery_is_bounded_and_preserves_evidence(tmp_path):
    import time
    def attempt(name):
        path = tmp_path/name; path.mkdir()
        (path/'runtime_controls.json').write_text('{}')
        (tmp_path/(name+'.log')).write_text('std::system_error FileThread abort')
        return path
    first = attempt('expert_0')
    (first/'started.json').write_text('{}')
    assert not recover_preaction_start(tmp_path, 'expert_0', time.time()+60)
    (first/'started.json').unlink()
    assert recover_preaction_start(tmp_path, 'expert_0', time.time()+60)
    assert (tmp_path/'startup_recoveries/expert_0/worker.log').is_file()
    attempt('expert_0')
    assert not recover_preaction_start(tmp_path, 'expert_0', time.time()+60)
    for name in ('expert_1', 'expert_2'):
        attempt(name)
        assert recover_preaction_start(tmp_path, name, time.time()+60)
    attempt('expert_3')
    assert not recover_preaction_start(tmp_path, 'expert_3', time.time()+60)


def test_partial_expert_results_are_not_lost(tmp_path):
    import importlib.util
    import json
    from pathlib import Path
    spec = importlib.util.spec_from_file_location('state_report', Path('tools/state_history_report.py'))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    directory = tmp_path/'expert'; directory.mkdir()
    (directory/'progress.json').write_text(json.dumps(dict(trials=[
        dict(cohort='fresh', legal=True, success=True, native_lift=True, native_subgoals=2)])))
    result = module.summarize(tmp_path, tmp_path/'report')
    assert result['stages']['expert']['complete'] is False
    assert result['stages']['expert']['cohorts']['fresh']['successes'] == 1
