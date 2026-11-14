import copy
import json
import time

import pytest
import torch

from tools import tight_trajectory_recovery_probe as probe
from visuotactile.checkpointing import training_state
from visuotactile.models.vendor.ema_model import EMAModel
from test_trajectory_training_gate import complete_fixture, artifacts


def objects():
    model = torch.nn.Linear(3, 2); model.contract = dict(test='diagnostic')
    averaged = copy.deepcopy(model).eval().requires_grad_(False)
    return (model, averaged, torch.optim.AdamW(model.parameters(), lr=3e-5, betas=(.95, .999)),
        EMAModel(averaged, power=.75), torch.Generator().manual_seed(11), torch.Generator().manual_seed(12))


VISIT = dict(episode='train_only', anchors=2, target_ticks=30, visual_frames=15, loss_weight=1.)


def update(values):
    model, _, opt, ema, _, noise = values
    opt.zero_grad(set_to_none=True)
    x = torch.randn(5, 3, generator=noise)
    loss = (model(x)-torch.randn(5, 2, generator=noise)).square().mean()
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step(); ema.step(model)
    return dict(loss=float(loss.detach()), gradient_norm=float(norm), visits=[dict(VISIT)])


def boundary():
    values = objects(); update(values)
    return training_state(*values, updates=1, visits_seen=2, full_trajectory_batch={'test': True})


def test_five_real_cpu_adam_ema_transactions_use_same_boundary_and_exact_gradients():
    saved = boundary(); unchanged = copy.deepcopy(saved)
    writes = {}; makes = []; updates = []; profiles = []; progress = dict(confirmed_updates=0, active=None)
    def make():
        values = objects(); makes.append(id(values[0])); return values
    def step(values):
        updates.append(id(values[0])); return update(values)
    result = probe.probe_updates(saved, make, step,
        lambda deterministic: profiles.append(deterministic) or {'deterministic': deterministic},
        lambda name, value: writes.__setitem__(name, value), lambda:None, lambda:None, progress, VISIT)
    assert len(result) == len(updates) == 5 and len(makes) == 4
    assert updates[0] == updates[1] and profiles == [False, False, False, True, True]
    assert progress == dict(confirmed_updates=5, active=None)
    assert all(row['state_exact'] is not False and row['gradients_exact'] is not False for row in result)
    assert probe.all_exact(probe.compare(saved, unchanged))
    assert all(probe.all_exact(writes[name+'_boundary.json']['comparisons']) for name,_,_ in probe.ARMS)
    assert writes['default_first.json']['gradient_capture'] == 'after clipping, before AdamW'
    squared = sum(v['l2']**2 for v in writes['default_first.json']['gradient_summary'].values())
    assert squared <= 1.00001
    assert probe.all_exact(writes['arithmetic_profiles_comparison.json']['states'])


def test_exact_failure_reports_size_without_relaxing_to_tolerance():
    a = torch.tensor([0., 1., 2.]); b = torch.tensor([1e-8, 1., 2.000001])
    row = probe.compare(a, b)['state']
    assert not row['exact'] and row['different_elements'] == 2
    assert row['maximum_absolute'] == pytest.approx(float((a-b).abs().max()))
    assert row['maximum_relative'] == 1.
    rng = probe.compare(torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([1, 3], dtype=torch.uint8))['state']
    assert not rng['exact'] and rng['maximum_absolute'] == 1. and rng['maximum_relative'] == pytest.approx(1/3)
    assert not probe.all_exact(probe.compare(torch.zeros(1), torch.tensor([float('nan')])))
    assert probe.compare(torch.zeros(1), torch.tensor([float('nan')]))['state']['maximum_absolute'] is None
    with pytest.raises(ValueError, match='structure'): probe.compare({'ema': a}, {})
    with pytest.raises(ValueError, match='structure'): probe.compare(a, b.double())


def test_unsupported_deterministic_branch_stops_without_retry_and_preserves_prior_receipts():
    writes = {}; progress = dict(confirmed_updates=0, active=None)
    def configure(deterministic):
        if deterministic: raise RuntimeError('unsupported deterministic CUDA operator')
        return {'deterministic': False}
    with pytest.raises(RuntimeError, match='unsupported'):
        probe.probe_updates(boundary(), objects, update, configure,
            lambda name, value: writes.__setitem__(name, value), lambda:None, lambda:None, progress, VISIT)
    assert progress['confirmed_updates'] == 3
    assert progress['active'] == dict(arm='deterministic_first', optimizer_step_may_be_partial=False)
    assert 'default_new_objects.json' in writes and 'deterministic_first.json' not in writes


def test_bad_boundary_and_bad_exposure_cannot_be_published():
    saved = boundary(); saved['updates'] = 2
    with pytest.raises(ValueError, match='first-update boundary'):
        probe.probe_updates(saved, objects, update, lambda _: {}, lambda *a:None,
            lambda:None, lambda:None, dict(confirmed_updates=0, active=None), VISIT)
    wrong = dict(VISIT, target_ticks=31)
    with pytest.raises(ValueError, match='exposure'):
        probe.probe_updates(boundary(), objects, update, lambda _: {}, lambda *a:None,
            lambda:None, lambda:None, dict(confirmed_updates=0, active=None), wrong)


def source_fixture(tmp_path, monkeypatch):
    root = tmp_path/'session'; root.mkdir()
    (root/'session.json').write_text(json.dumps(dict(deadline=time.time()+3600)))
    original = root/'gate'; original.mkdir()
    point = original/'diagnostic_boundary.pt'; torch.save(boundary(), point)
    receipt = dict(source_boundary=str(point))
    monkeypatch.setattr(probe, 'admission', lambda *args:receipt)
    return root, original


def test_check_only_does_not_initialize_cuda_and_never_replaces_output(tmp_path, monkeypatch):
    root, original = source_fixture(tmp_path, monkeypatch); output = root/'probe'
    result = probe.run(original, output, root, check_only=True)
    assert result['complete'] and not result['cuda_initialized']
    assert result['confirmed_optimizer_updates'] == 0 and result['adoption'] is False
    with pytest.raises(FileExistsError): probe.run(original, output, root, check_only=True)


@pytest.mark.parametrize('reason', ['stop', 'deadline'])
def test_original_deadline_and_stop_exit_before_admission(tmp_path, monkeypatch, reason):
    root, original = source_fixture(tmp_path, monkeypatch)
    if reason == 'stop': (root/'STOP').touch()
    else: (root/'session.json').write_text(json.dumps(dict(deadline=time.time()-1)))
    monkeypatch.setattr(probe, 'admission', lambda *a:pytest.fail('Stopped before admission'))
    with pytest.raises(InterruptedError): probe.run(original, root/'probe', root, check_only=True)
    failed = probe.read(root/'probe/failed.json')
    assert failed['confirmed_updates'] == 0 and failed['active'] is None


def test_missing_launcher_environment_fails_before_cuda(tmp_path, monkeypatch):
    root, original = source_fixture(tmp_path, monkeypatch)
    monkeypatch.delenv('CUBLAS_WORKSPACE_CONFIG', raising=False)
    with pytest.raises(ValueError, match='launcher CUBLAS'): probe.run(original, root/'probe', root)
    assert not torch.cuda.is_initialized()
    assert probe.read(root/'probe/failed.json')['confirmed_updates'] == 0


def test_profile_explicitly_separates_nondeterministic_and_deterministic(monkeypatch):
    monkeypatch.setenv('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    previous = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    try:
        for flag in (False, True):
            row = probe.configure_arithmetic(flag)
            assert row['deterministic_algorithms'] is flag and row['cudnn_deterministic'] is flag
            assert not row['cudnn_benchmark'] and not row['deterministic_warn_only']
            assert not row['matmul_tf32'] and not row['cudnn_tf32']
            assert row['cublas_workspace_config'] == ':4096:8'
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = previous[2:4]
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous[4:]
    assert not torch.cuda.is_initialized()


def test_source_failure_requires_exact_original_boundary_and_matching_rng():
    started = dict(version='trajectory_batch_two_training_recovery_gate_v1', role='B')
    result = dict(complete=True, passed=False, role='B',
        optimizer_steps_confirmed=dict(serial_reference=2, batched=2, restored=1),
        comparisons={'state': {'passed': True}}, boundary_restoration_comparisons={'state': {'passed': True}},
        recovery_comparisons={'model': {'passed': False}, 'noise_rng': {'passed': True}})
    probe.validate_failure(started, result)
    result['recovery_comparisons']['noise_rng']['passed'] = False
    with pytest.raises(ValueError, match='RNG'): probe.validate_failure(started, result)
    result['recovery_comparisons']['noise_rng']['passed'] = True
    result['boundary_restoration_comparisons']['state']['passed'] = False
    with pytest.raises(ValueError, match='boundary'): probe.validate_failure(started, result)


@pytest.mark.parametrize('drift', [None, 'projection', 'helper', 'boundary'])
def test_admission_binds_actual_original_inputs_projection_helpers_and_boundary(artifacts, drift):
    from tools import tight_trajectory_training_gate as gate
    root, frozen = complete_fixture(artifacts); original = root/'gate'
    gate.run(root, 'B', frozen, original, root, check_only=True)
    point = original/'diagnostic_boundary.pt'; torch.save(boundary(), point)
    result = dict(complete=True, passed=False, role='B',
        optimizer_steps_confirmed=dict(serial_reference=2, batched=2, restored=1),
        comparisons={'state': {'passed': True}}, boundary_restoration_comparisons={'state': {'passed': True}},
        recovery_comparisons={'model': {'passed': False}, 'noise_rng': {'passed': True}},
        boundary_checkpoint_sha256=probe.sha256(point), reports={'batched': [{},
            dict(visits=[dict(VISIT, episode='old1')])]})
    (original/'result.json').write_text(json.dumps(result))
    if drift == 'projection': (original/'train_only/numeric.pt').write_bytes(b'drift')
    if drift == 'boundary': point.write_bytes(b'drift')
    if drift == 'helper':
        receipt = probe.read(original/'admission.json')
        receipt['explicit_candidate_helpers']['trajectory_training']['sha256'] = '0'*64
        (original/'admission.json').write_text(json.dumps(receipt))
    if drift:
        with pytest.raises(ValueError, match='SHA256'): probe.admission(original, root, lambda:None)
    else:
        receipt = probe.admission(original, root, lambda:None)
        assert receipt['singleton'] == ['old1']
        assert receipt['source_boundary_sha256'] == probe.sha256(point)
        assert receipt['expected_visit'] == dict(VISIT, episode='old1')
        assert not torch.cuda.is_initialized()
