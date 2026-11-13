import pytest
from tools import tight_span_dose as runner


def test_budget_never_extends_compute_deadline():
    with pytest.raises(TimeoutError):
        runner.require_budget(runner.COMPUTE_DEADLINE-10, runner.COMPUTE_DEADLINE+100, 10)


def test_stage_reserve_is_enforced():
    with pytest.raises(TimeoutError):
        runner.require_budget(100, 200, 100)
    runner.require_budget(100, 201, 100)


def test_unique_root_and_existing_attempt_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError):
        runner.checked_root(tmp_path)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='overwrite'):
        runner.prepare(tmp_path)


def test_fixed_plan_has_fresh_unique_seeds(monkeypatch):
    monkeypatch.setattr(runner, 'direct_configuration', lambda domain, seed: {'seed': seed})
    plan = runner.evaluation_plan({})
    assert len(plan) == 10
    assert [row['spec']['seed'] for row in plan] == list(range(27200000, 27200010))
    assert len({row['identity'] for row in plan}) == 10
    assert all(row['spec']['split'] == 'new' for row in plan)
    assert sum(row['record_video'] for row in plan) == 1
    assert runner.ORDER == (('reference', 'QG01'), ('continued', 'G01'),
        ('continued', 'QG01'), ('reference', 'G01'))
