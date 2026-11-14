"""Metric sampling must retain the actual full student's 5Hz noise clock."""
import pytest
import torch

from tools import tight_fresh_train as fresh
from visuotactile.anchor_sampling import labels, planning_anchors
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW


class NoiseProbe:
    """Same sole generator draw as NativePolicy.predict's deterministic DDIM."""
    horizon = 24
    action_scale = torch.ones(29)

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def predict(self, condition, generator, *, scheduler_index_mode):
        assert scheduler_index_mode == 'host'
        self.calls.append(condition.clone())
        return torch.randn((len(condition), self.horizon, 29),
                           device=condition.device, generator=generator) + condition[:, None, :1]


def ordinary_plans(model, conditions):
    generator = torch.Generator(device=conditions.device).manual_seed(7091)
    return torch.cat([model.predict(row[None], generator, scheduler_index_mode='host')
                      for row in conditions])


@pytest.mark.parametrize('count', [1, 4, 51])
def test_original_demonstration_and_validation_without_prefix_are_bitwise_unchanged(count):
    conditions = torch.arange(count, dtype=torch.float32)[:, None] * .01
    anchors = torch.arange(count) * 12
    before = ordinary_plans(NoiseProbe(), conditions)
    after = fresh.predict_on_native_clock(NoiseProbe(), conditions, anchors)
    assert torch.equal(before, after)


@pytest.mark.parametrize('kept', [[6, 7, 8], [0, 2, 5], [25, 26]])
def test_every_skipped_native_plan_consumes_its_noise_draw(kept):
    all_conditions = torch.arange(kept[-1] + 1, dtype=torch.float32)[:, None] * .01
    reference = ordinary_plans(NoiseProbe(), all_conditions)[kept]
    model = NoiseProbe()
    observed = fresh.predict_on_native_clock(model, all_conditions[kept], torch.tensor(kept) * 12)
    assert torch.equal(reference, observed)
    # Only eligible anchors denoise; prefix and mask gaps advance noise only.
    assert len(model.calls) == len(kept)


def test_metric_with_student_prefix_and_real_queue_flush_matches_full_clock(monkeypatch):
    n = 130
    record = dict(time=torch.arange(n, dtype=torch.float64) / 60,
        command=torch.zeros(n, 29), clean_sample=torch.zeros(n, 29),
        supervision_mask=torch.arange(n) >= 82,
        control_segment=(torch.arange(n) >= 80).long() + (torch.arange(n) >= 82).long(),
        target_window_contract=RECOVERY_ONSET_WINDOW,
        joint_limits=torch.tensor([-1000., 1000.]).repeat(29, 1))
    anchors = planning_anchors(record)
    assert anchors[0] == 72  # Expert starts80, real queue clears at82.
    conditions = torch.zeros(len(anchors), 1)
    monkeypatch.setattr(fresh, 'supervised_conditions', lambda model, rec: (conditions, anchors))
    result = fresh.metrics(NoiseProbe(), {'correction': record}, ['correction'])
    full = ordinary_plans(NoiseProbe(), torch.zeros((n + 11) // 12, 1))
    selected = full[anchors // 12]
    target, valid = labels(record, anchors, 24)
    error = (selected[:, :12] - target[:, :12])[valid[:, :12]]
    assert result['arm_rmse_rad'] == float(error[:, :7].square().mean().sqrt())
    assert result['hand_rmse_rad'] == float(error[:, 7:].square().mean().sqrt())
    # Old metrics wrongly reused tick0's draw at the first supervised anchor.
    old = ordinary_plans(NoiseProbe(), conditions)
    assert not torch.equal(old[0], selected[0])
    assert not record['supervision_mask'][:82].any()


@pytest.mark.parametrize('anchors', [[12, 0], [0, 0], [1], [-12]])
def test_invalid_planning_clock_rejected(anchors):
    with pytest.raises(ValueError, match='planning clock'):
        fresh.predict_on_native_clock(NoiseProbe(), torch.zeros(len(anchors), 1), torch.tensor(anchors))
