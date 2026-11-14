from types import SimpleNamespace

import pytest
import torch

from tools.tight_fresh_train import bind_geometry_supervision
from visuotactile.anchor_continue import geometry_selection
from visuotactile.anchor_sampling import labels, planning_anchors


@pytest.mark.parametrize('phase', range(12))
def test_fresh_geometry_uses_actual_suffix_including_partial_onset_and_tail(phase):
    n = 89
    ticks = torch.arange(n)
    first = 60 + phase
    records = {'recovery': dict(command=torch.zeros(n, 29), time=ticks / 60,
        supervision_mask=ticks >= first, control_segment=(ticks >= first).long(),
        target_window_contract='recovery_onset_commit12_v1')}
    context = dict(stage=torch.ones(n, dtype=torch.long), contact_mask=torch.ones(n, 5, dtype=torch.bool))
    objective = SimpleNamespace(loss_context={'recovery': context})
    report = bind_geometry_supervision(objective, records)
    assert objective.supervision_records is records
    assert report['recovery']['eligible_commit_count'] == n - first
    anchors = planning_anchors(records['recovery'])
    _, valid = labels(records['recovery'], anchors, 24)
    for seed in range(50):
        selection, _ = geometry_selection(objective, [('recovery', int(t)) for t in anchors], {'recovery': n}, seed)
        assert not (selection & ~valid).any()
        assert not selection[:, 12:].any()
        assert int(selection.sum()) == 2
