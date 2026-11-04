"""Focused tests for new diagnostic boundaries, not repeated diffusion tests."""
import numpy as np
import torch

from tools.h24_diagnosis_common import first_sustained, prefix_comparison, prefix_feedback
from visuotactile.state_history import HistoryController


def test_sustained_threshold_ignores_single_spike_and_equality():
    assert first_sustained([0, 6, 0, 5, 6, 7, 8], 5) == 4
    assert first_sustained([6, 6], 5) is None
    assert first_sustained([5, 5, 5], 5) is None


def test_expert_feedback_uses_actual_postprocessed_targets_without_mutating_transition():
    class Controller:
        def observe_execution(self, transition):
            self.received = transition
    controller = Controller()
    target = torch.arange(29.)[None]
    transition = {'target_after_rad': target, 'command_clipped_pre_delay': -target}
    prefix_feedback(controller, transition)
    assert torch.equal(controller.command, target)
    assert torch.equal(controller.received['position_applied_rad'], target)
    assert 'position_applied_rad' not in transition
    assert torch.equal(transition['command_clipped_pre_delay'], -target)
    controller.command.add_(1)
    assert torch.equal(target, torch.arange(29.)[None])


def test_prefix_mismatch_rejected_before_student_control():
    keys = ('command', 'target', 'joint', 'velocity', 'object', 'contact', 'object_contact')
    rows = [{key:np.zeros(3) for key in keys} for _ in range(12)]
    ref = {key:np.zeros((12, 3)) for key in keys}
    assert prefix_comparison(rows, ref, 12)['passed']
    ref['joint'][5, 0] = 1e-3
    assert not prefix_comparison(rows, ref, 12)['passed']
