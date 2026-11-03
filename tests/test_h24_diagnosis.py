"""Focused tests for new diagnostic boundaries, not repeated diffusion tests."""
import numpy as np
import torch

from tools.h24_diagnosis_common import first_sustained, prefix_comparison, prefix_feedback
from tools.h24_diagnosis_worker import CapturedController, append_policy_inputs
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


def test_capture_returns_same_output_preserves_rng_and_restores_scheduler(monkeypatch):
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    original = DDIMScheduler.step
    holder = CapturedController.__new__(CapturedController)
    holder.generator = torch.Generator().manual_seed(7091)
    holder.bank = type('Bank', (), {'lanes': {}})()
    expected = torch.arange(29.)[None]
    def action(self, actor, packet, diagnostics):
        scheduler = DDIMScheduler(num_train_timesteps=100, prediction_type='sample', clip_sample=False)
        scheduler.set_timesteps(16)
        scheduler.step(torch.zeros((1, 24, 29)), scheduler.timesteps[0], torch.ones((1, 24, 29)), eta=0.)
        return expected
    monkeypatch.setattr(HistoryController, 'act', action)
    rng = torch.get_rng_state().clone()
    assert holder.act(None, None, None) is expected
    assert holder.normalized_plan.shape == (1, 24, 29)
    assert torch.equal(rng, torch.get_rng_state())
    assert torch.equal(holder.rng_before, holder.rng_after)
    assert DDIMScheduler.step is original


def test_actual_flat_policy_schema_roundtrip(tmp_path):
    import h5py
    import pytest
    path = tmp_path/'inputs.h5'
    with h5py.File(path, 'w') as stream:
        group = stream.create_group('episode')
        values = dict(actor=torch.arange(141.)[None],robot=torch.arange(178.)[None],
            geometry=torch.arange(85.)[None], hidden_before=torch.zeros(1,1,256),
            hidden_after=torch.ones(1,1,256), condition=torch.zeros(1,512),
            sampler_rng_before=torch.arange(16,dtype=torch.uint8)[None],
            sampler_rng_after=torch.arange(16,dtype=torch.uint8)[None],
            planning=torch.tensor([[True]]),candidate_requested_position=torch.zeros(1,29),raw_position=torch.ones(1,29))
        for tick in range(2):append_policy_inputs(group,values,tick)
        with pytest.raises(ValueError):append_policy_inputs(group,values,3)
    with h5py.File(path) as stream:
        for name,value in values.items():
            assert np.array_equal(stream['episode'][name][:],np.stack([value.numpy()[0]]*2))
