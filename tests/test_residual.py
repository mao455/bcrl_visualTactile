import copy
import torch
import pytest

from visuotactile.models.nominal import PolicyConfig, VisualHistoryEncoder
from visuotactile.models.residual import ResidualCorrection, compose_residual, residual_loss
from visuotactile.history import detach_state


def fixture_model(masked=False):
    config = PolicyConfig(views=3, own_plan_mode='none')
    encoder = VisualHistoryEncoder(config).requires_grad_(False)
    return encoder, ResidualCorrection(encoder, config, [2.]*29, force_masked=masked)


def inputs(steps=5):
    force = torch.randn(1, steps*2, 5, 3)
    return dict(features=torch.randn(1, steps, 512),
        tactile=dict(force_w_N=force, force_local_N=force.clone(), position_base_m=torch.randn_like(force),
            rotation_base=torch.eye(3).expand(1, steps*2, 5, 3, 3),
            time_s=torch.arange(steps*2, dtype=torch.float64)[None]/120),
        tactile_valid=torch.ones(1, steps*2, dtype=torch.bool),
        plans=torch.randn(1, steps, 12, 29), plan_valid=torch.ones(1, steps, 12, dtype=torch.bool),
        motion=dict(proprio=torch.randn(1, steps, 58), previous_command=torch.randn(1, steps, 29),
            command_valid=torch.ones(1, steps, dtype=torch.bool), actuator_target_rad=torch.randn(1, steps, 29),
            tracking_error_rad=torch.randn(1, steps, 29)))


def test_zero_residual_bypass_and_reachable_two():
    base = torch.tensor([[-1.]*29, [.2]*29])
    raw = torch.zeros_like(base, requires_grad=True)
    contact = torch.tensor([True, False])
    action, delta = compose_residual(base, raw, contact, torch.full((29,), 2.))
    assert torch.equal(action, base)
    target = torch.ones_like(base)
    loss = residual_loss(action, target, delta, contact)
    loss.backward()
    assert raw.grad[0].abs().sum() > 0
    assert raw.grad[1].abs().sum() == 0
    action, _ = compose_residual(base, torch.full_like(base, 8.), contact, torch.full((29,), 2.))
    assert torch.allclose(action[0], target[0], atol=1e-6)
    assert torch.equal(action[1], base[1])


def test_independent_freeze_and_full_prefix_equivalence():
    encoder, model = fixture_model()
    original = copy.deepcopy(encoder.state_dict())
    data = inputs()
    with torch.no_grad():
        model.policy[-1].weight.normal_(std=.01)
    output, final = model.forward_cached(**data)
    state, pieces = None, []
    for step in range(5):
        piece = {name:value[:, step:step+1] for name,value in data.items()
                 if name not in ('tactile', 'tactile_valid', 'motion')}
        piece['tactile'] = {name:value[:, 2*step:2*step+2] for name,value in data['tactile'].items()}
        piece['tactile_valid'] = data['tactile_valid'][:, 2*step:2*step+2]
        piece['motion'] = {name:value[:, step:step+1] for name,value in data['motion'].items()}
        value, state = model.forward_cached(**piece, state=state)
        pieces.append(value)
        state = detach_state(state)
    assert torch.allclose(output, torch.cat(pieces, 1), atol=1e-6)
    assert torch.allclose(final['visual'], state['visual'], atol=1e-6)
    output.square().mean().backward()
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(torch.equal(value, encoder.state_dict()[name]) for name,value in original.items())
    assert all(parameter.grad is None for name,parameter in model.visual.named_parameters() if not name.startswith('temporal.'))
    assert model.visual.temporal.weight_ih.grad is not None
    assert encoder.temporal.weight_ih.data_ptr() != model.visual.temporal.weight_ih.data_ptr()


def test_force_mask_removes_force_derived_features_but_retains_fk():
    _, model = fixture_model(True)
    with torch.no_grad():
        model.policy[-1].weight.normal_(std=.01)
    data = inputs()
    changed = copy.deepcopy(data)
    changed['tactile']['force_w_N'] *= 100
    changed['tactile']['force_local_N'] += 100
    reference, _ = model.forward_cached(**data)
    result, _ = model.forward_cached(**changed)
    assert torch.equal(reference, result)
    changed['tactile']['position_base_m'] += .1
    result, _ = model.forward_cached(**changed)
    assert not torch.equal(reference, result)


def test_v3_is_not_silently_connected():
    config = PolicyConfig(views=3, own_plan_mode='none', action_history_mode='issued')
    with pytest.raises(ValueError, match='v3'):
        ResidualCorrection(VisualHistoryEncoder(config), config, [2.]*29)


def test_untrained_checkpoint_cannot_execute(tmp_path, monkeypatch):
    from visuotactile import residual_controller
    from visuotactile.residual_data import BASE_HASH
    monkeypatch.setattr(residual_controller, 'load_base', lambda device: None)
    path = tmp_path / 'untrained.pt'
    torch.save(dict(residual_checkpoint_version=1, base_sha256=BASE_HASH,
                    trained=True, optimizer_updates=0), path)
    with pytest.raises(ValueError, match='Untrained'):
        residual_controller.ResidualController(path, device='cpu')


def test_acknowledged_mixed_feedback_is_private_to_correction():
    from visuotactile.action_history import ExecutionHistory
    from visuotactile.residual_controller import ResidualController
    controller = ResidualController.__new__(ResidualController)
    controller.base = object()
    controller.history = ExecutionHistory(1)
    controller.pending = True
    controller.last = dict(command=torch.full((1, 29), .3))
    controller.last_packet = dict(episode_id=torch.tensor([7]), step_id=torch.tensor([0]))
    transition = dict(episode_id=torch.tensor([7]), step_id=torch.tensor([0]),
        command_clipped_pre_delay=torch.full((1, 29), .3), time_before_s=torch.tensor([0.], dtype=torch.float64),
        time_after_s=torch.tensor([1/60], dtype=torch.float64), control_trace=dict(valid=torch.tensor([True]),
            delayed_command=torch.full((1, 29), .1), selected_command_time_s=torch.tensor([0.], dtype=torch.float64)))
    with pytest.raises(ValueError, match='differs'):
        controller.acknowledge(transition, torch.full((1, 29), .4))
    controller.acknowledge(transition, controller.last['command'])
    context = controller.history.context(dict(episode_id=torch.tensor([7]), step_id=torch.tensor([1]),
                                              time_s=torch.tensor([1/60], dtype=torch.float64)))
    assert torch.equal(context['previous_issued_command'], controller.last['command'])
    with pytest.raises(ValueError, match='pending'):
        controller.acknowledge(transition, controller.last['command'])


def test_diagnostic_continuation_never_repeats_a_started_allocation(tmp_path):
    import json
    from visuotactile.evaluate_residual import diagnostic_allocation
    plan = dict(diagnostics=[dict(mode='expert_rollin', seed=seed, variant=variant)
                            for seed in (981001, 981002) for variant in ('a', 'b', 'c')])
    previous = tmp_path / 'closed_loop'
    previous.mkdir()
    (previous / 'status.json').write_text(json.dumps(dict(state='failed', completed=1)))
    (previous / 'trials.json').write_text(json.dumps([dict(**plan['diagnostics'][0], attempt='01_reference')]))
    (previous / '02_failed_switch_error.json').write_text(json.dumps(dict(step=120, prefix_equal=True, max_state_error=4.)))
    output, offset, inherited = diagnostic_allocation(tmp_path, plan, True)
    assert offset == 2 and len(inherited) == 2
    assert plan['diagnostics'][offset:] == plan['diagnostics'][2:]
    assert inherited[-1]['success'] is None and inherited[-1]['student_steps'] == 0
    assert output != previous
    (previous / 'status.json').write_text(json.dumps(dict(state='failed', completed=0)))
    with pytest.raises(ValueError, match='preserved'):
        diagnostic_allocation(tmp_path, plan, True)


def test_pose_comparison_respects_quaternion_double_cover_and_units():
    import numpy as np
    from visuotactile.residual_report import pose_difference
    first = np.array([.1, .2, .3, 1., 0., 0., 0.])
    second = first.copy()
    second[3:] *= -1
    second[0] += .002
    result = pose_difference(first, second)
    assert result['position_distance_m'] == pytest.approx(.002)
    assert result['orientation_distance_rad'] == 0


def test_viewing_allocation_is_separate_and_never_continued(tmp_path):
    from visuotactile.evaluate_residual import diagnostic_allocation
    plan = dict(purpose='viewing_rerun')
    output, offset, inherited = diagnostic_allocation(tmp_path, plan, False)
    assert output == tmp_path / 'viewing'
    assert offset == 0 and inherited == []
    with pytest.raises(ValueError, match='cannot resume'):
        diagnostic_allocation(tmp_path, plan, True)


def test_video_status_does_not_modify_policy_images(tmp_path):
    import av
    from visuotactile.evaluate_baselines import RolloutVideo
    packet = dict(image=torch.rand(1, 3, 4, 24, 80))
    original = packet['image'].clone()
    path = tmp_path / 'view.mp4'
    video = RolloutVideo(path, 'b', 981002, 5, ['high', 'side', 'wrist'])
    for step in range(3):
        video.append(packet, step, status='student | gate=1')
    info = video.close(complete=True)
    assert torch.equal(packet['image'], original)
    assert info['frames'] == 3 and not info['terminal_image_available']
    with av.open(str(path)) as container:
        frames = list(container.decode(video=0))
        assert len(frames) == 3
        assert (frames[0].width, frames[0].height) == (240, 88)
        assert float(container.streams.video[0].average_rate) == 60


def test_resource_overlimit_intervals_preserve_unknown_gap(tmp_path):
    import json
    from visuotactile.viewing_report import resource_intervals
    path = tmp_path / 'samples.jsonl'
    values = [49., 51., 52., None, 53., 49.]
    path.write_text('\n'.join(json.dumps(dict(utc_epoch=100+index, rolling_60s_percent=value))
                              for index, value in enumerate(values)))
    intervals = resource_intervals(path)
    assert len(intervals) == 2
    assert intervals[0]['first_sample_epoch'] == 101
    assert intervals[0]['last_sample_epoch'] == 102
    assert intervals[0]['max_complete_60s_mean_percent'] == 52.
    assert intervals[1]['samples'] == 1


def test_recorder_resource_window_excludes_monitor_tail_and_keeps_missing(tmp_path):
    import json
    from visuotactile.viewing_report import worker_resource_window
    path = tmp_path / 'samples.jsonl'
    values = [(0, 80), (10, None), (20, 20), (30, 0), (40, 0)]
    path.write_text('\n'.join(json.dumps(dict(utc_epoch=100+stamp, elapsed_s=stamp,
        gpu_util_percent=value, rolling_60s_percent=None)) for stamp,value in values))
    result = worker_resource_window(path, 105, 125)
    assert result['measured_seconds'] == 10
    assert result['missing_seconds'] == 10
    assert result['time_weighted_mean_percent'] == 50
    assert result['max_complete_60s_mean_percent'] is None
