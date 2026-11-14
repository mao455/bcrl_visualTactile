"""CPU parity with the unchanged native mapper; no simulator claims."""

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from visuotactile.batched_position_bridge import BatchedExpertPositionBridge
from visuotactile.position_control import ExpertPositionBridge, POSITION_ACTION, apply_position_targets, bounded_targets


def native_pipeline():
    spec = importlib.util.spec_from_file_location('_batch_bridge_native_math',
        Path(__file__).resolve().parents[1]/'isaacsimenvs/tasks/play/utils/action_utils.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.apply_action_pipeline


def environment(count=8, delay=True, shared_bounds=False):
    generator = torch.Generator().manual_seed(413)
    perm = torch.cat((torch.randperm(7, generator=generator), 7+torch.randperm(22, generator=generator)))
    extent = 1.5+torch.rand(count, 29, generator=generator)
    limits = torch.stack((-extent, extent), -1)
    if shared_bounds:
        limits[:] = limits[:1].clone()
    previous = torch.rand(count, 29, generator=generator)*.4-.2
    env = NS(device='cpu', num_envs=count, step_dt=1/60, action_control_mode=POSITION_ACTION,
        _perm_canon_to_lab=perm, _perm_lab_to_canon=perm.argsort(),
        _arm_joint_ids=torch.arange(7), _hand_joint_ids=torch.arange(7, 29),
        _arm_lower=limits[:, :7, 0], _arm_upper=limits[:, :7, 1],
        _hand_lower=limits[:, 7:, 0], _hand_upper=limits[:, 7:, 1],
        robot=NS(data=NS(joint_pos_limits=limits)), _cur_targets=previous.clone(),
        _prev_targets=previous.clone(), _action_queue=torch.full((count, 4, 29), .91),
        episode_length_buf=torch.zeros(count, dtype=torch.long), _successes=torch.zeros(count),
        cfg=NS(domain_randomization=NS(use_action_delay=delay, action_delay_max=3),
            action=NS(dof_speed_scale=1.5, arm_moving_average=.1, hand_moving_average=.1)))
    if shared_bounds:
        for name in ('_arm_lower', '_arm_upper', '_hand_lower', '_hand_upper'):
            setattr(env, name, getattr(env, name)[0])
    return env


def serial_environment(env, lane):
    result = copy.deepcopy(env)
    result.num_envs = 1
    for name in ('_prev_targets', '_cur_targets', '_action_queue', 'episode_length_buf', '_successes'):
        setattr(result, name, getattr(env, name)[lane:lane+1].clone())
    for name in ('_arm_lower', '_arm_upper', '_hand_lower', '_hand_upper'):
        value = getattr(env, name)
        setattr(result, name, (value[lane:lane+1] if value.ndim > 1 else value).clone())
    result.robot.data.joint_pos_limits = env.robot.data.joint_pos_limits[lane:lane+1].clone()
    return result


def snapshot(bridge):
    return dict(tick=bridge.tick, segment=bridge.segment, previous_phase=bridge.previous_phase,
        queue=None if bridge.queue is None else bridge.queue.clone(),
        source_steps=None if bridge.source_steps is None else bridge.source_steps.clone(),
        trace=None if bridge.trace is None else {key: value.clone() for key, value in bridge.trace.items()})


def assert_snapshot(bridge, expected):
    assert bridge.tick == expected['tick']
    assert bridge.segment == expected['segment']
    assert bridge.previous_phase == expected['previous_phase']
    for name in ('queue', 'source_steps'):
        actual = getattr(bridge, name)
        assert (actual is None) if expected[name] is None else torch.equal(actual, expected[name])
    assert bridge.pending is None
    if expected['trace'] is None:
        assert bridge.trace is None
    else:
        assert bridge.trace.keys() == expected['trace'].keys()
        for key, value in expected['trace'].items():
            assert torch.equal(bridge.trace[key], value), key


@pytest.mark.parametrize('delay', [False, True])
@pytest.mark.parametrize('shared_bounds', [False, True])
def test_wave_matches_serial_queues_targets_phases_rng_and_completed_lanes(delay, shared_bounds):
    env = environment(delay=delay, shared_bounds=shared_bounds)
    takeovers = [2, 3, 5, 100000, 4, 8, 6, 10]
    bridge = BatchedExpertPositionBridge(env, takeovers, ExpertPositionBridge.speed_scheme,
        pipeline=native_pipeline())
    serial_envs = [serial_environment(env, lane) for lane in range(env.num_envs)]
    env._dagger_position_bridge = bridge
    serial = [ExpertPositionBridge(lane_env, switch, ExpertPositionBridge.speed_scheme,
        pipeline=native_pipeline()) for lane_env, switch in zip(serial_envs, takeovers)]
    native_queue = env._action_queue.clone()
    end = torch.tensor([7, 10, 13, 15, 17, 20, 22, 22])
    frozen = {}
    generator = torch.Generator().manual_seed(613)
    for tick in range(22):
        active = tick < end
        if tick == 3:
            bridge.schedule_takeover(3, 4)
            serial[3].schedule_takeover(4)
        for lane in range(env.num_envs):
            if not active[lane] and lane not in frozen:
                frozen[lane] = snapshot(bridge.bridges[lane])
                # A native auto-reset may replace target tensors. The adapter
                # must hold this real current target, not a stale pre-reset view.
                env._prev_targets = env._prev_targets.clone()
                env._prev_targets[lane] = .015*lane
                env._cur_targets[lane] = env._prev_targets[lane]
        previous = env._prev_targets[:, env._perm_lab_to_canon].clone()
        raw = torch.randn(env.num_envs, 29, generator=generator)*1.3
        student = torch.randn(env.num_envs, 29, generator=generator)*.1
        # Inactive policy values are not used or required to be meaningful.
        raw[~active] = float('nan')
        student[~active] = float('nan')
        clipped = raw.clamp(-1, 1)
        torch.manual_seed(719+tick)
        draw = torch.randint(0, 4, (env.num_envs,)) if delay else torch.full((env.num_envs,), -1)
        expected_rng = torch.get_rng_state().clone()
        expected_requested = previous.clone()
        for lane in active.nonzero().flatten().tolist():
            serial_envs[lane]._prev_targets = env._prev_targets[lane:lane+1].clone()
            serial_envs[lane]._cur_targets = env._cur_targets[lane:lane+1].clone()
            serial[lane].submit(raw[lane:lane+1], clipped[lane:lane+1], tick)
            expected_requested[lane:lane+1] = serial[lane].select(serial_envs[lane],
                student[lane:lane+1], draw[lane:lane+1])
        bridge.submit(raw, clipped, tick, active)
        torch.manual_seed(719+tick)
        applied, actual_draw = apply_position_targets(env, student)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(draw, actual_draw)
        assert torch.equal(bridge.requested, expected_requested)
        limits = env.robot.data.joint_pos_limits[:, env._perm_lab_to_canon]
        assert torch.equal(applied, bounded_targets(expected_requested, limits))
        assert torch.equal(applied[~active], previous[~active])
        assert torch.equal(bridge.trace['bridge_active'], active)
        assert not bridge.trace['expert_label_valid'][~active].any()
        assert not bridge.trace['switch_clean'][~active].any()
        assert all(value.shape[0] == env.num_envs for value in bridge.trace.values())
        assert torch.equal(env._action_queue, native_queue)
        for lane in active.nonzero().flatten().tolist():
            assert bridge.bridges[lane].tick == tick+1
            for key, value in serial[lane].trace.items():
                assert torch.equal(bridge.trace[key][lane:lane+1], value), (tick, lane, key)
        for lane, saved in frozen.items():
            assert_snapshot(bridge.bridges[lane], saved)


def test_lane_input_and_scheduled_takeover_cannot_change_other_lanes():
    first, second = environment(5), environment(5)
    wrappers = [BatchedExpertPositionBridge(env, [3]*5, ExpertPositionBridge.speed_scheme,
        pipeline=native_pipeline()) for env in (first, second)]
    wrappers[1].schedule_takeover(2, 1)
    generator = torch.Generator().manual_seed(831)
    for tick in range(12):
        raw = torch.randn(5, 29, generator=generator)
        student = torch.randn(5, 29, generator=generator)*.03
        draw = torch.randint(0, 4, (5,), generator=generator)
        for index, (env, wrapper) in enumerate(zip((first, second), wrappers)):
            proposal = raw.clone()
            if index:
                proposal[2] *= -1
            wrapper.submit(proposal, proposal.clamp(-1, 1), tick, torch.ones(5, dtype=torch.bool))
            target = wrapper.select(env, student, draw)
            env._cur_targets = target[:, env._perm_canon_to_lab].clone()
            env._prev_targets = env._cur_targets.clone()
        other = [0, 1, 3, 4]
        assert torch.equal(wrappers[0].requested[other], wrappers[1].requested[other])
        for key in wrappers[0].trace:
            assert torch.equal(wrappers[0].trace[key][other], wrappers[1].trace[key][other]), key
        for lane in other:
            assert torch.equal(wrappers[0].bridges[lane].queue, wrappers[1].bridges[lane].queue)


def test_no_initial_activity_and_no_lane_reuse_have_explicit_nonlabel_traces():
    env = environment(3)
    wrapper = BatchedExpertPositionBridge(env, [1]*3, ExpertPositionBridge.speed_scheme,
        pipeline=native_pipeline())
    active = torch.tensor([True, False, True])
    wrapper.submit(torch.zeros(3, 29), torch.zeros(3, 29), 0, active)
    wrapper.select(env, torch.zeros(3, 29), torch.zeros(3, dtype=torch.long))
    assert wrapper.trace['control_source'].tolist() == [0, -1, 0]
    assert wrapper.trace['expert_source_step'].tolist() == [0, -1, 0]
    assert not wrapper.trace['expert_label_valid'].any()
    assert wrapper.bridges[1].tick == 0 and wrapper.bridges[1].queue is None
    with pytest.raises(ValueError, match='reactivate'):
        wrapper.submit(torch.zeros(3, 29), torch.zeros(3, 29), 1, torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match='active lane'):
        wrapper.schedule_takeover(1, 1)
    wrapper.submit(torch.zeros(3, 29), torch.zeros(3, 29), 1, torch.zeros(3, dtype=torch.bool))
    held = wrapper.select(env, torch.full((3, 29), float('nan')), torch.zeros(3, dtype=torch.long))
    assert torch.equal(held, env._prev_targets[:, env._perm_lab_to_canon])
    assert [bridge.tick for bridge in wrapper.bridges] == [1, 0, 1]
    assert all(value.shape[0] == 3 for value in wrapper.trace.values())


def test_invalid_submit_or_delay_is_rejected_before_any_lane_is_advanced():
    env = environment(4)
    wrapper = BatchedExpertPositionBridge(env, [2]*4, ExpertPositionBridge.speed_scheme,
        pipeline=native_pipeline())
    raw, clipped = torch.zeros(4, 29), torch.zeros(4, 29)
    clipped[3, 0] = .1
    with pytest.raises(ValueError, match='provenance'):
        wrapper.submit(raw, clipped, 0, torch.ones(4, dtype=torch.bool))
    assert all(bridge.pending is None and bridge.tick == 0 for bridge in wrapper.bridges)
    with pytest.raises(ValueError, match='boolean active'):
        wrapper.submit(raw, raw, 0, torch.ones(4, dtype=torch.long))
    with pytest.raises(ValueError, match='exactly once'):
        wrapper.submit(raw, raw, 1, torch.ones(4, dtype=torch.bool))
    wrapper.submit(raw, raw, 0, torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match='before the next submit'):
        wrapper.schedule_takeover(0, 1)
    with pytest.raises(ValueError, match='exactly once'):
        wrapper.submit(raw, raw, 0, torch.ones(4, dtype=torch.bool))
    with pytest.raises(ValueError, match='delay index'):
        wrapper.select(env, raw, torch.tensor([0, 1, 2, 4]))
    assert all(bridge.tick == 0 and bridge.queue is None for bridge in wrapper.bridges)
    invalid_student = raw.clone()
    invalid_student[-1, 0] = float('nan')
    with pytest.raises(ValueError, match='must be finite'):
        wrapper.select(env, invalid_student, torch.zeros(4, dtype=torch.long))
    assert all(bridge.tick == 0 and bridge.queue is None for bridge in wrapper.bridges)
    wrapper.select(env, raw, torch.tensor([0, 1, 2, 3]))
    with pytest.raises(ValueError, match='Missing expert wave'):
        wrapper.select(env, raw, torch.zeros(4, dtype=torch.long))


def test_queue_flush_label_boundaries_remain_independent_for_a_wide_wave():
    env = environment(128)
    takeovers = [1+lane%4 for lane in range(128)]
    wrapper = BatchedExpertPositionBridge(env, takeovers, ExpertPositionBridge.speed_scheme,
        pipeline=native_pipeline())
    raw = torch.zeros(128, 29)
    for tick in range(9):
        wrapper.submit(raw, raw, tick, torch.ones(128, dtype=torch.bool))
        wrapper.select(env, raw, torch.arange(128)%4)
        assert wrapper.trace['control_source'].tolist() == [int(tick >= switch) for switch in takeovers]
        assert wrapper.trace['expert_label_valid'].tolist() == [tick >= switch+3 for switch in takeovers]
        assert wrapper.trace['switch_clean'].tolist() == [tick >= switch+3 for switch in takeovers]
