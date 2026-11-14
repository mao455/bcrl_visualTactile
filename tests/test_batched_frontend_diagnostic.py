"""CPU trajectory tests of real NativeController history and lane scheduling."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from diffusers import DDPMScheduler

from visuotactile import batched_frontend_diagnostic as candidate
from visuotactile.batched_controller import lane_tree
from visuotactile.models.native_dp import NativePolicy, TargetResidual
from visuotactile.native_controller import NativeController
from visuotactile.position_control import continuous_image_contract, REACHABLE_POSITION_EXECUTION
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT


class ToyPolicy(torch.nn.Module):
    """Tiny weights, but the real 16-step NativePolicy DDIM implementation."""
    group = 'I-HistoryDP'
    modality = 'G00'
    horizon = 24
    predict = NativePolicy.predict

    def __init__(self, contract):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.01))
        self.register_buffer('action_scale', torch.ones(29))
        self.register_buffer('action_offset', torch.zeros(29))
        self.contract = contract
        self.noise = DDPMScheduler(num_train_timesteps=100, beta_schedule='squaredcos_cap_v2',
            clip_sample=False, prediction_type='sample')
        self.denoiser = lambda sample, timestep, global_cond: sample*.01+global_cond[:, :1, None]*.1
        self.eval()

    def observe(self, robot, stamp, tick, frame, memory, *, tactile=None):
        assert tactile is None and frame is not None
        assert memory is None if tick == 0 else memory.next_tick == tick
        old = torch.zeros(1) if memory is None else memory.total
        # Include actual past command and per-lane current RGB/depth in the
        # next plan. Wrong lane/history feedback must change future outputs.
        total = old + self.weight*(robot[0]+robot[116]+robot[145]+frame['patches'].mean()+frame['depth'].mean())
        return SimpleNamespace(total=total, next_tick=tick+1,
            robots=([] if memory is None else memory.robots)+[robot.clone()])

    def condition(self, memory, stamp):
        return memory.total.reshape(1, 1).expand(1, 3)


class Dino:
    def __init__(self):
        self.calls = []

    def forward_lanes(self, rgb, *, vision_microbatch):
        assert vision_microbatch == 8
        self.calls.append(rgb.clone())
        return rgb.mean(dim=(-1, -2))


class EncoderDiagnostics(dict):
    def __getitem__(self, key):
        assert key in ('joint_position_rad', 'joint_velocity_rad_s'), 'Privileged diagnostic read'
        return super().__getitem__(key)

    def items(self):
        raise AssertionError('Do not traverse object/goal diagnostics')


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from visuotactile import native_controller
    from visuotactile.models import native_dp
    previous_threads = torch.get_num_threads(); torch.set_num_threads(1)
    monkeypatch.setattr(native_controller, 'load_policy',
        lambda saved, device: ToyPolicy(saved['contract']).to(device))
    monkeypatch.setattr(native_controller, 'depth_grid',
        lambda depth, valid: torch.where(valid, depth, 0.).mean(dim=(-1, -2)))
    monkeypatch.setattr(native_dp, 'load_residual', lambda *a, **k: pytest.fail('No rear may be loaded'))
    saved = dict(kind='dp', group='I-HistoryDP', contract=continuous_image_contract('u_expert'),
        epoch=5, updates=685, configuration={}, ema=dict(weight=torch.tensor(.01)))
    path = tmp_path/'best.pt'; torch.save(saved, path)
    digest = candidate.sha256(path)
    (tmp_path/'finished.json').write_text(json.dumps(dict(complete=True, checkpoint_sha256=digest)))
    metadata = dict(joint_limits_rad=[[[-2., 2.]]*29]*3,
        action_control_mode='absolute_joint_position_target_rad', control_dt=1/60, physics_dt=1/120,
        config=dict(action={key: REACHABLE_POSITION_EXECUTION[key] for key in
            ('dof_speed_scale', 'arm_moving_average', 'hand_moving_average')}))
    yield SimpleNamespace(path=path, sha=digest, saved=saved, metadata=metadata)
    torch.set_num_threads(previous_threads)


def build(x, mode):
    return candidate.BatchedFrontendDiagnostic(x.path, x.metadata, Dino(), 3, mode=mode,
        checkpoint_sha256=x.sha,
        execution_adapter=REACHABLE_POSITION_EXECUTION if mode == 'projection' else None, device='cpu')


def packet(global_tick, previous):
    local = [global_tick, global_tick, global_tick if global_tick < 14 else global_tick-14]
    n = 3
    image = (torch.arange(n).float()[:, None, None, None, None]*.1+global_tick*.001).expand(n, 3, 4, 2, 2).clone()
    stamp = torch.full((n,), global_tick/60, dtype=torch.float64)
    p = dict(image=image, depth_valid=torch.ones(n, 3, 1, 2, 2, dtype=torch.bool),
        episode_id=torch.tensor([1, 1, 1 if global_tick < 14 else 2]), step_id=torch.tensor(local),
        time_s=stamp, camera_time_s=stamp[:, None].expand(n, 3).clone(),
        camera_valid=torch.ones(n, 3, dtype=torch.bool), task_id=torch.zeros(n, dtype=torch.long),
        goal_valid=torch.zeros(n, dtype=torch.bool), actuator_target_rad=previous.clone())
    d = EncoderDiagnostics(joint_position_rad=torch.arange(n).float()[:, None].expand(n, 29)*.01,
        joint_velocity_rad_s=torch.zeros(n, 29), object_pose_w=object(), goal_pose_w=object())
    return p, d


def acknowledgement(p, commands):
    # Applied differs slightly from requested so the two history channels
    # cannot accidentally be replaced by one nominal/predicted sequence.
    return dict(episode_id=p['episode_id'].clone(), step_id=p['step_id'].clone(),
        position_requested_rad=commands.clone(), position_applied_rad=commands.clone()+.00003)


@pytest.mark.parametrize('mode', ['direct', 'projection'])
def test_real_native_history_batched_ddim_matches_independent_serial_lanes(setup, mode):
    x = setup; batched = build(x, mode)
    serial = [NativeController(x.saved, x.metadata, dino=lambda rgb: rgb.mean(dim=(-1, -2)),
        device='cpu', scheduler_index_mode='host') for _ in range(3)]
    previous = torch.zeros(3, 29)
    global_rng = torch.get_rng_state().clone()
    retired = None
    for tick in range(27):
        p, d = packet(tick, previous)
        active = torch.tensor([True, tick < 6, True])
        out = batched.act(p, d, active)
        expected = previous.clone()
        for lane in active.nonzero().flatten().tolist():
            nominal = serial[lane].act(lane_tree(p, lane, 3),
                {key: d[key][lane:lane+1] for key in ('joint_position_rad', 'joint_velocity_rad_s')})
            target = candidate.frontend_target(nominal, serial[lane].last_robot[None, 58:87],
                serial[lane].limits, mode=mode, execution_adapter=batched.execution_adapter)
            expected[lane:lane+1] = target
            serial[lane].command = target.clone()
        assert torch.equal(out, expected)
        transition = acknowledgement(p, out)
        batched.observe_execution(transition, active)
        for lane in active.nonzero().flatten().tolist():
            serial[lane].observe_execution(lane_tree(transition, lane, 3))
            assert torch.equal(batched.lanes[lane].generator.get_state(), serial[lane].generator.get_state())
            assert torch.equal(batched.lanes[lane].memory.total, serial[lane].memory.total)
            assert torch.equal(batched.lanes[lane].last_robot, serial[lane].last_robot)
        if tick == 5:
            retired = (batched.lanes[1].memory, batched.lanes[1].generator.get_state().clone())
        if tick > 5:
            assert batched.lanes[1].memory is retired[0]
            assert torch.equal(batched.lanes[1].generator.get_state(), retired[1])
        previous = transition['position_applied_rad']
    assert torch.equal(global_rng, torch.get_rng_state())
    assert batched.lanes[0].model is batched.lanes[2].model
    assert batched.lanes[0].memory is not batched.lanes[2].memory
    assert all(not p.requires_grad and p.grad is None for p in batched.lanes[0].model.parameters())
    report = batched.execution_report()
    assert [r['actual_visual_observations'] for r in report['lanes']] == [27, 6, 27]
    assert [r['history_updates'] for r in report['lanes']] == [27, 6, 27]
    assert [r['acknowledged_control_ticks'] for r in report['lanes']] == [27, 6, 27]
    assert [r['plans_generated'] for r in report['lanes']] == [3, 1, 4]
    assert [(v['episode_id'], v['step_id']) for v in batched.plan_records[2]] == [(1, 0), (1, 12), (2, 0), (2, 12)]
    assert report['rear_forward_ticks'] == 0 and not report['complete_student_pair']
    assert report['vision']['real_images'] == 60*3 and report['vision']['processed_images'] == 27*8*3
    before = copy.deepcopy(report)
    assert torch.equal(batched.act(p, d, [False]*3), p['actuator_target_rad'])
    assert batched.execution_report() == before
    saved_plan = batched.plan_records[0][-1]['plan'].clone()
    batched.lanes[0].plan.zero_()
    assert torch.equal(batched.plan_records[0][-1]['plan'], saved_plan)


def test_projection_feedback_uses_actual_submitted_and_applied_not_private_nominal(setup):
    c = build(setup, 'projection'); p, d = packet(0, torch.zeros(3, 29))
    # Deliberately request beyond the current reachable interval. A near-zero
    # toy prediction would not exercise the postprocessing/history boundary.
    c.lanes[0].model.action_offset.fill_(.5)
    command = c.act(p, d, [True]*3)
    assert not torch.equal(command[0:1], c.last_nominal[0])
    transition = acknowledgement(p, command)
    c.observe_execution(transition, [True]*3)
    following, joints = packet(1, transition['position_applied_rad'])
    c.act(following, joints, [True]*3)
    for lane in range(3):
        robot = c.lanes[lane].last_robot
        assert torch.equal(robot[116:145], command[lane])
        assert torch.equal(robot[145:174], transition['position_applied_rad'][lane])
        assert torch.equal(robot[58:87], transition['position_applied_rad'][lane])
    assert any(row['projection']['large_ticks'] for row in c.execution_report()['lanes'])


def test_projection_matches_exact_existing_composition_including_roundoff_anchor():
    limits = torch.tensor([[-2., 2.]]*29)
    previous = torch.tensor([[-2.]*29, [0.]*29, [2.]*29])
    from visuotactile.execution_learning import ideal_execution_bounds
    bounds = ideal_execution_bounds(previous, limits, REACHABLE_POSITION_EXECUTION)
    nominal = bounds[..., 1]+.5e-6
    # An arbitrary well-outside prediction exercises both arm and hand spans.
    nominal[1, 0] = .8; nominal[1, 8] = -.7
    stub = SimpleNamespace(output_contract=REACHABLE_CENTERED_OUTPUT, loss_contract=None,
        execution_adapter=REACHABLE_POSITION_EXECUTION, action_scale=torch.linspace(.2, 2., 29))
    expected = TargetResidual.compose_chain(stub, nominal, torch.zeros_like(nominal),
        torch.zeros(3, dtype=torch.bool), limits, previous=previous)['u_exec']
    actual = candidate.frontend_target(nominal, previous, limits, mode='projection',
        execution_adapter=REACHABLE_POSITION_EXECUTION)
    assert torch.equal(actual, expected)
    direct = candidate.frontend_target(nominal, previous, limits, mode='direct', execution_adapter=None)
    assert not torch.equal(actual, direct)


def test_terminal_lane_must_acknowledge_and_wrong_targets_or_tick_fail_before_feedback(setup):
    c = build(setup, 'direct'); p, d = packet(0, torch.zeros(3, 29))
    commands = c.act(p, d, [True]*3); receipt = acknowledgement(p, commands)
    with pytest.raises(ValueError, match='acknowledgement required'):c.act(p, d, [True]*3)
    with pytest.raises(ValueError, match='including terminal'):c.observe_execution(receipt, [True, False, True])
    broken = copy.deepcopy(receipt); broken['position_requested_rad'][1, 0] += .1
    with pytest.raises(ValueError, match='submitted targets differ'):c.observe_execution(broken, [True]*3)
    broken = copy.deepcopy(receipt); broken['step_id'][2] += 1
    with pytest.raises(ValueError, match='episode/tick differs'):c.observe_execution(broken, [True]*3)
    assert c.executed_ticks == [0, 0, 0] and all(x.position_previous is None for x in c.lanes)
    c.observe_execution(receipt, [True]*3)
    with pytest.raises(ValueError, match='one valid pending'):c.observe_execution(receipt, [True]*3)


def test_missing_native_observation_is_fail_closed_without_lane_reuse(setup):
    c = build(setup, 'direct'); p, d = packet(0, torch.zeros(3, 29))
    command = c.act(p, d, [True]*3); c.observe_execution(acknowledgement(p, command), [True]*3)
    skipped, joints = packet(2, command)
    with pytest.raises(ValueError, match='skipped or duplicated'):c.act(skipped, joints, [True]*3)
    with pytest.raises(ValueError, match='Failed diagnostic'):c.act(p, d, [True]*3)


@pytest.mark.parametrize('fault', ['A', 'sparse_vision', 'untrained', 'disposable', 'rear', 'sha', 'not_selected'])
def test_wrong_or_untrained_checkpoint_rejected_before_controller_construction(setup, monkeypatch, fault):
    x = setup; saved = copy.deepcopy(x.saved)
    if fault == 'A':saved['contract'] = continuous_image_contract('q_reached')
    elif fault == 'sparse_vision':saved['contract']['visual_stride'] = 12
    elif fault == 'untrained':saved['updates'] = 0
    elif fault == 'disposable':saved['disposable'] = True
    elif fault == 'rear':saved['residual_version'] = 'anything'
    torch.save(saved, x.path); x.sha = candidate.sha256(x.path)
    (x.path.parent/'finished.json').write_text(json.dumps(dict(complete=True,
        checkpoint_sha256='0'*64 if fault == 'not_selected' else x.sha)))
    if fault == 'sha':x.sha = '0'*64
    monkeypatch.setattr(candidate, 'NativeController', lambda *a, **k: pytest.fail('Invalid admission reached model'))
    with pytest.raises(ValueError):build(x, 'direct')


@pytest.mark.parametrize('mode,adapter', [('direct', REACHABLE_POSITION_EXECUTION), ('projection', None), ('rear', None)])
def test_mode_and_adapter_are_explicit_not_inferred(setup, mode, adapter):
    with pytest.raises(ValueError):
        candidate.BatchedFrontendDiagnostic(setup.path, setup.metadata, Dino(), 3,
            mode=mode, checkpoint_sha256=setup.sha, execution_adapter=adapter, device='cpu')


def test_projection_requires_actual_environment_adapter_and_original_clock(setup):
    setup.metadata['config']['action']['arm_moving_average'] = .2
    with pytest.raises(ValueError, match='actual environment'):build(setup, 'projection')
    setup.metadata['physics_dt'] = 1/60
    with pytest.raises(ValueError, match='120Hz'):build(setup, 'direct')
