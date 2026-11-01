"""Native-command causal boundaries, independent identities and admission tests."""
import copy

import pytest
import torch
from torch import nn

from visuotactile.native_data import CONTRACT, GEOMETRY_DIM, ROBOT_DIM, geometry, labels, limits_affine, robot_rows
from visuotactile.long_data import shift_actions
from visuotactile.models.native_dp import NativePolicy, encode_segment, load_policy
from visuotactile.native_training import common_budget, fit_passed


class EpisodeFixture:
    def __init__(self, length=109):
        self.group = 'S-NativeDP'
        times = torch.arange(length, dtype=torch.float64)/60
        commands = torch.linspace(-1, 1, length)[:, None].expand(-1, 29).clone()
        position, velocity, target = torch.randn(3, length, 29)
        action = shift_actions(commands, -commands, times, times, times)
        self.record = dict(time=times, command=commands,
            robot=robot_rows(position, velocity, target, action, times),
            geometry=torch.randn((length+11)//12, GEOMETRY_DIM))

    def __len__(self):
        return len(self.record['time'])

    def frame(self, tick, device):
        return dict(geometry=self.record['geometry'][tick//12].to(device),
                    time=self.record['time'][tick].expand(3).to(device),
                    valid=torch.ones(3, dtype=torch.bool, device=device))


@pytest.fixture
def condition_model(monkeypatch):
    import visuotactile.models.native_dp as module
    monkeypatch.setattr(module, 'ConditionalUnet1D', lambda **kwargs: nn.Linear(29, 29))
    return NativePolicy('S-NativeDP')


def test_label_first_action_commit_and_tail():
    episode = EpisodeFixture(25)
    commands, valid = labels(episode.record, torch.tensor([0, 12, 24]))
    assert commands.shape == (3, 64, 29)
    assert torch.equal(commands[:, 0], episode.record['command'][[0, 12, 24]])
    assert valid.sum(1).tolist() == [25, 13, 1]
    assert torch.equal(torch.cat((commands[0, :12], commands[1, :12], commands[2, :1])), episode.record['command'])
    assert not commands[~valid].any()


def test_strict_past_commands_and_first_missing_mask():
    episode = EpisodeFixture()
    assert not episode.record['robot'][0, 116:177].any()
    assert episode.record['robot'][1, 176] == 1
    assert torch.equal(episode.record['robot'][1, 116:145], episode.record['command'][0])
    changed = episode.record['command'].clone()
    changed[12:] = 100
    times = episode.record['time']
    original = shift_actions(episode.record['command'], -episode.record['command'], times, times, times)
    altered = shift_actions(changed, -changed, times, times, times)
    assert torch.equal(original[:13], altered[:13])
    assert not torch.equal(original[13], altered[13])


def test_current_target_changes_condition_with_gradient(condition_model):
    episode = EpisodeFixture()
    robot = episode.record['robot'][0].clone().requires_grad_()
    stamp = episode.record['time'][0]
    frame = episode.frame(0, 'cpu')
    condition = condition_model.condition(condition_model.observe(robot, stamp, 0, frame), stamp)
    gradient = torch.autograd.grad(condition.square().sum(), robot)[0]
    assert gradient[58:87].norm() > 0
    altered = robot.detach().clone()
    altered[58:116] += .1
    result = condition_model.condition(condition_model.observe(altered, stamp, 0, frame), stamp)
    assert not torch.equal(condition, result)
    with pytest.raises(ValueError, match='allowlist'):
        condition_model.observe(robot, stamp, 0, frame | {'reward': torch.ones(1)})


def test_full_episode_online_and_segment_prefix_match(condition_model):
    episode = EpisodeFixture()
    condition_model.eval()
    with torch.no_grad():
        full, anchors, _ = encode_segment(condition_model, episode, 0, len(episode))
        tail, tail_anchors, _ = encode_segment(condition_model, episode, 96, len(episode))
        torch.testing.assert_close(full[anchors >= 96], tail, atol=1e-5, rtol=0)
        memory, online = None, []
        for tick in range(len(episode)):
            stamp = episode.record['time'][tick]
            memory = condition_model.observe(episode.record['robot'][tick], stamp, tick,
                episode.frame(tick, 'cpu') if tick % 12 == 0 else None, memory)
            if tick % 12 == 0:
                online.append(condition_model.condition(memory, stamp))
        torch.testing.assert_close(full, torch.cat(online), atol=1e-5, rtol=0)
        assert len(memory.robot) == 13 and len(memory.frames) == 3
        with pytest.raises(ValueError, match='Skipped'):
            condition_model.observe(episode.record['robot'][0], stamp, 0, episode.frame(0, 'cpu'), memory)


def test_prefix_uses_current_weights_and_does_not_receive_gradients(condition_model):
    episode = EpisodeFixture()
    episode.record['robot'].requires_grad_()
    condition_model.train()
    condition, _, _ = encode_segment(condition_model, episode, 96, 109)
    condition.square().sum().backward()
    assert not episode.record['robot'].grad[:96].any()
    assert episode.record['robot'].grad[96:].norm() > 0
    before = condition.detach().clone()
    with torch.no_grad():
        condition_model.robot_encoder[0].weight.add_(.05)
    after, _, _ = encode_segment(condition_model, episode, 96, 109)
    assert not torch.equal(before, after)


def test_limits_affine_constant_and_train_only_roundtrip():
    values = torch.tensor([[-.9, .25], [.7, .25]])
    scale, offset = limits_affine(values)
    normalized = values*scale+offset
    torch.testing.assert_close(normalized[:, 0], torch.tensor([-1., 1.]))
    assert not normalized[:, 1].any()
    torch.testing.assert_close((normalized-offset)/scale, values)


def test_geometry_has_no_world_translation_or_future_goal():
    pose = torch.tensor([[1., 2., 3., 1., 0., 0., 0.]])
    diagnostics = {name: pose.clone() for name in ('base_pose_w', 'object_pose_w', 'goal_pose_w',
        'hole_pose_w', 'table_pose_w', 'palm_body_pose_w')}
    diagnostics.update(finger_position_w=pose[:, None, :3].expand(-1, 5, -1),
        finger_quat_wxyz=pose[:, None, 3:].expand(-1, 5, -1),
        object_velocity_w_m_s_rad_s=torch.zeros(1, 6), goal_velocity_w_m_s_rad_s=torch.zeros(1, 6))
    result = geometry(diagnostics, torch.ones(3))
    shifted = copy.deepcopy(diagnostics)
    for name in shifted:
        if name.endswith('pose_w'):
            shifted[name][..., :3] += 4
    shifted['finger_position_w'] = shifted['finger_position_w']+4
    torch.testing.assert_close(result, geometry(shifted, torch.ones(3)))
    assert result.shape == (1, 85)


def test_old_checkpoint_and_gate_budget_cannot_be_promoted():
    with pytest.raises(ValueError, match='old weights'):
        load_policy({'version': 2, 'config': {'group': 'G00-new'}})
    good = dict(arm_rmse=.1, hand_rmse=.05, zero_arm_rmse=.8, zero_hand_rmse=.7)
    assert fit_passed(good, 1)
    assert not fit_passed(good | {'hand_rmse': .11}, 1)
    assert not fit_passed(good | {'zero_arm_rmse': .1}, 4)
    pilots = {group: 1. for group in ('S-NativeDP', 'I-OfficialDP', 'I-HistoryDP')}
    assert common_budget(80000, pilots, pilots) == 20000
    with pytest.raises(RuntimeError, match='Insufficient'):
        common_budget(100, pilots, pilots)


def test_history_native_and_visual_update_rates(monkeypatch):
    import visuotactile.models.native_dp as module
    monkeypatch.setattr(module, 'ConditionalUnet1D', lambda **kwargs: nn.Linear(29, 29))
    model = NativePolicy('I-HistoryDP').train()
    episode = EpisodeFixture(25)
    frame = dict(patches=torch.randn(3, 384, 18, 32), depth=torch.randn(3, 2, 18, 32),
                 time=torch.zeros(3), valid=torch.ones(3, dtype=torch.bool))
    state = model.observe(episode.record['robot'][0], torch.tensor(0.), 0, frame)
    visual = state.visual.clone()
    native = state.native.clone()
    state = model.observe(episode.record['robot'][1], torch.tensor(1/60), 1, None, state)
    assert torch.equal(visual, state.visual)
    assert not torch.equal(native, state.native)
    model.condition(state, torch.tensor(1/60)).square().sum().backward()
    for prefix in ('native_gru', 'visual_gru', 'vision.rgb', 'vision.depth'):
        assert any(parameter.grad is not None and parameter.grad.norm() > 0
                   for name, parameter in model.named_parameters() if name.startswith(prefix))
    with pytest.raises(ValueError, match='complete causal'):
        model.observe(episode.record['robot'][12], torch.tensor(.2), 12, frame)


def test_official_full_field_spatial_keypoints_and_depth_gradient(monkeypatch):
    import visuotactile.models.native_dp as module
    monkeypatch.setattr(module, 'ConditionalUnet1D', lambda **kwargs: nn.Linear(29, 29))
    model = NativePolicy('I-OfficialDP')
    rgb = torch.rand(3, 3, 252, 448)
    depth = torch.ones(3, 1, 252, 448, requires_grad=True)
    encoded = model.vision(rgb, depth, torch.ones_like(depth, dtype=torch.bool))
    assert encoded.shape == (384,)
    encoded.square().sum().backward()
    assert depth.grad.norm() > 0
    assert model.vision.keypoints.out_channels == 32
    assert not any(isinstance(layer, nn.BatchNorm2d) for layer in model.vision.modules())


def test_filtered_object_contact_is_required_for_grasp():
    import numpy as np
    from visuotactile.native_evaluation import stable_grasp
    pose = np.zeros((12, 7), dtype=np.float32)
    pose[:, 2], pose[:, 3] = .53, 1
    palm = pose.copy()
    forces = np.zeros((12, 2, 5, 3), dtype=np.float32)
    forces[:, :, :2, 2] = .2
    arrays = dict(object=pose, palm=palm, object_contact=forces)
    assert stable_grasp(arrays, .5)['stable_grasp_diagnostic']
    forces[4] = 0
    assert not stable_grasp(arrays, .5)['stable_grasp_diagnostic']


def test_stage_completion_requires_verified_artifacts(tmp_path):
    from visuotactile.native_pipeline import valid_artifacts
    from visuotactile.compact import sha256
    result = tmp_path/'result.json'
    result.write_text('{"passed":true}')
    stage = dict(state='complete', exit_code=0, artifacts={str(result): sha256(result)})
    assert valid_artifacts(stage)
    assert not valid_artifacts(stage | {'exit_code': 1})
    result.write_text('{"passed":false}')
    assert not valid_artifacts(stage)


def test_reviewed_pre_freeze_fix_preserves_original_deadline(tmp_path, monkeypatch):
    import json
    import visuotactile.native_pipeline as pipeline
    session = dict(started_epoch=100, deadline=86500, source_hashes={'visuotactile/native_data.py': 'before'})
    (tmp_path/'session.json').write_text(json.dumps(session))
    monkeypatch.setattr(pipeline, 'sources', lambda: {'visuotactile/native_data.py': 'after'})
    pipeline.review_pre_freeze_update(tmp_path, 'Reviewed numeric-cache atomicity repair')
    changed = json.loads((tmp_path/'session.json').read_text())
    assert changed['deadline'] == 86500 and changed['started_epoch'] == 100
    assert changed['pre_freeze_source_reviews'][0]['previous'] == session['source_hashes']
    (tmp_path/'frozen.json').write_text('{}')
    with pytest.raises(ValueError, match='frozen'):
        pipeline.review_pre_freeze_update(tmp_path, 'Must reject formal migration')


def test_monitor_resume_retains_previous_time_weighted_exposure(tmp_path):
    import json
    from visuotactile.native_resources import restore_telemetry
    from visuotactile.training_resources import ResourceGuard
    rows = [dict(elapsed_s=elapsed, utc_epoch=100+elapsed, gpu_util_percent=util,
        gpu_used_mib=1000, ram_available_bytes=20*1024**3, rolling_60s_percent=None)
        for elapsed, util in ((0, 20), (1, 40), (2, 60))]
    (tmp_path/'resources.jsonl').write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    guard = ResourceGuard(tmp_path, cuda_metrics=False)
    restore_telemetry(guard)
    assert guard.area == 60 and guard.measured_s == 2
    assert guard.last == rows[-1]


def test_oracle_runtime_watch_control_is_explicit():
    import ctypes
    import visuotactile.native_evaluation as evaluation
    assert hasattr(ctypes, 'CDLL')
    assert callable(evaluation.disable_omniclient_watches)


@pytest.mark.parametrize('total_deadline,expected', [(90000, 'gate_deadline'), (100, 'deadline')])
def test_expired_resume_preserves_attempt_and_never_spawns(tmp_path, monkeypatch, total_deadline, expected):
    import json
    import visuotactile.native_pipeline as module
    from visuotactile.compact import sha256
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(module, 'sources', lambda: {})
    monkeypatch.setattr(module.time, 'time', lambda: 101.)
    monkeypatch.setattr(module, 'video_tools', lambda: {})
    session = dict(source_hashes={}, started_epoch=1, gate_deadline=100,
                   training_deadline=80000, deadline=total_deadline)
    (tmp_path/'session.json').write_text(json.dumps(session))
    result = tmp_path/'data_result.json'
    result.write_text('{"passed":true}')
    previous = dict(state='blocked', exit_code=-6, output=str(tmp_path/'old_attempt'),
                    attempts=[dict(number=1, exit_code=-6, reason='native abort')])
    manifest = dict(state='blocked', stages={
        'data': dict(state='complete', exit_code=0, output=str(tmp_path), artifacts={str(result): sha256(result)}),
        'oracle_base': copy.deepcopy(previous)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    runner = module.Pipeline(tmp_path, resume=True)
    monkeypatch.setattr(runner, 'start_gpu_clock', lambda: None)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: pytest.fail('Expired run spawned a worker'))
    runner.run()
    assert runner.manifest['state'] == 'blocked'
    assert runner.manifest['stages']['oracle_base'] == previous
    assert f'{expected} expired' in runner.manifest['reason']
    assert not runner.manifest['last_failure']['attempt_started']
    assert json.loads((tmp_path/'session.json').read_text()) == session
    assert not (tmp_path/'stages').exists()


def test_stage_records_native_crash_exit_code_in_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import visuotactile.native_pipeline as module
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(module, 'sources', lambda: {})
    (tmp_path/'session.json').write_text('{"source_hashes": {}}')
    (tmp_path/'manifest.json').write_text('{"stages": {}}')
    runner = module.Pipeline(tmp_path)
    child = SimpleNamespace(pid=123, returncode=-6, poll=lambda: -6)
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **kw: child)
    with pytest.raises(RuntimeError, match='Worker failed'):
        runner.stage('oracle_base', 'unused', [], gpu=False)
    assert runner.manifest['stages']['oracle_base']['attempts'][0]['exit_code'] == -6


def test_monitor_exit_is_not_reported_as_deadline_expiry(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import visuotactile.native_pipeline as module
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(module.time, 'time', lambda: 50.)
    (tmp_path/'session.json').write_text('{"deadline": 200, "gate_deadline": 100}')
    (tmp_path/'manifest.json').write_text('{"stages": {}}')
    runner = module.Pipeline(tmp_path)
    runner.monitor = SimpleNamespace(poll=lambda: 2)
    with pytest.raises(RuntimeError, match='Resource monitor is not running'):
        runner.check_stage_budget('gate_deadline')


def test_missing_video_acceptance_tools_block_before_gpu_clock(tmp_path, monkeypatch):
    import sys
    import visuotactile.native_pipeline as module
    monkeypatch.setattr(module.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(module.shutil, 'which', lambda name: None)
    monkeypatch.setitem(sys.modules, 'imageio_ffmpeg', None)
    (tmp_path/'session.json').write_text('{"deadline": null, "gate_deadline": null, "training_deadline": null}')
    (tmp_path/'manifest.json').write_text('{"stages": {}}')
    runner = module.Pipeline(tmp_path)
    monkeypatch.setattr(runner, 'stage', lambda *a, **kw: tmp_path)
    monkeypatch.setattr(runner, 'start_gpu_clock', lambda: pytest.fail('Missing tools consumed GPU budget'))
    runner.run()
    assert runner.manifest['state'] == 'blocked'
    assert runner.manifest['last_failure']['stage'] == 'video_preflight'
    assert 'ffmpeg unavailable' in runner.manifest['reason']


def test_video_preflight_checks_explicit_tools_and_library_failures(monkeypatch):
    from types import SimpleNamespace
    import visuotactile.native_pipeline as module
    monkeypatch.setenv('NATIVE_DP_FFPROBE', '/explicit/ffprobe')
    monkeypatch.setenv('NATIVE_DP_FFMPEG', '/explicit/ffmpeg')
    monkeypatch.setattr(module.shutil, 'which', lambda name: name if name.startswith('/explicit/') else None)
    calls = []
    def execute(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', execute)
    assert module.video_tools() == {'ffprobe': '/explicit/ffprobe', 'ffmpeg': '/explicit/ffmpeg'}
    assert calls == [['/explicit/ffprobe', '-version'], ['/explicit/ffmpeg', '-version']]
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw:
        SimpleNamespace(returncode=127, stderr='error while loading shared libraries: libass.so.9'))
    with pytest.raises(RuntimeError, match='ffprobe cannot run.*shared libraries'):
        module.video_tools()


def test_late_gate_authorization_preserves_original_deadlines(tmp_path, monkeypatch):
    import json
    import visuotactile.native_pipeline as module
    monkeypatch.setattr(module.time, 'time', lambda: 250.)
    session = dict(started_epoch=1, gate_deadline=100, training_deadline=800,
                   deadline=1000, source_commit='old', source_hashes={})
    (tmp_path/'session.json').write_text(json.dumps(session))
    result = module.authorize_late_gate(tmp_path, 'Emergency recovery after Kit initialization crash')
    saved = json.loads((tmp_path/'session.json').read_text())
    assert result['authorized'] and result['deadline_unchanged']
    assert saved['gate_deadline'] == 100 and saved['training_deadline'] == 800 and saved['deadline'] == 1000
    assert json.loads((tmp_path/'late_gate_recovery.json').read_text()) == result
    with pytest.raises(ValueError, match='already authorized'):
        module.authorize_late_gate(tmp_path, 'A second late gate is forbidden')


def test_late_gate_authorization_rejects_before_gate_or_after_total(tmp_path, monkeypatch):
    import json
    import visuotactile.native_pipeline as module
    session = dict(started_epoch=1, gate_deadline=100, training_deadline=800,
                   deadline=1000, source_commit='old', source_hashes={})
    (tmp_path/'session.json').write_text(json.dumps(session))
    monkeypatch.setattr(module.time, 'time', lambda: 50.)
    with pytest.raises(ValueError, match='only valid after'):
        module.authorize_late_gate(tmp_path, 'Too early late gate')
    monkeypatch.setattr(module.time, 'time', lambda: 1001.)
    with pytest.raises(TimeoutError, match='total deadline'):
        module.authorize_late_gate(tmp_path, 'Too late late gate')


def test_kit_crash_is_classified_from_worker_log(tmp_path):
    import visuotactile.native_pipeline as module
    log = tmp_path/'worker.log'
    log.write_text("terminate called after throwing an instance of 'std::system_error'\n")
    assert module.failure_class(-6, log) == 'kit_native_crash'
    assert module.failure_class(-9, log) == 'resource_kill_or_oom'
    assert module.failure_class(1, log) == 'worker_failure'
