"""CPU provenance gates; actual Kit recovery is a separate bounded live gate."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from visuotactile.expert_speed import cap_hand_target, progress_target
from visuotactile.position_control import ExpertPositionBridge, POSITION_ACTION, apply_position_targets
from visuotactile.state_benchmark import dagger_configuration, record_dagger_transition


def native_pipeline():
    # Load the pure upstream function without importing the simulator registry.
    spec = importlib.util.spec_from_file_location('_dagger_native_math',
        Path(__file__).resolve().parents[1]/'isaacsimenvs/tasks/play/utils/action_utils.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.apply_action_pipeline


def environment(delay=True):
    generator = torch.Generator().manual_seed(31)
    perm = torch.cat((torch.randperm(7, generator=generator), 7+torch.randperm(22, generator=generator)))
    limits = torch.tensor([[-2., 2.]]*29)[None]
    return NS(device='cpu', num_envs=1, step_dt=1/60, action_control_mode=POSITION_ACTION,
        _perm_canon_to_lab=perm, _perm_lab_to_canon=perm.argsort(),
        _arm_joint_ids=torch.arange(7), _hand_joint_ids=torch.arange(7, 29),
        _arm_lower=limits[0, :7, 0], _arm_upper=limits[0, :7, 1],
        _hand_lower=limits[0, 7:, 0], _hand_upper=limits[0, 7:, 1],
        robot=NS(data=NS(joint_pos_limits=limits)), _cur_targets=torch.zeros(1, 29),
        _prev_targets=torch.zeros(1, 29), _action_queue=torch.full((1, 4, 29), .91),
        episode_length_buf=torch.zeros(1, dtype=torch.long), _successes=torch.zeros(1),
        cfg=NS(domain_randomization=NS(use_action_delay=delay, action_delay_max=3),
            action=NS(dof_speed_scale=1.5, arm_moving_average=.1, hand_moving_average=.1)))


@pytest.mark.parametrize('delay', [True, False])
def test_bridge_matches_native_mapping_without_stale_queue_or_extra_rng(delay):
    env = environment(delay); reference = copy.deepcopy(env)
    pipeline = native_pipeline()
    bridge = ExpertPositionBridge(env, 3, ExpertPositionBridge.speed_scheme, pipeline=pipeline)
    env._dagger_position_bridge = bridge
    original_queue = env._action_queue.clone()
    for tick in range(14):
        raw = torch.linspace(-1.3, 1.4, 29)[None]+tick*.02
        clipped = raw.clamp(-1, 1)
        previous = env._prev_targets[:, env._perm_lab_to_canon].clone()
        reference._prev_targets = env._prev_targets.clone()
        reference._cur_targets = env._cur_targets.clone()
        torch.manual_seed(140+tick)
        pipeline(reference, clipped)
        expected_rng = torch.get_rng_state()
        native = reference._cur_targets[:, env._perm_lab_to_canon]
        mapped = cap_hand_target(previous, progress_target(previous, native, .5, 1.), 2., env.step_dt)
        student = torch.full((1, 29), .1*tick)
        bridge.submit(raw, clipped, tick)
        torch.manual_seed(140+tick)
        applied, draw = apply_position_targets(env, student)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        assert torch.equal(bridge.trace['expert_native_target_rad'], native)
        assert torch.equal(bridge.trace['expert_mapped_target_rad'], mapped)
        assert torch.equal(applied, student if tick < 3 else mapped)
        assert torch.equal(env._action_queue, original_queue)
        assert env.action_control_mode == POSITION_ACTION
        assert bool(bridge.trace['expert_label_valid'][0]) == (tick >= (6 if delay else 3))
        assert int(bridge.trace['segment_id'][0]) == (0 if tick < 3 else 1 if delay and tick < 6 else 2 if delay else 1)
        if delay:
            assert int(bridge.trace['expert_source_step'][0]) == max(0, tick-int(draw[0]))
        assert ((mapped[:, 7:]-previous[:, 7:]).abs() <= 2/60+1e-7).all()
        reference.episode_length_buf += 1


def test_bridge_rejects_missing_shadow_duplicate_ticks_and_raw_as_position():
    env = environment(); bridge = ExpertPositionBridge(env, 2, ExpertPositionBridge.speed_scheme, pipeline=native_pipeline())
    with pytest.raises(ValueError, match='Missing expert'):
        bridge.select(env, torch.zeros(1, 29), torch.zeros(1, dtype=torch.long))
    with pytest.raises(ValueError, match='provenance'):
        bridge.submit(torch.ones(1, 29)*2, torch.ones(1, 29)*2, 0)
    bridge.submit(torch.ones(1, 29), torch.ones(1, 29), 0)
    with pytest.raises(ValueError, match='exactly once'):
        bridge.submit(torch.ones(1, 29), torch.ones(1, 29), 0)
    env.action_control_mode = 'native_command_clipped_pre_delay'
    with pytest.raises(ValueError, match='changed position'):
        bridge.select(env, torch.zeros(1, 29), torch.zeros(1, dtype=torch.long))


@pytest.mark.parametrize('clean_tick',[48,72,96,120,156,192,240,300])
def test_takeover_clean_boundary_can_align_with_unchanged_native_plan_phase(clean_tick):
    env=environment();env._action_queue=env._action_queue[:,:3].clone()
    switch=clean_tick-(env._action_queue.shape[1]-1)
    bridge=ExpertPositionBridge(env,switch,ExpertPositionBridge.speed_scheme,pipeline=native_pipeline())
    env._dagger_position_bridge=bridge
    mask=[];sources=[]
    for tick in range(clean_tick+1):
        bridge.submit(torch.zeros(1,29),torch.zeros(1,29),tick)
        apply_position_targets(env,torch.zeros(1,29))
        mask.append(bool(bridge.trace['expert_label_valid'][0]))
        sources.append(int(bridge.trace['control_source'][0]))
    assert sources.index(1)==switch and mask.index(True)==clean_tick
    assert next(t for t in range(0,len(mask),12) if mask[t])==clean_tick
    assert not any(mask[switch:clean_tick])
    assert clean_tick%12==0 and bridge.queue_length==3


def test_actual_expert_acknowledgement_and_q_reached_are_not_student_counterfactuals():
    env = environment(False); bridge = ExpertPositionBridge(env, 1, ExpertPositionBridge.speed_scheme, pipeline=native_pipeline())
    env._dagger_position_bridge = bridge
    controller = NS(nominal=torch.full((1, 29), .2), command=torch.full((1, 29), .7),
        base=NS(command=torch.full((1, 29), .7)), residual_diagnostics=dict(applied_rad=torch.full((1, 29), .5)))
    controller.action_scale_cpu = torch.ones(29); controller.residual_scales_cpu = torch.full((29,), .05)
    acknowledgements = []
    controller.observe_execution = lambda tr: acknowledgements.append(controller.base.command.clone())
    previous_nominal = None
    for tick in range(3):
        bridge.submit(torch.ones(1, 29), torch.ones(1, 29), tick)
        applied, _ = apply_position_targets(env, torch.full((1, 29), .7))
        reached = applied*.8  # Distinct observed integration outcome, not a target.
        tr = dict(dagger=copy.deepcopy(bridge.trace), target_after_rad=applied.clone(),
            position_requested_rad=bridge.requested.clone(), position_applied_rad=applied.clone(),
            actuator_substep_target_rad=applied[:, None].repeat(1, 2, 1),
            diagnostics_after=dict(joint_position_rad=reached))
        previous_nominal = record_dagger_transition(controller, tr, previous_nominal)
        assert torch.equal(acknowledgements[-1], applied)
        assert torch.equal(tr['dagger']['q_reached'], reached)
        assert not torch.equal(tr['dagger']['q_reached'], tr['dagger']['expert_mapped_target_rad'])
        assert bool(tr['dagger']['previous_nominal_valid'][0]) == (tick > 0)
        assert torch.equal(tr['dagger']['u_expert'], applied if tick else torch.zeros_like(applied))
        if tick:
            assert torch.equal(tr['dagger']['previous_nominal'], controller.nominal)
            assert not torch.equal(controller.base.command, controller.nominal)
        tr['actuator_substep_target_rad'][:, 1, 0] += .01
        with pytest.raises(ValueError, match='physics substep'):
            record_dagger_transition(controller, tr, previous_nominal)


def test_train_only_frozen_takeover_job_validation():
    config = dict(version=ExpertPositionBridge.version, expert_speed=ExpertPositionBridge.speed_scheme)
    spec = dict(cohort='dagger_train', split='train', narrow_reset={}, source_identity='train_a',
        takeover_tick=120, event_seed=1001)
    source = dict(splits=dict(train=['train_a'], validation=['val_a']))
    assert dagger_configuration(config, spec, source)['takeover_tick'] == 120
    for changed in (dict(cohort='fresh'), dict(split='test'), dict(source_identity='val_a'), dict(takeover_tick=0)):
        with pytest.raises(ValueError):
            dagger_configuration(config, dict(spec, **changed), source)
    with pytest.raises(ValueError, match='speed scheme'):
        dagger_configuration(dict(config, expert_speed=dict(ExpertPositionBridge.speed_scheme, arm=.67)), spec, source)


@pytest.mark.parametrize('steps,native_success,accepted', [(8, True, True), (1, True, False), (8, False, False)])
def test_existing_trial_writes_full_compact_episode_and_separates_mixed_success(
        tmp_path, monkeypatch, steps, native_success, accepted):
    """Execute the shared runner with CPU fake physics, not a second collector."""
    import contextlib
    import json
    import h5py
    import visuotactile.native_controller as controllers
    import visuotactile.position_control as positions
    import visuotactile.state_benchmark as benchmark
    from visuotactile.recording import verify_stream_fields
    env = environment()
    env.env_max_goals = torch.tensor([1]); env.max_episode_length = 10
    env.scene = NS(env_origins=torch.zeros(1, 3)); env.cfg.to_dict = lambda: {}
    env.research_metadata = lambda: {}
    env.research_step = 0

    def diagnostics():
        pose = torch.tensor([[0., 0., .5, 1., 0., 0., 0.]])
        return dict(joint_position_rad=env._prev_targets[:, env._perm_lab_to_canon]*.8,
            joint_velocity_rad_s=torch.ones(1, 29)*.03, object_pose_w=pose,
            palm_body_pose_w=pose.clone(), finger_position_w=torch.zeros(1, 5, 3),
            lifted_object=torch.zeros(1, dtype=torch.bool), successes=torch.zeros(1),
            retract_succeeded=torch.zeros(1, dtype=torch.bool))

    def packet(tick):
        image = torch.full((1, 3, 4, 2, 2), (tick % 255)/255.)
        return dict(image=image, depth_m=image[:, :, 3:].clone(), depth_valid=torch.ones(1, 3, 1, 2, 2, dtype=torch.bool),
            camera_time_s=torch.full((1, 3), tick/60., dtype=torch.float64), camera_frame=torch.full((1, 3), tick),
            camera_valid=torch.ones(1, 3, dtype=torch.bool), time_s=torch.tensor([tick/60.], dtype=torch.float64),
            episode_id=torch.tensor([7]), step_id=torch.tensor([tick]))

    env._diagnostics = diagnostics; env.research_packet = packet(0)

    def prepare(env, isolated, spec, source, output):
        (output/'initial_observable.json').write_text(json.dumps({'object_pose_w': [[0., 0., .5, 1., 0., 0., 0.]]}))
        return dict(policy=torch.zeros(1, 3)), dict(legal=True)

    def step(proposal):
        tick = env.research_step
        before = env._prev_targets[:, env._perm_lab_to_canon].clone()
        applied, draw = apply_position_targets(env, proposal)
        bridge = env._dagger_position_bridge
        done = tick+1 == steps
        env.last_transition = dict(episode_id=torch.tensor([7]), step_id=torch.tensor([tick]),
            time_before_s=torch.tensor([tick/60.], dtype=torch.float64),
            time_after_s=torch.tensor([(tick+1)/60.], dtype=torch.float64),
            target_before_rad=before, target_after_rad=applied.clone(),
            position_requested_rad=bridge.requested.clone(), position_applied_rad=applied.clone(),
            dagger=copy.deepcopy(bridge.trace), control_trace=dict(delay_index=torch.zeros(1, dtype=torch.long)),
            diagnostics_after=diagnostics(), actuator_substep_target_rad=applied[:, None].repeat(1, 2, 1),
            contact_substeps_after_w_N=torch.zeros(1, 2, 5, 3), object_contact_force_w=torch.zeros(1, 2, 5, 3),
            applied_wrench=dict(force_w_N=torch.zeros(1, 2, 3), torque_w_Nm=torch.zeros(1, 2, 3)),
            termination_reasons=dict(max_successes=torch.tensor([done and native_success]), timeout=torch.tensor([done and not native_success])),
            native_alignment_condition=torch.zeros(1, dtype=torch.bool))
        env.research_step += 1; env.research_packet = packet(env.research_step)
        return dict(policy=torch.full((1, 3), float(env.research_step))), None, torch.tensor([done]), torch.tensor([False]), {}

    env.step = step
    class Expert(torch.nn.Module):
        def forward(self, actor):
            return dict(mus=torch.full((1, 29), 1.+float(actor[0, 0])*.01))
    player = NS(model=Expert(), intr_reward_coef_embd=torch.zeros(1, 1), reset=lambda: None)
    calls = []
    def get_action(actor, is_deterministic):
        calls.append(float(actor[0, 0])); return player.model(actor)['mus'].clamp(-1, 1)
    player.get_action = get_action
    instances = []
    class Student:
        def __init__(self, *args, **kwargs):
            self.base = NS(command=None); self.model = NS(modality='G00'); self.seen = []; instances.append(self)
            self.action_scale_cpu = torch.ones(29); self.residual_scales_cpu = torch.full((29,), .05)
        def act(self, packet, joints):
            self.seen.append(self.base.command.clone() if self.base.command is not None else None)
            tick = int(packet['step_id'][0]); self.nominal = torch.full((1, 29), .1+tick*.01)
            self.command = self.nominal+.1; self.plan = self.nominal[:, None].repeat(1, 24, 1)
            self.delta = torch.full((1, 29), .1); self.contact_gate = False
            self.residual_diagnostics = dict(proposed_rad=self.delta, applied_rad=self.delta,
                clip_mask=torch.zeros(1, 29, dtype=torch.bool), tanh_fraction=self.delta, contact_valid=torch.tensor(False))
            return self.command
        def observe_execution(self, transition):
            assert torch.equal(self.base.command, transition['position_requested_rad'])
    monkeypatch.setattr(controllers, 'ResidualController', Student)
    original_bridge = positions.ExpertPositionBridge
    class CPUBridge(original_bridge):
        def __init__(self, *args, **kwargs): super().__init__(*args, pipeline=native_pipeline(), **kwargs)
    monkeypatch.setattr(positions, 'ExpertPositionBridge', CPUBridge)
    monkeypatch.setattr(benchmark, 'prepare', prepare)
    monkeypatch.setattr(benchmark, 'environment_identity', lambda env: ({}, 'test_configuration'))
    monkeypatch.setattr(benchmark, 'export_initial', lambda *args: {})
    from visuotactile.models.native_dp import TargetResidual
    from visuotactile.compact import sha256
    frontend = tmp_path/'frontend.pt'
    contract = dict(supervision='q_reached', modality='G00')
    torch.save(dict(contract=contract), frontend)
    saved = dict(group='I-OfficialDP', residual_version=TargetResidual.version, kind='test', updates=1,
        configuration=dict(base_checkpoint=str(frontend), base_sha256=sha256(frontend)), contract=contract)
    checkpoint = tmp_path/'student.pt'; torch.save(saved, checkpoint)
    expert = tmp_path/'expert.pt'; expert.write_bytes(b'test_expert_identity')
    result = benchmark.trial(env, NS(block=lambda: contextlib.nullcontext()), tmp_path/'trial',
        dict(identity='attempt', seed=99, cohort='dagger_train', split='train', source_identity='train_a',
            narrow_reset={}, event_seed=1001, takeover_tick=2), dict(splits=dict(train=['train_a'])),
        player, str(checkpoint), saved=saved,
        position=True, capture_dataset=True, expert_checkpoint=str(expert),
        dagger=dict(version=original_bridge.version, expert_speed=original_bridge.speed_scheme))
    assert calls == list(range(steps)) and len(instances[0].seen) == steps
    assert not hasattr(env, '_dagger_position_bridge')
    assert result['success'] == native_success
    assert result['dataset_capture']['accepted_success'] == accepted
    assert result['dagger']['expert_valid_ticks'] == max(0, steps-5)
    assert result['dagger']['student_autonomous_success'] == (native_success and steps==1)
    assert result['complete_student_policy']['frontend_sha256'] == sha256(frontend)
    assert result['complete_student_policy']['backend_sha256'] == sha256(checkpoint)
    acceptance = benchmark.audit_dagger_capture(tmp_path/'trial')
    assert acceptance['passed'] and acceptance['student_ticks'] == min(2, steps)
    assert acceptance['recovery_coverage']['population']['first24']['valid_expert_ticks'] == max(0, steps-5)
    path = tmp_path/'trial/demonstration/data.h5'
    assert verify_stream_fields(path) > 30
    with h5py.File(path) as stream:
        g = stream['episodes/attempt']; d = g['transition/dagger']
        assert g.attrs['length'] == steps and g.attrs['complete']
        assert d['expert_label_valid'][:].tolist() == [tick >= 5 for tick in range(steps)]
        assert d['segment_id'][:].tolist() == [0 if tick < 2 else 1 if tick < 5 else 2 for tick in range(steps)]
        assert (d['q_reached'][:] == g['transition/diagnostics_after/joint_position_rad'][:]).all()
        assert (g['frames/post_index'][-1:] == -1).all()
        for tick in range(1, steps):
            assert torch.equal(instances[0].seen[tick], torch.from_numpy(d['actual_issued'][tick-1:tick]))
    import numpy as np
    with np.load(tmp_path/'trial/trace.npz') as archive:
        trace = {key: archive[key] for key in archive.files}
    trace['nominal'][0, 0] += .1
    np.savez_compressed(tmp_path/'trial/trace.npz', **trace)
    with pytest.raises(ValueError, match='numerical chain mismatch'):
        benchmark.audit_dagger_capture(tmp_path/'trial')


def test_complete_policy_identity_rejects_bare_or_unpaired_checkpoint(tmp_path):
    from visuotactile.state_benchmark import dagger_policy_identity
    from visuotactile.compact import sha256
    from visuotactile.models.native_dp import TargetResidual
    front = tmp_path/'front.pt'; front.write_bytes(b'identity only')
    back = tmp_path/'back.pt'; back.write_bytes(b'identity only')
    saved = dict(group='I-HistoryDP', residual_version=TargetResidual.version, updates=5,
        configuration=dict(base_checkpoint=str(front), base_sha256=sha256(front)),
        contract=dict(supervision='u_expert', modality='G00'))
    assert dagger_policy_identity(saved, back)['zero_correction_is_valid']
    for change in (dict(residual_version=None), dict(group='state_dp'), dict(updates=0),
            dict(configuration={}), dict(contract={})):
        with pytest.raises(ValueError):
            dagger_policy_identity(dict(saved, **change), back)


def test_zero_correction_is_complete_policy_and_wrong_proposal_is_rejected():
    env = environment(False); bridge = ExpertPositionBridge(env, 1, ExpertPositionBridge.speed_scheme, pipeline=native_pipeline())
    env._dagger_position_bridge = bridge
    command = torch.full((1, 29), .1)
    controller = NS(nominal=command.clone(), command=command.clone(), base=NS(command=command.clone()),
        residual_diagnostics=dict(applied_rad=torch.zeros_like(command)), action_scale_cpu=torch.ones(29),
        residual_scales_cpu=torch.ones(29), observe_execution=lambda transition: None)
    bridge.submit(torch.zeros(1, 29), torch.zeros(1, 29), 0)
    applied, _ = apply_position_targets(env, command)
    transition = dict(dagger=copy.deepcopy(bridge.trace), target_after_rad=applied,
        position_requested_rad=applied.clone(), position_applied_rad=applied.clone(),
        actuator_substep_target_rad=applied[:, None].repeat(1, 2, 1),
        diagnostics_after=dict(joint_position_rad=applied*.8))
    record_dagger_transition(controller, transition, None)
    controller.residual_diagnostics['applied_rad'] = torch.ones_like(command)
    with pytest.raises(ValueError, match='nominal/residual/issued'):
        record_dagger_transition(controller, transition, None)


def test_only_exact_legacy_command_contract_can_infer_missing_supervision(tmp_path):
    from visuotactile.state_benchmark import dagger_policy_identity
    from visuotactile.position_control import image_contract
    from visuotactile.compact import sha256
    from visuotactile.models.native_dp import TargetResidual
    front = tmp_path/'front.pt'; front.write_bytes(b'identity only')
    back = tmp_path/'back.pt'; back.write_bytes(b'identity only')
    saved = dict(group='I-HistoryDP', residual_version=TargetResidual.version, updates=5,
        configuration=dict(base_checkpoint=str(front), base_sha256=sha256(front)))
    for modality in ('G00', 'G10'):
        legacy = image_contract('I-HistoryDP', 24, modality)
        identity = dagger_policy_identity(dict(saved, contract=legacy), back)
        assert identity['supervision'] == 'u_expert'
        assert identity['supervision_inference'] == 'legacy_v4_command_contract'
        for changed in (dict(version=5), dict(horizon=32), dict(normalization_identity='q_reached_train166_affine_v1'),
                dict(clean_sample_field='transition/diagnostics_after/joint_position_rad'), dict(supervision=None)):
            with pytest.raises(ValueError):
                dagger_policy_identity(dict(saved, contract=dict(legacy, **changed)), back)
        reached = image_contract('I-HistoryDP', 24, modality, 'q_reached')
        identity = dagger_policy_identity(dict(saved, contract=reached), back)
        assert identity['supervision'] == 'q_reached' and identity['supervision_inference'] is None
        # The same 29D action interface cannot recover a deleted Q supervision field.
        del reached['supervision']
        with pytest.raises(ValueError):
            dagger_policy_identity(dict(saved, contract=reached), back)


@pytest.mark.parametrize('boundary', ['mask_gap', 'segment_change', 'continuous', 'no_valid_labels'])
def test_recovery_initial_windows_never_join_across_mask_or_segment_boundary(boundary):
    import numpy as np
    from visuotactile.state_benchmark import dagger_recovery_coverage
    n = 72; source = np.zeros(n, dtype=int); source[10:] = 1
    valid = np.zeros(n, dtype=bool); valid[12:] = True
    segment = np.zeros(n, dtype=int); segment[10:12] = 1; segment[12:] = 2
    if boundary == 'mask_gap':
        valid[20] = False  # Later valid labels must not backfill either early window.
    elif boundary == 'segment_change':
        segment[20:] = 3
    elif boundary == 'no_valid_labels':
        valid[:] = False
    arrays = dict(dagger_control_source=source, dagger_expert_label_valid=valid,
        dagger_segment_id=segment, residual_contact_gate=np.zeros(n, dtype=bool),
        residual_contact_valid=np.zeros(n, dtype=bool))
    population = dagger_recovery_coverage(arrays, {})['population']
    if boundary in ('mask_gap', 'segment_change'):
        assert population['initial_valid_segment_ticks'] == 8
        for size in (24, 48):
            assert population['first'+str(size)] == dict(valid_expert_ticks=8, dp_anchors=1, valid_h24_window_targets=8)
        assert population['valid_expert_ticks'] > 8
    elif boundary == 'continuous':
        assert population['initial_valid_segment_ticks'] == 60
        assert population['first24'] == dict(valid_expert_ticks=24, dp_anchors=2, valid_h24_window_targets=48)
        assert population['first48'] == dict(valid_expert_ticks=48, dp_anchors=4, valid_h24_window_targets=96)
    else:
        assert population['initial_valid_segment_ticks'] == 0 and population['first_valid_expert_tick'] is None
        assert population['first24'] == population['first48'] == dict(valid_expert_ticks=0, dp_anchors=0, valid_h24_window_targets=0)
