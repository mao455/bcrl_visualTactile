"""Opt-in B frontend component diagnostics; never a complete DP/rear pair.

No runner imports this candidate. Live sensor/command acceptance is still
required before using it in Kit. Neither mode instantiates a rear network.
"""
import copy
import json
from pathlib import Path

import torch

from .action_history import ExecutionHistory
from .batched_controller import CurrentFrame, lane_tree
from .compact import sha256
from .execution_learning import ideal_execution_bounds
from .models.native_vision import vision_lane_budget
from .native_controller import NativeController, InferenceTimings
from .position_control import (POSITION_ACTION, bounded_targets, continuous_image_contract,
    project_processed_position_targets, validate_execution_adapter, validate_execution_environment)
from .residual_output import REACHABLE_CENTERED_OUTPUT

VERSION = 'batched_trained_B_frontend_diagnostic_v1'
VISION_MICROBATCH = 8
LARGE_PROJECTION_RAD = .01  # Reporting threshold only; never a control/success rule.


def require(value, message):
    if not value:
        raise ValueError(message)


def selected_frontend(checkpoint, expected_sha256):
    """Load an actual completed training best EMA, not an update-zero prior."""
    path = Path(checkpoint).resolve(strict=True)
    require(path.name == 'best.pt', 'Diagnostic requires the selected best.pt frontend')
    require(isinstance(expected_sha256, str) and len(expected_sha256) == 64 and
        sha256(path) == expected_sha256, 'Registered frontend checkpoint SHA differs')
    finished_path = path.parent/'finished.json'
    finished = json.loads(finished_path.read_text())
    require(finished.get('complete') is True and finished.get('checkpoint_sha256') == expected_sha256,
        'Frontend training completion does not bind the selected best')
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    require(saved.get('kind') == 'dp' and saved.get('group') == 'I-HistoryDP' and
        saved.get('contract') == continuous_image_contract('u_expert') and
        'residual_version' not in saved and 'ema' in saved,
        'Only the explicit B u_expert G00 H24 every-frame frontend is supported')
    require(type(saved.get('updates')) is int and saved['updates'] > 0 and
        type(saved.get('epoch')) is int and saved['epoch'] > 0 and
        not saved.get('disposable') and not saved.get('configuration', {}).get('disposable'),
        'A trained nondisposable selected frontend is required')
    return saved, dict(frontend_checkpoint=str(path), frontend_sha256=expected_sha256,
        finished_sha256=sha256(finished_path), selected_epoch=saved['epoch'],
        selected_updates=saved['updates'], checkpoint_weights='validation_selected_best_ema')


def frontend_target(nominal, previous, limits, *, mode, execution_adapter):
    """Deterministic target map, separate from learned rear inference.

    Projection uses the same nominal-centered anchor and final projection as
    reachable_nominal_centered_linear_v1, with no learned proposal. Applying
    only the final tolerance-aware projection to the unclipped nominal would
    differ at its roundoff boundary, so preserve the exact two-stage map.
    """
    require(mode in ('direct', 'projection'), 'Explicit direct/projection diagnostic mode required')
    adapter = validate_execution_adapter(execution_adapter)
    require((mode == 'projection') == (adapter is not None), 'Projection adapter and mode differ')
    if mode == 'direct':
        return bounded_targets(nominal, limits)
    bounds = ideal_execution_bounds(previous, limits, adapter)
    anchor = torch.maximum(torch.minimum(nominal, bounds[..., 1]), bounds[..., 0])
    return project_processed_position_targets(bounded_targets(anchor, limits), previous, limits, adapter)


class BatchedFrontendDiagnostic:
    """Shared frozen EMA, M8 current vision; independent native lane histories.

    act/observe_execution form a transaction. A terminal transition still
    acknowledges the pre-step active mask, before that lane becomes inactive.
    New episode IDs require step0; no skipped native observation is accepted.
    """
    def __init__(self, checkpoint, metadata, dino, count, *, mode, checkpoint_sha256,
                 execution_adapter=None, device='cuda:0'):
        require(type(count) is int and count > 0, 'Positive integer lane count required')
        require(mode in ('direct', 'projection'), 'Explicit direct/projection diagnostic mode required')
        adapter = validate_execution_adapter(execution_adapter)
        require((mode == 'projection') == (adapter is not None), 'Projection adapter and mode differ')
        require(metadata.get('action_control_mode') == POSITION_ACTION and
            abs(metadata.get('control_dt', 0)-1/60) < 1e-10 and
            abs(metadata.get('physics_dt', 0)-1/120) < 1e-10,
            'Original absolute-target 120Hz physics / 60Hz control metadata required')
        validate_execution_environment(adapter, metadata)
        limits = torch.as_tensor(metadata['joint_limits_rad'])
        require(limits.shape == (count, 29, 2) and torch.isfinite(limits).all() and
            (limits[..., 0] <= limits[..., 1]).all(), 'Independent finite canonical29 lane limits required')
        saved, selected = selected_frontend(checkpoint, checkpoint_sha256)
        self.count, self.device, self.dino = count, torch.device(device), dino
        self.mode, self.execution_adapter = mode, adapter
        first = NativeController(saved, metadata, dino=None, device=device, scheduler_index_mode='host')
        first.model.eval().requires_grad_(False)
        self.lanes = [first] + [copy.copy(first) for _ in range(count-1)]
        for lane, controller in enumerate(self.lanes):
            controller.limits = limits[lane].detach().cpu().clone()
            controller.dino = CurrentFrame()
            controller.generator = torch.Generator(device=device).manual_seed(saved['contract']['sampler_seed'])
            controller.execution = ExecutionHistory(1)
            controller.timings = InferenceTimings(device)
            controller.position_previous = None
        self.identity = dict(version=VERSION, controller=type(self).__name__, mode=mode,
            **selected, complete_student_pair=False, rear_network_loaded=False,
            supervision='u_expert', modality='G00', horizon=24, commit=12,
            physics_hz=120, control_hz=60, plan_hz=5, visual_stride=1,
            vision_microbatch=VISION_MICROBATCH, vision_tail_profile=None,
            scheduler_index_mode='host', execution_adapter=copy.deepcopy(adapter),
            projection_anchor_contract=copy.deepcopy(REACHABLE_CENTERED_OUTPUT) if adapter else None,
            projection_reporting_threshold_rad=LARGE_PROJECTION_RAD,
            history='Every real observation; strict past actually submitted/applied targets',
            scope='Frontend component diagnostic, not the trained complete DP/rear policy')
        self.visual_frames = [0]*count
        self.history_ticks = [0]*count
        self.executed_ticks = [0]*count
        self.plan_records = [[] for _ in range(count)]
        self.projection = [dict(ticks=0, changed_ticks=0, large_ticks=0, large_joint_components=0,
            absolute_sum_rad=0., maximum_abs_rad=0., arm_max_abs_rad=0., hand_max_abs_rad=0.) for _ in range(count)]
        self.vision_work = vision_lane_budget(0, VISION_MICROBATCH)
        self.vision_calls = 0
        self.last_nominal = [None]*count
        self.last_projection = [None]*count
        self._pending_active = None
        self._failed = False

    def _active(self, active):
        active = torch.as_tensor(active, device='cpu')
        require(active.dtype == torch.bool and active.shape == (self.count,), 'Boolean lane mask required')
        return active

    @torch.no_grad()
    def act(self, packet, diagnostics, active):
        require(not self._failed, 'Failed diagnostic controller cannot continue')
        require(self._pending_active is None, 'Actual execution acknowledgement required before next observation')
        active = self._active(active)
        require(len(packet['episode_id']) == self.count, 'Packet lane population differs')
        commands = packet['actuator_target_rad'].detach().cpu().clone()
        require(commands.shape == (self.count, 29) and torch.isfinite(commands).all(),
            'Finite actual actuator targets required')
        ids = active.nonzero().flatten().tolist()
        if not ids:
            return commands
        try:
            # Do not iterate or slice the diagnostic object/goal dictionary.
            # Only actual encoder joint position/velocity cross this boundary.
            joints = {key: diagnostics[key] for key in ('joint_position_rad', 'joint_velocity_rad_s')}
            rgb = packet['image'][ids, :, :3].to(self.device)
            patches = self.dino.forward_lanes(rgb, vision_microbatch=VISION_MICROBATCH)
            require(len(patches) == len(ids), 'Current DINO batch lost lane identity')
            work = vision_lane_budget(len(ids), VISION_MICROBATCH)
            for key in ('forward_features_calls', 'real_lanes', 'padding_lanes', 'real_images', 'processed_images'):
                self.vision_work[key] += work[key]
            self.vision_calls += 1
            replanners, conditions, packets = [], [], {}
            for local, lane in enumerate(ids):
                controller = self.lanes[lane]
                controller.dino.set(patches[local])
                packets[lane] = lane_tree(packet, lane, self.count)
                condition = controller.prepare_observation(packets[lane], lane_tree(joints, lane, self.count))
                require(controller.dino.consumed, 'Every active current observation must enter learned history')
                self.visual_frames[lane] += 1
                self.history_ticks[lane] += 1
                if condition is not None:
                    replanners.append(lane); conditions.append(condition)
            plans = {}
            if replanners:
                controllers = [self.lanes[lane] for lane in replanners]
                tokens = [controller.timings.begin('ddim_plan', int(packets[lane]['step_id'][0]))
                    for lane, controller in zip(replanners, controllers)]
                predicted = controllers[0].model.predict(torch.cat(conditions),
                    [controller.generator for controller in controllers], scheduler_index_mode='host').cpu()
                require(predicted.shape == (len(replanners), 24, 29) and torch.isfinite(predicted).all(),
                    'Expected finite batched H24 canonical29 plans')
                for index, (lane, controller, token) in enumerate(zip(replanners, controllers, tokens)):
                    plans[lane] = predicted[index:index+1]
                    controller.timings.finish(token)
                    self.plan_records[lane].append(dict(episode_id=int(packets[lane]['episode_id'][0]),
                        step_id=int(packets[lane]['step_id'][0]), plan=plans[lane][0].clone()))
            for lane in ids:
                controller = self.lanes[lane]
                nominal = controller.finish_plan(plans.get(lane))
                target = frontend_target(nominal.to(self.device),
                    controller.last_robot[None, 58:87].to(self.device), controller.limits.to(self.device),
                    mode=self.mode, execution_adapter=self.execution_adapter).cpu()
                require(torch.isfinite(target).all(), 'Nonfinite diagnostic target')
                # The next NativeController observation must see the executed
                # postprocessed target, not the original private DP proposal.
                controller.command = target.clone()
                commands[lane:lane+1] = target
                self.last_nominal[lane] = nominal.clone()
                delta = target-nominal
                self.last_projection[lane] = delta.clone()
                absolute = delta.abs(); row = self.projection[lane]
                row['ticks'] += 1
                row['changed_ticks'] += int(bool((absolute > 0).any()))
                row['large_ticks'] += int(bool((absolute > LARGE_PROJECTION_RAD).any()))
                row['large_joint_components'] += int((absolute > LARGE_PROJECTION_RAD).sum())
                row['absolute_sum_rad'] += float(absolute.sum())
                for key, values in (('maximum_abs_rad', absolute),
                        ('arm_max_abs_rad', absolute[:, :7]), ('hand_max_abs_rad', absolute[:, 7:])):
                    row[key] = max(row[key], float(values.max()))
            self._pending_active = active.clone()
            self._pending_commands = commands.clone()
            return commands
        except BaseException:
            self._failed = True
            raise

    def observe_execution(self, transition, active):
        require(not self._failed and self._pending_active is not None,
            'Execution acknowledgement requires one valid pending action')
        active = self._active(active)
        require(torch.equal(active, self._pending_active), 'Acknowledge the pre-step active lanes, including terminal lanes')
        for key in ('position_requested_rad', 'position_applied_rad'):
            value = transition[key].detach().cpu()
            require(value.shape == (self.count, 29) and torch.isfinite(value).all(),
                'Actual requested/applied target receipt differs')
        require(torch.equal(transition['position_requested_rad'].detach().cpu()[active],
            self._pending_commands[active]), 'Actual submitted targets differ from diagnostic command')
        ids = active.nonzero().flatten().tolist()
        for lane in ids:
            controller = self.lanes[lane]
            for key in ('episode_id', 'step_id'):
                require(int(transition[key][lane]) == int(controller.last_packet[key][0]),
                    'Execution receipt episode/tick differs from current lane')
        for lane in ids:
            self.lanes[lane].observe_execution(lane_tree(transition, lane, self.count))
            self.executed_ticks[lane] += 1
        self._pending_active = None
        del self._pending_commands

    def execution_report(self):
        require(self.visual_frames == self.history_ticks and
            sum(self.visual_frames) == self.vision_work['real_lanes'], 'Frontend observation accounting differs')
        return dict(identity=copy.deepcopy(self.identity), failed=self._failed,
            all_submitted_acknowledged=self._pending_active is None,
            vision=dict(self.vision_work, controller_calls=self.vision_calls),
            lanes=[dict(lane=lane, actual_visual_observations=self.visual_frames[lane],
                history_updates=self.history_ticks[lane], acknowledged_control_ticks=self.executed_ticks[lane],
                plans_generated=len(self.plan_records[lane]), projection=copy.deepcopy(self.projection[lane]))
                for lane in range(self.count)],
            rear_forward_ticks=0, complete_student_pair=False,
            limits='Controller accounting only; live sensor clocks, contact physics and autonomous outcomes require separate acceptance')

    def inference_metrics(self, lane):
        result = self.lanes[lane].inference_metrics()
        result['batch_timing_scope'] = 'Shared DDIM durations overlap across lanes; never sum lane times for throughput'
        result['controller_kind'] = VERSION
        return result
