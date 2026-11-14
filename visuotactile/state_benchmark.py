"""Frozen, pre-action legal State benchmark and shared native command runner."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback

import h5py
import numpy as np
import torch

from .compact import sha256
from .environment_rng import IsolatedEnvironment, RandomStream
from .video import RolloutVideo
from .expert import restore_observable_state
from .initial_geometry import inspect, boxes as object_boxes
from .native_data import CAMERAS
from .native_evaluation import client_startup_controls, make_env, kit_process_snapshot, exit_completed_kit_worker
from .expert import create_expert, environment_identity, export_initial
from .state_resources import lease_for
from .recording import append_numeric, BufferedNumericWriter
from .state_history import HistoryController
from .state_motion import plan_metrics, distribution
from .training_resources import atomic_json


def native_policy_action(controller, packet, diagnostics_provider):
    """Route actual joint observations without exposing task geometry to vision."""
    if controller.model.modality in ('G00', 'G10'):
        observed = diagnostics_provider()
        joints = {key: observed[key] for key in ('joint_position_rad', 'joint_velocity_rad_s')}
        return controller.act(packet, joints)
    if controller.model.group.startswith('I-'):
        return controller.act(packet)
    return controller.act(packet, diagnostics_provider())


def dagger_configuration(configuration, spec, source):
    """Validate a predeclared train-family takeover; never choose by outcome."""
    from .position_control import ExpertPositionBridge
    if configuration.get('version') != ExpertPositionBridge.version:
        raise ValueError('Explicit expert takeover position contract required')
    tick = spec.get('takeover_tick', configuration.get('takeover_tick'))
    if isinstance(tick, bool) or not isinstance(tick, int) or tick < 1:
        raise ValueError('Positive preregistered takeover tick required')
    if configuration.get('expert_speed') != ExpertPositionBridge.speed_scheme:
        raise ValueError('Existing validated expert speed scheme must be explicit')
    if spec.get('cohort') != 'dagger_train' or spec.get('split') != 'train' or 'narrow_reset' not in spec:
        raise ValueError('DAgger requires a fresh, explicitly train-only family reset')
    identity = spec.get('source_identity')
    if spec.get('training_expansion') is not None:
        from .training_expansion import validate_expansion_spec
        validate_expansion_spec(spec, source)
    elif identity not in source['splits']['train']:
        raise ValueError('DAgger family must come from the training split')
    if spec.get('event_seed') is None:
        raise ValueError('A separately registered new event seed is required')
    if configuration.get('trigger') is not None:
        from .takeover_trigger import validate_trigger
        validate_trigger(configuration['trigger'], tick)
    return dict(configuration, takeover_tick=tick, trajectory_kind='expert_takeover_v1',
        control_source_codes=dict(student=0, expert=1),
        segment_rule='Increase at control-source or supervision-validity changes; never join across episode reset',
        supervision_rule='Student prefix and the complete expert-delay queue flush have mask0; clean executed expert suffix only',
        q_reached_rule='Actual same-transition post-integration joint state; never a counterfactual expert response',
        expert_delay_rule='Continuous shadow queue; same single draw retained by the position executor; native queue untouched')


def dagger_policy_identity(saved, checkpoint, *, payload_audit=None):
    """Bind the actual two-part deployment, before any student action is allowed."""
    if payload_audit is not None and not isinstance(payload_audit,DaggerCaptureAuditMemo):
        raise TypeError('Explicit process-local capture auditor required')
    from .models.native_dp import TargetResidual
    from .state_history import trained_continuation_updates
    if (not isinstance(saved, dict) or saved.get('residual_version') != TargetResidual.version
            or saved.get('group') not in ('I-HistoryDP', 'I-OfficialDP')
            or trained_continuation_updates(saved) < 1):
        raise ValueError('DAgger requires a trained complete DP plus Transformer student')
    cfg = saved.get('configuration', {}); contract = saved.get('contract', {})
    frontend = Path(cfg.get('base_checkpoint', ''))
    payload_sha256 = sha256 if payload_audit is None else payload_audit.sha256
    if not frontend.is_file() or payload_sha256(frontend) != cfg.get('base_sha256'):
        raise ValueError('DAgger complete policy frontend identity mismatch')
    supervision = contract.get('supervision'); inference = None
    if 'supervision' not in contract:
        from .position_control import image_contract
        modality = contract.get('modality')
        # Exactly the historical command contract accepted by load_image_continuation.
        # Equal tensor dimensions, a version number alone, or an unknown Q contract
        # are not evidence that a checkpoint learned commands.
        if (saved['group'] != 'I-HistoryDP' or modality not in ('G00', 'G10')
                or contract != image_contract('I-HistoryDP', 24, modality)):
            raise ValueError('DAgger missing supervision is not a known legacy command contract')
        supervision = 'u_expert'; inference = 'legacy_v4_command_contract'
    if supervision not in ('u_expert', 'q_reached'):
        raise ValueError('DAgger complete policy requires known frontend supervision')
    from .position_control import validate_execution_adapter
    adapter = validate_execution_adapter(cfg.get('execution_adapter'))
    identity = dict(version='complete_student_policy_v1', controller='ResidualController',
        frontend=str(frontend.resolve()), frontend_sha256=cfg['base_sha256'],
        backend=str(Path(checkpoint).resolve()), backend_sha256=payload_sha256(checkpoint),
        residual_version=saved['residual_version'], supervision=supervision, supervision_inference=inference,
        modality=contract.get('modality'), output='absolute_joint_position_target_rad',
        composition='bounded nominal plus Transformer correction; actual issued/applied feedback',
        execution_adapter=adapter,
        zero_correction_is_valid=True)
    from .residual_output import REACHABLE_PREVIOUS_TARGET_OUTPUT
    output_contract = cfg.get('output_contract')
    if (output_contract == REACHABLE_PREVIOUS_TARGET_OUTPUT
            or saved.get('residual_output_version') == REACHABLE_PREVIOUS_TARGET_OUTPUT['version']):
        if (output_contract != REACHABLE_PREVIOUS_TARGET_OUTPUT
                or saved.get('residual_output_version') != REACHABLE_PREVIOUS_TARGET_OUTPUT['version']):
            raise ValueError('Complete policy output-anchor identity differs from checkpoint')
        identity.update(residual_output_contract=dict(output_contract),
            output_anchor='actual_pre_action_actuator_target_rad',
            composition='previous actual actuator target plus Transformer increment; full nominal plan input; common final reachable projection')
    return identity


def dagger_recovery_coverage(arrays, group):
    """Observed switch coverage only; never a policy input or a success criterion."""
    source = np.asarray(arrays['dagger_control_source']); valid = np.asarray(arrays['dagger_expert_label_valid'], dtype=bool)
    segment = np.asarray(arrays['dagger_segment_id']); starts = np.flatnonzero(source == 1)
    supervised = np.flatnonzero(valid); anchors = np.flatnonzero(valid & (np.arange(len(valid)) % 12 == 0))
    first = int(supervised[0]) if len(supervised) else None
    initial_end = first
    if first is not None:
        while initial_end < len(valid) and valid[initial_end] and segment[initial_end] == segment[first]:
            initial_end += 1
    def window_population(anchor):
        end = int(anchor)
        while end < min(int(anchor)+24, len(valid)) and valid[end] and segment[end] == segment[anchor]:
            end += 1
        return end-int(anchor)
    population = dict(valid_expert_ticks=int(valid.sum()), dp_anchors=len(anchors),
        first_valid_expert_tick=first,
        initial_valid_segment_ticks=initial_end-first if first is not None else 0,
        early_window_scope='First valid tick onward, stopping at the first invalid tick or control-segment change')
    for size in (24, 48):
        selected = np.arange(first, min(first+size, initial_end)) if first is not None else np.empty(0, dtype=int)
        population['first'+str(size)] = dict(valid_expert_ticks=len(selected),
            dp_anchors=int(np.isin(anchors, selected).sum()),
            valid_h24_window_targets=sum(window_population(t) for t in anchors if t in selected))
    coverage = dict(takeover_starts=int(len(starts) > 0), population=population,
        tick_index='zero-based pre-action switch; supervision starts only after the real queue flush',
        diagnostic_only=True, exact_simulator_state_claim=False)
    if not len(starts):
        return dict(coverage, switch_tick=None, switch_observable_sha256=None, phase='no_takeover', deviation=None)
    tick = int(starts[0]); fields = {}; missing = []
    paths = ('transition/diagnostics_before/joint_position_rad', 'transition/diagnostics_before/joint_velocity_rad_s',
        'transition/diagnostics_before/object_pose_w', 'transition/diagnostics_before/object_velocity_w_m_s_rad_s',
        'obs/actuator_target_rad', 'obs/contact_substeps_w_N', 'obs/contact_valid')
    digest = hashlib.sha256()
    for name in paths:
        if name not in group:
            missing.append(name); continue
        value = np.asarray(group[name][tick]); fields[name] = value
        digest.update(name.encode()); digest.update(str(value.dtype).encode()); digest.update(str(value.shape).encode())
        digest.update(np.ascontiguousarray(value).tobytes())
    def observed(name):
        return np.asarray(group['transition/diagnostics_before/'+name][tick]) if 'transition/diagnostics_before/'+name in group else None
    lifted = observed('lifted_object'); subgoals = observed('successes'); stage = observed('native_stage_step')
    contact_known = bool(np.asarray(arrays['residual_contact_valid'])[tick])
    contact = bool(np.asarray(arrays['residual_contact_gate'])[tick]) if contact_known else None
    previous_contact = np.asarray(arrays['residual_contact_gate'])[:tick]
    previous_known = np.asarray(arrays['residual_contact_valid'])[:tick]
    seen_contact = bool(np.any(previous_contact & previous_known))
    contact_lost = bool(seen_contact and not contact) if contact_known else None
    q = fields.get(paths[0]); target = fields.get('obs/actuator_target_rad')
    tracking = None if q is None or target is None else q-target
    object_pose = fields.get(paths[2]); initial_z = observed('object_initial_z_m')
    z_delta = float(object_pose[2]-initial_z.reshape(-1)[0]) if object_pose is not None and initial_z is not None else None
    # These are observable buckets, not inferred expert-relative trajectory deviation.
    phase = ('subgoal_observed' if subgoals is not None and bool((subgoals > 0).any()) else
        'native_lift_observed' if lifted is not None and bool(lifted.any()) else
        'contact_observed' if contact is True else 'noncontact_observed' if contact is False else 'unknown')
    deviation = dict(contact_loss_before_switch=contact_lost, object_height_change_from_initial_m=z_delta,
        arm_tracking_rmse_rad=float(np.sqrt(np.mean(tracking[:7]**2))) if tracking is not None else None,
        hand_tracking_rmse_rad=float(np.sqrt(np.mean(tracking[7:]**2))) if tracking is not None else None,
        expert_trajectory_deviation=None,
        type='observed_contact_loss' if contact_lost else 'unclassified',
        interpretation='Contact loss and target tracking are observations, not a diagnosed failure cause')
    return dict(coverage, switch_tick=tick, switch_observable_sha256=digest.hexdigest() if fields else None,
        switch_hash_fields=list(fields), switch_hash_missing_fields=missing,
        switch_hash_scope='Recorded pre-action observable subset only; hidden PhysX/RNG state excluded',
        phase=phase, native_stage_step=stage.tolist() if stage is not None else None,
        native_lift=bool(lifted.any()) if lifted is not None else None,
        native_subgoals=float(subgoals.max()) if subgoals is not None else None,
        contact_valid=contact_known, observed_contact=contact, deviation=deviation)


CAPTURE_AUDIT_MEMO_PROFILE = dict(version='process_local_closed_capture_audit_v1',
    max_payloads=256, max_pairs=4, retains_tensors=False, accepts_external_digests=False)


def validate_capture_audit_profile(profile):
    if profile is not None and (not isinstance(profile, dict) or profile != CAPTURE_AUDIT_MEMO_PROFILE
            or any(type(profile[key]) is not type(value) for key,value in CAPTURE_AUDIT_MEMO_PROFILE.items())):
        raise ValueError('Unknown registered capture-audit memo profile')
    return None if profile is None else dict(CAPTURE_AUDIT_MEMO_PROFILE)


class DaggerCaptureAuditMemo:
    """Bounded metadata only; every digest is first read in this process.

    Full numeric/control/camera semantics are never memoized. Only immutable
    payload SHA and the matched checkpoint contract inspection can be reused.
    """
    def __init__(self, *, max_payloads=256, max_pairs=4):
        from collections import OrderedDict
        from .frozen_payload_audit import FrozenPayloadAudit
        if type(max_pairs) is not int or max_pairs < 1:
            raise ValueError('Positive matched-pair memo bound required')
        self.payload_audit = FrozenPayloadAudit(max_entries=max_payloads)
        self.max_pairs = max_pairs
        self._pairs = OrderedDict()
        self._counts = dict(pair_checks=0, pair_loads=0, pair_hits=0, pair_evictions=0,
            pair_validation_seconds=0., source_checks=0)

    def _check_process(self):
        import os
        if os.getpid() != self.payload_audit.pid:
            raise RuntimeError('Capture-audit memo cannot cross a process boundary')

    @staticmethod
    def _fingerprint(path):
        import os
        import stat
        value = os.stat(path)
        if not stat.S_ISREG(value.st_mode):
            raise ValueError('Capture audit requires regular immutable files')
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)

    def sha256(self, path, *, expected=None):
        self._check_process()
        before = self._fingerprint(path)
        digest = self.payload_audit.sha256(path)
        if self._fingerprint(path) != before:
            raise ValueError('Capture payload changed during SHA validation')
        if expected is not None and digest != expected:
            raise ValueError('Capture payload differs from its registered expected SHA')
        return digest

    def validate_pair(self, pair):
        self._check_process(); started = time.perf_counter()
        self._counts['pair_checks'] += 1
        try:
            key = []
            for name in ('frontend', 'backend'):
                path = Path(pair[name])
                before = self._fingerprint(path)
                self.sha256(path, expected=pair[name+'_sha256'])
                if self._fingerprint(path) != before:
                    raise ValueError('Frozen pair changed during inspection')
                key.append((str(path.resolve()), before, pair[name+'_sha256']))
            key = tuple(key)
            if key in self._pairs:
                self._pairs.move_to_end(key)
                self._counts['pair_hits'] += 1
                return
            backend = torch.load(pair['backend'], map_location='cpu', weights_only=False, mmap=True)
            frontend = torch.load(pair['frontend'], map_location='cpu', weights_only=False, mmap=True)
            actual = dagger_policy_identity(backend, pair['backend'], payload_audit=self)
            if (actual['frontend_sha256'] != pair['frontend_sha256']
                    or Path(actual['frontend']).resolve() != Path(pair['frontend']).resolve()
                    or frontend['contract'] != backend['contract']):
                raise ValueError('DAgger actual loaded checkpoints are not a matched complete policy')
            for name, (_, fingerprint, expected) in zip(('frontend', 'backend'), key):
                self.sha256(pair[name], expected=expected)
                if self._fingerprint(pair[name]) != fingerprint:
                    raise ValueError('Frozen pair changed while loading checkpoint contracts')
            # Never retain checkpoints, model tensors, optimizer state or history.
            self._pairs[key] = True
            self._counts['pair_loads'] += 1
            if len(self._pairs) > self.max_pairs:
                self._pairs.popitem(last=False); self._counts['pair_evictions'] += 1
        finally:
            self._counts['pair_validation_seconds'] += time.perf_counter()-started

    def closed_source(self, output, result):
        """The verifier receipt is binding evidence, never a trusted SHA cache."""
        self._check_process(); output = Path(output)
        receipt = json.loads((output/'capture_complete.json').read_text())
        capture = result['dataset_capture']; bound = receipt.get('dataset_capture') or {}
        if (receipt.get('status') != 'source_verified' or receipt.get('steps') != result['steps']
                or Path(receipt.get('numeric_path', '')).resolve() != (output/'numeric.h5').resolve()
                or Path(bound.get('path', '')).resolve() != Path(capture['path']).resolve()
                or not bound.get('sha256') or bound['sha256'] != capture.get('sha256')):
            raise ValueError('Closed capture source verification receipt differs')
        self._counts['source_checks'] += 1
        return self.sha256(capture['path'], expected=bound['sha256'])

    def statistics(self):
        self._check_process()
        return dict(version=CAPTURE_AUDIT_MEMO_PROFILE['version'], **self._counts,
            max_pairs=self.max_pairs, retained_pairs=len(self._pairs), retains_tensors=False,
            payload=self.payload_audit.statistics(),
            semantic_scope='Numeric/action/history/camera audits still run for every capture; only immutable file and pair metadata reused')


def audit_dagger_capture(output, expected_pair=None, *, verify_weights=True, audit_memo=None):
    """CPU readback of real completed collection, including zero-valued corrections.

    Historical captures use their immutable initial_pair ledger. New captures also
    embed the paired identity. No simulator, model forward, or labels are generated.
    """
    if audit_memo is not None and (not isinstance(audit_memo, DaggerCaptureAuditMemo) or not verify_weights):
        raise ValueError('Capture-audit memo requires explicit full checkpoint verification')
    output = Path(output); result = json.loads((output/'result.json').read_text())
    if not result.get('software_complete') or not result.get('legal') or not result.get('dagger'):
        raise ValueError('Complete legal DAgger capture required for deployment audit')
    identity = result.get('complete_student_policy')
    pair = expected_pair or identity
    if pair is None:
        raise ValueError('Historical DAgger capture requires its explicit paired checkpoint ledger')
    adapter = (identity or pair).get('execution_adapter')
    if 'execution_adapter' in pair and pair['execution_adapter'] != adapter:
        raise ValueError('DAgger recorded execution adapter differs from expected pair')
    for name in ('frontend', 'backend'):
        if identity is not None and (Path(identity[name]).resolve() != Path(pair[name]).resolve()
                or identity[name+'_sha256'] != pair[name+'_sha256']):
            raise ValueError('DAgger recorded paired identity differs from expected collector')
    if result['checkpoint_sha256'] != pair['backend_sha256']:
        raise ValueError('DAgger actual backend checkpoint identity differs')
    if audit_memo is not None:
        audit_memo.validate_pair(pair)
        audit_memo.closed_source(output, result)
    elif verify_weights:
        for name in ('frontend', 'backend'):
            if sha256(pair[name]) != pair[name+'_sha256']:
                raise ValueError('DAgger actual '+name+' checkpoint bytes changed')
        backend = torch.load(pair['backend'], map_location='cpu', weights_only=False, mmap=True)
        frontend = torch.load(pair['frontend'], map_location='cpu', weights_only=False, mmap=True)
        actual = dagger_policy_identity(backend, pair['backend'])
        if (actual['frontend_sha256'] != pair['frontend_sha256']
                or Path(actual['frontend']).resolve() != Path(pair['frontend']).resolve()
                or frontend['contract'] != backend['contract']):
            raise ValueError('DAgger actual loaded checkpoints are not a matched complete policy')
    with np.load(output/'trace.npz') as archive:
        arrays = {key: archive[key] for key in archive.files}
    n = result['steps']; student = arrays['dagger_control_source'] == 0
    valid = arrays['dagger_expert_label_valid'].astype(bool)
    if (len(student) != n or valid[student].any() or result['residual_forward_ticks'] != n
            or result['residual_executed_ticks'] != int(student.sum())):
        raise ValueError('DAgger actual controller execution/label counts differ')
    errors = {}
    def match(name, a, b, tolerance=0.):
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('DAgger '+name+' has malformed or nonfinite values')
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        errors[name] = error
        if error > tolerance:
            raise ValueError('DAgger '+name+' numerical chain mismatch')
    with h5py.File(output/'numeric.h5', 'r') as stream:
        group = stream['episode']; transition = group['transition']; data = transition['dagger']
        nominal = arrays['nominal']; residual = arrays['residual_applied_rad']
        proposed = data['student_proposed_target_rad'][:]; issued = data['actual_issued'][:]
        applied = data['actual_applied'][:]; target = transition['target_after_rad'][:]
        match('nominal_plus_residual_to_proposal', nominal+residual, proposed, 1e-6)
        match('recorded_nominal', nominal, data['nominal'][:])
        match('student_proposal_to_issued', proposed[student], issued[student])
        match('issued_to_applied', issued, applied); match('applied_to_actuator_target', applied, target)
        match('trace_issued', arrays['command'], issued)
        match('expert_command_to_applied', data['u_expert'][:][valid], target[valid])
        if adapter is not None:
            from .position_control import project_processed_position_targets
            metadata=json.loads(stream.attrs['metadata_json'])
            previous=torch.from_numpy(transition['target_before_rad'][:])
            limits=torch.tensor(metadata['joint_limits_rad'][0])
            projected=project_processed_position_targets(torch.from_numpy(target),previous,limits,adapter)
            match('new_shared_reachable_command_identity',target,projected.numpy())
        match('q_reached_to_observed_after', data['q_reached'][:], transition['diagnostics_after/joint_position_rad'][:])
        match('physics_substep_targets', transition['actuator_substep_target_rad'][:],
            np.repeat(target[:, None], transition['actuator_substep_target_rad'].shape[1], axis=1))
        with h5py.File(result['dataset_capture']['path'], 'r') as compact:
            recorded = compact['episodes'][result['identity']]['transition']
            for name in ('actual_issued', 'actual_applied', 'u_expert', 'q_reached'):
                match('compact_'+name, recorded['dagger/'+name][:], data[name][:])
            if not np.array_equal(recorded['dagger/expert_label_valid'][:], valid):
                raise ValueError('DAgger compact expert supervision mask differs')
        coverage = dagger_recovery_coverage(arrays, group)
    if audit_memo is not None:
        audit_memo.closed_source(output, result)
    return dict(passed=True, finding='No bare-DP collection problem found in actual numeric execution',
        identity=result['identity'], frontend_sha256=pair['frontend_sha256'], backend_sha256=pair['backend_sha256'],
        checkpoint_files_checked=verify_weights, embedded_pair_identity=identity is not None,
        student_ticks=int(student.sum()), residual_forward_ticks=n, valid_expert_ticks=int(valid.sum()),
        zero_correction_student_ticks=int((np.abs(residual[student]).max(-1) == 0).sum()),
        correction_must_be_nonzero=False, maximum_absolute_errors_rad=errors, recovery_coverage=coverage,
        trace_sha256=sha256(output/'trace.npz'), result_sha256=sha256(output/'result.json'))


def record_dagger_transition(controller, transition, previous_nominal):
    """Attach factual labels, then acknowledge the actual command to the student."""
    value = transition['dagger']
    target = transition['target_after_rad']
    requested = transition['position_requested_rad']
    applied = transition['position_applied_rad']
    expert = value['control_source'].bool()
    if not torch.equal(requested, applied) or not torch.equal(applied, target):
        raise ValueError('DAgger requested/applied/actuator target mismatch')
    if not torch.equal(transition['actuator_substep_target_rad'], target[:, None].expand_as(transition['actuator_substep_target_rad'])):
        raise ValueError('DAgger target was not applied at every physics substep')
    if expert.any() and not torch.equal(target[expert], value['expert_mapped_target_rad'][expert]):
        raise ValueError('DAgger expert mapping differs from actual actuator target')
    adapter=getattr(getattr(controller,'residual',None),'execution_adapter',None)
    if adapter is not None:
        from .position_control import project_processed_position_targets
        projected=project_processed_position_targets(target,transition['target_before_rad'],
            controller.base.limits.to(target.device),adapter)
        if not torch.equal(projected,target):
            raise ValueError('DAgger actual command violates shared expert reachable targets')
    reached = transition['diagnostics_after']['joint_position_rad']
    if not torch.isfinite(reached).all():
        raise ValueError('Nonfinite actual reached configuration')
    nominal = controller.nominal.to(target.device)
    residual = controller.residual_diagnostics['applied_rad'].to(target.device)
    proposal = value['student_proposed_target_rad']
    if (not torch.allclose(nominal+residual, proposal, rtol=0, atol=1e-6)
            or not torch.equal(controller.command.to(target.device), proposal)
            or not torch.equal(proposal[~expert], requested[~expert])):
        raise ValueError('DAgger complete student nominal/residual/issued chain mismatch')
    required = torch.where(expert[:, None], target-nominal, torch.zeros_like(target))
    required_normalized = required*controller.action_scale_cpu.to(target.device)
    residual_bound = controller.residual_scales_cpu.to(target.device)
    value.update(u_expert=torch.where(expert[:, None], target, torch.zeros_like(target)),
        q_reached=reached.clone(), nominal=nominal.clone(), student_residual_applied_rad=residual.clone(),
        required_expert_residual_rad=required, required_expert_residual_normalized=required_normalized,
        required_expert_residual_bound_fraction=required_normalized.abs()/residual_bound,
        required_expert_residual_out_of_bounds=required_normalized.abs()>residual_bound,
        previous_nominal=torch.zeros_like(nominal) if previous_nominal is None else previous_nominal.to(target.device).clone(),
        previous_nominal_valid=torch.full_like(expert, previous_nominal is not None),
        actual_issued=requested.clone(), actual_applied=applied.clone())
    if adapter is not None:
        value.update({key:controller.residual_diagnostics[key].to(target.device).clone() for key in
            ('execution_projection_rad','execution_preprojection_target_rad','execution_previous_target_rad')})
    # During the expert suffix the student's proposal remains a diagnostic.
    # Its next history must contain the command the position executor received.
    controller.base.command = requested.detach().cpu().clone()
    controller.observe_execution(transition)
    return nominal.detach().cpu().clone()


def residual_episode_metrics(arrays, control_hz=60, *, output_contract=None):
    """Read-only correction/contact diagnostics, never inputs or success gates."""
    contact=np.asarray(arrays['residual_contact_gate'],dtype=bool)
    valid=np.asarray(arrays['residual_contact_valid'],dtype=bool)
    applied=np.asarray(arrays['residual_applied_rad'])
    proposed=np.asarray(arrays['residual_proposed_rad'])
    clipped=np.asarray(arrays['residual_clip_mask'],dtype=bool)
    tanh=np.asarray(arrays['residual_tanh_fraction'])
    if any(value.shape!=(len(contact),29) for value in (applied,proposed,clipped,tanh)) or valid.shape!=contact.shape:
        raise ValueError('Residual diagnostic dimensions differ from native ticks')
    def amplitudes(mask):
        result=dict(ticks=int(mask.sum()))
        for name,part in (('arm',slice(0,7)),('hand',slice(7,29))):
            a=applied[mask,part];p=proposed[mask,part]
            result[name]=dict(applied_rms_rad=float(np.sqrt(np.mean(a*a))) if a.size else None,
                proposed_rms_rad=float(np.sqrt(np.mean(p*p))) if p.size else None,
                applied_max_abs_rad=float(np.max(np.abs(a))) if a.size else None)
        return result
    gaps=[];unknown_interruptions=[];previous=None;start=None
    for tick,(seen,known) in enumerate(zip(contact,valid)):
        if not known:
            if start is not None:unknown_interruptions.append(dict(loss_tick=start,invalid_tick=tick))
            previous=None;start=None
            continue
        if previous is True and not seen:start=tick
        if start is not None and seen:
            gaps.append(dict(loss_tick=start,regain_tick=tick,noncontact_ticks=tick-start,
                noncontact_seconds=(tick-start)/control_hz,
                correction_active_ticks=int((np.abs(applied[start:tick]).max(-1)>1e-8).sum())))
            start=None
        previous=bool(seen)
    result = dict(all_ticks=amplitudes(np.ones(len(contact),dtype=bool)),
        observed_contact=amplitudes(valid&contact),observed_noncontact=amplitudes(valid&~contact),
        unknown_contact_ticks=int((~valid).sum()),
        joint_limit_clip=dict(joint_components=int(clipped.sum()),ticks=int(clipped.any(-1).sum()),
            joint_component_fraction=float(clipped.mean()) if clipped.size else None),
        tanh_saturation=dict(threshold=.95,joint_components=int((tanh>=.95).sum()),
            ticks=int((tanh>=.95).any(-1).sum()),maximum_fraction=float(tanh.max()) if tanh.size else None),
        observed_contact_loss_regain=dict(completed=gaps,completed_count=len(gaps),
            interrupted_by_invalid_observation=unknown_interruptions,
            open_loss_tick=start,open_noncontact_ticks=len(contact)-start if start is not None else 0,
            tick_index='zero-based pre-action observation tick',force_threshold_N=.1,
            definition='Any finger observed normal resultant >0.1N, valid current observation; gaps require uninterrupted valid observations',
            interpretation='Recontact is a sensor event, not successful recovery; no recovery supervision or causal capability is established'))
    if output_contract is not None:
        from .residual_output import validate_output_contract
        validate_output_contract(output_contract)
        # This legacy trace records |delta|/old_scales, not tanh(raw), for
        # linear-tail and centered proposals. Large ratios cannot establish
        # saturation or a vanishing logit gradient under either new map.
        result['legacy_normalized_amplitude'] = result['tanh_saturation']
        result['legacy_normalized_amplitude']['definition'] = 'abs(proposed_delta_normalized)/historical_scales; not a derivative or model bound'
        result['tanh_saturation'] = dict(applicable=False,
            reason='Versioned proposal is linear or has a true linear tail; legacy amplitude ratio does not measure saturation')
        result['output_contract'] = output_contract
    return result


def collision_evidence(env):
    """Query real cooked PhysX shapes before any action; no integration probe."""
    import omni.usd
    import carb
    from omni.physx import get_physx_scene_query_interface
    from pxr import PhysicsSchemaTools, Sdf, UsdPhysics
    stage = omni.usd.get_context().get_stage()
    root = env.object.root_physx_view.prim_paths[0]
    if not env.cfg.sim.enable_scene_query_support:
        raise RuntimeError('Headless scene queries disabled: empty hits cannot prove legality')
    shapes = [str(prim.GetPath()) for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(root+'/') and prim.HasAPI(UsdPhysics.CollisionAPI)
        and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not False]
    usd_hits = []
    for shape in shapes:
        def report(hit):
            usd_hits.append(dict(shape=shape, collider=str(hit.collision), rigid_body=str(hit.rigid_body)))
            return True
        encoded = PhysicsSchemaTools.encodeSdfPath(Sdf.Path(shape))
        get_physx_scene_query_interface().overlap_shape(*encoded, report, False)
    if not shapes:
        raise RuntimeError('No cooked object shapes found: '+root)
    # A large box around this lane must hit the actual object. This detects a
    # disabled/stale query backend without moving anything or integrating time.
    positive = []
    def positive_hit(hit):
        positive.append(str(hit.rigid_body)); return True
    position = env.object.data.root_pos_w[0].tolist()
    get_physx_scene_query_interface().overlap_box(carb.Float3(1., 1., 1.), carb.Float3(*position),
        carb.Float4(0., 0., 0., 1.), positive_hit, False)
    if root not in positive:
        raise RuntimeError('PhysX query positive control did not see the actual object')
    from pxr import UsdGeom
    initial = {key: value[0].cpu().numpy() for key, value in env._diagnostics().items()}
    boxes = inspect(dict(config=dict(assets=env.cfg.assets.to_dict())), initial)
    live_boxes = object_boxes(env.cfg.assets.object_urdf, initial['object_pose_w'])
    if len(live_boxes) != len(shapes):
        raise RuntimeError('Original object collider census changed')
    live_hits = []; transforms = []
    body = UsdGeom.Xformable(stage.GetPrimAtPath(root)).ComputeLocalToWorldTransform(0)
    q = initial['object_pose_w'][3:]
    for shape, (center, rotation, half) in zip(shapes, live_boxes):
        prim = stage.GetPrimAtPath(shape)
        if not prim.IsA(UsdGeom.Cube):
            raise RuntimeError('Live pose query requires original box GPrim')
        world = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)
        local = world*body.GetInverse()
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get())
        usd_half = np.array([np.linalg.norm(np.array(local)[i, :3]) for i in range(3)])*size/2
        if not np.allclose(usd_half, half, rtol=1e-5, atol=1e-7):
            raise RuntimeError('USD/URDF object collider dimensions differ')
        expected_center = np.asarray(initial['object_pose_w'][:3])+rotation@np.asarray(local.ExtractTranslation())
        if not np.allclose(center, expected_center, rtol=0, atol=1e-6):
            raise RuntimeError('USD/URDF object collider local positions differ')
        transforms.append(dict(shape=shape, usd_world_center=list(world.ExtractTranslation()),
            live_world_center=center.tolist(), half_extents=half.tolist()))
        def live_report(hit):
            live_hits.append(dict(shape=shape, collider=str(hit.collision), rigid_body=str(hit.rigid_body)))
            return True
        get_physx_scene_query_interface().overlap_box(carb.Float3(*half.tolist()), carb.Float3(*center.tolist()),
            carb.Float4(float(q[1]), float(q[2]), float(q[3]), float(q[0])), live_report, False)
        if not any(row['shape'] == shape and row['rigid_body'] == root for row in live_hits):
            raise RuntimeError('Live object box query failed self-hit positive control')
    external = [row for row in live_hits if row['rigid_body'] != root and not row['rigid_body'].startswith(root+'/')]
    return dict(object_body=root, shapes=shapes, query_positive_control=positive,
        usd_query_hits=usd_hits, collider_transforms=transforms, query_hits=live_hits, external_hits=external, boxes=boxes,
        legal=boxes['valid_box_initialization'] and not external,
        definition='Original verified object boxes at live PhysX pose vs all scene shapes; OBB check also passes',
        limitation='Overlap query is conservative: touching may be rejected; not a signed penetration-depth oracle')


def prepare(env, isolated, spec, source, output):
    """One reset and optional observable initialization, never trajectory forcing."""
    env.clear_recorded_events()
    if 'narrow_reset' in spec:
        from .narrow_data import configure_direct_reset
        configure_direct_reset(env,spec['narrow_reset'])
    teacher, _ = isolated.reset(spec['seed'])
    restoration = dict(kind='fresh_native_reset')
    if spec['cohort'] not in ('fresh','formal_test','dagger_train'):
        entry = source['entries'][spec['identity']]
        last = {}; original = env._get_observations
        def observed():
            result = original(); last['teacher'] = result
            return result
        env._get_observations = observed
        try:
            with h5py.File(entry['path'], 'r') as stream, isolated.stream.scope():
                restoration = restore_observable_state(env, stream['episodes'][entry['episode']],
                    json.loads(stream.attrs['metadata_json']))
            teacher = last['teacher']
        finally:
            env._get_observations = original
        # restore_observable_state historically restores a source CPU RNG. That
        # source included RL inference draws; this new contract deliberately
        # starts a separately registered native-event stream after initialization.
        isolated.stream = RandomStream(spec['seed']+100000)
        restoration['event_stream_seed'] = spec['seed']+100000
        restoration['historical_physics_exact'] = False
    if spec.get('event_seed') is not None:
        isolated.begin_events(spec['event_seed'])
        restoration.update(event_stream_seed=spec['event_seed'],
            event_seed_scope='Post-reset native draws only; initial state and reset-sampled DR probabilities retain initial seed')
    if spec.get('collision_gate')=='live_collision_mesh_bounds_v1':
        from .live_collision import clearance
        legal=clearance(env)
        # Retain the demonstrated unreliable query as diagnostic evidence only.
        if env.cfg.sim.enable_scene_query_support:
            atomic_json(output/'scene_query_diagnostic.json',collision_evidence(env))
    else:
        legal = collision_evidence(env)
    atomic_json(output/'initial_collision.json', legal)
    atomic_json(output/'restoration.json', restoration)
    atomic_json(output/'initial_observable.json', {k: v.cpu().tolist() for k, v in env._diagnostics().items()})
    return teacher, legal


@torch.no_grad()
def trial(env, lease, output, spec, source, player, checkpoint, *, saved=None, replay=None, video=False,position=False,
          inference=None, takeover_steps=None, expert_speed=None, capture_dataset=False, video_title=None,replay_horizon=None,
          capture_compression='lzf', expert_checkpoint=None, dagger=None, numeric_buffer_steps=32, expert_shadow=True,
          scheduler_index_mode='legacy', first_plan_forensics=False):
    if type(first_plan_forensics) is not bool:
        raise ValueError('First-plan forensics must be explicitly enabled or disabled')
    if first_plan_forensics and (saved is None or replay is not None
            or saved.get('contract',{}).get('visual_stride')!=1
            or saved.get('contract',{}).get('modality')!='G00'):
        raise ValueError('First-plan forensics requires a current every-frame G00 student')
    first_plan_recorder=None
    if replay_horizon is not None and (replay_horizon!=24 or replay is None or not position or saved is not None):
        raise ValueError('Diagnostic oracle requires explicit H24 position replay without student weights')
    dagger = dagger_configuration(dagger, spec, source) if dagger is not None else None
    if dagger is not None and (saved is None or not saved.get('residual_version') or not position or not capture_dataset
            or replay is not None or takeover_steps is not None or expert_speed is not None or expert_checkpoint is None):
        raise ValueError('DAgger requires a trained combined position student, compact capture and explicit expert identity')
    complete_identity = dagger_policy_identity(saved, checkpoint) if dagger is not None else None
    shadow_required = saved is None or dagger is not None or takeover_steps is not None or expert_shadow
    trial_started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output/'started.json', dict(spec=spec, utc=time.time(), checkpoint=checkpoint))
    isolated = IsolatedEnvironment(env, spec['seed'])
    with lease.block():
        env.action_control_mode = 'native_command_clipped_pre_delay'
        teacher, legal = prepare(env, isolated, spec, source, output)
        env.action_control_mode = 'absolute_joint_position_target_rad' if position else 'native_command_clipped_pre_delay'
        if not legal['legal']:
            result = dict(**spec, software_complete=True, legal=False, success=False,
                not_started=True, reason='pre-action collision rejection; no replacement by outcome')
            atomic_json(output/'result.json', result)
            return result
        metadata = env.research_metadata(); metadata['config'] = env.cfg.to_dict()
        metadata['inference_scheduler_index_mode']=scheduler_index_mode
        if saved is not None and saved.get('configuration',{}).get('execution_adapter') is not None:
            from .position_control import validate_execution_environment
            metadata['execution_adapter']=saved['configuration']['execution_adapter']
            validate_execution_environment(metadata['execution_adapter'],metadata)
        metadata['env_origins_w'] = env.scene.env_origins.cpu().tolist()
        metadata['expert_model_sha256'] = sha256(checkpoint if expert_checkpoint is None else expert_checkpoint)
        if expert_checkpoint is not None:
            metadata['policy_checkpoint_sha256'] = sha256(checkpoint)
        metadata['initial_family'] = spec.get('family', spec['identity'])
        metadata['dataset_split'] = spec.get('split', spec['cohort'])
        if dagger is not None:
            metadata.update(dagger=dagger, trajectory_kind='expert_takeover_v1', source_identity=spec['source_identity'],
                complete_student_policy=complete_identity)
            if spec.get('training_expansion') is not None:
                metadata.update(training_expansion=spec['training_expansion'],
                    initial_configuration_sha256=spec['initial_configuration_sha256'])
        if expert_speed is not None:
            metadata['expert_speed']=expert_speed
            metadata['supervision_contract']='transition/target_after_rad: actual issued actuator target, not measured joint position'
        config, config_hash = environment_identity(env)
        atomic_json(output/'metadata.json', json.loads(json.dumps(metadata, default=str)))
        player.reset()
        torch.save(export_initial(env, player), output/'initial_export.pt')
        torch.save(isolated.stream.state, output/'environment_rng_initial.pt')
        if first_plan_forensics:
            from .first_plan_forensics import FirstPlanRecorder
            frontend=saved.get('configuration',{}).get('base_checkpoint',checkpoint)
            first_plan_recorder=FirstPlanRecorder(output/'first_plan_forensics',binding=dict(
                spec=spec,checkpoint=str(checkpoint),checkpoint_sha256=sha256(checkpoint),
                frontend=str(frontend),frontend_sha256=sha256(frontend),
                frontend_contract=saved['contract'],
                metadata_sha256=sha256(output/'metadata.json'),scheduler_index_mode=scheduler_index_mode,
                source_sha256={name:sha256(Path(__file__).resolve().parent/name) for name in
                    ('native_controller.py','first_plan_forensics.py','models/native_dp.py','models/native_vision.py','native_data.py')}))
        if saved is not None and saved.get('group','').startswith('I-'):
            from .native_controller import NativeController
            dino=None
            if saved['group']=='I-HistoryDP':
                from .models.native_vision import NativeDinoSpatial
                dino=NativeDinoSpatial().to('cuda:0').eval()
            if saved.get('residual_version'):
                from .native_controller import ResidualController
                controller=ResidualController(saved,metadata,dino=dino,scheduler_index_mode=scheduler_index_mode,
                    first_plan_recorder=first_plan_recorder)
            else:controller=NativeController(saved,metadata,dino=dino,scheduler_index_mode=scheduler_index_mode,
                first_plan_recorder=first_plan_recorder)
        else:
            controller = None if saved is None else HistoryController(saved, metadata, inference=inference)
    setup_wall_s = time.perf_counter()-trial_started
    rows = []; plans = []; raw_plans = []; timings = []; first_success_tick = None
    profile=dict(expert_inference_s=0.,environment_step_render_s=0.,numeric_write_s=0.,compact_append_s=0.,finalize_verify_s=0.)
    numeric = h5py.File(output/'numeric.h5', 'w')
    numeric.attrs['metadata_json'] = json.dumps(metadata, default=str)
    group = numeric.create_group('episode')
    numeric_writer = BufferedNumericWriter(group, numeric_buffer_steps) if numeric_buffer_steps else None
    stream=None
    if capture_dataset:
        from .recording import CompactEpisodeStream,flatten,numpy_tree,verify_stream_fields
        stream=CompactEpisodeStream(output/'demonstration',metadata,spec['identity'],compression=capture_compression)
        stream.group.attrs['action_source']='recorded_position_retime' if expert_speed and expert_speed['kind']=='recorded_retime' else 'frozen_expert_slow_execution'
        if dagger is not None:
            stream.group.attrs['action_source']='student_then_expert_position_bridge_v1'
    movie = RolloutVideo(output/'rollout.mp4', 'expert' if saved is None else saved.get('kind', 'oldState'),
        spec['seed'], 0, CAMERAS) if video else None
    if movie is not None and video_title is not None:
        movie.title = video_title
    actor_parity = 0.
    raw_policy={};raw_handle=None
    if saved is None or dagger is not None:
        def capture_raw_mean(module,inputs,result):
            raw_policy['mean']=result['mus'].detach().clone()
        raw_handle=player.model.register_forward_hook(capture_raw_mean)
    bridge = None; previous_nominal = None; takeover_trigger = None
    try:
        if dagger is not None:
            from .position_control import ExpertPositionBridge
            if hasattr(env, '_dagger_position_bridge'):
                raise ValueError('A prior DAgger bridge is still attached')
            bridge = ExpertPositionBridge(env, dagger['takeover_tick'], dagger['expert_speed'])
            env._dagger_position_bridge = bridge
            if dagger.get('trigger') is not None:
                from .takeover_trigger import make_trigger,observe_transition
                takeover_trigger = make_trigger(dagger['trigger'], dagger['takeover_tick'])
        bound = (int(env.env_max_goals.max())+2)*(int(env.max_episode_length)+1)
        if replay is not None and expert_speed is None:
            bound = len(replay)
        for tick in range(bound):
            packet = env.research_packet
            if movie:
                movie.append(packet, tick)
            with lease.block():
                actor = torch.cat((teacher['policy'], player.intr_reward_coef_embd), -1)
                # Expert shadow follows the actual observed student trajectory;
                # it never executes and its RNG is outside the environment stream.
                measured=time.perf_counter()
                expert_command = (player.get_action(actor, is_deterministic=True).clamp(-1, 1)
                    if shadow_required else torch.zeros((1,29),device=actor.device))
                profile['expert_inference_s']+=time.perf_counter()-measured
                if replay is not None:
                    index=min(tick,len(replay)-1)
                    command = torch.from_numpy(replay[index:index+1])
                    if replay_horizon is not None:
                        if tick%12==0:
                            from .position_control import recorded_target_plan
                            oracle_plan=recorded_target_plan(replay,tick)
                            plans.append(oracle_plan[0].numpy());raw_plans.append(oracle_plan[0].numpy())
                        command=oracle_plan[:,tick%12]
                    # Deliberately perturb policy RNG in this second runner.
                    if expert_speed is None:torch.rand(37+tick % 17)
                elif controller is None:
                    command = expert_command
                elif takeover_steps is not None and tick >= takeover_steps:
                    env.action_control_mode = 'native_command_clipped_pre_delay'
                    command = expert_command
                elif isinstance(controller, HistoryController):
                    if inference is not None: torch.cuda.synchronize()
                    planning_started = time.perf_counter()
                    command = controller.act(actor, packet, env._diagnostics())
                    if inference is not None:
                        torch.cuda.synchronize()
                        timings.append(dict(tick=tick,wall_s=time.perf_counter()-planning_started,planning=tick%12==0))
                    expected = player.model.norm_obs(actor)[:, :140]
                    actual = controller.model.actor_input(actor.to(controller.device))[:, :140].cpu()
                    actor_parity = max(actor_parity, float((expected-actual).abs().max()))
                else:
                    command = native_policy_action(controller, packet, env._diagnostics)
                if controller is not None and tick % 12 == 0 and (takeover_steps is None or tick < takeover_steps):
                    plans.append(controller.plan[0].cpu().numpy())
                    if isinstance(controller, HistoryController): raw_plans.append(controller.raw_plan[0].numpy())
                if takeover_steps is not None and tick >= takeover_steps:
                    # Diagnostic hand-back to the real expert, with its recurrent
                    # shadow continuously updated above. Native queue stays native.
                    env.action_control_mode = 'native_command_clipped_pre_delay'
                    command = expert_command
                if not torch.isfinite(command).all():
                    raise FloatingPointError('Nonfinite policy output')
                if bridge is not None:
                    bridge.submit(raw_policy['mean'], expert_command, tick)
                before_rng = isolated.stream.state[2].clone()
                measured=time.perf_counter()
                teacher, _, terminated, truncated, _ = isolated.step(command)
                profile['environment_step_render_s']+=time.perf_counter()-measured
                transition = env.last_transition
                transition.update(expert_actor_input_diagnostic=actor.clone(),
                    torch_rng_before_step=before_rng[None], expert_shadow_command=expert_command.clone())
                if raw_handle is not None:
                    transition['expert_raw_policy_mean_before_clip']=raw_policy['mean']
                if bridge is not None:
                    previous_nominal = record_dagger_transition(controller, transition, previous_nominal)
                    command = transition['position_requested_rad']
                    if takeover_trigger is not None and not bool(transition['dagger']['control_source'][0]):
                        scheduled = observe_transition(takeover_trigger,tick,transition)
                        if scheduled is not None: bridge.schedule_takeover(scheduled)
                elif controller is not None and (takeover_steps is None or tick < takeover_steps):
                    controller.observe_execution(transition)
                before_packet = {k: v for k, v in packet.items() if torch.is_tensor(v)
                    and k not in ('image', 'depth_m', 'depth_valid')}
                measured=time.perf_counter()
                if numeric_writer is None:
                    append_numeric(group, dict(obs=before_packet, transition=transition), tick)
                else:
                    numeric_writer.append(dict(obs=before_packet, transition=transition), tick)
                profile['numeric_write_s']+=time.perf_counter()-measured
                if stream is not None:
                    measured=time.perf_counter()
                    from .recording import post_observation
                    # The compact writer deduplicates real adjacent camera frames;
                    # automatic-reset images must be masked on the terminal tick.
                    transition['observation_after'] = post_observation(env.research_packet, terminated | truncated)
                    row=flatten(numpy_tree(dict(obs={k:v for k,v in packet.items() if torch.is_tensor(v)},
                        transition=transition,terminated=terminated,truncated=truncated)))
                    stream.append({k:v[0] for k,v in row.items()})
                    profile['compact_append_s']+=time.perf_counter()-measured
                after = transition['diagnostics_after']
                # Read the pre-reset native outcome captured by _get_rewards;
                # never inspect an auto-reset scene to decide the prior result.
                complete = bool(transition['termination_reasons']['max_successes'][0])
                if complete and first_success_tick is None:
                    first_success_tick = tick+1
                if complete and not bool(terminated[0]):
                    raise RuntimeError('Native complete-success/terminated disagreement; no next action allowed')
                def take(value):
                    return value[0].cpu().numpy().copy()
                rows.append(dict(actor=take(actor), command=take(command), expert_shadow=take(expert_command),
                    delayed=take(transition['control_trace'].get('delayed_command',transition.get('position_applied_rad'))), delay_index=take(transition['control_trace']['delay_index']),
                    target=take(transition['target_after_rad']), joint=take(after['joint_position_rad']),
                    actuator_targets=take(transition['actuator_substep_target_rad']),
                    velocity=take(after['joint_velocity_rad_s']), object=take(after['object_pose_w']),
                    contact=take(transition['contact_substeps_after_w_N']), object_contact=take(transition['object_contact_force_w']),
                    force=take(transition['applied_wrench']['force_w_N']), torque=take(transition['applied_wrench']['torque_w_Nm']),
                    lifted=take(after['lifted_object']), subgoals=take(after['successes']),
                    retract=take(after['retract_succeeded']), alignment=take(transition['native_alignment_condition'])))
                if bridge is not None:
                    rows[-1].update({'dagger_'+key: take(transition['dagger'][key]) for key in
                        ('control_source','segment_id','expert_label_valid','switch_clean','expert_source_step','expert_delay_index',
                         'required_expert_residual_bound_fraction','required_expert_residual_out_of_bounds')})
                if inference is not None or (saved is not None and saved.get('residual_version')):
                    rows[-1]['palm']=take(after['palm_body_pose_w'])
                    rows[-1]['fingertip_distance']=np.linalg.norm(take(after['finger_position_w'])-
                        take(after['object_pose_w'])[:3],axis=-1).min()
                if saved is not None and saved.get('residual_version'):
                    rows[-1].update(nominal=take(controller.nominal),residual_normalized=take(controller.delta),
                        residual_contact_gate=np.asarray(controller.contact_gate))
                    diagnostics=controller.residual_diagnostics
                    rows[-1].update(residual_proposed_rad=take(diagnostics['proposed_rad']),
                        residual_applied_rad=take(diagnostics['applied_rad']),
                        residual_clip_mask=take(diagnostics['clip_mask']),
                        residual_tanh_fraction=take(diagnostics['tanh_fraction']),
                        residual_contact_valid=np.asarray(bool(diagnostics['contact_valid'])))
                    if 'execution_projection_rad' in diagnostics:
                        rows[-1].update({key:take(diagnostics[key]) for key in
                            ('execution_projection_rad','execution_preprojection_target_rad','execution_previous_target_rad')})
            if tick % 100 == 0:
                atomic_json(output/'progress.json', dict(tick=tick, utc=time.time()))
            if first_success_tick is not None or bool((terminated | truncated)[0]):
                break
        measured=time.perf_counter()
        if numeric_writer is not None:numeric_writer.flush()
        profile['numeric_write_s']+=time.perf_counter()-measured
        arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        np.savez_compressed(output/'trace.npz', **arrays)
        if plans:
            np.savez_compressed(output/'plans.npz', plans=np.stack(plans),
                **({'raw_plans':np.stack(raw_plans)} if raw_plans else {}))
        reasons = {k: bool(v[0]) for k, v in transition['termination_reasons'].items()}
        result = dict(**spec, software_complete=True, legal=True, success=reasons['max_successes'],
            reasons=reasons, steps=len(rows), native_lift=bool(arrays['lifted'].any()),
            native_subgoals=int(arrays['subgoals'].max()), native_alignment=bool(arrays['alignment'].any()),
            actor_normalization_max_error=actor_parity, environment_sha256=config_hash,
            checkpoint_sha256=sha256(checkpoint), action_exhausted=not any(reasons.values()),
            simulation_step_hz=60, physics_hz=120, plan_hz=5 if saved is not None or replay_horizon is not None else 60)
        if replay_horizon is not None:
            result['oracle']=dict(horizon=24,commit=12,open_loop=True,final_padding_executed=False)
        result.update(ever_complete_success=first_success_tick is not None,first_success_tick=first_success_tick,
            termination_tick=len(rows) if any(reasons.values()) else None,
            duration_sim_s=len(rows)*env.step_dt,tick_index='1-based completed control transitions',
            post_success_actions=0 if first_success_tick is None else len(rows)-first_success_tick)
        result['action_control_mode']=env.action_control_mode
        if first_plan_recorder is not None:
            result['first_plan_forensics']=first_plan_recorder.summary()
        result['actuator_target_max_error']=float(np.abs(arrays['actuator_targets']-arrays['target'][:,None,:]).max())
        result['tracking_rmse_rad']=float(np.sqrt(np.mean((arrays['joint']-arrays['target'])**2)))
        if expert_speed is not None:
            result['expert_speed']=expert_speed
            result['rl_observation_source']='actual current simulator observations every 60Hz tick'
            result['replay_sequence_ticks']=len(replay) if replay is not None else None
            result['replay_final_hold_ticks']=max(0,len(rows)-len(replay)) if replay is not None else 0
            result['trajectory_retiming_is_not_state_retiming']=True
        if position:
            result['submitted_target_max_error']=float(np.abs(arrays['command']-arrays['target']).max())
        if controller is not None and hasattr(controller,'inference_metrics'):
            timing=controller.inference_metrics()
            atomic_json(output/'native_inference_timing.json',timing)
            result['native_inference']=dict(summary=timing['summary'],pending_cuda_events=timing['pending_cuda_events'],
                timing_scope=timing['timing_scope'],wall_scope=timing['wall_scope'],excludes=timing['excludes'],
                raw_file='native_inference_timing.json',control_dt_changed=False)
        if saved is not None and saved.get('residual_version'):
            from .native_evaluation import stable_grasp
            initial=json.loads((output/'initial_observable.json').read_text())
            result['contact']=stable_grasp(arrays,float(initial['object_pose_w'][0][2]))
            result['residual_hz']=60
            result['residual_version']=saved['residual_version']
            result['residual_ticks']=len(rows)
            result['contact_gate_ticks']=int(arrays['residual_contact_gate'].sum())
            result['residual_diagnostics']=residual_episode_metrics(arrays,
                output_contract=saved.get('configuration',{}).get('output_contract'))
        if inference is not None:
            from .native_evaluation import stable_grasp
            initial=json.loads((output/'initial_observable.json').read_text())
            result['contact']=stable_grasp(arrays,float(initial['object_pose_w'][0][2]))
            result['approach_proxy']=bool((arrays['fingertip_distance']<=.08).any())
            result['inference'] = inference
            result['plan_consistency'] = plan_metrics(np.stack(plans)) if len(plans)>1 else {}
            result['planning_latency_s'] = distribution([r['wall_s'] for r in timings if r['planning']])
            result['native_history_latency_s'] = distribution([r['wall_s'] for r in timings if not r['planning']])
            atomic_json(output/'inference_timing.json',dict(rows=timings,scope='synchronized wall including observation history, inference and host transfer; excludes resource waits and simulation'))
        if takeover_steps is not None:
            result['takeover'] = dict(student_ticks=takeover_steps,expert_recovery=True,
                limitation='Direct position to original native expert mode; original delay queue resumes unchanged. Not autonomous success.')
        if dagger is not None:
            expert_ticks = int(arrays['dagger_control_source'].sum())
            valid = arrays['dagger_expert_label_valid'].astype(bool)
            outside = arrays['dagger_required_expert_residual_out_of_bounds'][valid]
            fractions = arrays['dagger_required_expert_residual_bound_fraction'][valid]
            result['dagger'] = dict(configuration=dagger, expert_shadow_queries=len(rows),
                student_ticks=len(rows)-expert_ticks, expert_ticks=expert_ticks, takeover_triggered=expert_ticks>0,
                expert_valid_ticks=int(arrays['dagger_expert_label_valid'].sum()),
                expert_switch_masked_ticks=expert_ticks-int(arrays['dagger_expert_label_valid'].sum()),
                student_autonomous_success=bool(result['success'] and expert_ticks==0),
                expert_takeover_success=bool(result['success'] and expert_ticks>0),
                position_executor_continuous=env.action_control_mode=='absolute_joint_position_target_rad',
                provenance_passed=True, autonomous_evaluation=False,
                residual_capacity=dict(valid_expert_ticks=int(valid.sum()),
                    out_of_bounds_ticks=int(outside.any(-1).sum()),
                    out_of_bounds_tick_fraction=float(outside.any(-1).mean()) if len(outside) else None,
                    out_of_bounds_joint_components=int(outside.sum()),
                    max_required_bound_fraction=float(fractions.max()) if fractions.size else None,
                    scope='Executed valid expert command minus contemporaneous frozen frontend output; original residual normalized bounds unchanged'),
                admission='Root aggregator requires native success and nonzero valid expert supervision; failures retained')
            result['residual_forward_ticks'] = len(rows)
            result['residual_executed_ticks'] = len(rows)-expert_ticks
            result['complete_student_policy'] = complete_identity
            result['dagger']['recovery_coverage'] = dagger_recovery_coverage(arrays, group)
            if takeover_trigger is not None:
                result['dagger']['trigger_receipt'] = takeover_trigger.receipt()
            result['residual_diagnostics']['scope'] = 'Student proposals at every tick, including unexecuted expert-suffix shadow proposals'
        if movie:
            movie.close(complete=True)
        if stream is not None:
            measured=time.perf_counter()
            stream.finish(result)
            count=verify_stream_fields(output/'demonstration/data.h5')
            profile['finalize_verify_s']+=time.perf_counter()-measured
            result['dataset_capture']=dict(path=str(output/'demonstration/data.h5'),verified_source_fields=count,
                accepted_success=bool(result['success']),checkpoint_sha256=result['checkpoint_sha256'])
            if dagger is not None:
                result['dataset_capture'].update(sha256=sha256(output/'demonstration/data.h5'),
                    trajectory_kind='expert_takeover_v1', accepted_success=bool(
                        result['dagger']['expert_takeover_success'] and result['dagger']['expert_valid_ticks']>0))
        result['expert_shadow_computed']=shadow_required
        result['expert_shadow_placeholder_zero']=not shadow_required
        result['wall_profile']=dict(profile,setup_s=setup_wall_s,through_result_s=time.perf_counter()-trial_started,scope='Synchronous component call times exclude resource waits; setup_s and through_result_s include waits. Env step includes physics/render/sensors; compact includes encode/I/O.')
        atomic_json(output/'result.json', result)
        return result
    finally:
        if bridge is not None and getattr(env, '_dagger_position_bridge', None) is bridge:
            del env._dagger_position_bridge
        try:
            if numeric_writer is not None:numeric_writer.flush()
        finally:
            numeric.close()
        if raw_handle is not None:raw_handle.remove()
        if stream is not None and stream.file.id.valid:stream.file.close()
        if movie:
            movie.close()


def run(root, output, mode, job=None):
    output.mkdir(parents=True, exist_ok=False)
    session = json.loads((root/'session.json').read_text())
    job_data = json.loads(job.read_text()) if job else None
    speed_config=job_data.get('expert_speed') if job_data else None
    dagger = job_data.get('dagger') if job_data else None
    source = json.loads((Path(session['source_root'])/'data.json').read_text())
    if dagger is not None:
        if mode != 'dp' or not job_data.get('capture_dataset') or speed_config is not None or job_data.get('takeover_steps') is not None:
            raise ValueError('DAgger uses the existing dp job with explicit compact capture and its own takeover contract')
        if not job_data.get('specs'):
            raise ValueError('Finite preregistered DAgger attempts required')
        for spec in job_data['specs']:
            dagger_configuration(dagger, spec, source)
        if sha256(session['expert_checkpoint']) != job_data['expert_checkpoint_sha256']:
            raise ValueError('Frozen expert identity changed')
    lease = lease_for(root, deadline=session['deadline'])
    atomic_json(output/'runtime_controls.json', client_startup_controls(session.get('kit_gate',{}).get('omniclient_startup')))
    atomic_json(output/'runtime_before.json', kit_process_snapshot())
    atomic_json(output/'lifecycle.json',dict(state='launching',utc=time.time()))
    with lease.block(cuda=False):
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device='cpu', fast_shutdown=False, kit_args=(
            '--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 '
            '--/persistent/physics/numThreads=2 --/app/extensions/fsWatcherEnabled=false')).app
    env = None
    try:
        from .native_evaluation import omniclient_watch_state
        atomic_json(output/'app_ready.json',dict(utc=time.time(),watch_state=omniclient_watch_state()))
        atomic_json(output/'runtime_ready.json', kit_process_snapshot())
        if mode=='startup':
            for _ in range(10):app.update()
            atomic_json(output/'result.json',dict(software_complete=True,mode=mode,watch_state=omniclient_watch_state(),trials=[]))
            atomic_json(output/'lifecycle.json',dict(state='results_committed',utc=time.time()))
            return
        import isaacsimenvs
        env = make_env(True, lease, scene_queries=True,
            **({'camera_content_validation':True} if dagger is not None else {}),
            **({'expert_speed':speed_config} if speed_config and speed_config['kind']=='closed_loop' else {}))
        with lease.block():
            player, config = create_expert(env, session['expert_checkpoint'])
        expert_fingerprint=None
        if mode in ('expert','expert_slow','expert_retime') or dagger is not None:
            from .recording import model_fingerprint
            expert_fingerprint=model_fingerprint(player.model)
        atomic_json(output/'actor_configuration.json', config)
        env_config, digest = environment_identity(env)
        atomic_json(output/'environment.json', dict(config=env_config, sha256=digest))
        atomic_json(output/'lifecycle.json',dict(state='environment_ready',utc=time.time()))
        if mode == 'smoke':
            with lease.block():
                isolated = IsolatedEnvironment(env, 982501)
                teacher, _ = isolated.reset(982501)
                actor = torch.cat((teacher['policy'], player.intr_reward_coef_embd), -1)
                action = player.get_action(actor, is_deterministic=True).clamp(-1,1)
                isolated.step(action)
                if not torch.isfinite(env.last_transition['target_after_rad']).all():
                    raise FloatingPointError('Nonfinite smoke actuator target')
            atomic_json(output/'result.json',dict(software_complete=True,mode=mode,control_steps=1,trials=[]))
            return
        trials = []; saved = None; checkpoint = session['expert_checkpoint']
        if mode == 'freeze':
            specs = []
            # Initial-state-only screen. Every rejection is retained before any
            # expert/student outcome is observed. Never search by task success.
            for seed in session['fresh_candidates']:
                spec = dict(identity=f'fresh_{seed}', seed=seed, cohort='fresh')
                directory = output/spec['identity']; directory.mkdir()
                with lease.block():
                    _, legal = prepare(env, IsolatedEnvironment(env, seed), spec, source, directory)
                if legal['legal']:
                    specs.append(spec)
                if len(specs) == session['fresh_count']:
                    break
            for spec in session['saved_specs']:
                directory = output/spec['identity']; directory.mkdir()
                with lease.block():
                    _, legal = prepare(env, IsolatedEnvironment(env, spec['seed']), spec, source, directory)
                if not legal['legal']:
                    raise RuntimeError('Registered saved initialization illegal: '+spec['identity'])
                specs.append(spec)
            for spec in session.get('validation_candidates',[]):
                directory=output/spec['identity'];directory.mkdir()
                with lease.block():
                    _,legal=prepare(env,IsolatedEnvironment(env,spec['seed']),spec,source,directory)
                if legal['legal']:specs.append(spec)
                if sum(s['cohort']=='validation' for s in specs)==session['validation_count']:break
            if len(specs) != session['fresh_count']+len(session['saved_specs'])+session.get('validation_count',0):
                raise RuntimeError('Predeclared candidate pool lacks enough legal starts')
            atomic_json(output/'benchmark.json', dict(specs=specs, configuration_sha256=digest,
                outcome_selected=False, frozen_before_expert=True))
        else:
            benchmark = json.loads((root/'freeze/benchmark.json').read_text())
            if benchmark['configuration_sha256'] != digest:
                raise ValueError('Frozen environment configuration changed')
            specs = benchmark['specs']
            if mode in ('bc', 'dp', 'old'):
                specs = [s for s in specs if s['cohort'] != 'fresh']
                checkpoint = (job_data['checkpoint'] if job_data else
                    session['old_state_checkpoint'] if mode == 'old' else str(root/(mode+'_train')/'best.pt'))
                if job_data and sha256(checkpoint)!=job_data['checkpoint_sha256']:
                    raise ValueError('Checkpoint hash drift')
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            if job_data:
                specs = job_data.get('specs', specs)
            if mode in ('replay','position_replay'):
                specs = specs[:2]
            for i, spec in enumerate(specs):
                replay = None
                if mode=='expert_retime':
                    from .expert_speed import retime_targets
                    assert speed_config['kind']=='recorded_retime'
                    entry=source['entries'].get(spec['identity'])
                    path=job_data['replay_numeric'] if job_data.get('replay_numeric') else entry['path']
                    speed_config=dict(speed_config,replay_source_path=str(path),replay_source_sha256=sha256(path),
                        source_target_time='q(k/stretch), q(0)=initial target; each interpolated command is physically reexecuted',
                        expert_shadow_is_not_executed=True)
                    with h5py.File(path) as f:
                        g=f['episode'] if job_data.get('replay_numeric') else f['episodes'][entry['episode']]
                        replay=retime_targets(g['transition/target_before_rad'][0],
                            g['transition/target_after_rad'][:],speed_config['stretch'])
                if mode in ('replay','position_replay'):
                    replay = np.load(root/'expert'/spec['identity']/'trace.npz')['target' if mode=='position_replay' else 'command']
                position=mode in ('position_replay','expert_retime') or (saved is not None and saved.get('contract',{}).get('action')=='absolute_joint_position_target_rad')
                result = trial(env, lease, output/spec['identity'], spec, source, player, checkpoint,
                    saved=saved, replay=replay,position=position, video=(job_data.get('video', i == 0) if job_data else i == 0) and mode not in ('replay','position_replay'),
                    inference=job_data.get('inference') if job_data else None,
                    takeover_steps=job_data.get('takeover_steps') if job_data else None,
                    expert_speed=speed_config,capture_dataset=bool(job_data and job_data.get('capture_dataset')),
                    capture_compression=job_data.get('capture_compression','lzf') if job_data else 'lzf',
                    expert_checkpoint=session['expert_checkpoint'] if dagger is not None else None,dagger=dagger)
                if mode in ('replay','position_replay') and result.get('legal'):
                    with np.load(root/'expert'/spec['identity']/'trace.npz') as original, np.load(output/spec['identity']/'trace.npz') as current:
                        result['bytewise_fields'] = {key: original[key].shape == current[key].shape and
                            original[key].dtype == current[key].dtype and original[key].tobytes() == current[key].tobytes()
                            for key in original.files if key not in (('expert_shadow','command','delayed','delay_index') if position else ('expert_shadow',))}
                trials.append(result)
                atomic_json(output/'progress.json', dict(completed=len(trials), trials=trials))
        if expert_fingerprint is not None:
            after_fingerprint=model_fingerprint(player.model)
            assert expert_fingerprint==after_fingerprint,'Expert parameters/buffers changed during inference'
            atomic_json(output/'expert_weight_audit.json',dict(before=expert_fingerprint,after=after_fingerprint,
                passed=True,optimizer_updates=0,raw_mean_recorded_before_player_clipping=True))
        atomic_json(output/'result.json', dict(software_complete=True, mode=mode, trials=trials))
        atomic_json(output/'lifecycle.json',dict(state='results_committed',utc=time.time()))
    except BaseException as error:
        atomic_json(output/'worker_error.json', dict(error=repr(error), traceback=traceback.format_exc()))
        raise
    finally:
        atomic_json(output/'lifecycle.json',dict(state='closing',utc=time.time()))
        atomic_json(output/'runtime_closing.json', kit_process_snapshot())
        if session.get('kit_gate',{}).get('exit_on_error') and (output/'worker_error.json').exists():
            import os
            import sys
            atomic_json(output/'failed_exit_requested.json', dict(pid=os.getpid(), utc=time.time(),
                reason='Durable worker error; process isolation replaces known hanging SDK cleanup'))
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(1)
        if session.get('kit_gate',{}).get('shutdown') == 'isolated_process_exit' and (output/'result.json').exists() and not (output/'worker_error.json').exists():
            exit_completed_kit_worker(output)
        if env is not None:
            env.close()
        app.close()
        atomic_json(output/'closed.json',dict(utc=time.time()))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['freeze', 'expert', 'replay', 'position_replay','startup', 'smoke', 'old', 'bc', 'dp'], required=True)
    p.add_argument('--job', type=Path)
    a = p.parse_args(); torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    run(a.root, a.output, a.mode, a.job)
