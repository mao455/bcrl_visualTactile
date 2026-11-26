"""One registered own-student paired collection; production workers stay unchanged.

``prepare`` writes a finite 96-opportunity queue without starting Kit. ``project``
adds a process-local tight_project entry; only ``worker`` installs the collection
screens and paired native-reset wrapper. All action, camera and suffix admission
checks still run through tight_fresh_batch_run.admit_capture.
"""
import argparse
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
BASE = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01'
CONTRACT = 'paired_own_student_recovery_collection_v1'
TRIGGER_VERSION = 'paired_causal_recovery_screen_v1'
NEVER = 1_000_000_000
OPPORTUNITIES = ('support', 'alignment')
SAMPLER_SEED = 7091


def read(path):
    return json.loads(Path(path).read_text())


def trigger_profile(opportunity):
    from visuotactile.adaptive_takeover import profile
    if opportunity not in OPPORTUNITIES:
        raise ValueError('Exactly support/alignment opportunities required')
    return dict(version=TRIGGER_VERSION, opportunity=opportunity,
        screens={s: profile(s) for s in ('contact', 'instability', 'alignment', 'post_subgoal')},
        gate_m=.015, native_confirmation_ticks=10,
        priority=['support_instability', 'contact_stall' if opportunity == 'support' else 'never_entered_alignment',
                  'alignment_after_gate_exit', 'post_subgoal'],
        no_fixed_time_fallback=True,
        interpretation='Privileged collection screen only; no change to student input, task success or native cumulative confirmation')


def validate_trigger(configuration, fallback):
    if (not isinstance(configuration, dict) or configuration != trigger_profile(configuration.get('opportunity'))
            or type(fallback) is not int or fallback != NEVER):
        raise ValueError('Exact paired recovery profile and never-fallback required')
    return copy.deepcopy(configuration)


class PairedRecoveryTrigger:
    """Reuse existing adaptive screens and retain causal gate history explicitly.

    Both opportunities catch an already lifted support loss with at least one
    current contact. Only support uses the existing contact-stall screen. Only
    alignment takes a never-entered alignment stall. Both retain natural gate-exit
    alignment and post-subgoal screens. Recovery is measured afterwards, never
    assumed by this screen. Trigger ticks count completed student transitions.
    """
    def __init__(self, configuration, fallback):
        from visuotactile.adaptive_takeover import AdaptiveTrigger
        self.configuration = validate_trigger(configuration, fallback)
        self.fallback_tick = self.scheduled_tick = fallback
        self.screens = {k: AdaptiveTrigger(v, fallback) for k, v in configuration['screens'].items()}
        self.count = self.gate_hits = 0
        self.ever_entered = self.ever_left = False
        self.first_gate_tick = self.first_exit_tick = None
        self.trigger_completed_tick = self.at_trigger = self.last = None
        self.reason = 'registered_conditions_not_reached'

    def observe(self, tick, pose, palm, force, **extra):
        if tick != self.count or tick >= self.scheduled_tick:
            raise ValueError('Every completed student transition must be observed once')
        goal_before, goal_after = int(extra['goal_index_before']), int(extra['goal_index_after'])
        key = float(extra['keypoint_max_distance_m'])
        if not np.isfinite(key):
            raise ValueError('Finite actual native keypoint distance required')
        self.count += 1
        entered_now = goal_before == 0 and key <= self.configuration['gate_m']
        if entered_now:
            self.gate_hits += 1
            if not self.ever_entered:
                self.first_gate_tick = self.count
            self.ever_entered = True
        if self.ever_entered and goal_before == 0 and not entered_now:
            if not self.ever_left:
                self.first_exit_tick = self.count
            self.ever_left = True
        ready = {}
        for name, screen in self.screens.items():
            screen.observe(tick, pose, palm, force, **extra)
            ready[name] = screen.confirm >= screen.configuration['confirm_ticks']
            # Child screens are causal predicates, not independent authorities.
            # An unselected screen must keep observing the same real student.
            screen.scheduled_tick = NEVER
            screen.trigger_completed_tick = screen.at_trigger = None
        last = self.screens['alignment'].last
        self.last = dict(completed_tick=self.count, native_goal_before=goal_before, native_goal_after=goal_after,
            current_gate=bool(entered_now), ever_entered_15mm=self.ever_entered,
            ever_entered_then_left_15mm=self.ever_left, never_entered_15mm=not self.ever_entered,
            observed_stage0_gate_hit_ticks=self.gate_hits,
            native_first_subgoal_confirmed=bool(goal_after >= 1),
            screen_ready=ready, geometry=copy.deepcopy(last))
        category = None
        instability = self.screens['instability'].last
        if ready['instability'] and instability is not None and instability['fingers'] >= 1:
            category = 'support_instability'
        elif self.configuration['opportunity'] == 'support' and ready['contact']:
            category = 'contact_stall'
        elif (self.configuration['opportunity'] == 'alignment' and ready['alignment'] and not self.ever_entered):
            category = 'never_entered_alignment'
        elif ready['alignment'] and self.ever_entered:
            category = 'alignment_after_gate_exit'
        elif ready['post_subgoal']:
            category = 'post_subgoal'
        if category is None:
            return None
        self.trigger_completed_tick = self.scheduled_tick = self.count
        self.reason = category
        self.at_trigger = dict(copy.deepcopy(self.last), selected_category=category)
        return self.count

    def receipt(self):
        alignment = self.screens['alignment']
        return dict(configuration=self.configuration, fallback_tick=self.fallback_tick,
            actual_scheduled_takeover_tick=self.scheduled_tick,
            trigger_completed_tick=self.trigger_completed_tick,
            completed_student_transitions_observed=self.count,
            first_held_completed_tick=alignment.first_held,
            settled_height_m=alignment.height, reason=self.reason,
            first_gate_completed_tick=self.first_gate_tick, first_exit_completed_tick=self.first_exit_tick,
            ever_entered_15mm=self.ever_entered, ever_entered_then_left_15mm=self.ever_left,
            never_entered_15mm=not self.ever_entered, observed_stage0_gate_hit_ticks=self.gate_hits,
            at_trigger=self.at_trigger, last=self.last,
            native_confirmation='Original cumulative10 task counter is read, never set; observed gate hits are a separate diagnostic',
            native_success_changed=False, student_inputs_changed=False, no_expert_fallback=True,
            future_information_used=False,
            phase_alignment='Next action after completed causal confirmation; native planning phase preserved',
            limitation='A collection screen does not prove harmful error or expert recoverability')


def observe_transition(trigger, tick, transition, index=0):
    def value(path):
        current = transition
        for key in path.split('/'):
            current = current[key]
        current = current[index]
        return current.detach().cpu().numpy() if hasattr(current, 'detach') else current
    return trigger.observe(tick, value('diagnostics_after/object_pose_w'), value('diagnostics_after/palm_body_pose_w'),
        value('object_contact_force_w'), keypoint_max_distance_m=value('native_keypoints_max_dist'),
        goal_index_before=value('diagnostics_before/successes'), goal_index_after=value('diagnostics_after/successes'),
        goal_pose_before=value('diagnostics_before/goal_pose_w'), goal_pose_after=value('diagnostics_after/goal_pose_w'))


def audit_trigger(directory, result):
    """Independent closed-file replay; production provenance audit is additional."""
    import h5py
    dagger = result['dagger']
    cfg = dagger['configuration']
    trigger = PairedRecoveryTrigger(cfg['trigger'], cfg['takeover_tick'])
    with h5py.File(Path(directory)/'numeric.h5') as f:
        transitions = f['episode/transition']
        codes = transitions['dagger/control_source'][:]
        indices = np.flatnonzero(codes == 0)
        if not np.array_equal(indices, np.arange(len(indices))):
            raise ValueError('Student prefix must be contiguous from tick0')
        for tick in indices:
            observe_transition(trigger, int(tick), transitions, int(tick))
    if trigger.receipt() != dagger['trigger_receipt']:
        raise ValueError('Paired causal trigger cannot be reproduced from actual student prefix')
    actual = dagger['recovery_coverage']['switch_tick']
    if actual is not None and (actual != trigger.scheduled_tick or actual != trigger.trigger_completed_tick):
        raise ValueError('Authority switch differs from causally confirmed paired screen')
    if actual is None and (dagger['expert_ticks'] or dagger['expert_valid_ticks']):
        raise ValueError('Unswitched opportunity has expert authority or targets')
    return dict(passed=True, category='ended_before_takeover' if actual is None else 'student_paired_recovery_then_takeover',
        receipt=trigger.receipt(), actual_switch_tick=actual, numeric_sha256=sha256(Path(directory)/'numeric.h5'),
        student_input=False)


def classify_capture(directory, case, result, plan):
    if not result.get('legal'):
        return 'paired_recovery_audit.json', 'invalid_initialization', None
    if not result.get('software_complete'):
        return 'paired_recovery_audit.json', 'software_incomplete', None
    if result['dagger']['configuration']['trigger'] != case['dagger']['trigger']:
        raise ValueError('Executed trigger differs from predeclared opportunity')
    audit = audit_trigger(directory, result)
    dagger = result['dagger']
    if audit['actual_switch_tick'] is None:
        category = 'student_success_without_takeover' if dagger['student_autonomous_success'] else 'no_trigger_student_failure'
    else:
        category = 'expert_recovered' if dagger['expert_takeover_success'] else 'expert_failed'
    return 'paired_recovery_audit.json', category, audit


@contextmanager
def trigger_scope():
    """Process-local dispatch only; restore every production binding on exit."""
    from visuotactile import takeover_trigger as triggers
    old_validate, old_make, old_observe = triggers.validate_trigger, triggers.make_trigger, triggers.observe_transition
    def validate(value, fallback):
        return validate_trigger(value, fallback) if isinstance(value, dict) and value.get('version') == TRIGGER_VERSION else old_validate(value, fallback)
    def make(value, fallback):
        return PairedRecoveryTrigger(value, fallback) if isinstance(value, dict) and value.get('version') == TRIGGER_VERSION else old_make(value, fallback)
    def observe(trigger, tick, transition, index=0):
        return observe_transition(trigger, tick, transition, index) if isinstance(trigger, PairedRecoveryTrigger) else old_observe(trigger, tick, transition, index)
    triggers.validate_trigger, triggers.make_trigger, triggers.observe_transition = validate, make, observe
    try:
        yield
    finally:
        triggers.validate_trigger, triggers.make_trigger, triggers.observe_transition = old_validate, old_make, old_observe


def _source_hashes():
    return {str(p.relative_to(PROJECT)): sha256(p) for folder in ('tools', 'visuotactile')
            for p in sorted((PROJECT/folder).rglob('*.py'))}


def validate_plan(plan):
    from visuotactile.state_benchmark import dagger_configuration
    from visuotactile.training_expansion import read_registry, validate_expansion_spec
    if (plan.get('stage_takeover_contract') != CONTRACT or plan.get('kind') != 'takeover'
            or plan.get('test_used') is not False or plan.get('never_fallback') != NEVER):
        raise ValueError('Exact finite train-only paired recovery contract required')
    registry = read_registry(plan['training_expansion_registry'])
    if len(registry['families']) != 24 or plan['utc'] < registry['created_utc']:
        raise ValueError('All24 families must have been registered before collection')
    if set(plan['pairs']) != {'A', 'B'} or plan['sampler_seed'] != SAMPLER_SEED:
        raise ValueError('Both own complete parents and sampler7091 required')
    if plan['pairs']['A']['supervision'] != 'q_reached' or plan['pairs']['B']['supervision'] != 'u_expert':
        raise ValueError('A/B target contracts differ from registration')
    for role, pair in plan['pairs'].items():
        if (pair.get('version') != 'complete_student_policy_v1'
                or pair.get('controller') != 'ResidualController'
                or pair.get('residual_version') != 'target_residual_transformer60_v1'
                or pair.get('modality') != 'G00'
                or pair.get('output') != 'absolute_joint_position_target_rad'):
            raise ValueError('Both roles require the registered complete RGB-D DP+Transformer student')
    if plan['pairs']['A']['execution_adapter'] != plan['pairs']['B']['execution_adapter']:
        raise ValueError('Both complete students must use the same reachable command projection')
    batch = plan['batch']
    if (batch['lanes'] not in (2, 4, 8) or batch['vision_microbatch'] != 2
            or batch['version'] != 'whole_wave_complete_student_v1'
            or batch.get('vision_tail_profile') is not None or batch.get('inactive_render_profile') is not None):
        raise ValueError('Native microbatch2/tail-duplicate2/4/8lane queue required')
    from visuotactile.queued_rollout import PROFILE
    from visuotactile.gpu_batch_controller import PROFILE as GPU_PROFILE
    if batch.get('queue_profile') != PROFILE or batch.get('gpu_pipeline') != GPU_PROFILE:
        raise ValueError('Only the verified queued GPU complete-controller path is supported')
    if len(plan['cases']) != 96 or len({c['name'] for c in plan['cases']}) != 96:
        raise ValueError('Exactly96 unique finite opportunities required')
    expected = {(family, opportunity, role) for family in registry['families'] for opportunity in OPPORTUNITIES for role in 'AB'}
    actual = set()
    with trigger_scope():
        for case in plan['cases']:
            spec, role, opportunity = case['spec'], case['role'], case['opportunity']
            key = (spec['family'], opportunity, role)
            if key not in expected or key in actual:
                raise ValueError('Duplicate, missing or unregistered family/opportunity/role')
            actual.add(key)
            if (spec['training_expansion'] != plan['training_expansion_registry'] or spec['takeover_tick'] != NEVER
                    or spec['split'] != 'train' or spec['cohort'] != 'dagger_train'
                    or case['dagger']['trigger'] != trigger_profile(opportunity)
                    or spec['identity'] != case['name'] or case.get('video') is not False):
                raise ValueError('Opportunity binding, trigger, phase or capture scope changed')
            source = read(case['source_data'])
            validate_expansion_spec(spec, source)
            dagger_configuration(case['dagger'], spec, source)
    if actual != expected:
        raise ValueError('Complete paired opportunity product is required')
    for i in range(0, 96, 2):
        a, b = plan['cases'][i:i+2]
        if (a['spec']['family'], a['opportunity']) != (b['spec']['family'], b['opportunity']) or {a['role'], b['role']} != {'A', 'B'}:
            raise ValueError('Adjacent paired chances must retain identical family and opportunity')
        for key in ('seed', 'event_seed', 'narrow_reset', 'initial_configuration_sha256'):
            if a['spec'][key] != b['spec'][key]:
                raise ValueError('Paired native reset or noise schedule differs')
    return True


def prepare(root=DEFAULT_ROOT, output=None, *, lanes=8):
    """Freeze plans and a job; this function performs no Kit or GPU work."""
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.training_expansion import read_registry, registry_binding
    from visuotactile.state_benchmark import CAPTURE_AUDIT_MEMO_PROFILE
    from visuotactile.visual_arithmetic import profile as arithmetic_profile
    root = Path(root).resolve()
    output = Path(output or root/'collection').resolve()
    if output.exists():
        raise FileExistsError('Keep every attempted collection identity; no automatic retries')
    pairs = read(root/'parent_pairs.json')
    session = read(root/'session.json')
    if time.time() >= session['collection_deadline']:
        raise ValueError('Registered finite collection deadline has passed')
    from tools.tight_paired_dagger import active_registry
    binding = registry_binding(active_registry(root))
    registry = read_registry(binding)
    template = read(PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01/wave00/plan.json')
    plan = copy.deepcopy(template)
    for key in ('holdout_contract', 'evaluation_scope', 'paired_initial_scope'):
        plan.pop(key, None)
    cases = []
    for family_index, (family, row) in enumerate(registry['families'].items()):
        for chance, opportunity in enumerate(OPPORTUNITIES):
            roles = 'AB' if (family_index + chance) % 2 == 0 else 'BA'
            for role in roles:
                name = f'{role}_s{row["seed"]}_{opportunity}'
                spec = dict(identity=name, family=family, source_identity=row['source_identity'],
                    split='train', cohort='dagger_train', seed=row['seed'],
                    event_seed=row['seed']+100000+chance, narrow_reset=copy.deepcopy(row['narrow_reset']),
                    initial_configuration_sha256=row['configuration_sha256'], training_expansion=binding,
                    collision_gate='live_collision_mesh_bounds_v1', takeover_tick=NEVER, layout=session['layout'])
                cases.append(dict(name=name, role=role, opportunity=opportunity, spec=spec, video=False,
                    source_data=str(BASE/role/'state_data/data.json'),
                    dagger=dict(version=ExpertPositionBridge.version, expert_speed=ExpertPositionBridge.speed_scheme,
                                trigger=trigger_profile(opportunity))))
    plan.update(kind='takeover', stage_takeover_contract=CONTRACT, never_fallback=NEVER,
        pairs=pairs, cases=cases, source_root=str(BASE), resource_root=str(root), round_index=1,
        training_expansion_registry=binding, source_data_sha256={r: sha256(BASE/r/'state_data/data.json') for r in 'AB'},
        source_session_sha256=sha256(root/'session.json'), source_sha256=_source_hashes(),
        parent_pairs_path=str(root/'parent_pairs.json'), parent_pairs_sha256=sha256(root/'parent_pairs.json'),
        source_commit=session['source_commit'], test_used=False, utc=time.time(),
        sampler_seed=SAMPLER_SEED, inference_scheduler_index_mode='host', record_sampler_rng=True,
        visual_arithmetic=arithmetic_profile(2), expert_sha256=session['expert_sha256'],
        selection='All96 predeclared opportunities retained; only audited successful legal expert suffixes admitted to the same role',
        paired_initial_scope='Same requested reset and per-family native-reset seed; observed starts checked at1e-5; no hidden PhysX or future event-stream equality claim')
    plan['batch'].update(lanes=lanes, seed=28920000, vision_microbatch=2, writer_bytes=128<<20,
        verification_workers=0, verification_max_pending=4,
        capture_audit_memo=dict(CAPTURE_AUDIT_MEMO_PROFILE),
        ordering='Adjacent A/B paired chance; cyclic preset A/B order; persistent queue with native indexed refills; realized lane assignments retained')
    for key in ('vision_tail_profile', 'inactive_render_profile'):
        plan['batch'].pop(key, None)
    validate_plan(plan)
    output.mkdir(parents=True)
    # Execution shares the finite experiment session and its owner telemetry.
    (output/'session.json').write_bytes((root/'session.json').read_bytes())
    atomic_json(output/'plan.json', plan)
    owner = root/'owners/collection'
    if owner.exists():
        raise FileExistsError('Keep attempted owner identity')
    owner.mkdir(parents=True)
    owner_session = copy.deepcopy(session)
    owner_session['resource_root'] = str(owner)
    atomic_json(owner/'session.json', owner_session)
    atomic_json(owner/'cuda_allocator_policy.json', dict(version='reclaim_unused_on_low_headroom_v1'))
    # The runner checks root/session bytes; resource leases use the shared root.
    shared = root/'shared_resources.json'
    if shared.is_symlink():
        # Do not replace an active expert owner's telemetry during preparation.
        telemetry_note = 'Root telemetry symlink must be changed to this owner only after expert owner closes'
    else:
        telemetry_note = 'Before execution root/shared_resources.json must point to this owner/shared_resources.json'
    job = dict(version='tight_project_job_v1', operation='dagger', resource_root=str(owner),
               stages=[dict(entry='paired_collection', args=['--output', str(output)])])
    job_path = root/'collection_job.json'
    atomic_json(job_path, job)
    receipt = dict(version=CONTRACT, root=str(root), output=str(output), owner=str(owner), job=str(job_path),
        plan_sha256=sha256(output/'plan.json'), parent_pairs_sha256=sha256(root/'parent_pairs.json'),
        opportunities=96, opportunities_per_role=48, lanes=lanes, trigger_profiles={o: trigger_profile(o) for o in OPPORTUNITIES},
        wrapper_sha256=sha256(Path(__file__)), utc=time.time(), telemetry_note=telemetry_note,
        command=f'python -m tools.tight_paired_collection project dagger --job {job_path}',
        no_physics_or_gpu_started=True)
    atomic_json(output/'registration.json', receipt)
    print(json.dumps(receipt), flush=True)
    return receipt


def _initial_values(packet, diagnostics, lane):
    """Only pre-action observable physical fields, normalized by lane base origin."""
    result = {key: diagnostics[key][lane].detach().cpu().numpy().copy()
              for key in ('joint_position_rad', 'joint_velocity_rad_s', 'object_velocity_w_m_s_rad_s', 'goal_velocity_w_m_s_rad_s')}
    result['actuator_target_rad'] = packet['actuator_target_rad'][lane].detach().cpu().numpy().copy()
    base = diagnostics['base_pose_w'][lane, :3]
    for key in ('object_pose_w', 'hole_pose_w', 'goal_pose_w', 'table_pose_w', 'palm_body_pose_w'):
        pose = diagnostics[key][lane].detach().clone()
        pose[:3] -= base
        result[key] = pose.cpu().numpy()
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError('Pre-action paired observations must be finite')
    return result


@contextmanager
def pairing_scope(plan, output):
    """Reset scopes and lane generator seeds; native step/event math is retained."""
    import torch
    from visuotactile import narrow_data, live_collision
    from visuotactile.environment_rng import LaneRandomStreams
    from visuotactile.gpu_batch_controller import MultiPolicyController
    from visuotactile.training_expansion import configuration_digest, read_registry
    registry = read_registry(plan['training_expansion_registry'])
    reset_seeds = {configuration_digest(row['narrow_reset']): row['seed']+500000 for row in registry['families'].values()}
    old_configure = narrow_data.configure_direct_reset_lanes
    old_assign, old_act = MultiPolicyController.assign, MultiPolicyController.act
    old_clearance = live_collision.clearance
    assignments, lane_cases, checked, starts, checks = [], {}, set(), {}, []
    def configure(env, configurations):
        old_configure(env, configurations)
        native_reset = env._reset_idx
        streams = LaneRandomStreams([], device=str(env.device))
        def reset(env_ids):
            ids = list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
            for lane in ids:
                digest = configuration_digest(env._direct_lane_reset_configurations[lane])
                if digest not in reset_seeds:
                    raise ValueError('A reset outside registered training families is forbidden')
                streams.replace(lane, reset_seeds[digest])
                with streams.scope(lane):
                    native_reset(torch.tensor([lane], device=env.device, dtype=torch.long))
        env._reset_idx = reset
    def assign(self, lane, role):
        case = plan['cases'][len(assignments)]
        if role != case['role']:
            raise ValueError('Queue assignment differs from frozen case order')
        old_assign(self, lane, role)
        lane_cases[lane] = case
        self.lanes[lane].base.generator.manual_seed(SAMPLER_SEED)
        assignments.append(dict(case=case['name'], lane=lane, role=role, ordinal=len(assignments), sampler_seed=SAMPLER_SEED))
        atomic_json(output/'paired_queue_assignments.json', assignments)
    def record_start(lane, packet, diagnostics, legal):
        case = lane_cases[lane]
        if case['name'] in checked:
            raise ValueError('Duplicate initial-state admission check')
        if int(packet['step_id'][lane]) != 0:
            raise ValueError('Paired observed-state check must precede first action')
        values = _initial_values(packet, diagnostics, lane)
        key = (case['spec']['family'], case['opportunity'])
        if key in starts:
            reference, reference_name, reference_legal = starts[key]
            differences = {}
            for field, value in values.items():
                if field.endswith('pose_w'):
                    differences[field+'_position'] = float(np.max(np.abs(value[:3]-reference[field][:3])))
                    differences[field+'_quaternion'] = float(min(np.max(np.abs(value[3:]-reference[field][3:])), np.max(np.abs(value[3:]+reference[field][3:]))))
                else:
                    differences[field] = float(np.max(np.abs(value-reference[field])))
            row = dict(family=key[0], opportunity=key[1], reference=reference_name, case=case['name'],
                       reference_legal=reference_legal, legal=bool(legal), differences=differences,
                       passed=bool(max(differences.values()) <= 1e-5 and reference_legal == bool(legal)))
            checks.append(row)
            atomic_json(output/'paired_initial_check.json', dict(rows=checks, passed=all(r['passed'] for r in checks),
                scope='All legal and rejected pre-action observable fields and legal masks;1e-5 each unit; hidden state/future events not asserted equal'))
            if not row['passed']:
                raise ValueError('Observed paired initial state differs; this lane received no action')
        else:
            starts[key] = (values, case['name'], bool(legal))
        checked.add(case['name'])
    def clearance(env, *args, **kwargs):
        admission = old_clearance(env, *args, **kwargs)
        lane = kwargs.get('lane', 0)
        record_start(lane, env.research_packet, env._diagnostics(), admission['legal'])
        return admission
    def act(self, packet, diagnostics, active):
        for lane, running in enumerate(active):
            if running and lane_cases[lane]['name'] not in checked:
                raise ValueError('An active lane lacks its pre-action paired admission check')
        return old_act(self, packet, diagnostics, active)

    narrow_data.configure_direct_reset_lanes = configure
    live_collision.clearance = clearance
    MultiPolicyController.assign, MultiPolicyController.act = assign, act
    try:
        yield
    finally:
        narrow_data.configure_direct_reset_lanes = old_configure
        live_collision.clearance = old_clearance
        MultiPolicyController.assign, MultiPolicyController.act = old_assign, old_act


def _resource_admission(root, output):
    import psutil
    import shutil
    import subprocess
    session = read(root/'session.json')
    resources = session['resources']
    ram = psutil.virtual_memory().available/2**30
    disk = shutil.disk_usage(root).free/2**30
    gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], text=True).splitlines()[0])
    # Leave the registered reserves plus startup headroom; rejection is technical,
    # never permission to interrupt the unrelated test or silently change rates.
    lanes = read(output/'plan.json')['batch']['lanes']
    ready = ram >= resources['min_ram_gib']+10+lanes and disk >= resources['min_disk_gib']+4 and gpu >= resources['min_gpu_free_mib']+4096+512*lanes
    atomic_json(output/'resource_admission.json', dict(ready=ready, ram_available_gib=ram, disk_free_gib=disk,
        vram_free_mib=gpu, lanes=lanes, utc=time.time(), external_processes_untouched=True))
    if not ready:
        raise RuntimeError('Insufficient coexistence startup headroom; no Kit started and no external process touched')


def worker(output):
    import fcntl
    from tools import tight_fresh_batch_run as runner
    output = Path(output).resolve()
    plan = read(output/'plan.json')
    root = Path(plan['resource_root'])
    registration = read(output/'registration.json')
    if (sha256(output/'plan.json') != registration['plan_sha256']
            or sha256(Path(__file__)) != registration['wrapper_sha256']
            or sha256(plan['parent_pairs_path']) != plan['parent_pairs_sha256']):
        raise ValueError('Frozen paired collection plan/wrapper/parent binding changed')
    validate_plan(plan)
    _resource_admission(root, output)
    with (root/'collection_worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            os.nice(10)
        except OSError:
            pass
        original_validate, original_classify, original_admit = runner.validate_takeover_plan, runner.classify_capture, runner.admit_capture
        def admit(out, case, result, pair, frozen, entries, **kwargs):
            original_admit(out, case, result, pair, frozen, entries, **kwargs)
            # Admission remains the production action/camera/suffix audit. Add
            # role ownership and gate-stratum provenance without changing labels.
            if case['name'] in entries:
                entry = entries[case['name']]
                if entry['source_group'] != case['role']:
                    raise ValueError('A correction cannot enter the other student pool')
                entry.update(opportunity=case['opportunity'], paired_collection_contract=CONTRACT)
                receipt = result['dagger']['trigger_receipt']
                entry.update(ever_entered_15mm_at_takeover=receipt['ever_entered_15mm'],
                    ever_entered_then_left_15mm_at_takeover=receipt['ever_entered_then_left_15mm'],
                    never_entered_15mm_at_takeover=receipt['never_entered_15mm'])
                atomic_json(out/'correction_entries.json', entries)
        runner.validate_takeover_plan, runner.classify_capture, runner.admit_capture = validate_plan, classify_capture, admit
        try:
            with trigger_scope(), pairing_scope(plan, output):
                runner.run(output)
        finally:
            runner.validate_takeover_plan, runner.classify_capture, runner.admit_capture = original_validate, original_classify, original_admit


def summarize(output):
    output = Path(output).resolve()
    plan, rows, entries = read(output/'plan.json'), read(output/'results.json'), read(output/'correction_entries.json')
    if {r['case'] for r in rows} != {c['name'] for c in plan['cases']} or len(rows) != 96:
        raise ValueError('Full96 opportunity denominator required; retain partial runs separately')
    cases = {c['name']: c for c in plan['cases']}
    if any(r['role'] != cases[r['case']]['role'] for r in rows):
        raise ValueError('Actual opportunity role differs from frozen own-student plan')
    accepted = {r['case'] for r in rows if r['result'].get('accepted_for_training')}
    if accepted != set(entries) or any(e.get('source_group') != cases[i]['role'] for i,e in entries.items()):
        raise ValueError('Accepted corrections and own-student manifest differ')
    assignments = read(output/'paired_queue_assignments.json')
    starts = read(output/'paired_initial_check.json')
    expected_pairs = {(c['spec']['family'],c['opportunity']) for c in plan['cases']}
    checked_pairs = {(r['family'],r['opportunity']) for r in starts['rows']}
    if (len(assignments) != 96 or [r['case'] for r in assignments] != [c['name'] for c in plan['cases']]
            or len(starts['rows']) != 48 or checked_pairs != expected_pairs
            or not starts['passed'] or not all(r['passed'] for r in starts['rows'])):
        raise ValueError('Complete96 assignments and48 pre-action paired checks required')
    result = dict(version=CONTRACT, opportunities=96, roles={})
    for role in 'AB':
        selected = [r for r in rows if r['role'] == role]
        if len(selected) != 48:
            raise ValueError('Exactly48 registered opportunities per own student required')
        totals = dict(opportunities=len(selected), invalid_initialization=0, software_incomplete=0, no_trigger=0,
                      takeovers=0, expert_recovered=0, student_success_without_takeover=0,
                      valid_trajectories=0, legal_expert_ticks=0, admitted_legal_expert_ticks=0, stages={})
        for row in selected:
            r, c = row['result'], cases[row['case']]
            d = r.get('dagger', {})
            receipt = d.get('trigger_receipt', {})
            switched = d.get('recovery_coverage', {}).get('switch_tick') is not None
            totals['invalid_initialization'] += not r.get('legal', False)
            totals['software_incomplete'] += not r.get('software_complete', False)
            totals['no_trigger'] += bool(r.get('legal') and not switched)
            totals['takeovers'] += switched
            totals['expert_recovered'] += bool(switched and d.get('expert_takeover_success'))
            totals['student_success_without_takeover'] += bool(not switched and d.get('student_autonomous_success'))
            totals['valid_trajectories'] += row['case'] in entries
            totals['legal_expert_ticks'] += int(d.get('expert_valid_ticks', 0))
            if row['case'] in entries:
                totals['admitted_legal_expert_ticks'] += int(d.get('expert_valid_ticks', 0))
            stage = receipt.get('reason', 'invalid_or_incomplete')
            group = totals['stages'].setdefault(stage, dict(opportunities=0, takeovers=0, recovered=0, legal_ticks=0, never_entered=0, entered_then_left=0))
            group['opportunities'] += 1
            group['takeovers'] += switched
            group['recovered'] += bool(switched and d.get('expert_takeover_success'))
            group['legal_ticks'] += int(d.get('expert_valid_ticks', 0))
            group['never_entered'] += bool(switched and receipt.get('never_entered_15mm'))
            group['entered_then_left'] += bool(switched and receipt.get('ever_entered_then_left_15mm'))
        result['roles'][role] = totals
    result['expert_recovery_is_autonomous_success'] = False
    atomic_json(output/'collection_summary.json', result)
    print(json.dumps(result), flush=True)
    return result



def diagnose(output):
    """Replay this exact frozen rule on existing development, never terminal data."""
    import h5py
    source = BASE/'development_20261122_01'
    rows = []
    for wave in sorted(source.glob('wave*')):
        plan = read(wave/'plan.json')
        for case in plan['cases']:
            if case['role'] not in ('A_entry', 'B_entry'):
                continue
            folder = wave/case['name']
            result = read(folder/'result.json')
            if not result.get('legal') or not result.get('software_complete'):
                raise ValueError('Historical development replay requires closed legal numeric evidence')
            screens = {}
            with h5py.File(folder/'numeric.h5') as f:
                transitions = f['episode/transition']
                for opportunity in OPPORTUNITIES:
                    trigger = PairedRecoveryTrigger(trigger_profile(opportunity), NEVER)
                    for tick in range(int(result['steps'])):
                        observe_transition(trigger, tick, transitions, tick)
                        if trigger.trigger_completed_tick is not None:
                            break
                    screens[opportunity] = trigger.receipt()
            rows.append(dict(role=case['role'], seed=case['spec']['seed'], historical_success=result['success'],
                numeric_path=str(folder/'numeric.h5'), numeric_sha256=sha256(folder/'numeric.h5'), screens=screens))
    if len(rows) != 32:
        raise ValueError('Expect all32 entry development executions; no terminal trajectories or selection')
    summary = {r: {op: {reason: sum(row['screens'][op]['reason'] == reason for row in rows if row['role'] == r)
                       for reason in sorted({row['screens'][op]['reason'] for row in rows if row['role'] == r})}
                   for op in OPPORTUNITIES} for r in ('A_entry', 'B_entry')}
    receipt = dict(version=CONTRACT, exact_profiles={o: trigger_profile(o) for o in OPPORTUNITIES},
        source=str(source), rows=rows, summary=summary, source_wrapper_sha256=sha256(Path(__file__)), utc=time.time(),
        threshold_tuning=False, terminal_trajectories_opened=0, simulation_steps=0, optimizer_updates=0,
        limitation='Screen coverage only. Historical autonomous continuation does not establish expert recoverability or causal harm.')
    atomic_json(Path(output), receipt)
    print(json.dumps(dict(output=str(output), cases=len(rows), summary=summary)), flush=True)
    return receipt


def self_check():
    """CPU-only semantic checks; no simulator import or optimizer update."""
    def transition(trigger, tick, *, height=.65, fingers=3, key=.03, stage=0):
        pose = np.array([0., 0., height, 1., 0., 0., 0.])
        palm = np.array([0., 0., .65, 1., 0., 0., 0.])
        force = np.zeros((2, 5, 3)); force[:, :fingers, 2] = 1.
        goal = np.array([0., 0., .68, 1., 0., 0., 0.])
        return trigger.observe(tick, pose, palm, force, keypoint_max_distance_m=key,
            goal_index_before=stage, goal_index_after=stage, goal_pose_before=goal, goal_pose_after=goal)
    alignment = PairedRecoveryTrigger(trigger_profile('alignment'), NEVER)
    for tick in range(48):
        assert transition(alignment, tick, height=.63, fingers=0) is None
    for tick in range(48, 100):
        scheduled = transition(alignment, tick)
        if scheduled is not None:
            break
    assert alignment.reason == 'never_entered_alignment' and alignment.scheduled_tick == tick+1
    assert alignment.receipt()['never_entered_15mm'] and not alignment.receipt()['native_success_changed']
    gated = PairedRecoveryTrigger(trigger_profile('alignment'), NEVER)
    for tick in range(48):
        transition(gated, tick, height=.63, fingers=0)
    transition(gated, 48, key=.01)
    for tick in range(49, 130):
        scheduled = transition(gated, tick)
        if scheduled is not None:
            break
    assert gated.reason == 'alignment_after_gate_exit' and gated.receipt()['ever_entered_then_left_15mm']
    assert gated.gate_hits == 1 and not gated.receipt()['never_entered_15mm']
    support = PairedRecoveryTrigger(trigger_profile('support'), NEVER)
    for tick in range(120):
        scheduled = transition(support, tick, height=.63, fingers=0)
        if scheduled is not None:
            break
    assert support.reason == 'contact_stall' and support.count >= 96
    unstable = PairedRecoveryTrigger(trigger_profile('alignment'), NEVER)
    for tick in range(48):
        transition(unstable, tick, height=.63, fingers=0, key=.15)
    for tick in range(48, 68):
        assert transition(unstable, tick, key=.10-.001*(tick-48)) is None
    assert transition(unstable, 68, height=.644, fingers=1) is None
    assert transition(unstable, 69, height=.644, fingers=1) == 70
    assert unstable.reason == 'support_instability'
    quiet = PairedRecoveryTrigger(trigger_profile('alignment'), NEVER)
    for tick in range(180):
        assert transition(quiet, tick, height=.63, fingers=0) is None
    assert quiet.scheduled_tick == NEVER
    # Profiles with changed thresholds/fallback must fail before execution.
    bad = trigger_profile('alignment'); bad['gate_m'] = .016
    for cfg, fallback in ((bad, NEVER), (trigger_profile('support'), 120)):
        try:
            validate_trigger(cfg, fallback)
        except ValueError:
            pass
        else:
            raise AssertionError('Altered profile accepted')
    from visuotactile import takeover_trigger as prod
    original = prod.make_trigger
    with trigger_scope():
        assert isinstance(prod.make_trigger(trigger_profile('alignment'), NEVER), PairedRecoveryTrigger)
    assert prod.make_trigger is original
    print(json.dumps(dict(passed=True, checks=['never_entered_alignment', 'gate_exit_stratum', 'contact_stall',
        'support_recovery_window', 'no_time_fallback', 'profile_tamper_rejection', 'dispatch_restoration'], simulation_steps=0, optimizer_updates=0)), flush=True)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'project':
        from tools import tight_project
        tight_project.ENTRIES['paired_collection'] = ('dagger', 'tools.tight_paired_collection', ['worker'])
        sys.argv.pop(1)
        return tight_project.main()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'validate', 'worker', 'summarize', 'self-check', 'diagnose'])
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--output', type=Path)
    p.add_argument('--lanes', type=int, default=8)
    a = p.parse_args()
    if a.mode == 'prepare':
        prepare(a.root, a.output, lanes=a.lanes)
    elif a.mode == 'diagnose':
        diagnose(a.output or PROJECT/'outputs/diagnostics/tight_paired_dagger_20261122_01/collection_development_screen_replay.json')
    elif a.mode == 'self-check':
        self_check()
    elif a.mode == 'validate':
        validate_plan(read((a.output or a.root/'collection')/'plan.json'))
        print(json.dumps(dict(passed=True, simulation_steps=0)), flush=True)
    elif a.mode == 'summarize':
        summarize(a.output or a.root/'collection')
    else:
        if a.output is None:
            p.error('--output required for worker')
        try:
            worker(a.output)
        except BaseException as error:
            atomic_json(a.output/'paired_worker_failure.json', dict(error=repr(error), utc=time.time(), traceback=traceback.format_exc()))
            traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush(); os._exit(1)


if __name__ == '__main__':
    main()
