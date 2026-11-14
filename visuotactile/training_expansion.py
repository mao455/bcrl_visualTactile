"""Explicit pre-outcome admission of additional narrow-domain training families.

Only JSON registration/split metadata are inspected here. Held-out trajectory
files are never opened. Different requested reset hashes establish distinct
registered configurations, not broad physics generalization or exact hidden
simulator-state independence. Native resets remain the only state writer.
"""
import copy
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .compact import sha256

VERSION = 'preregistered_training_families_v1'
COLLECTION_CONTRACT = 'registered_training_expansion_takeover_v1'
NEVER_FALLBACK = 1_000_000_000


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def configuration_digest(configuration):
    """Hash requested physical values at the simulator's float32 precision.

    Quaternion sign is canonicalized; seed is checked separately. This is a
    duplicate screen, not a metric separation or a hidden-state replay claim.
    """
    clean = {}
    for key, value in sorted(configuration.items()):
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError('Nonfinite registered reset configuration')
        if key == 'fixed_start_pose':
            if array.shape != (7,):
                raise ValueError('Registered fixed pose must be xyz plus quaternion')
            array = array.copy()
            quat = array[3:]
            norm = np.linalg.norm(quat)
            if norm < .5:
                raise ValueError('Invalid registered reset quaternion')
            quat /= norm
            if quat[np.flatnonzero(quat)[0]] < 0:
                quat *= -1
        array = np.where(array == 0, np.float32(0), array)
        clean[key] = array.tolist()
    return canonical_digest(clean)


def _read(path):
    return json.loads(Path(path).read_text())


def _file_binding(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=sha256(path))


def _partitions(data):
    from .narrow_data import aggregate_family_partitions
    return aggregate_family_partitions(data)


def make_registry(registry_id, domain, seeds, source_manifests, partition_manifest,
                  frozen_families, *, created_utc):
    """Build an immutable registration before any collection directory exists."""
    from .narrow_data import direct_configuration
    if not isinstance(registry_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}', registry_id):
        raise ValueError('Explicit safe training expansion identity required')
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('Unique nonnegative new initial seeds required')
    if set(source_manifests) != {'A', 'B'}:
        raise ValueError('Both complete-student source manifests required')
    sources = {role: _read(path) for role, path in source_manifests.items()}
    partitions = {role: _partitions(data) for role, data in sources.items()}
    if partitions['A'] != partitions['B'] or sources['A']['splits'] != sources['B']['splits']:
        raise ValueError('A/B source family partitions differ')
    source = sources['A']
    if len({source['entries'][identity]['family'] for identity in source['splits']['validation']}) != 6:
        raise ValueError('The six independent validation families must remain registered')
    original = _read(partition_manifest)
    original_partitions = _partitions(original)
    if len(original['splits'].get('test', [])) != 22:
        raise ValueError('Full original partition metadata must retain all test22 identities')
    protected = {family for family, split in original_partitions.items() if split in ('validation', 'test')}
    if any(partitions['A'][family] == 'train' for family in protected & partitions['A'].keys()):
        raise ValueError('Source already leaks an original held-out family')
    validation = {family for family, split in partitions['A'].items() if split == 'validation'}
    if not validation <= protected:
        raise ValueError('Original partition metadata does not protect current validation families')
    prior_rows = _read(frozen_families)
    if not isinstance(prior_rows, list) or not prior_rows:
        raise ValueError('Original frozen family configuration list required')
    prior = {row['family']: row for row in prior_rows}
    if len(prior) != len(prior_rows) or not set(original_partitions) <= set(prior):
        raise ValueError('Original family reset metadata is missing or duplicated')
    for family, split in original_partitions.items():
        if prior[family]['split'] != split:
            raise ValueError('Original frozen family and split metadata disagree')
    hashes = {configuration_digest(row['narrow_reset']) for row in prior_rows}
    used_seeds = {row['seed'] for row in prior_rows}
    existing_families = set(original_partitions) | set(partitions['A'])
    # Previous expansion registries remain exclusion evidence even if some of
    # their preregistered attempts were not successful training records.
    previous_bindings = source.get('training_expansion_registries', [])
    if previous_bindings != sources['B'].get('training_expansion_registries', []):
        raise ValueError('A/B previous training expansion registries differ')
    for binding in previous_bindings:
        if binding['registry_id'] == registry_id:
            raise ValueError('Preserve the existing training expansion registry identity')
        old = read_registry(binding)
        for family, row in old['families'].items():
            existing_families.add(family)
            hashes.add(row['configuration_sha256'])
            used_seeds.add(row['seed'])
    families = {}
    for seed in seeds:
        family = f'expanded_{registry_id}_s{seed}'
        reset = direct_configuration(domain, seed)
        digest = configuration_digest(reset)
        if seed in used_seeds or digest in hashes or family in existing_families:
            raise ValueError('New initial family duplicates an existing registered seed or physical configuration')
        families[family] = dict(family=family, split='train', seed=seed, narrow_reset=reset,
            source_identity=f'preregistered_{registry_id}_s{seed}', configuration_sha256=digest)
        used_seeds.add(seed)
        hashes.add(digest)
    return dict(version=VERSION, registry_id=registry_id, created_utc=float(created_utc),
        domain=copy.deepcopy(domain), domain_sha256=canonical_digest(domain), families=families,
        source_manifests={role: dict(_file_binding(path), canonical_sha256=canonical_digest(sources[role]))
            for role, path in source_manifests.items()},
        original_partition_manifest=_file_binding(partition_manifest), original_frozen_families=_file_binding(frozen_families),
        prior_training_registries=copy.deepcopy(previous_bindings),
        protected_families=sorted(protected), source_validation_families=sorted(validation),
        original_test_identities=list(original['splits']['test']),
        initial_configuration_hash_scope='float32 requested native reset fields, quaternion sign normalized; not hidden simulator state',
        source_domain='unchanged narrow initial-state domain; no mass or friction generalization',
        outcomes_used=False, test_trajectories_opened=0,
        registration='Family and train assignment fixed before collection; every attempted opportunity retained')


def registry_binding(path):
    path = Path(path).resolve(strict=True)
    registry = _read(path)
    if registry.get('version') != VERSION:
        raise ValueError('Unknown training expansion registry version')
    return dict(version=VERSION, registry_id=registry['registry_id'], path=str(path), sha256=sha256(path))


def read_registry(binding, source=None):
    if not isinstance(binding, dict) or binding.get('version') != VERSION:
        raise ValueError('Explicit versioned training expansion binding required')
    path = Path(binding['path'])
    if sha256(path) != binding.get('sha256'):
        raise ValueError('Training expansion registry bytes changed')
    registry = _read(path)
    if registry.get('version') != VERSION or registry.get('registry_id') != binding.get('registry_id'):
        raise ValueError('Training expansion registry identity differs')
    if registry.get('outcomes_used') is not False or registry.get('test_trajectories_opened') != 0:
        raise ValueError('Expansion must be registered without outcomes or held-out trajectory reads')
    if canonical_digest(registry['domain']) != registry['domain_sha256']:
        raise ValueError('Registered narrow domain changed')
    references = [registry['original_partition_manifest'], registry['original_frozen_families'],
        *registry['source_manifests'].values()]
    for reference in references:
        if sha256(reference['path']) != reference['sha256']:
            raise ValueError('Training expansion source metadata changed')
    if source is not None and canonical_digest(source) not in {
            item['canonical_sha256'] for item in registry['source_manifests'].values()}:
        raise ValueError('Training expansion belongs to a different source dataset')
    protected = set(registry['protected_families'])
    seen_hashes = set()
    from .narrow_data import direct_configuration
    for family, row in registry['families'].items():
        if family in protected or family != row['family'] or row['split'] != 'train':
            raise ValueError('Expansion registry contains a held-out or inconsistent family')
        expected = direct_configuration(registry['domain'], row['seed'])
        digest = configuration_digest(expected)
        if expected != row['narrow_reset'] or digest != row['configuration_sha256'] or digest in seen_hashes:
            raise ValueError('Registered new family draw or configuration identity differs')
        seen_hashes.add(digest)
    return registry


def _registered_row(binding, family, source):
    registry = read_registry(binding, source)
    if family not in registry['families']:
        raise ValueError('Family is absent from its preregistered training expansion')
    return registry['families'][family]


def validate_expansion_spec(spec, source):
    row = _registered_row(spec.get('training_expansion'), spec.get('family'), source)
    for key in ('source_identity', 'seed', 'narrow_reset'):
        if spec.get(key) != row[key]:
            raise ValueError('Takeover reset differs from pre-outcome training registration: '+key)
    if spec.get('initial_configuration_sha256') != row['configuration_sha256']:
        raise ValueError('Takeover physical configuration binding differs')
    return row


def validate_expansion_entry(entry, source, registered_binding):
    if entry.get('training_expansion') != registered_binding:
        raise ValueError('Accepted episode must carry the explicitly selected expansion registry')
    row = _registered_row(registered_binding, entry.get('family'), source)
    for key in ('seed', 'source_identity'):
        if entry.get(key) != row[key]:
            raise ValueError('Accepted episode differs from registered new family: '+key)
    if entry.get('initial_configuration_sha256') != row['configuration_sha256'] or entry.get('split') != 'train':
        raise ValueError('Accepted episode physical configuration/split differs')
    return row


def validate_capture_binding(entry, metadata):
    """Bind real captured sensor/action data to the pre-action declaration."""
    binding = entry.get('training_expansion')
    if binding is None:
        if metadata.get('training_expansion') is not None:
            raise ValueError('Expanded capture lost its registry binding in the training entry')
        return
    registry = read_registry(binding)
    row = registry['families'].get(entry.get('family'))
    if row is None or metadata.get('training_expansion') != binding:
        raise ValueError('Capture and accepted entry disagree on training expansion')
    expected = dict(initial_family=row['family'], source_identity=row['source_identity'],
        initial_configuration_sha256=row['configuration_sha256'], dataset_split='train')
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError('Captured family/split/reset provenance differs from registration')


def validate_schedule(schedule):
    if schedule in ('fixed40', 'fixed80', 'contact0', 'contact12','near60','held60','held120'):
        return schedule
    raise ValueError('Schedule must be fixed40, fixed80, contact0, contact12, near60, held60 or held120')


def schedule_trigger(schedule):
    from .takeover_trigger import CONTACT_HEIGHT_VERSION,HELD_NEAR_GOAL_VERSION,HELD_RECHECK_VERSION
    validate_schedule(schedule)
    if schedule=='near60':
        return dict(version=HELD_NEAR_GOAL_VERSION,delay_ticks=0,max_keypoint_distance_m=.06)
    if schedule.startswith('held'):
        return dict(version=HELD_RECHECK_VERSION,min_delay_ticks=int(schedule[4:]))
    if schedule.startswith('contact'):
        return dict(version=CONTACT_HEIGHT_VERSION,delay_ticks=int(schedule[7:]))
    return None


def validate_collection_plan(plan):
    if plan.get('stage_takeover_contract') != COLLECTION_CONTRACT or plan.get('test_used') is not False:
        raise ValueError('Explicit non-test expansion collection contract required')
    binding = plan.get('training_expansion_registry')
    registry = read_registry(binding)
    if plan.get('utc', 0) < registry['created_utc']:
        raise ValueError('Collection cannot precede its family registration')
    cases = plan['cases']
    if not cases or len({case['name'] for case in cases}) != len(cases):
        raise ValueError('Unique finite expansion opportunities required')
    family_roles = set()
    for case in cases:
        spec = case['spec']
        schedule = validate_schedule(case.get('expansion_schedule'))
        key = (spec['family'], case['role'])
        if case['role'] not in ('A', 'B') or key in family_roles:
            raise ValueError('At most one opportunity per family and complete student role')
        family_roles.add(key)
        if spec.get('split') != 'train' or spec.get('cohort') != 'dagger_train':
            raise ValueError('Expansion collection must remain explicitly training-only')
        source = _read(case['source_data'])
        if spec.get('training_expansion') is not None:
            if spec['training_expansion'] != binding:
                raise ValueError('Collection contains another expansion registry')
            validate_expansion_spec(spec, source)
        else:
            identity = spec.get('source_identity')
            if identity not in source['splits']['train'] or source['entries'][identity]['family'] != spec['family']:
                raise ValueError('Existing opportunity lacks a source training-family identity')
        if schedule.startswith('fixed'):
            if spec['takeover_tick'] != int(schedule[5:]) or case['dagger'].get('trigger') is not None:
                raise ValueError('Fixed-time opportunity changed its preregistered takeover')
        elif spec['takeover_tick'] != NEVER_FALLBACK or case['dagger'].get('trigger') != schedule_trigger(schedule):
            raise ValueError('Contact/near-goal opportunity changed its trigger or acquired a fallback')
    planned_new = {case['spec']['family'] for case in cases if case['spec'].get('training_expansion') is not None}
    if planned_new != set(registry['families']):
        raise ValueError('Every preregistered new family must retain its opportunities')
    for family, _ in family_roles:
        if (family, 'A') not in family_roles or (family, 'B') not in family_roles:
            raise ValueError('Both complete students must have one registered family opportunity')


def classify_case(case, result, trigger_audit=None):
    if not result.get('legal'):
        return 'invalid_initialization'
    if not result.get('software_complete'):
        return 'software_incomplete'
    if schedule_trigger(case['expansion_schedule']) is not None:
        from tools.tight_fresh_stage_takeover import classify
        if trigger_audit is None:
            raise ValueError('Contact/near-goal opportunity requires actual trigger audit')
        return classify(result, trigger_audit)
    dagger = result['dagger']
    switched = bool(dagger['expert_ticks'])
    switch_tick = dagger['recovery_coverage']['switch_tick']
    if switched:
        if switch_tick != case['spec']['takeover_tick']:
            raise ValueError('Actual expert switch differs from registered fixed-time opportunity')
        return 'expert_rescue_after_fixed_takeover' if dagger['expert_takeover_success'] else 'expert_failed_after_fixed_takeover'
    if switch_tick is not None or dagger['expert_valid_ticks']:
        raise ValueError('Unswitched opportunity has expert labels or a switch')
    return 'student_success_without_takeover' if dagger['student_autonomous_success'] else 'ended_before_fixed_takeover'
