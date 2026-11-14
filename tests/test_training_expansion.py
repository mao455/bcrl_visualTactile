"""New training families require immutable metadata-only registration."""
import copy
import json
import time

import h5py
import numpy as np
import pytest
import torch

from visuotactile.compact import sha256
from visuotactile.narrow_data import direct_configuration, validate_aggregate_families, build_dagger_aggregate
from visuotactile.position_control import ExpertPositionBridge
from visuotactile.state_benchmark import dagger_configuration
from visuotactile.training_expansion import (canonical_digest, classify_case, configuration_digest,
    make_registry, read_registry, registry_binding, validate_capture_binding, validate_collection_plan)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


@pytest.fixture
def setup_registry(tmp_path):
    domain = dict(center=dict(object_xyz=[-.03, -.01, .63], relative_xyz=[.07, -.03, -.08],
        quaternion=[1., 0., 0., 0.]), object_xy_halfwidth=[.002, .002], relative_xy_halfwidth=[.003, .003],
        rotation_radius_degrees=1.5, native_joint_lerp=.0075, table_z_m=[.379, .381],
        object_height_above_table_origin_m=[.248, .252], joint_velocity_halfwidth=.03)
    splits = dict(train=['old'], validation=[f'val{i}' for i in range(6)], test=[f'test{i}' for i in range(22)])
    entries = {'old': dict(family='train_original', path='/DO_NOT_OPEN_TRAIN', source_sha256='fixture')}
    entries.update({identity: dict(family=f'validation_{i}', path='/DO_NOT_OPEN_VALIDATION', source_sha256='fixture')
        for i, identity in enumerate(splits['validation'])})
    entries.update({identity: dict(family=f'test_{i % 7}', path='/DO_NOT_OPEN_TEST', source_sha256='fixture')
        for i, identity in enumerate(splits['test'])})
    partition = dict(entries=entries, splits=splits)
    family_split = {entries[identity]['family']: split for split, identities in splits.items() for identity in identities}
    family_rows = [dict(family=family, split=split, seed=1000+i,
        narrow_reset=direct_configuration(domain, 1000+i)) for i, (family, split) in enumerate(family_split.items())]
    data = dict(supervision='q_reached', entries={identity: entries[identity] for identity in splits['train']+splits['validation']},
        splits=dict(splits, test=[]), normalization_train_ids=['old'])
    source = tmp_path/'source'
    manifests = {role: write(source/role/'state_data/data.json', dict(data, supervision='q_reached' if role=='A' else 'u_expert'))
        for role in ('A', 'B')}
    partition_path = write(tmp_path/'original_partition.json', partition)
    family_path = write(tmp_path/'original_families.json', family_rows)
    registry = make_registry('unit_expansion', domain, [5000, 5001], manifests,
        partition_path, family_path, created_utc=time.time())
    binding = registry_binding(write(tmp_path/'registry.json', registry))
    return dict(root=tmp_path, source=source, domain=domain, source_data=data, manifests=manifests,
        partition=partition_path, families=family_path, registry=registry, binding=binding)


def entry_for(setup, index=0):
    row = list(setup['registry']['families'].values())[index]
    return dict(family=row['family'], seed=row['seed'], source_identity=row['source_identity'],
        initial_configuration_sha256=row['configuration_sha256'], training_expansion=setup['binding'],
        split='train', trajectory_kind='expert_takeover_v1', path='/DO_NOT_OPEN_CORRECTION')


def test_new_families_admit_only_explicit_registry_without_reading_trajectories(setup_registry, monkeypatch):
    setup = setup_registry
    monkeypatch.setattr(h5py, 'File', lambda *a, **k: pytest.fail('No trajectory may be read for family admission'))
    entry = entry_for(setup)
    with pytest.raises(ValueError, match='explicit aggregate registry'):
        validate_aggregate_families(setup['source_data'], {'new': entry})
    admitted = validate_aggregate_families(setup['source_data'], {'new': entry},
        training_expansion_registry=setup['binding'])
    assert admitted[entry['family']] == 'train'
    assert all(admitted[f'validation_{i}'] == 'validation' for i in range(6))
    assert 'test_0' in setup['registry']['protected_families']
    assert setup['registry']['test_trajectories_opened'] == 0


def test_all_new_episode_entries_need_binding_even_after_same_family_admitted(setup_registry):
    entry = entry_for(setup_registry)
    unbound = dict(entry)
    del unbound['training_expansion']
    with pytest.raises(ValueError, match='known source training family'):
        validate_aggregate_families(setup_registry['source_data'], {'a': entry, 'b': unbound},
            training_expansion_registry=setup_registry['binding'])


@pytest.mark.parametrize('fault', ['family', 'seed', 'configuration', 'source', 'registry_bytes', 'partition_bytes'])
def test_registry_and_source_mutation_fail_before_payloads(setup_registry, fault, monkeypatch):
    setup = setup_registry
    entry = entry_for(setup)
    data = copy.deepcopy(setup['source_data'])
    if fault == 'family':
        entry['family'] = 'test_0'
    elif fault == 'seed':
        entry['seed'] += 1
    elif fault == 'configuration':
        entry['initial_configuration_sha256'] = 'wrong'
    elif fault == 'source':
        data['normalization_train_ids'] = []
    elif fault == 'registry_bytes':
        with open(setup['binding']['path'], 'a') as stream:
            stream.write('\n')
    else:
        with setup['partition'].open('a') as stream:
            stream.write('\n')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Rejected expansion must not read numeric payloads'))
    write(setup['source']/'A/state_data/data.json', data)
    with pytest.raises(ValueError):
        build_dagger_aggregate(setup['source']/'A/state_data', {'new': entry}, setup['root']/'aggregate',
            'q_reached', include_geometry=False, include_frontend_tactile=False,
            training_expansion_registry=setup['binding'])
    assert not (setup['root']/'aggregate').exists()


@pytest.mark.parametrize('seeds', [[1000], [5000, 5000], [-1], [True]])
def test_duplicate_old_seed_or_invalid_seed_list_rejected(setup_registry, seeds):
    setup = setup_registry
    with pytest.raises(ValueError):
        make_registry('next', setup['domain'], seeds, setup['manifests'], setup['partition'], setup['families'], created_utc=time.time())


def test_different_seed_cannot_relabel_identical_physical_configuration(setup_registry, monkeypatch):
    setup = setup_registry
    old = json.loads(setup['families'].read_text())[0]['narrow_reset']
    monkeypatch.setattr('visuotactile.narrow_data.direct_configuration', lambda domain, seed: copy.deepcopy(old))
    with pytest.raises(ValueError, match='physical configuration'):
        make_registry('next', setup['domain'], [9000], setup['manifests'], setup['partition'], setup['families'], created_utc=time.time())
    equivalent = copy.deepcopy(old)
    equivalent['fixed_start_pose'][3:] = [-v for v in equivalent['fixed_start_pose'][3:]]
    assert configuration_digest(old) == configuration_digest(equivalent)


def test_expanded_takeover_keeps_actual_reset_and_source_binding(setup_registry):
    setup = setup_registry
    row = list(setup['registry']['families'].values())[0]
    spec = dict(entry_for(setup), cohort='dagger_train', narrow_reset=row['narrow_reset'], takeover_tick=40, event_seed=8000)
    bridge = dict(version=ExpertPositionBridge.version, expert_speed=ExpertPositionBridge.speed_scheme)
    assert dagger_configuration(bridge, spec, setup['source_data'])['takeover_tick'] == 40
    for altered in (dict(spec, split='test'), dict(spec, narrow_reset={}), dict(spec, source_identity='old')):
        with pytest.raises(ValueError):
            dagger_configuration(bridge, altered, setup['source_data'])
    metadata = dict(initial_family=row['family'], source_identity=row['source_identity'], dataset_split='train',
        initial_configuration_sha256=row['configuration_sha256'], training_expansion=setup['binding'])
    validate_capture_binding(entry_for(setup), metadata)
    with pytest.raises(ValueError, match='lost its registry'):
        validate_capture_binding({}, metadata)
    with pytest.raises(ValueError, match='provenance'):
        validate_capture_binding(entry_for(setup), dict(metadata, dataset_split='validation'))


def test_prepare_registers_both_roles_and_mixed_schedules_before_any_outcome(setup_registry, monkeypatch):
    from tools.tight_fresh_expansion import prepare
    setup = setup_registry
    session = dict(deadline=time.time()+1000, domain=setup['domain'], expert_sha256='expert_fixture')
    write(setup['source']/'session.json', session)
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair', lambda root, role: dict(role=role))
    monkeypatch.setattr(h5py, 'File', lambda *a, **k: pytest.fail('Preparation must use metadata only'))
    output = setup['root']/'prepared'
    plan = prepare(setup['source'], output, session_root=setup['source'], registry_id='prepared_fixture',
        seeds=[6000, 6001], partition_manifest=setup['partition'], frozen_families=setup['families'],
        schedules=['fixed40', 'contact0', 'fixed80'], existing_identities=['old'], lanes=8,
        event_seed=8000, round_index=7)
    assert len(plan['cases']) == 6
    assert plan['registered_new_training_families'] == 2
    assert {case['expansion_schedule'] for case in plan['cases']} == {'fixed40', 'fixed80', 'contact0'}
    assert all(case['spec']['split'] == 'train' for case in plan['cases'])
    assert not (output/'run_started.json').exists()
    assert not (output/'results.json').exists()
    validate_collection_plan(plan)
    malformed = copy.deepcopy(plan)
    malformed['cases'].pop()
    with pytest.raises(ValueError, match='Both complete students'):
        validate_collection_plan(malformed)
    malformed = copy.deepcopy(plan)
    contact = next(case for case in malformed['cases'] if case['expansion_schedule'] == 'contact0')
    contact['spec']['takeover_tick'] = 360
    with pytest.raises(ValueError, match='fallback'):
        validate_collection_plan(malformed)
    with pytest.raises(FileExistsError):
        prepare(setup['source'], output, session_root=setup['source'], registry_id='prepared_fixture',
            seeds=[6000], partition_manifest=setup['partition'], frozen_families=setup['families'],
            schedules=['fixed40'], lanes=8, event_seed=8000, round_index=7)


def test_fixed_opportunity_classification_does_not_call_rescue_student_success():
    case = dict(expansion_schedule='fixed40', spec={'takeover_tick': 40})
    result = dict(legal=True, software_complete=True, dagger=dict(expert_ticks=200, expert_valid_ticks=198,
        expert_takeover_success=True, student_autonomous_success=False, recovery_coverage={'switch_tick': 40}))
    assert classify_case(case, result) == 'expert_rescue_after_fixed_takeover'
    result['dagger']['recovery_coverage']['switch_tick'] = 39
    with pytest.raises(ValueError, match='differs'):
        classify_case(case, result)


def test_actual_hdf5_expansion_keeps_masks_statistics_and_validation(setup_registry):
    from test_dagger_data import takeover
    from visuotactile.narrow_data import dagger_record
    setup = setup_registry
    directory = setup['root']/'capture'
    directory.mkdir()
    entry = takeover(directory)
    original, _, _ = dagger_record(entry, include_geometry=False)
    entry.update(entry_for(setup))
    entry['path'] = str(directory/'takeover.h5')
    row = list(setup['registry']['families'].values())[0]
    with h5py.File(entry['path'], 'r+') as stream:
        metadata = json.loads(stream.attrs['metadata_json'])
        metadata.update(initial_family=row['family'], source_identity=row['source_identity'],
            initial_configuration_sha256=row['configuration_sha256'], training_expansion=setup['binding'])
        stream.attrs['metadata_json'] = json.dumps(metadata)
    entry['source_sha256'] = sha256(entry['path'])
    base = {key: original[key].clone() for key in ('robot', 'command', 'time', 'joint_limits', 'q_reached')}
    source = setup['source']/'A/state_data'
    records = {identity: base for identity in setup['source_data']['splits']['train']+setup['source_data']['splits']['validation']}
    torch.save(dict(contract={}, data_sha256=sha256(source/'data.json'), records=records), source/'numeric.pt')
    torch.save({'action_scale': torch.arange(29, dtype=torch.float32)+1}, source/'statistics.pt')
    output = setup['root']/'aggregate'
    data = build_dagger_aggregate(source, {'expanded_capture': entry}, output, 'q_reached',
        include_geometry=False, include_frontend_tactile=False, training_expansion_registry=setup['binding'])
    assert data['splits']['validation'] == setup['source_data']['splits']['validation']
    assert data['normalization_train_ids'] == ['old']
    assert data['training_expansion_registries'] == [setup['binding']]
    assert sha256(source/'statistics.pt') == sha256(output/'statistics.pt')
    recorded = torch.load(output/'numeric.pt', weights_only=False)['records']['expanded_capture']
    assert recorded['supervision_mask'].sum() == 16
    assert not recorded['supervision_mask'][:24].any()
    assert torch.equal(recorded['robot'], original['robot'])
    assert torch.equal(recorded['command'], original['command'])
    assert not torch.equal(recorded['q_reached'][24:], recorded['u_expert'][24:])
