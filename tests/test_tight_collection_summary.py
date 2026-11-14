"""Live census cannot open payloads or promote verification to admission."""
import json
from pathlib import Path

import h5py
import pytest

from test_batch_collection_review import completed_mixed, numeric
from test_training_expansion import setup_registry, write, entry_for
from tools import tight_collection_summary as summary
from visuotactile.training_expansion import (COLLECTION_CONTRACT, NEVER_FALLBACK,
    schedule_trigger, registry_binding)


def registered_plan(setup):
    cases = []
    for index in range(2):
        for role in ('A', 'B'):
            entry = entry_for(setup, index)
            name = f'{role}_{index}'
            spec = dict(entry, identity=name, cohort='dagger_train', takeover_tick=NEVER_FALLBACK,
                narrow_reset=setup['registry']['families'][entry['family']]['narrow_reset'])
            cases.append(dict(name=name, role=role, spec=spec, expansion_schedule='held60',
                dagger=dict(trigger=schedule_trigger('held60')), source_data=str(setup['manifests'][role])))
    return dict(cases=cases, stage_takeover_contract=COLLECTION_CONTRACT, test_used=False,
        training_expansion_registry=setup['binding'], utc=setup['registry']['created_utc']+1,
        pairs={'A': {'frontend_sha256':'actual_A', 'backend_sha256':'rear_A'},
            'B': {'frontend_sha256':'actual_B', 'backend_sha256':'rear_B'}},
        source_data_sha256={'A':'Adata','B':'Bdata'}, source_commit='fixture')


def test_metadata_family_disjointness_without_any_payload_reads(setup_registry, monkeypatch):
    monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('Payload must remain closed'))
    plan = registered_plan(setup_registry)
    result = summary.source_audit(plan)
    assert result['new_family_count'] == 2 and result['registered_opportunities'] == 4
    assert result['original_test_identity_count'] == 22
    assert result['distinct_family_seed_configuration'] is True
    assert result['large_payload_sha_recomputed'] is False
    assert result['source_pairs']['B']['frontend_sha256'] == 'actual_B'


def test_metadata_rejects_original_configuration_overlap(setup_registry, monkeypatch):
    plan = registered_plan(setup_registry)
    rows = json.loads(setup_registry['families'].read_text())
    rows[0]['narrow_reset'] = next(iter(setup_registry['registry']['families'].values()))['narrow_reset']
    write(setup_registry['families'], rows)
    registry = setup_registry['registry']
    registry['original_frozen_families']['sha256'] = summary.sha256(setup_registry['families'])
    write(Path(setup_registry['binding']['path']), registry)
    binding = registry_binding(setup_registry['binding']['path'])
    plan['training_expansion_registry'] = binding
    for case in plan['cases']:
        case['spec']['training_expansion'] = binding
    with pytest.raises(ValueError, match='duplicates prior'):
        summary.source_audit(plan)


def test_live_snapshot_keeps_verified_success_unadmitted_and_never_opens_hdf(setup_registry, monkeypatch):
    root = setup_registry['root']/'live'; root.mkdir()
    plan = registered_plan(setup_registry)
    write(root/'plan.json', plan)
    case = plan['cases'][0]; directory = root/case['name']; directory.mkdir()
    result = numeric(directory, 120, length=150, trigger=schedule_trigger('held60'))
    result['dataset_capture']['verified_source_fields'] = 185
    write(directory/'execution_complete.json', result)
    write(directory/'capture_complete.json', dict(status='source_verified', dataset_capture=result['dataset_capture']))
    # A live publication can expose entries before the root result index.
    write(root/'correction_entries.json', {case['name']:dict(episode=case['name'])})
    (directory/'numeric.h5').write_bytes(b'active or closed payload MUST NOT be opened')
    monkeypatch.setattr(h5py, 'File', lambda *a, **kw: pytest.fail('Snapshot opens no HDF5'))
    value = summary.snapshot(root)
    assert value['complete'] is False and value['rates'] is None
    assert value['committed_opportunities'] == value['counters']['committed_accepted'] == 0
    assert value['counters']['source_verified'] == 1
    assert value['counters']['manifest_ahead_of_result_index'] == [case['name']]
    assert value['cases'][0]['status'] == 'source_verified_pending_admission'
    assert value['cases'][0]['held_recheck']['category'] == 'actual_rechecked_takeover'
    result['accepted_for_training'] = True
    entry=dict(episode=case['name'],family=case['spec']['family'],
        source_sha256=result['dataset_capture']['sha256'],**plan['pairs'][case['role']])
    write(root/'correction_entries.json', {case['name']:entry})
    write(root/'results.json', [dict(case=case['name'], role=case['role'], result=result)])
    value = summary.snapshot(root)
    assert value['counters']['committed_accepted'] == 1
    assert value['counters']['committed_accepted_families'] == 1
    assert value['counters']['manifest_ahead_of_result_index'] == []
    entry['source_sha256']='wrong_capture'
    write(root/'correction_entries.json', {case['name']:entry})
    with pytest.raises(ValueError,match='case/capture/policy'):
        summary.snapshot(root)


def test_snapshot_rejects_unverified_accepted_capture(setup_registry):
    root = setup_registry['root']/'live';root.mkdir();plan=registered_plan(setup_registry)
    write(root/'plan.json', plan); case=plan['cases'][0]
    write(root/'results.json', [dict(case=case['name'], role=case['role'],
        result=dict(software_complete=True, legal=True, accepted_for_training=True))])
    write(root/'correction_entries.json', {case['name']:dict(episode=case['name'])})
    with pytest.raises(ValueError, match='verified successful'):
        summary.snapshot(root)


def test_closed_boundary_uses_true_student_reference_and_both_physics_samples(tmp_path):
    numeric(tmp_path, 120, length=150, trigger=schedule_trigger('held60'))
    window=summary.held_boundary_window(tmp_path,120)
    assert window['completed_tick']==120
    assert window['conditions']['two_contact_fingers_every_substep'] is True
    assert window['conditions']['height'] is True
    assert window['conditions']['rotation'] is False  # actual turning is allowed
    with h5py.File(tmp_path/'numeric.h5','r+') as stream:
        stream['episode/transition/dagger/control_source'][30]=1
    with pytest.raises(ValueError,match='only actual student'):
        summary.held_boundary_window(tmp_path,120)


def receipt_case(*, first, completed, rechecked=None, expert=0, delay=60):
    case = dict(expansion_schedule=f'held{delay}', dagger=dict(trigger=schedule_trigger(f'held{delay}')))
    receipt = dict(configuration=case['dagger']['trigger'],
        first_contact_height_window_end_completed_tick=first,
        earliest_allowed_completed_tick=None if first is None else first+delay,
        first_rechecked_contact_height_window_end_completed_tick=rechecked,
        completed_student_transitions_observed=completed,
        actual_scheduled_takeover_tick=rechecked if rechecked is not None else NEVER_FALLBACK,
        latest_contact_height_window=rechecked is not None, nonheld_windows_after_first=88)
    result = dict(legal=True, reasons=dict(timeout=True, fall=False, dropped=False),
        dagger=dict(trigger_receipt=receipt, expert_ticks=expert, expert_takeover_success=expert>0))
    return case, result


@pytest.mark.parametrize('first,completed,rechecked,expert,category', [
    (None,601,None,0,'no_completed_held_window'),
    (580,601,None,0,'minimum_delay_not_reached'),
    (535,601,None,0,'minimum_elapsed_without_current_held_window'),
    (183,305,305,0,'rechecked_without_next_action'),
    (183,305,305,144,'actual_rechecked_takeover')])
def test_recheck_outcomes_do_not_infer_drop_or_autonomous_success(first,completed,rechecked,expert,category):
    case,result=receipt_case(first=first,completed=completed,rechecked=rechecked,expert=expert)
    value=summary.held_recheck(case,result)
    assert value['category']==category
    assert value['native_termination_reasons']['dropped'] is False
    if rechecked:
        assert value['additional_wait_after_minimum_ticks']==62
        assert value['wait_after_first_held_ticks']==122
    assert 'not drop events' in value['interpretation']


@pytest.mark.parametrize('fault', ['latest_not_held','too_early','numeric_boundary'])
def test_recheck_cannot_count_stale_window_as_actual_takeover(fault):
    case,result=receipt_case(first=183,completed=305,rechecked=305,expert=144)
    boundary=dict(completed_student_tick=305,held_window=True)
    if fault=='latest_not_held':result['dagger']['trigger_receipt']['latest_contact_height_window']=False
    elif fault=='too_early':result['dagger']['trigger_receipt']['first_contact_height_window_end_completed_tick']=300
    else:boundary['held_window']=False
    with pytest.raises(ValueError):summary.held_recheck(case,result,boundary=boundary)


def test_complete_reuses_real_numeric_stage_checks_and_wall_denominators(setup_registry,monkeypatch):
    root=completed_mixed(setup_registry,monkeypatch)
    monkeypatch.setattr(summary,'source_audit',lambda plan:dict(fixture=True))
    plan=json.loads((root/'plan.json').read_text());plan.update(kind='takeover',test_used=False)
    for case in plan['cases']:
        case['spec']['narrow_reset']=setup_registry['registry']['families'][case['spec']['family']]['narrow_reset']
    write(root/'plan.json',plan)
    for case in plan['cases']:
        path=root/case['name']/'training_expansion_audit.json'
        value=json.loads(path.read_text());value['source_plan_sha256']=summary.sha256(root/'plan.json');write(path,value)
    write(root/'owner_finished.json',dict(software_complete=True,returncode=0,seconds=100))
    write(root/'result.json',dict(software_complete=True,cases=3,elapsed_seconds=95))
    write(root/'session.json',dict(deadline=200))
    write(root/'initialization.json',dict(seconds=5))
    write(root/'wave_profiles.json',[dict(wave=0,cases=3,control_ticks=160,seconds=80,admission_s=3)])
    rows=json.loads((root/'results.json').read_text())
    rows[0]['result']['async_capture']=dict(profile=dict(worker_verify_s=120,verification_slot_wait_s=2))
    write(root/'results.json',rows);write(root/'fixed/result.json',rows[0]['result'])
    value=summary.complete(root)
    assert value['complete'] is True and value['behavior']['summary']['recovered']==2
    assert value['behavior']['summary']['accepted']==0
    assert value['by_schedule']['fixed40']['all_valid_expert_ticks']==38
    assert value['by_schedule']['fixed40']['student_window_reference_unavailable']==1
    assert value['throughput']['profile']['owner_seconds']==100
    assert value['throughput']['profile']['async_profile_sums']['worker_verify_s']==120
    assert value['additional_async_profile_sums']['verification_slot_wait_s']==2
    assert value['throughput']['profile']['primary_rates']['attempts_per_hour']==108
    assert 'not schedule entrance labels' in value['definitions']['stage_counts']


def test_complete_refuses_partial_before_numeric_open(setup_registry,monkeypatch):
    root=completed_mixed(setup_registry,monkeypatch)
    monkeypatch.setattr(summary,'source_audit',lambda plan:dict(fixture=True))
    (root/'result.json').unlink()
    monkeypatch.setattr(h5py,'File',lambda *a,**kw:pytest.fail('Partial completion may not read HDF5'))
    with pytest.raises(ValueError):summary.complete(root)
