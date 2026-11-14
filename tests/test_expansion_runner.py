"""Expansion registration must survive the production runner's admission path."""
import copy
import json
import time

import h5py
import pytest

from test_training_expansion import setup_registry
from test_batched_rollout import metadata
from tools.tight_fresh_batch_run import admit_capture,classify_capture,validate_takeover_plan
from visuotactile.batched_rollout import lane_metadata
from visuotactile.compact import sha256
from visuotactile.training_expansion import COLLECTION_CONTRACT,validate_capture_binding
from visuotactile.training_resources import atomic_json


def prepared(setup,monkeypatch):
    from tools.tight_fresh_expansion import prepare
    session=dict(deadline=time.time()+1000,domain=setup['domain'],expert_sha256='expert_fixture')
    atomic_json(setup['source']/'session.json',session)
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair',lambda root,role:dict(role=role))
    output=setup['root']/'expansion'
    plan=prepare(setup['source'],output,session_root=setup['source'],registry_id='runner_fixture',
        seeds=[6000,6001],partition_manifest=setup['partition'],frozen_families=setup['families'],
        schedules=['fixed40','contact0','fixed80'],existing_identities=['old'],lanes=8,event_seed=8000,round_index=7)
    return output,plan


def test_runner_dispatches_real_prepared_mixed_plan_without_old_stage_guard(setup_registry,monkeypatch):
    from tools.tight_fresh_stage_takeover import validate_plan as original_stage_validator
    output,plan=prepared(setup_registry,monkeypatch)
    monkeypatch.setattr('tools.tight_fresh_stage_takeover.validate_plan',
        lambda *args:pytest.fail('Expansion must not enter old never-fallback-only validator'))
    validate_takeover_plan(plan)
    changed=copy.deepcopy(plan)
    next(case for case in changed['cases'] if case['expansion_schedule']=='fixed40')['spec']['takeover_tick']=41
    with pytest.raises(ValueError,match='Fixed-time'):
        validate_takeover_plan(changed)
    monkeypatch.setattr('tools.tight_fresh_stage_takeover.validate_plan',original_stage_validator)
    changed=copy.deepcopy(plan);changed['stage_takeover_contract']='invented_expansion'
    with pytest.raises(ValueError,match='never-fallback'):
        validate_takeover_plan(changed)


def test_new_fixed_family_binding_reaches_capture_and_accepted_entry(setup_registry,monkeypatch):
    output,plan=prepared(setup_registry,monkeypatch)
    case=next(case for case in plan['cases'] if case['expansion_schedule']=='fixed80' and case['role']=='A')
    directory=output/case['name'];directory.mkdir()
    pair=dict(frontend_sha256='f',backend_sha256='b')
    local=lane_metadata(metadata(8),3,8,case,pair,case['dagger'],dict(lane=3))
    path=directory/'data.h5'
    with h5py.File(path,'x') as stream:stream.attrs['metadata_json']=json.dumps(local)
    result=dict(**case['spec'],legal=True,software_complete=True,success=True,steps=90,
        dataset_capture=dict(path=str(path)),dagger=dict(configuration=case['dagger'],expert_ticks=10,
            expert_valid_ticks=7,expert_takeover_success=True,student_autonomous_success=False,
            recovery_coverage=dict(switch_tick=80)))
    atomic_json(directory/'result.json',result)
    monkeypatch.setattr('visuotactile.state_benchmark.audit_dagger_capture',
        lambda directory,pair:dict(passed=True,result_sha256=sha256(directory/'result.json')))
    monkeypatch.setattr('tools.tight_fresh_report.audit_trigger_case',
        lambda *args:pytest.fail('A fixed opportunity must not call causal contact trigger audit'))
    seen=[]
    def load_record(entry,include_geometry):
        with h5py.File(entry['path']) as stream:
            validate_capture_binding(entry,json.loads(stream.attrs['metadata_json']))
        seen.append(copy.deepcopy(entry))
        return {'mask_count':7},None,None
    monkeypatch.setattr('visuotactile.narrow_data.dagger_record',load_record)
    monkeypatch.setattr('visuotactile.anchor_sampling.supervision_population',lambda *args:dict(supervised_ticks=7))
    monkeypatch.setattr('visuotactile.camera_validity.inspect_capture',lambda *args:dict(passed=True))
    entries={};admit_capture(output,case,result,pair,plan,entries)
    assert result['accepted_for_training'] and entries[case['spec']['identity']]==seen[0]
    for key in ('training_expansion','initial_configuration_sha256','source_identity'):
        assert seen[0][key]==case['spec'][key]
    assert local['training_expansion']==case['spec']['training_expansion']
    assert seen[0]['expansion_schedule']=='fixed80'
    audit=json.loads((directory/'training_expansion_audit.json').read_text())
    assert audit['category']=='expert_rescue_after_fixed_takeover'
    assert audit['causal_trigger'] is None and audit['no_expert_fallback'] is None
    assert audit['expert_schedule']=='preregistered_fixed_tick'
    assert not (directory/'contact_height_stage_audit.json').exists()
    assert json.loads((directory/'result.json').read_text())['accepted_for_training']


def test_expansion_contact_case_audits_actual_trigger_and_rejects_changed_binding(setup_registry,monkeypatch):
    output,plan=prepared(setup_registry,monkeypatch)
    case=next(case for case in plan['cases'] if case['expansion_schedule']=='contact0')
    result=dict(legal=True,software_complete=True,dagger=dict(configuration=copy.deepcopy(case['dagger'])))
    calls=[];trigger_audit=dict(receipt='actual_observed_window')
    def audit(directory,observed):
        calls.append((directory,observed));return trigger_audit
    monkeypatch.setattr('tools.tight_fresh_report.audit_trigger_case',audit)
    monkeypatch.setattr('tools.tight_fresh_stage_takeover.classify',
        lambda result,audit:'verified_contact_category' if audit is trigger_audit else pytest.fail('Lost actual trigger audit'))
    name,category,receipt=classify_capture(output/case['name'],case,result,plan)
    assert name=='training_expansion_audit.json' and category=='verified_contact_category' and receipt is trigger_audit
    assert len(calls)==1
    result['dagger']['configuration']['trigger']['delay_ticks']=12
    with pytest.raises(ValueError,match='differs from registered'):
        classify_capture(output/case['name'],case,result,plan)
    assert len(calls)==1
