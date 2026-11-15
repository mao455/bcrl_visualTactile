"""Unified short check for the early-expert schedule and new-session disk budget."""
import copy

import pytest

from test_training_expansion import setup_registry, write
from visuotactile.position_control import ExpertPositionBridge
from visuotactile.state_benchmark import dagger_configuration
from visuotactile.training_expansion import classify_case, schedule_trigger, validate_collection_plan
from tools.tight_fresh_followup_session import followup_storage


def test_first_step_expert_schedule_retains_full_student_and_legal_mask_contract(setup_registry, monkeypatch):
    import time
    from tools.tight_fresh_expansion import prepare
    setup = setup_registry
    write(setup['source'] / 'session.json', dict(deadline=time.time()+1000,
        domain=setup['domain'], expert_sha256='expert_fixture'))
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair', lambda root, role: dict(role=role))
    plan = prepare(setup['source'], setup['root'] / 'first_step', session_root=setup['source'],
        registry_id='first_step_fixture', seeds=[6000, 6001], partition_manifest=setup['partition'],
        frozen_families=setup['families'], schedules=['fixed1', 'held0'], lanes=16,
        event_seed=8000, round_index=14, verification_workers=8, verification_max_pending=16)
    validate_collection_plan(plan)
    assert plan['batch']['verification_workers'] == 8
    assert schedule_trigger('fixed1') is None
    cases = [case for case in plan['cases'] if case['expansion_schedule'] == 'fixed1']
    assert len(cases) == 2 and {case['role'] for case in cases} == {'A', 'B'}
    for case in cases:
        assert case['spec']['takeover_tick'] == 1
        config = dagger_configuration(dict(version=ExpertPositionBridge.version,
            expert_speed=ExpertPositionBridge.speed_scheme), case['spec'], setup['source_data'])
        assert config['takeover_tick'] == 1 and 'mask0' in config['supervision_rule']
        result = dict(legal=True, software_complete=True, dagger=dict(expert_ticks=200,
            expert_valid_ticks=198, expert_takeover_success=True, student_autonomous_success=False,
            recovery_coverage={'switch_tick': 1}))
        assert classify_case(case, result) == 'expert_rescue_after_fixed_takeover'


def test_new_storage_budget_preserves_old_session():
    old = dict(reserve_bytes=60*1024**3, resources=dict(min_disk_gib=60, min_ram_gib=10),
               deadline=1234, followup=dict(source_root='old'))
    before = copy.deepcopy(old)
    new = followup_storage(old, 32)
    assert old == before and followup_storage(old, None) == before
    assert new['reserve_bytes'] == 32*1024**3 and new['resources']['min_disk_gib'] == 32
    assert new['deadline'] == 1234 and new['resources']['min_ram_gib'] == 10
    assert new['followup']['storage_override']['previous']['reserve_bytes'] == 60*1024**3
    for invalid in (True, 0, 15, 32.0):
        with pytest.raises(ValueError, match='disk reserve'):
            followup_storage(old, invalid)
