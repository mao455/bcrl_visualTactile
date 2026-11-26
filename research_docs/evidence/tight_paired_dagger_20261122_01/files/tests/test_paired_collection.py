"""CPU contracts for the finite own-student collection wrapper."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tools import tight_paired_collection as collection


def put(path, value):
    path.write_text(json.dumps(value))


def population(tmp_path):
    cases, rows, assignments, starts, entries = [], [], [], [], {}
    for family in range(24):
        for op in collection.OPPORTUNITIES:
            pair=[]
            for role in 'AB':
                name=f'{role}{family}_{op}'
                cases.append(dict(name=name,role=role,opportunity=op,spec=dict(family=str(family))))
                assignments.append(dict(case=name))
                takeover=role=='A'
                dagger=dict(recovery_coverage=dict(switch_tick=100 if takeover else None),
                    trigger_receipt=dict(reason='support_instability' if takeover else 'registered_conditions_not_reached',
                        never_entered_15mm=True,ever_entered_then_left_15mm=False),
                    expert_takeover_success=takeover,student_autonomous_success=not takeover,expert_valid_ticks=7 if takeover else 0)
                rows.append(dict(case=name,role=role,result=dict(legal=True,software_complete=True,
                    accepted_for_training=takeover,dagger=dagger)))
                if takeover:entries[name]=dict(source_group=role)
                pair.append(name)
            starts.append(dict(family=str(family),opportunity=op,reference=pair[0],case=pair[1],passed=True))
    put(tmp_path/'plan.json',dict(cases=cases));put(tmp_path/'results.json',rows)
    put(tmp_path/'correction_entries.json',entries);put(tmp_path/'paired_queue_assignments.json',assignments)
    put(tmp_path/'paired_initial_check.json',dict(rows=starts,passed=True))
    return cases,rows,assignments,starts,entries


def test_full_opportunity_summary_separates_expert_recovery_from_student_autonomy(tmp_path):
    population(tmp_path);r=collection.summarize(tmp_path)
    assert r['opportunities']==96 and not r['expert_recovery_is_autonomous_success']
    assert r['roles']['A']['opportunities']==r['roles']['B']['opportunities']==48
    assert r['roles']['A']['expert_recovered']==48
    assert r['roles']['A']['student_success_without_takeover']==0
    assert r['roles']['B']['student_success_without_takeover']==48
    assert r['roles']['B']['expert_recovered']==0
    assert r['roles']['A']['admitted_legal_expert_ticks']==336


@pytest.mark.parametrize('bad',['missing_opportunity','wrong_role','wrong_owner','missing_admission','missing_pair','failed_pair','missing_assignment','assignment_order'])
def test_summary_rejects_incomplete_or_cross_owned_denominators(tmp_path,bad):
    cases,rows,assignments,starts,entries=population(tmp_path)
    if bad=='missing_opportunity':put(tmp_path/'results.json',rows[:-1])
    if bad=='wrong_role':rows[0]['role']='B';put(tmp_path/'results.json',rows)
    if bad=='wrong_owner':entries[cases[0]['name']]['source_group']='B';put(tmp_path/'correction_entries.json',entries)
    if bad=='missing_admission':entries.pop(cases[0]['name']);put(tmp_path/'correction_entries.json',entries)
    if bad=='missing_pair':put(tmp_path/'paired_initial_check.json',dict(rows=starts[:-1],passed=True))
    if bad=='failed_pair':starts[0]['passed']=False;put(tmp_path/'paired_initial_check.json',dict(rows=starts,passed=True))
    if bad=='missing_assignment':put(tmp_path/'paired_queue_assignments.json',assignments[:-1])
    if bad=='assignment_order':put(tmp_path/'paired_queue_assignments.json',list(reversed(assignments)))
    with pytest.raises(ValueError):collection.summarize(tmp_path)


def test_cli_defaults_root_collection_and_eight_lanes(monkeypatch,tmp_path):
    seen=[]
    monkeypatch.setattr(collection,'prepare',lambda root,output,lanes:seen.append((root,output,lanes)))
    monkeypatch.setattr(sys,'argv',['collection','prepare','--root',str(tmp_path)])
    collection.main();assert seen==[(tmp_path,None,8)]
    monkeypatch.setattr(collection,'summarize',lambda output:seen.append(output))
    monkeypatch.setattr(sys,'argv',['collection','summarize','--root',str(tmp_path)])
    collection.main();assert seen[-1]==tmp_path/'collection'


def test_causal_trigger_semantics_and_exact_native_gate_contract(capsys):
    collection.self_check()
    assert json.loads(capsys.readouterr().out)['passed']
    for opportunity in collection.OPPORTUNITIES:
        cfg=collection.trigger_profile(opportunity)
        assert cfg['native_confirmation_ticks']==10 and cfg['gate_m']==.015
        assert cfg['no_fixed_time_fallback']
        cfg['native_confirmation_ticks']=9
        with pytest.raises(ValueError):collection.PairedRecoveryTrigger(cfg,collection.NEVER)


def test_pre_action_pairing_rejects_nonfinite_fields():
    diagnostics={k:torch.zeros(1,n) for k,n in (
        ('joint_position_rad',29),('joint_velocity_rad_s',29),('object_velocity_w_m_s_rad_s',6),
        ('goal_velocity_w_m_s_rad_s',6),('base_pose_w',7),('object_pose_w',7),('hole_pose_w',7),
        ('goal_pose_w',7),('table_pose_w',7),('palm_body_pose_w',7))}
    packet=dict(actuator_target_rad=torch.zeros(1,29))
    assert collection._initial_values(packet,diagnostics,0)
    diagnostics['joint_position_rad'][0,0]=float('nan')
    with pytest.raises(ValueError,match='finite'):collection._initial_values(packet,diagnostics,0)
