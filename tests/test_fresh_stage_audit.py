import json

import numpy as np
import pytest

from tools import tight_fresh_stage_audit as audit


def fields(*, switch=60, length=80, queue_length=3):
    tick = np.arange(length)
    source = (tick >= switch).astype(int) if switch is not None else np.zeros(length, int)
    delay = tick % queue_length
    selected = np.maximum(tick-delay, 0)
    clean = (tick >= switch+queue_length-1) if switch is not None else np.zeros(length, bool)
    expert_source = (selected >= switch) if switch is not None else np.zeros(length, bool)
    valid = clean & expert_source
    phases = np.column_stack((source.astype(bool), valid))
    return dict(control_source=source, expert_label_valid=valid,
        expert_queue_length=np.full(length, queue_length), expert_delay_index=delay,
        expert_source_step=selected, switch_clean=clean,
        expert_source_is_expert_issued=expert_source,
        segment_id=np.r_[0, np.cumsum((phases[1:] != phases[:-1]).any(-1))])


def test_first_legal_label_waits_for_whole_queue_even_if_newest_expert_suggestion_selected():
    result = audit.label_alignment(fields())
    assert result['switch_action_index'] == 60
    assert result['first_legal_label_action_index'] == 62
    assert result['expert_flush_masked_ticks'] == 2
    assert result['legal_expert_ticks'] == 18
    wrong = fields(); wrong['expert_label_valid'][60] = True
    with pytest.raises(ValueError, match='supervision mask'):
        audit.label_alignment(wrong)


def test_never_stable_case_has_no_expert_labels_and_no_synthetic_switch():
    result = audit.label_alignment(fields(switch=None))
    assert result['switch_action_index'] is None
    assert result['first_legal_label_action_index'] is None
    assert result['legal_expert_ticks'] == result['expert_ticks'] == 0
    assert result['student_history_ticks'] == 80


def test_queue_length_one_has_no_artificial_two_tick_flush():
    result = audit.label_alignment(fields(queue_length=1))
    assert result['first_legal_label_action_index'] == 60
    assert result['expert_flush_masked_ticks'] == 0


@pytest.mark.parametrize('field', ['expert_source_step', 'segment_id', 'switch_clean', 'expert_source_is_expert_issued'])
def test_actual_delay_and_boundary_corruptions_are_rejected(field):
    wrong = fields()
    wrong[field][65] = 0
    with pytest.raises(ValueError):
        audit.label_alignment(wrong)


def test_expert_then_student_interleaving_is_rejected():
    wrong = fields(); wrong['control_source'][70] = 0
    with pytest.raises(ValueError, match='one complete student prefix'):
        audit.label_alignment(wrong)


def test_unindexed_active_case_is_pending_even_after_result_json_exists(tmp_path, monkeypatch):
    plan = dict(cases=[dict(name='active', role='A')], source_commit='frozen')
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'active').mkdir()
    (tmp_path/'active/result.json').write_text(json.dumps(dict(software_complete=True)))
    (tmp_path/'active/numeric.h5').write_bytes(b'active HDF5 must never be opened')
    monkeypatch.setattr(audit, 'validate_plan', lambda plan: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Opened an active HDF5')
    monkeypatch.setattr(audit.h5py, 'File', forbidden)
    result = audit.report(tmp_path)
    assert result['completed'] == 0 and result['pending'] == ['active']
    assert not result['collection_complete']


def test_final_marker_cannot_hide_unresolved_registered_opportunities(tmp_path, monkeypatch):
    (tmp_path/'plan.json').write_text(json.dumps(dict(cases=[dict(name='missing', role='B')], source_commit='frozen')))
    (tmp_path/'result.json').write_text(json.dumps(dict(software_complete=True)))
    monkeypatch.setattr(audit, 'validate_plan', lambda plan: None)
    with pytest.raises(ValueError, match='Final opportunity accounting'):
        audit.report(tmp_path)


@pytest.mark.parametrize('contract,event',[
    ('stable_only_takeover_v1','stable'),('contact_height_only_takeover_v1','contact_height')])
def test_report_requires_the_registered_stage_audit_even_for_invalid_starts(tmp_path,monkeypatch,contract,event):
    plan=dict(cases=[dict(name='invalid',role='A')],source_commit='frozen',stage_takeover_contract=contract)
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    (tmp_path/'result.json').write_text(json.dumps(dict(software_complete=True)))
    result=dict(software_complete=True,legal=False,success=False)
    (tmp_path/'results.json').write_text(json.dumps([dict(case='invalid',role='A',result=result)]))
    directory=tmp_path/'invalid';directory.mkdir()
    (directory/'result.json').write_text(json.dumps(result))
    monkeypatch.setattr(audit,'validate_plan',lambda plan:None)
    proof=dict(category='invalid_initialization',causal_trigger=None,source_plan_sha256=audit.sha256(tmp_path/'plan.json'))
    wrong='stable' if event=='contact_height' else 'contact_height'
    (directory/(wrong+'_stage_audit.json')).write_text(json.dumps(proof))
    with pytest.raises(FileNotFoundError):audit.report(tmp_path)
    (directory/(event+'_stage_audit.json')).write_text(json.dumps(proof))
    report=audit.report(tmp_path)
    assert report['collection_complete'] and report['completed']==1
    assert report['version']=='completed_'+event+'_takeover_audit_v1'
