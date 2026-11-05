import copy
import json
from pathlib import Path

import pytest

from tools.narrow_followup import dataset_union, predecessor_released
from visuotactile.narrow_decision import decide, early_stop, continuation_rate, failure_stage
from tools.narrow_delivery import outside, replace, wilson


def curve(values, train=None):
    train=train or [x/10 for x in values]
    return [dict(update=(i+1)*500,metrics=dict(selection_mse=v,arm_rmse=v**.5,ee_distance_rmse_mm=v**.5*1000),
        train8=dict(selection_mse=train[i]),early={k:[dict(arm_rmse_rad=.005 if k=='train' else .02,
            hold_arm_rmse_rad=.01,direction_cos=.6,amplitude_ratio=3.)] for k in ('train','validation')})
        for i,v in enumerate(values)]


COHORTS=dict(train=dict(valid=4,successes=0),unseen=dict(valid=12,successes=0))


def test_improvement_allows_only_one_complete_state_extension():
    rr=curve([.01,.008,.006,.004])
    assert decide(rr,COHORTS,audit_passed=True)['action']=='extend'
    assert decide(rr,COHORTS,audit_passed=True,extension_used=True)['action']=='stop'


def test_data_expansion_requires_plateau_fit_gap_and_contract_audit():
    rr=curve([.005,.00498,.00497,.00496])
    assert decide(rr,COHORTS,audit_passed=False)['action']=='stop'
    assert decide(rr,COHORTS,audit_passed=True)['action']=='expand'
    unfit=copy.deepcopy(rr)
    for row in unfit:row['early']['train'][0]['arm_rmse_rad']=.02
    assert decide(unfit,COHORTS,audit_passed=True)['action']=='stop'
    assert early_stop(rr,1500)
    assert not early_stop(rr,2500)


def test_promising_small_cohort_goes_to_new_test_without_more_data():
    cc=copy.deepcopy(COHORTS);cc['train']['successes']=3;cc['unseen']['successes']=7
    assert decide(curve([.005]*4),cc,audit_passed=True)['action']=='final_validate'
    cc['unseen']['valid']=5
    assert decide(curve([.005]*4),cc,audit_passed=True)['action']=='stop'


def test_resume_schedule_never_rewarms_or_changes_old_lr_values():
    from visuotactile.anchor_sampling import continuation_lr
    config=dict(target=5735,warmup=200,lr_peak=3e-5,lr_floor=3e-6)
    for step in (1,200,1000,5735):
        assert continuation_rate(step,config)==continuation_lr(step,5735,200,3e-5,3e-6)
    assert continuation_rate(8000,config)==3e-6


def test_union_preserves_every_original_test_record_and_rejects_split_leakage():
    old=[dict(id='a',family='test_family',split='test',path='original')]
    added=[dict(id='b',family='new_train',split='train',path='new')]
    result=dataset_union(old,added)
    assert result[0]==old[0] and old[0]['split']=='test'
    with pytest.raises(ValueError):dataset_union(old,[dict(id='b',family='test_family',split='train')])
    with pytest.raises(ValueError):dataset_union(old,[dict(id='a',family='test_family',split='test')])


def test_dependency_requires_finished_owner_and_no_live_reused_identity(tmp_path,monkeypatch):
    def put(name,value):(tmp_path/name).write_text(json.dumps(value))
    put('status.json',dict(state='running',owned_workers_exited=False))
    assert not predecessor_released(tmp_path)
    put('status.json',dict(state='complete',owned_workers_exited=True));put('PIPELINE_COMPLETE.json',{})
    put('owner.json',dict(owner=dict(pid=77,start_ticks=100)))
    monkeypatch.setattr('tools.narrow_followup.proc',lambda pid:dict(pid=pid,start_ticks=100))
    assert not predecessor_released(tmp_path)
    monkeypatch.setattr('tools.narrow_followup.proc',lambda pid:dict(pid=pid,start_ticks=200))
    assert predecessor_released(tmp_path)
    put('status.json',dict(state='blocked',last_error='contract'))
    with pytest.raises(RuntimeError):predecessor_released(tmp_path)


def test_failure_stages_do_not_call_contact_success_or_stable_grasp():
    row=dict(legal=True,success=False,approach_proxy=True)
    assert failure_stage(row)=='grasp_or_lift_unresolved'
    assert failure_stage(dict(row,native_subgoals=1))=='second_insertion'
    assert failure_stage(dict(row,success=True))=='complete'


def test_current_observed_plateau_is_not_a_reason_for_blind_extension():
    root=Path('outputs/visuotactile/run/narrow_state_20261105_02/training/validation.jsonl')
    if not root.exists():pytest.skip('Local evidence')
    rr=[json.loads(line) for line in root.read_text().splitlines() if 3000<=json.loads(line)['update']<=4500]
    assert len(rr)==4
    assert decide(rr,COHORTS,audit_passed=True)['action']=='expand'


def test_publication_preserves_manual_text_and_reports_small_sample_uncertainty():
    original='# Results\n\nPreserved historical evidence.\n'
    once=replace(original,'New evidence')
    # Subsequent generated updates leave the manually reviewed remainder fixed.
    assert outside(once)==outside(replace(once,'Final evidence'))
    assert 'Preserved historical evidence.' in replace(once,'Final evidence')
    ci=wilson(8,12)
    assert ci[0]<.40 and ci[1]>.85
    assert wilson(0,0) is None
