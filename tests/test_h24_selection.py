import copy
import json

import numpy as np
import pytest

from visuotactile.h24_selection import choose
from tools.h24_comparison_run import cohort


def result(arm=.1, hand=.2, ee=.02, jump=.1, success=False):
    return dict(offline=dict(summary=dict(A0={key:dict(mean=value) for key,value in
        [('precontact_ee12_target_fk_rmse_m',ee),('arm_boundary_jump_rms_rad',jump)]}),
        planning_cycle_latency_s=dict(A0=dict(p95=.1)),
        rows=[dict(valid12_positions=10,arm12_rmse_rad=arm,hand12_rmse_rad=hand)]),
        loop=dict(trials=[dict(identity='val',cohort='validation',legal=True,success=success)]))


def test_selection_uses_common_legal_validation_only_and_no_losses():
    sample=result();epsilon=result(ee=.01)
    sample['loop']['trials'] += [dict(identity='train',cohort='train',legal=True,success=True),
        dict(identity='unmatched',cohort='validation',legal=True,success=True)]
    sample['offline']['loss']=-100;epsilon['offline']['loss']=1000
    selection=choose(dict(sample=sample,epsilon=epsilon))
    assert selection['selected']=='epsilon'
    assert selection['paired_validation_ids']==['val']
    assert selection['tentative'] and not selection['statistical_superiority_established']


def test_degradation_guard_can_outweigh_one_success_but_not_two():
    sample=result();epsilon=result(arm=.2,success=True)
    assert choose(dict(sample=sample,epsilon=epsilon))['selected']=='sample'
    for value in (sample,epsilon):
        row=copy.deepcopy(value['loop']['trials'][0]);row['identity']='val2';value['loop']['trials'].append(row)
    assert choose(dict(sample=sample,epsilon=epsilon))['selected']=='epsilon'


def test_pooled_balanced_metric_weights_actual_positions():
    sample=result();epsilon=result()
    sample['offline']['rows'].append(dict(valid12_positions=30,arm12_rmse_rad=.3,hand12_rmse_rad=.4))
    selection=choose(dict(sample=sample,epsilon=epsilon))
    assert selection['metrics']['sample']['balanced_mse']==pytest.approx(.1)
    epsilon['offline']['summary']['A0']['precontact_ee12_target_fk_rmse_m']['mean']=float('nan')
    with pytest.raises(ValueError):choose(dict(sample=sample,epsilon=epsilon))


def test_empty_paired_cohort_is_tentative_offline_selection():
    sample=result();epsilon=result(ee=.01)
    sample['loop']['trials']=[]
    selection=choose(dict(sample=sample,epsilon=epsilon))
    assert selection['selected']=='epsilon' and not selection['paired_validation_ids']
    assert all(row['successes']==0 for row in selection['metrics'].values())


def test_startup_faults_preserve_missing_without_retry(tmp_path,monkeypatch):
    specs=[dict(identity=str(index),cohort='validation',seed=index) for index in range(12)]
    (tmp_path/'run_plan.json').write_text(json.dumps(dict(formal_cohort=specs)))
    (tmp_path/'kit_acceptance.json').write_text('{"persistent_reset_parity_passed":false}')
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'checkpoint')
    calls=[]
    def failed(root,name,*args,**kwargs):
        calls.append(name)
        raise RuntimeError('pre-scene native fault')
    monkeypatch.setattr('tools.h24_comparison_run.worker',failed)
    monkeypatch.setattr('tools.h24_comparison_run.startup_failure_evidence',lambda *args:dict(confirmed=True))
    observed=cohort(tmp_path,'sample_3000',checkpoint)
    assert len(calls)==2 and len(set(calls))==2
    assert len(observed['missing'])==12 and observed['trials']==[]
    assert not observed['software_complete']
    assert all(row['native_outcome'] is None for row in observed['missing'])


def test_phase_report_uses_validated_record_phase_and_masks(tmp_path,monkeypatch):
    from tools.h24_comparison_report import phases
    folder=tmp_path/'sample_offline/numeric_plans';folder.mkdir(parents=True)
    truth=np.zeros((1,24,29),dtype=np.float32);valid=np.zeros((1,24),dtype=bool);valid[:,:5]=True
    np.savez(folder/'episode.npz',truth=truth,valid=valid,anchors=np.array([0]),before=np.zeros((1,29)),A0=truth)
    monkeypatch.setattr('tools.h24_comparison_report.read_episode',lambda entry:dict(meta={},phase=np.zeros(5,dtype=int)))
    monkeypatch.setattr('tools.h24_comparison_report.validated_fk',lambda metadata:None)
    monkeypatch.setattr('tools.h24_comparison_report.target_errors',lambda *args:{})
    monkeypatch.setattr('tools.h24_comparison_report.error_summary',lambda error,mask:dict(ticks=int(mask.sum())))
    rows=phases(tmp_path,'sample',dict(entries=dict(episode={})))
    assert len(rows)==3 and {row['model'] for row in rows}=={'Student','Hold','Expert'}
    assert all(row['ticks']==5 for row in rows)
