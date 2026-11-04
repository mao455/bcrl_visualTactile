from pathlib import Path
import pytest
from tools.final_dp_report import aggregate
from tools.final_dp_eval import classify, equal_values


def row(valid=True, code=0, ordinal=0):
    return dict(model='StateDP',ordinal=ordinal,finished=1,valid=valid,exit_code=code,
        trial=dict(success=False,native_lift=True,native_subgoals=0,native_alignment=False,
            reasons={'timeout':True,'fall':False},contact={}))


def test_missing_and_shutdown_are_distinct_from_policy_failures():
    a=aggregate([row(),row(False,-6,1),row(True,-11,2)],'StateDP')
    assert a['valid']==2 and a['success']==0 and a['rate']==0
    assert a['attempts']==3 and a['attempts_without_result']==1
    assert a['software_faults']==2 and a['shutdown_faults']==1
    assert a['pending']==10 and a['reasons']=={'timeout':2}


def test_missing_only_has_no_success_rate():
    a=aggregate([row(False,-6)],'StateDP')
    assert a['valid']==0 and a['rate'] is None
    with pytest.raises(AssertionError):aggregate([row(),row()],'StateDP')


def test_fault_phase_and_fast_exit(tmp_path):
    assert classify(tmp_path,-6,None,'FileThread',False)=='startup_FileThread_before_scene'
    (tmp_path/'environment.json').write_text('{}')
    assert classify(tmp_path,-6,None,'FileThread',False)=='runtime_FileThread'
    assert classify(tmp_path,0,None,'',True)=='sdk_fast_exit_zero'
    assert classify(tmp_path,-11,None,'',True)=='shutdown_fault_after_complete_runtime'


def test_initial_rng_comparison_preserves_dtype_and_values():
    import torch
    a={'x':[torch.tensor([1,2]),('rng',42)]}
    assert equal_values(a,a)
    assert not equal_values(a,{'x':[torch.tensor([1.,2.]),('rng',42)]})
    assert not equal_values(a,{'x':[torch.tensor([1,3]),('rng',42)]})
