import numpy as np
import pytest

from tools.tight_diagnostics import control_audit_masks,control_audit_metrics


def trace(n):
    return dict(object_contact=np.zeros((n,2,5,3)),lifted=np.zeros(n,bool),subgoals=np.zeros(n))


def test_recovery_prefix_stops_at_invalid_or_segment_and_preserves_student_mask():
    n=64;source=np.ones(n,int);source[:10]=0
    valid=np.ones(n,bool);valid[:12]=False;valid[20]=False
    segment=np.zeros(n,int);segment[12:]=2
    masks,phases=control_audit_masks(source,valid,segment,trace(n))
    assert masks['student_prefix'].sum()==10
    assert masks['expert_flush'].sum()==3
    assert masks['expert_early48'].sum()==8
    assert masks['expert_later'].sum()==43
    assert sum(v.astype(int) for v in masks.values()).tolist()==[1]*n
    valid[20]=True;segment[18:]=3
    masks,_=control_audit_masks(source,valid,segment,trace(n))
    assert masks['expert_early48'].sum()==6
    with pytest.raises(ValueError,match='authority'):
        control_audit_masks(source,np.ones(n,bool),segment,trace(n))


def test_phase_events_do_not_infer_native_lift_from_success():
    a=trace(4);a['object_contact'][1,0,0,0]=.2
    a['object_contact'][2,0,:2,0]=.2
    a['subgoals'][3]=2
    _,phases=control_audit_masks(np.zeros(4),np.zeros(4,bool),np.zeros(4),a)
    assert phases.tolist()==['noncontact','contact','multifinger_contact','native_complete']
    assert not a['lifted'].any()


def test_metrics_pool_sums_before_rmse_and_keep_exact_denominator():
    x=np.arange(3*29,dtype=float).reshape(3,29)
    result=control_audit_metrics({'error_squared':x*x},np.array([True,False,True]))
    assert result['ticks']==2
    np.testing.assert_array_equal(result['error_squared'],x[0]**2+x[2]**2)
    assert control_audit_metrics({'x':x},np.zeros(3,bool)) is None
    x[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        control_audit_metrics({'x':x},np.ones(3,bool))
