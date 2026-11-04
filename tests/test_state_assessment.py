import numpy as np

from tools.anchor_interim_analysis import motion_window, aggregate_windows


def test_static_target_accuracy_cannot_hide_missing_small_motion():
    before=np.zeros(29);truth=np.zeros((24,29));truth[:,:7]=np.arange(1,25)[:,None]*.001
    xyz=np.zeros((24,3));xyz[:,0]=np.arange(1,25)*.0001
    exact=motion_window(truth,truth,before,xyz,xyz,np.zeros(3),12)
    held=motion_window(np.zeros_like(truth),truth,before,np.zeros_like(xyz),xyz,np.zeros(3),12)
    assert exact['arm_delta_rmse_rad']==0 and exact['ee_rmse_mm']==0
    assert np.isclose(exact['arm_motion']['cosine'],1) and np.isclose(exact['arm_motion']['amplitude_ratio'],1)
    assert held['arm_motion']['amplitude_ratio']==0 and held['arm_motion']['cosine']==0
    assert np.isclose(held['arm_delta_rmse_rad'],.001)
    assert held['ee_integrated_error_mm_s']>0
    # Includes initial tick against the preceding target, not just internal differences.
    assert exact['arm_motion']['moving_ticks']==12


def test_zero_length_precontact_and_static_motion_are_explicit():
    q=np.zeros((5,29));x=np.zeros((5,3));b=np.zeros(29)
    assert motion_window(q,q,b,x,x,np.zeros(3),0)==dict(ticks=0)
    row=motion_window(q,q,b,x,x,np.zeros(3),5)
    assert row['arm_motion']['moving_ticks']==0 and row['arm_motion']['cosine'] is None
    assert aggregate_windows([dict(ticks=0),row])['ticks']==5


def test_pooled_rmse_weights_actual_valid_ticks():
    y=np.zeros((3,29));x=np.zeros((3,3));p=y.copy();p[:,:7]=1
    a=motion_window(p,y,np.zeros(29),x,x,np.zeros(3),1)
    b=motion_window(y,y,np.zeros(29),x,x,np.zeros(3),3)
    assert aggregate_windows([a,b])['arm_rmse_rad']==.5
