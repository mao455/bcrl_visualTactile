import copy

import h5py
import numpy as np
import pytest

from tools.tight_fresh_takeover_pose_audit import timeline_checks, window_components
from visuotactile.takeover_trigger import make_trigger


def sample():
    pose=np.zeros((12,7));pose[:,2]=.1;pose[:,3]=1
    palm=np.zeros_like(pose);palm[:,3]=1
    force=np.zeros((12,2,5,3));force[:,:,:2,2]=1
    return pose,palm,force


def test_relative_reorientation_can_fail_window_without_losing_contact_or_height():
    pose,palm,force=sample();angle=np.linspace(0,.3,12)
    pose[:,3]=np.cos(angle/2);pose[:,6]=np.sin(angle/2)
    row=window_components(pose,palm,force,0.,12)
    assert not row['stable_window']
    assert row['conditions']==dict(two_contact_fingers_every_substep=True,height=True,translation=True,rotation=False)
    assert row['palm_relative_max_rotation_rad']==pytest.approx(.3)


def test_translation_and_contact_failures_remain_distinct():
    pose,palm,force=sample();pose[:,0]=np.linspace(0,.012,12)
    row=window_components(pose,palm,force,0.,12)
    assert not row['conditions']['translation'] and row['conditions']['two_contact_fingers_every_substep']
    pose,palm,force=sample();force[3,0,1]=0
    row=window_components(pose,palm,force,0.,12)
    assert row['conditions']['translation'] and not row['conditions']['two_contact_fingers_every_substep']


def test_complete_unchanged_window_and_incomplete_window():
    pose,palm,force=sample()
    assert window_components(pose,palm,force,0.,12)['stable_window']
    with pytest.raises(ValueError):window_components(pose,palm,force,0.,11)


def test_moving_rotating_palm_and_quaternion_sign_changes_preserve_relative_grasp():
    pose,palm,force=sample();yaw=np.linspace(0,1,12)
    palm[:,0]=np.linspace(0,.1,12);palm[:,3]=np.cos(yaw/2);palm[:,6]=np.sin(yaw/2)
    pose[:,:3]=palm[:,:3]+np.column_stack((.02*np.cos(yaw),.02*np.sin(yaw),np.full(12,.1)))
    pose[:,3]=np.cos((yaw+.15)/2);pose[:,6]=np.sin((yaw+.15)/2)
    pose[::2,3:]*=-1;palm[1::2,3:]*=-1
    result=window_components(pose,palm,force,0.,12)
    assert result['stable_window']
    assert result['palm_relative_root_max_translation_m']<1e-12
    assert result['palm_relative_max_rotation_rad']<1e-7


def test_contact_threshold_is_strict_and_completed_tick_window_excludes_future():
    pose,palm,force=sample();pose=np.vstack((pose,pose[-1]));palm=np.vstack((palm,palm[-1]))
    force=np.concatenate((force,np.zeros_like(force[:1])))
    assert window_components(pose,palm,force,0.,12)['stable_window']
    assert not window_components(pose,palm,force,0.,13)['conditions']['two_contact_fingers_every_substep']
    force[:12,:,1,2]=.1
    assert not window_components(pose,palm,force,0.,12)['conditions']['two_contact_fingers_every_substep']


@pytest.mark.parametrize('mutation',['misaligned','nan','boolean_tick'])
def test_rejects_malformed_or_nonfinite_window(mutation):
    pose,palm,force=sample();tick=12
    if mutation=='misaligned':force=force[:-1]
    elif mutation=='nan':pose[0,0]=np.nan
    else:tick=True
    with pytest.raises(ValueError):window_components(pose,palm,force,0.,tick)


@pytest.mark.parametrize('mutation',[None,'time','episode','pose','authority','result','receipt'])
@pytest.mark.parametrize('version',['stable_grasp_then_delay_v1','contact_height_then_delay_v1'])
def test_actual_h5_timeline_checks_native_counts_and_expert_switch(tmp_path,mutation,version):
    n=72;student=67
    pose=np.zeros((n,7));pose[:,3]=1;pose[48:,2]=.1
    palm=np.zeros_like(pose);palm[:,3]=1
    force=np.zeros((n,2,5,3));force[:,:,:2,2]=1
    trigger=make_trigger(dict(version=version,delay_ticks=7),1000)
    receipt_key=('first_stable_window_end_completed_tick' if version=='stable_grasp_then_delay_v1'
        else 'first_contact_height_window_end_completed_tick')
    for tick in range(student):trigger.observe(tick,pose[tick],palm[tick],force[tick])
    result=dict(steps=n,dagger=dict(student_ticks=student,expert_ticks=n-student,trigger_receipt=trigger.receipt()))
    expected=copy.deepcopy(result)
    with h5py.File(tmp_path/'numeric.h5','w') as stream:
        t=stream.create_group('transition');o=stream.create_group('obs');before=t.create_group('diagnostics_before');after=t.create_group('diagnostics_after')
        t['time_before_s']=np.arange(n)/60;t['time_after_s']=np.arange(1,n+1)/60;o['time_s']=np.arange(n)/60
        t['step_id']=np.arange(n);t['episode_id']=np.full(n,42)
        for key,value in [('object_pose_w',pose),('palm_body_pose_w',palm),('goal_pose_w',pose),('successes',np.zeros(n))]:
            after[key]=value;before[key]=np.concatenate((value[:1],value[:-1]))
        t.create_group('dagger')['control_source']=(np.arange(n)>=student).astype(int)
        if mutation=='time':t['time_after_s'][10]+=.001
        elif mutation=='episode':t['episode_id'][10]=43
        elif mutation=='pose':before['object_pose_w'][10,0]=1
        elif mutation=='authority':t['dagger/control_source'][10]=1
        elif mutation=='result':result['dagger']['student_ticks']-=1
        elif mutation=='receipt':result['dagger']['trigger_receipt'][receipt_key]+=1
        if mutation is None:
            checked=timeline_checks(t,o,result,pose,palm,force)
            assert checked['trigger_replay_exact'] and result==expected
            assert checked['student_last_completed_tick']==67
            assert checked['expert_first_action_index']==67 and checked['expert_first_completed_tick']==68
            assert checked['trigger_receipt'][receipt_key]==60
        else:
            with pytest.raises(ValueError):timeline_checks(t,o,result,pose,palm,force)
