import numpy as np
import pytest

from tools import tight_fresh_expert_stability_coverage as report
from tools.tight_fresh_takeover_pose_audit import window_components
from visuotactile.takeover_trigger import stable_windows


def example():
    n=84
    obj=np.zeros((n,7),np.float32);obj[:,2]=.6;obj[48:,2]=.7;obj[:,3]=1.
    palm=np.zeros_like(obj);palm[:,3]=1.
    theta=np.zeros(n,np.float32);theta[64:]=np.linspace(0,1.3,n-64)
    obj[:,3]=np.cos(theta/2);obj[:,5]=np.sin(theta/2)
    force=np.zeros((n,2,5,3),np.float32);force[:,:,:2,2]=1.
    goal=obj.copy();goal[:,:3]=[0,0,.7]
    a=dict(object=obj,palm=palm,goal_before=goal.copy(),goal_after=goal.copy(),
        near=np.zeros(n,bool),align=np.zeros(n,bool),complete=np.zeros(n,bool),
        subgoals_before=np.zeros(n,np.int64),subgoals_after=np.zeros(n,np.int64),
        lifted=np.arange(n)>=48,two=np.ones(n,bool),
        terminations={'dropped':np.zeros(n,bool),'fall':np.zeros(n,bool)})
    a['near'][78:]=True;a['align'][-1]=True;a['complete'][-1]=True;a['subgoals_after'][-1]=1
    stable,_=stable_windows(obj,palm,force,.6)
    components=[None]*11+[window_components(obj,palm,force,.6,t) for t in range(12,n+1)]
    return a,stable,components


def test_windows_cannot_cross_student_queue_legal_or_segment_boundaries():
    mask=np.arange(84)>=52;segment=np.where(np.arange(84)<50,0,np.where(np.arange(84)<52,1,2))
    eligible,counts=report.isolated_windows(mask,segment)
    assert np.flatnonzero(eligible)[0]==63
    assert counts['no_complete_same_phase_segment_window']==11
    segment[70:]=3
    eligible,_=report.isolated_windows(mask,segment)
    assert not eligible[70:81].any() and eligible[81]
    # Completed48 is the first causal point with the frozen height reference.
    eligible,counts=report.isolated_windows(np.ones(84,bool),np.zeros(84,np.int64))
    assert np.flatnonzero(eligible)[0]==47
    assert counts['full_window_before_height_reference']==36


def test_successful_rotation_can_leave_stable_without_a_drop_label():
    arrays,stable,components=example()
    result=report.phase_summary(np.ones(84,bool),np.zeros(84,np.int64),arrays,stable,components)
    approach=result['first_stable_to_first_later_near_goal']
    assert approach['available'] and approach['left_stable_window']
    assert approach['first_loss']['failed_conditions']==['rotation']
    assert result['false_contact_and_height_pass_rotation_only_endpoints']>0
    assert result['events']['native_complete']['ticks']==1
    assert result['events']['native_complete']['false_eligible_ticks']==1
    assert result['native_termination_flags']=={'dropped':0,'fall':0}


def test_snapshot_uses_active_goal_before_native_goal_reset():
    arrays,stable,components=example();index=83
    arrays['goal_after'][index,:3]=[100,100,100]
    eligible,_=report.isolated_windows(np.ones(84,bool),np.zeros(84,np.int64))
    value=report.snapshot(index,arrays,stable,components,eligible)
    assert value['goal_changed_during_transition']
    assert value['object_to_active_goal_root_distance_m']==pytest.approx(0,abs=1e-6)
    assert value['object_to_active_goal_quaternion_angle_rad']==pytest.approx(0,abs=1e-3)


def test_mixed_boundary_stable_endpoint_is_not_legal_expert_stability():
    arrays,stable,components=example()
    mask=np.arange(84)>=62;segment=(np.arange(84)>=62).astype(np.int64)
    value=report.phase_summary(mask,segment,arrays,stable,components)
    assert value['raw_full_history_stable_phase_endpoints']>0
    assert value['rejected_boundary_or_early_stable_endpoints']>0
    assert value['stable_endpoints']==0
    assert value['first_stable_to_first_later_near_goal']['reason']=='no_same_phase_first_goal_stable_window'


def test_aggregate_keeps_tick_and_episode_denominators_separate():
    arrays,stable,components=example()
    value=report.phase_summary(np.ones(84,bool),np.zeros(84,np.int64),arrays,stable,components)
    result=report.aggregate([{'phase':{'original_expert':value}}]*2,'original_expert')
    assert result['episodes']==result['episodes_with_stable_window']==2
    assert result['phase_ticks']==168
    assert result['events']['native_complete']['ticks']==2
    assert result['first_stable_to_first_later_near_goal']['episodes_leaving_stable_window']==2


def test_contact_height_screen_is_fixed_and_excludes_after_first_near_even_if_flag_clears():
    arrays,stable,components=example()
    arrays['near'][78:]=False;arrays['near'][75]=True
    result=report.phase_summary(np.ones(84,bool),np.zeros(84,np.int64),arrays,stable,components)
    screens=result['prospective_screens']
    assert screens['contact_height12']['first_completed_tick']==60
    assert screens['contact_height12']['window_endpoints']==16  # completed60 through75 only
    assert screens['contact_height12']['additional_false_stable_endpoints']>0
    assert screens['contact_native_lift12']['window_endpoints']==16
    assert result['events_by_preaction_subgoals']['0']['near_goal']['ticks']==1


def test_frozen_loader_pins_both_files_and_explanation_dependency(tmp_path):
    trigger=tmp_path/'visuotactile/takeover_trigger.py'
    explain=tmp_path/'tools/tight_fresh_takeover_pose_audit.py'
    trigger.parent.mkdir();explain.parent.mkdir()
    trigger.write_text('def stable_windows(*args): return "frozen"\n')
    explain.write_text('stable_windows = None\ndef window_components(*args): return stable_windows()\n')
    registered={'source_sha256':{str(path.relative_to(tmp_path)):report.sha256(path)
        for path in (trigger,explain)}}
    predicate,components,bindings=report.frozen_functions(tmp_path,registered)
    assert predicate()==components()=='frozen'
    assert len(bindings)==2
    trigger.write_text('def stable_windows(*args): return "changed"\n')
    with pytest.raises(ValueError,match='Frozen diagnostic dependency changed'):
        report.frozen_functions(tmp_path,registered)
