from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visuotactile import geometry_sampling as sampling
from visuotactile.anchor_continue import geometry_selection, projection_batch
from visuotactile.state_image_loss import GeometryObjective, HandGeometrySoftLoss, geometry_loss_context


def fixture(n=120, first=0, phase=None):
    ticks=torch.arange(n)
    record=dict(command=torch.zeros(n,29),time=ticks/60,supervision_mask=ticks>=first,
        control_segment=(ticks>=first).long(),target_window_contract='recovery_onset_commit12_v1')
    context=dict(stage=torch.ones(n,dtype=torch.long),contact_mask=torch.ones(n,5,dtype=torch.bool),
        object_pose_base=torch.zeros(n,7),sampling_phase=torch.zeros(n,dtype=torch.long) if phase is None else phase)
    obj=object.__new__(GeometryObjective);obj.supervision_records={'episode':record};obj.loss_context={'episode':context}
    obj.gain=1.;obj.weights={'hand_geometry':1.};obj.hand_geometry=object.__new__(HandGeometrySoftLoss)
    obj.hand_geometry.per_configuration=lambda q,**kw:(q[:,7:].square().sum(-1),dict(attempted=len(q),rows=[]))
    return obj


def chosen(obj,anchors,seed=7,configuration=None):
    return geometry_selection(obj,[('episode',int(t)) for t in anchors],
        {'episode':len(obj.supervision_records['episode']['time'])},generator=torch.Generator().manual_seed(seed),
        sampling_profile=configuration or sampling.profile())


def test_half_sample_proportional_phases_changes_each_visit_without_global_rng():
    obj=fixture(100,phase=torch.tensor([0]*60+[1]*30+[2]*10));anchors=list(range(0,100,12))
    state=torch.get_rng_state().clone()
    a,report=chosen(obj,anchors);b,repeated=chosen(obj,anchors);c,other=chosen(obj,anchors,seed=8)
    assert torch.equal(a,b) and report==repeated and not torch.equal(a,c)
    assert torch.equal(torch.get_rng_state(),state)
    assert report['selected_count']==50 and report['eligible_unique_commit_count']==100
    assert {k:v['selected'] for k,v in report['phase_counts'].items()}=={
        'grasp_without_contact':30,'contact_not_held':15,'held_far':5}
    assert all(v['fraction']==.5 for v in report['phase_counts'].values())
    assert len({(r['identity'],r['tick']) for r in report['selected_ticks']})==50
    assert len({r['tick'] for r in report['selected_ticks']}|{r['tick'] for r in other['selected_ticks']})>50


def test_population_is_selected_commit_ticks_not_episode_or_overlapping_h24_slots():
    obj=fixture(120,first=24)
    mask,row=chosen(obj,[24,30,24])
    assert row['episode_legal_tick_count']==96
    assert row['eligible_commit_count']==36 and row['eligible_unique_commit_count']==18
    assert row['duplicate_commit_slots_removed']==18 and row['selected_count']==9
    assert not mask[:,12:].any() and not mask[2].any()
    assert all(24<=r['tick']<42 for r in row['selected_ticks'])
    # Student-prefix actions and an incomplete final target window are excluded.
    mask,row=chosen(obj,[12,24,114])
    assert not mask[0].any() and not mask[-1,6:].any()
    assert row['eligible_unique_commit_count']==18


def test_cap_and_rare_phase_rates_are_explicit_without_extra_loss_weights():
    obj=fixture(100,phase=torch.tensor([0]*98+[1,2]));anchors=list(range(0,100,12))
    _,row=chosen(obj,anchors,configuration=sampling.profile(16))
    assert row['selected_count']==16 and row['phase_counts']['grasp_without_contact']['selected']==14
    assert row['phase_counts']['contact_not_held']['fraction']==1.
    assert row['phase_counts']['held_far']['fraction']==1.
    assert not row['population_unbiased']
    with pytest.raises(ValueError,match='cover all'):chosen(obj,anchors,configuration=sampling.profile(2))
    for value in [True,0,-1,1.5]:
        with pytest.raises(ValueError):sampling.profile(value)
    with pytest.raises(ValueError):sampling.validate_profile({})


def test_projection_exposes_actual_receipt_and_forward_never_resamples(monkeypatch):
    obj=fixture(60);obj.geometry_sampling_profile=sampling.profile()
    with pytest.raises(ValueError,match='generator'):projection_batch(obj,[('episode',0)],'cpu')
    context=projection_batch(obj,[('episode',0),('episode',12)],'cpu',
        geometry_generator=torch.Generator().manual_seed(93))
    assert context['update_selected_count']==12 and len(context['selection_report']['selected_ticks'])==12
    monkeypatch.setattr(sampling,'select',lambda *a,**k:pytest.fail('Forward must never resample'))
    prediction=torch.ones(2,24,29,requires_grad=True);valid=torch.zeros(2,24,dtype=torch.bool);valid[:,:12]=True
    loss=obj._soft_hand_component(prediction,valid,torch.ones(29),torch.zeros(29),context)
    assert loss==22.  # Mean over selected configurations, not their sum.
    grad=torch.autograd.grad(loss,prediction)[0]
    assert grad[~context['selected_mask']].abs().sum()==0 and grad[:,:,:7].abs().sum()==0
    assert torch.allclose(grad[context['selected_mask']][:,7:],torch.full((12,22),2/12))
    assert obj.last_geometry_metrics['selection_report']==context['selection_report']


def test_half_mean_and_gradient_are_microbatch_additive_and_old_limit_stays():
    obj=fixture(60);configuration=sampling.profile()
    context=projection_batch(obj,[('episode',0),('episode',12)],'cpu',sampling_profile=configuration,
        geometry_generator=torch.Generator().manual_seed(7))
    prediction=torch.arange(2*24*29,dtype=torch.float32).reshape(2,24,29).requires_grad_()/100
    valid=torch.zeros(2,24,dtype=torch.bool);valid[:,:12]=True
    full=obj._soft_hand_component(prediction,valid,torch.ones(29),torch.zeros(29),context)
    parts=[]
    for row in range(2):
        ctx={k:v[row:row+1] if torch.is_tensor(v) else v for k,v in context.items()}
        parts.append(obj._soft_hand_component(prediction[row:row+1],valid[row:row+1],torch.ones(29),torch.zeros(29),ctx))
    torch.testing.assert_close(sum(parts),full)
    torch.testing.assert_close(torch.autograd.grad(full,prediction,retain_graph=True)[0],torch.autograd.grad(sum(parts),prediction)[0])
    legacy={k:v for k,v in context.items() if k not in ('sampling_profile','selection_report')}
    with pytest.raises(ValueError,match='One/two'):obj._soft_hand_component(prediction,valid,torch.ones(29),torch.zeros(29),legacy)


def test_explicit_locations_reject_duplicate_actual_tick_and_illegal_prefix():
    obj=fixture(60,first=12);obj.geometry_sampling_profile=sampling.profile()
    with pytest.raises(ValueError,match='illegal'):
        projection_batch(obj,[('episode',0)],'cpu',selected_indices=[(0,1)])
    with pytest.raises(ValueError,match='duplicated real'):
        projection_batch(obj,[('episode',12),('episode',12)],'cpu',selected_indices=[(0,1),(1,1)])
    context=projection_batch(obj,[('episode',12)],'cpu',selected_indices=[(0,1),(0,2),(0,3)])
    assert context['selection_report']['selected_ticks']==[
        dict(identity='episode',tick=t,stage='grasp_without_contact') for t in (13,14,15)]


def diagnostics():
    n=85;pose=np.zeros((n,7));pose[:,3]=1;pose[:,2]=.04;pose[48:,2]=.065
    palm=np.zeros_like(pose);palm[:,3]=1
    force=np.zeros((n,2,5,3));force[10:,:,0:2,2]=1.;force[73:75]=0.
    goal=np.zeros_like(pose);goal[:,3]=1
    successes=np.zeros(n);successes[80:]=1
    angle=np.linspace(0,.4,12);pose[60:72,3]=np.cos(angle/2);pose[60:72,6]=np.sin(angle/2)
    keypoint=np.ones(n)*.2;keypoint[72]=.04
    return pose,palm,force,goal,goal.copy(),successes,successes.copy(),keypoint


def test_phases_distinguish_contact_held_rotation_near_goal_and_contact_loss():
    values=diagnostics();phase=sampling.diagnostic_phases(*values)
    assert phase[0]==0 and phase[30]==1 and phase[59]==2 and phase[71]==3
    assert phase[72]==4 and phase[73]==6 and phase[81]==5
    changed=list(values);changed[4]=changed[4].copy();changed[4][72,0]=1
    assert sampling.diagnostic_phases(*changed)[72]!=4


def test_context_phases_opt_in_never_change_old_geometry_fields(tmp_path):
    import h5py
    pose,palm,force,bg,ag,bs,ass,kp=diagnostics();n=len(pose);path=tmp_path/'example.h5'
    with h5py.File(path,'w') as stream:
        g=stream.create_group('episodes/train');tr=g.create_group('transition')
        base=np.zeros_like(pose);base[:,3]=1
        for where,value in [('diagnostics_after/base_pose_w',base),('diagnostics_after/object_pose_w',pose),
                ('diagnostics_after/palm_body_pose_w',palm),('diagnostics_before/goal_pose_w',bg),
                ('diagnostics_after/goal_pose_w',ag),('diagnostics_before/successes',bs),('diagnostics_after/successes',ass),
                ('object_contact_force_w',force),('native_keypoints_max_dist',kp),
                ('time_before_s',np.arange(n)/60),('time_after_s',(np.arange(n)+1)/60),
                ('contact_time_after_s',(np.arange(n)[:,None]+np.array([.5,1]))/60)]:tr.create_dataset(where,data=value)
        g.create_dataset('terminated',data=np.arange(n)==n-1);g.create_dataset('truncated',data=np.zeros(n,bool))
    entry=dict(path=str(path),episode='train')
    old=geometry_loss_context(entry);new=geometry_loss_context(entry,sampling_profile=sampling.profile())
    assert set(new)==set(old)|{'sampling_phase'}
    for key in old:assert torch.equal(old[key],new[key])
    assert torch.equal(new['sampling_phase'],sampling.diagnostic_phases(*diagnostics()))
