"""One focused check: causal timing, family isolation, and complete-history exposure."""
import copy
import time
import numpy as np
import pytest
import torch

from test_training_expansion import setup_registry, write
from visuotactile.grasp_curriculum import EARLY_GRASP, EarlyGraspTrigger, holding_metrics
from visuotactile.latest_takeover_sampling import SAMPLING_VERSION, episode_deck
from visuotactile.rear_episode_sampling import plan, exposure
from visuotactile.training_expansion import make_registry, registry_binding, read_registry, validate_collection_plan


def test_approach_trigger_requires_real_near_moving_fingers_and_replays_hdf5(tmp_path):
    import h5py
    from visuotactile.grasp_curriculum import APPROACH
    from visuotactile.takeover_trigger import make_trigger
    from tools.tight_fresh_report import audit_trigger_case
    from tools.tight_fresh_stage_takeover import classify
    pose=np.array([0,0,.5,1,0,0,0.]);zero=np.zeros((2,5,3))
    for close_at in (40,57):
        trigger=make_trigger(APPROACH,10**9);positions=[]
        for tick in range(close_at+2):
            distance=.2 if tick<close_at else .059-(tick-close_at)*.001
            tips=np.tile(pose[:3]+[distance,0,0],(5,1));positions.append(tips)
            result=trigger.observe(tick,pose,pose,zero,finger_position_w=tips)
            assert result==(close_at+2 if tick==close_at+1 else None)
        assert trigger.reason=='approaching_first_contact'
        assert trigger.first_contact is None
        directory=tmp_path/str(close_at);directory.mkdir();count=close_at+4
        with h5py.File(directory/'numeric.h5','w') as f:
            t=f.create_group('episode/transition')
            t['diagnostics_after/object_pose_w']=np.tile(pose,(count,1))
            t['diagnostics_after/palm_body_pose_w']=np.tile(pose,(count,1))
            t['diagnostics_after/finger_position_w']=np.asarray(positions+[positions[-1]]*2)
            t['object_contact_force_w']=np.tile(zero,(count,1,1,1))
            t['dagger/control_source']=np.array([0]*(close_at+2)+[1]*2)
        result=dict(legal=True,software_complete=True,dagger=dict(
            configuration=dict(trigger=APPROACH,takeover_tick=10**9),trigger_receipt=trigger.receipt(),
            recovery_coverage=dict(switch_tick=close_at+2),expert_ticks=2,expert_takeover_success=True))
        assert classify(result,audit_trigger_case(directory,result))=='expert_rescue_after_early_grasp'
    trigger=make_trigger(APPROACH,10**9)
    stationary=np.tile(pose[:3]+[.05,0,0],(5,1))
    for tick in range(80):
        assert trigger.observe(tick,pose,pose,zero,finger_position_w=stationary) is None


def test_contact_formation_trigger_uses_real_persistent_contact_not_clock():
    from visuotactile.grasp_curriculum import CONTACT_ONSET
    from visuotactile.takeover_trigger import make_trigger
    from visuotactile.training_expansion import schedule_trigger
    assert schedule_trigger('grasp_touch') == CONTACT_ONSET
    pose=np.array([0,0,.5,1,0,0,0.]);zero=np.zeros((2,5,3));touch=zero.copy();touch[:,0,2]=.2
    for first_touch in (56,73):
        trigger=make_trigger(CONTACT_ONSET,10**9)
        for tick in range(first_touch+1):
            assert trigger.observe(tick,pose,pose,touch if tick==first_touch else zero) is None
        assert trigger.observe(first_touch+1,pose,pose,touch)==first_touch+2
        assert trigger.reason=='first_persistent_contact_formation'
        assert trigger.receipt()['future_information_used'] is False
    trigger=make_trigger(CONTACT_ONSET,10**9)
    for tick in range(95): assert trigger.observe(tick,pose,pose,zero) is None
    assert trigger.observe(95,pose,pose,zero)==96
    assert trigger.reason=='no_contact_by96'


def test_causal_loss_switch_between_plans_and_no_rotation_veto(tmp_path):
    trigger=EarlyGraspTrigger(EARLY_GRASP,10**9)
    pose=np.array([0,0,.5,1,0,0,0.]);palm=pose.copy();zero=np.zeros((2,5,3))
    contact=zero.copy();contact[:,:2,2]=.2
    for tick in range(70):
        force=contact if 60<=tick<69 else zero
        assert trigger.observe(tick,pose,palm,force) is None
    assert trigger.observe(70,pose,palm,zero)==71 and 71%12!=0
    assert trigger.reason=='recent_two_finger_contact_lost'
    with pytest.raises(ValueError):trigger.observe(71,pose,palm,zero)
    import h5py
    from tools.tight_fresh_report import audit_trigger_case
    from tools.tight_fresh_stage_takeover import classify
    with h5py.File(tmp_path/'numeric.h5','w') as f:
        t=f.create_group('episode/transition')
        t['diagnostics_after/object_pose_w']=np.tile(pose,(73,1))
        t['diagnostics_after/palm_body_pose_w']=np.tile(palm,(73,1))
        forces=np.tile(zero,(73,1,1,1));forces[60:69]=contact
        t['object_contact_force_w']=forces;t['dagger/control_source']=np.array([0]*71+[1]*2)
    result=dict(legal=True,software_complete=True,dagger=dict(
        configuration=dict(trigger=EARLY_GRASP,takeover_tick=10**9),trigger_receipt=trigger.receipt(),
        recovery_coverage=dict(switch_tick=71),expert_ticks=2,expert_takeover_success=True))
    audit=audit_trigger_case(tmp_path,result)
    assert classify(result,audit)=='expert_rescue_after_early_grasp'
    rotation=EarlyGraspTrigger(EARLY_GRASP,10**9)
    for tick in range(200):
        q=pose.copy()
        if tick>=60:q[2]=.53;q[3:]=[np.cos(tick*.04),np.sin(tick*.04),0,0]
        assert rotation.observe(tick,q,palm,contact if tick>=60 else zero) is None


def test_one_second_metric_rejects_brief_lift_and_long_contact_gap():
    p=np.zeros((140,7));p[:,2]=.5;p[:,3]=1
    f=np.zeros((140,2,5,3));f[60:72,:,:2,2]=.2;p[60:72,2]=.53
    assert not holding_metrics(p,f)['sustained']
    p[60:,2]=.53;f[60:,:,:2,2]=.2
    p[60:,3]=np.cos(np.arange(80)*.05);p[60:,4]=np.sin(np.arange(80)*.05)
    assert holding_metrics(p,f)['sustained']
    f[95:101]=0
    assert not holding_metrics(p,f)['sustained']


@pytest.mark.parametrize('schedule',['grasp_early','grasp_touch','grasp_approach'])
def test_revisit_registered_failed_training_family_without_fake_success(setup_registry,monkeypatch,schedule):
    from tools.tight_fresh_expansion import prepare
    s=setup_registry;source=s['root']/'revisit_source'
    data=copy.deepcopy(s['source_data']);data['training_expansion_registries']=[s['binding']]
    manifests={r:write(source/r/'state_data/data.json',data) for r in ('A','B')}
    write(source/'session.json',dict(deadline=time.time()+1000,domain=s['domain'],expert_sha256='fixture'))
    family=next(iter(s['registry']['families']))
    assert family not in {data['entries'][i]['family'] for i in data['splits']['train']}
    monkeypatch.setattr('tools.tight_fresh_expansion.completed_pair',lambda root,role:dict(role=role))
    result=prepare(source,s['root']/'new_attempt',session_root=source,registry_id='revisit',seeds=[6000],
        partition_manifest=s['partition'],frozen_families=s['families'],schedules=[schedule],
        lanes=16,event_seed=8100,round_index=15,revisit_families=[family])
    validate_collection_plan(result)
    assert len(result['cases'])==4 and result['registered_new_training_families']==1
    for c in result['cases']:
        from visuotactile.training_expansion import schedule_trigger
        assert c['dagger']['trigger']==schedule_trigger(schedule) and c['spec']['takeover_tick']==10**9
    with pytest.raises(ValueError,match='prior registered training family'):
        make_registry('bad',s['domain'],[6001],manifests,s['partition'],s['families'],
                      created_utc=time.time(),revisit_families=['validation_0'])


def test_rear_repeat_increases_real_valid_exposure_and_preserves_prefix_masks():
    data=dict(splits=dict(train=['old','new'],validation=['val']),normalization_train_ids=['old'],
        aggregation=dict(version='expert_takeover_aggregate_v1',student_prefix_supervised=False,new_training_ids=['new']),
        entries=dict(old={},new=dict(trajectory_kind='expert_takeover_v1')))
    records=dict(old=dict(supervision_mask=torch.ones(5,dtype=torch.bool)),
                 new=dict(supervision_mask=torch.tensor([False,False,True,True,True])))
    before=records['new']['supervision_mask'].clone()
    spec=dict(version=SAMPLING_VERSION,latest_repeat=4)
    _,counts,unique,repeated=plan(data,records,spec)
    deck=episode_deck(data['splits']['train'],counts,torch.Generator().manual_seed(8))
    assert unique==8 and repeated==17 and deck.count('new')==4 and deck.count('old')==1
    report=exposure(records,counts,{i:deck.count(i) for i in counts})
    assert report['episodes']['new']['supervised_ticks']==12
    assert torch.equal(records['new']['supervision_mask'],before)
