import copy
import torch

from visuotactile.anchor_sampling import AnchorSampler, labels, mask_totals
from visuotactile.narrow_data import inside, near, angle, admission_conflict
from visuotactile.recording import post_observation


def test_direct_relative_domain_preserves_variation_without_wide_reset_rejection():
    import numpy as np
    from visuotactile.narrow_data import direct_configuration
    center=dict(object_xyz=[-.035542,-.009880,.616962],relative_xyz=[.067669,-.024071,-.082286],
        quaternion=np.array([.153762,.896514,-.370529,.187958]))
    center['quaternion']/=np.linalg.norm(center['quaternion'])
    domain=dict(center=center,object_xy_halfwidth=[.025,.025],relative_xy_halfwidth=[.035,.035],
        rotation_radius_degrees=20.,native_joint_lerp=.03)
    rows=[direct_configuration(domain,25110000+i) for i in range(80)]
    for i,row in enumerate(rows):
        assert row==direct_configuration(domain,25110000+i)
        obj=np.array(row['fixed_start_pose']);hole=np.array([row['hole_x_range'][0],row['hole_y_range'][0]])
        assert np.all(abs(obj[:2]-center['object_xyz'][:2])<=.025)
        assert np.all(abs(hole-obj[:2]-center['relative_xyz'][:2])<=.035)
        assert abs(hole[0])<=.1875 and abs(hole[1])<=.1
        assert angle(obj[3:],center['quaternion'])<=20.00001
    assert np.ptp([r['table_reset_z'] for r in rows])>.007
    assert np.ptp([r['fixed_start_pose'][0] for r in rows])>.045


def test_live_collision_bounds_reject_overlap_and_allow_separation_under_rotation():
    import numpy as np
    from visuotactile.live_collision import separation
    from visuotactile.initial_geometry import rotation
    a=(np.zeros(3),np.eye(3),np.array([.125,.015,.01]))
    assert separation(a,a)<0
    b=(np.array([.2,0,0]),np.eye(3),np.array([.02,.02,.02]))
    assert abs(separation(a,b)-.055)<1e-10
    r=rotation(np.array([np.cos(.4),0,0,np.sin(.4)]))
    assert abs(separation((r@a[0],r@a[1],a[2]),(r@b[0],r@b[1],b[2]))-.055)<1e-10


def test_nondivisible_deck_visits_each_episode_without_replication_or_drop():
    sampler=AnchorSampler({str(i):25+i for i in range(15)},torch.Generator().manual_seed(5),8)
    first,tail=sampler.draw(),sampler.draw()
    assert [len(first),len(tail)]==[8,7]
    assert len(set(i for i,_ in first+tail))==15
    state=copy.deepcopy(sampler.state_dict());rng=sampler.generator.get_state()
    resumed=AnchorSampler(sampler.lengths,torch.Generator(),8)
    resumed.load_state_dict(state);resumed.generator.set_state(rng)
    assert resumed.draw()==sampler.draw()


def test_short_cohort_and_padding_never_create_targets():
    sampler=AnchorSampler({'one':13,'two':29},torch.Generator().manual_seed(1),64)
    assert len(sampler.draw())==2
    rec=dict(time=torch.arange(13),command=torch.arange(13*29).reshape(13,29))
    y,m=labels(rec,torch.tensor([12]),24)
    assert m.sum()==1 and torch.equal(y[0,0],rec['command'][12])
    assert torch.count_nonzero(y[0,1:])==0
    assert mask_totals({'one':13},[('one',12)])==(1,0)


def test_rotation_sign_wrap_and_relative_geometry():
    row=dict(object_xyz=[0,0,.63],relative_xyz=[.05,0,-.1],hole_xyz=[.05,0,.53],
        quaternion=[1,0,0,0],arm=[0]*7)
    domain=dict(center=row,object_xy_halfwidth=[.08,.08],relative_xy_halfwidth=[.12,.1],
        rotation_radius_degrees=75,arm_rms_radius_rad=.22)
    flipped=dict(row,quaternion=[-1,0,0,0])
    assert inside(flipped,domain) and near(row,flipped) and angle(row['quaternion'],flipped['quaternion'])==0
    assert not inside(dict(row,relative_xyz=[.3,0,-.1]),domain)
    assert not inside(dict(row,quaternion=[0,1,0,0]),domain)


def test_terminal_post_frame_mask_cannot_import_reset_images():
    packet={'image':torch.ones(2,3,4,2,2),'depth_valid':torch.ones(2,3,1,2,2,dtype=torch.bool),
        'step_id':torch.tensor([17,0]),'episode_id':torch.tensor([2,9])}
    post=post_observation(packet,torch.tensor([False,True]))
    assert post['available'].tolist()==[True,False]
    assert torch.equal(post['image'][0],packet['image'][0]) and post['image'][1].sum()==0
    assert not post['depth_valid'][1].any()


def test_real_reset_query_contradiction_is_not_a_task_failure():
    import json
    from pathlib import Path
    root=Path('outputs/visuotactile/run/narrow_state_20261105_01/episodes')
    family='r931005_1793209376_e01_p000144'
    if not root.exists():
        import pytest
        pytest.skip('Local physical evidence is intentionally not in Git')
    first=json.loads((root/(family+'_native')/'initial_observable.json').read_text())
    second=json.loads((root/(family+'_cal058')/'initial_observable.json').read_text())
    report=admission_conflict(first,second,True,False)
    assert report['conflict'] and max(report['observable_max_abs'].values())==0
    assert not admission_conflict(first,second,True,True)['conflict']
