import copy
import torch

from visuotactile.anchor_sampling import AnchorSampler, labels, mask_totals
from visuotactile.narrow_data import inside, near, angle
from visuotactile.recording import post_observation


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
