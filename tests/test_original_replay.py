import pytest,torch
from visuotactile.latest_takeover_sampling import sampling_plan,episode_deck
from visuotactile.rear_episode_sampling import plan

def data():
 ids=['o%d'%i for i in range(30)]+['correction']
 return dict(splits=dict(train=ids,validation=['val']),normalization_train_ids=ids[:30],entries={i:dict(trajectory_kind='expert_takeover_v1' if i=='correction' else 'expert') for i in ids})
def test_replay_front_rear_same_legal_populations():
 d=data();spec=dict(version='original_demonstration_replay_v1',original_repeat=3);_,m=sampling_plan(d,spec)
 assert m['correction']==1 and sum(m.values())==91
 deck=episode_deck(d['splits']['train'],m,torch.Generator().manual_seed(12));assert deck.count('o0')==3 and deck.count('correction')==1
 records={i:dict(supervision_mask=torch.tensor([False,False,True])) for i in m}
 _,n,unique,repeated=plan(d,records,spec);assert n==m and (unique,repeated)==(31,91)
def test_no_relabel_or_heldout_replay():
 d=data();d['splits']['validation']=['o0']
 with pytest.raises(ValueError):sampling_plan(d,dict(version='original_demonstration_replay_v1',original_repeat=3))
def test_default_unchanged():
 assert set(sampling_plan(data(),None)[1].values())=={1}
