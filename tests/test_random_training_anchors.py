import copy

import pytest
import torch

from visuotactile.anchor_sampling import (sampling_profile, validate_training_anchor_profile,
    sample_training_anchors, planning_anchors, labels, validate_selected_anchors, validate_training_anchors)
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW


def record():
    n=97
    mask=torch.arange(n)>=17
    mask[43:46]=False
    segment=torch.zeros(n,dtype=torch.long)
    segment[17:43]=1;segment[43:46]=2;segment[46:]=3
    return dict(time=torch.arange(n)/60,command=torch.arange(n*29).reshape(n,29).float(),
        clean_sample=-torch.arange(n*29).reshape(n,29).float(),supervision_mask=mask,
        control_segment=segment,target_window_contract=RECOVERY_ONSET_WINDOW)


def test_resample_each_visit_all_legal_ticks_without_replacement_and_no_external_rng_use():
    rec=record();profile=sampling_profile(seed=42);state=torch.get_rng_state().clone()
    selections=[];seeds=[]
    for visit in range(1,6):
        a,receipt=sample_training_anchors(rec,profile,identity='train_only_1',epoch=2,visit_index=visit)
        assert torch.equal(torch.get_rng_state(),state)
        assert len(a)==len(planning_anchors(rec)) and rec['supervision_mask'][a].all()
        assert (a[1:]>a[:-1]).all() and len(a.unique())==len(a)
        assert receipt['selected_anchors']==a.tolist() and receipt['non_native_grid_count']==int((a%12!=0).sum())
        selections.append(a.tolist());seeds.append(receipt['derived_seed'])
    assert len(set(seeds))==5 and len({tuple(a) for a in selections})==5
    assert any(t%12 for a in selections for t in a)
    again,receipt=sample_training_anchors(rec,profile,identity='train_only_1',epoch=2,visit_index=3)
    assert again.tolist()==selections[2] and receipt['derived_seed']==seeds[2]
    for changed in (dict(identity='train_only_2',epoch=2,visit_index=3),dict(identity='train_only_1',epoch=3,visit_index=3)):
        _,other=sample_training_anchors(rec,profile,**changed)
        assert other['derived_seed']!=receipt['derived_seed']


def test_small_legal_population_keeps_singleton_and_no_supervision_rejected():
    rec=dict(time=torch.arange(5)/60,command=torch.ones(5,29),supervision_mask=torch.tensor([False,False,False,False,True]))
    assert len(planning_anchors(rec))==0
    a,r=sample_training_anchors(rec,sampling_profile(),identity='tail',epoch=1,visit_index=1)
    assert a.tolist()==[4] and r['original_commit12_anchor_count']==0
    truth,valid=labels(rec,a,24)
    assert valid.sum()==1 and torch.equal(truth[0,0],rec['command'][4]) and not truth[0,1:].any()
    rec['supervision_mask'].zero_()
    with pytest.raises(ValueError,match='No legal'):sample_training_anchors(rec,sampling_profile(),identity='tail',epoch=1,visit_index=2)


def test_non_grid_labels_retain_reached_or_executed_targets_and_segment_masks():
    rec=record();anchors=torch.tensor([17,39,46,96])
    for reached in (False,True):
        source=copy.deepcopy(rec)
        if not reached:source.pop('clean_sample')
        truth,valid=labels(source,anchors,24)
        assert valid.sum(-1).tolist()==[24,4,24,1]
        expected=source.get('clean_sample',source['command'])
        for row,tick in enumerate(anchors):
            assert torch.equal(truth[row,:int(valid[row].sum())],expected[tick:tick+int(valid[row].sum())])
        assert not truth[~valid].any()
    # Existing onset-grid contract may supervise a future suffix from mask0 tick12;
    # the new explicit method samples only true-mask action starts.
    assert 12 in planning_anchors(rec).tolist()
    with pytest.raises(ValueError,match='true action'):validate_training_anchors(rec,torch.tensor([12,17]))


@pytest.mark.parametrize('bad', [True,-1,2**63,1.5,None])
def test_profile_rejects_ambiguous_seed(bad):
    with pytest.raises(ValueError):sampling_profile(seed=bad)


def test_profile_is_opt_in_and_exact():
    assert validate_training_anchor_profile(None) is None
    p=sampling_profile();assert validate_training_anchor_profile(p)==p
    for key,value in [('replacement',True),('replacement',0),('horizon',24.0),('horizon',64),('count','all'),('extra',1)]:
        with pytest.raises(ValueError):validate_training_anchor_profile(dict(p,**{key:value}))


@pytest.mark.parametrize('bad', [torch.tensor([]),torch.tensor([1.,2.]),torch.tensor([2,1]),
    torch.tensor([1,1]),torch.tensor([-1]),torch.tensor([7]),torch.tensor([[1]]),[1,2]])
def test_explicit_times_reject_bad_type_order_range(bad):
    with pytest.raises(ValueError):validate_selected_anchors(bad,start=0,stop=7,device='cpu')
