import torch
import pytest
from visuotactile.anchor_sampling import (native_entry_sampling_profile,coverage_sampling_profile,
    sample_training_anchors,labels,validate_training_anchors)
from visuotactile.entry_exposure import EntryLedger
from visuotactile.fresh_visit_sampling import visit
from visuotactile.geometry_sampling import profile as geometry_profile
from visuotactile.dp_objectives import masked_objectives


def record(first=110,length=137):
    mask=torch.arange(length)>=first
    return dict(time=torch.arange(length)/60,supervision_mask=mask,command=torch.zeros(length,29),
        control_segment=mask.long(),clean_sample=torch.arange(length*29).reshape(length,29).float(),
        target_window_contract='recovery_onset_commit12_v1')


@pytest.mark.parametrize('first,length,anchor,slots',[(110,137,108,list(range(2,12))),
    (108,137,108,list(range(12))),(110,111,108,[2])])
def test_entry_selected_and_only_legal_committed_slots_train(first,length,anchor,slots):
    r=record(first,length)
    a,receipt=sample_training_anchors(r,native_entry_sampling_profile(),identity='x',epoch=1,visit_index=1)
    assert anchor in a.tolist()
    old,old_receipt=sample_training_anchors(r,coverage_sampling_profile(),identity='x',epoch=1,visit_index=1)
    assert receipt['original_coverage_anchors']==old.tolist()
    assert receipt['original_random_anchors']==old_receipt['original_random_anchors']
    validate_training_anchors(r,torch.tensor([anchor]))
    y,m=labels(r,torch.tensor([anchor]),24)
    assert m[0,:12].nonzero().flatten().tolist()==slots
    assert torch.equal(y[0,first-anchor],r['clean_sample'][first])
    p=torch.zeros_like(y,requires_grad=True)
    masked_objectives(p,y,m)['B'].backward()
    assert not p.grad[~m].any()
    assert p.grad[m].abs().sum()>0
    if first==110:assert not p.grad[0,:2].any()
    with pytest.raises(ValueError):validate_training_anchors(r,torch.tensor([anchor-1]))


def test_entry_ledger_deduplicates_step_does_not_count_other_anchors_and_restores():
    records={'x':record()};ledger=EntryLedger(records)
    ledger.step(1,ledger.contributions(records,[('x',[110,111])]))
    assert ledger.summary()['minimum_entry_steps']==0
    for step in range(2,52):
        c=ledger.contributions(records,[('x',[108]),('x',[108,110])])
        ledger.step(step,c)
    assert ledger.summary()['minimum_entry_steps']==50
    assert ledger.summary()['eligible']
    assert EntryLedger(records,ledger.state_dict()).state_dict()==ledger.state_dict()
    with pytest.raises(ValueError):ledger.step(52,[('x',[108])])
    assert ledger.updates==51


def test_geometry_rng_keeps_original_coverage_seed_and_original_data_sampler_unchanged():
    for first in (0,110):
        r=record(first)
        common=dict(version='fresh_random_visit_training_v1',geometry=geometry_profile())
        old=dict(common,anchors=coverage_sampling_profile());new=dict(common,anchors=native_entry_sampling_profile())
        a,ar,g=visit(r,old,identity='x',epoch=3,visit_index=2)
        b,br,h=visit(r,new,identity='x',epoch=3,visit_index=2)
        assert torch.equal(g.get_state(),h.get_state())
        if first==0:assert torch.equal(a,b)


def test_production_b2_update_reads_108_history_and_masks_its_first_two_slots(monkeypatch):
    import copy
    from types import SimpleNamespace
    from test_full_trajectory_training import TinyNative,install_small_vision
    from visuotactile.trajectory_training import execute_update
    from visuotactile.models.vendor.ema_model import EMAModel
    install_small_vision(monkeypatch)
    from visuotactile import anchor_continue
    monkeypatch.setattr(anchor_continue,'projection_batch',lambda *args,**kwargs:{})
    r=record();n=len(r['time']);r['robot']=torch.zeros(n,178)
    ep=SimpleNamespace(visual_stride=1,record=dict(camera_time=r['time'][:,None].expand(-1,3),
        camera_valid=torch.ones(n,3,dtype=torch.bool)),perception=dict(patches=torch.zeros(n,3),depth=torch.zeros(n,1)))
    a,receipt=sample_training_anchors(r,native_entry_sampling_profile(),identity='x',epoch=1,visit_index=1)
    model=TinyNative();opt=torch.optim.AdamW(model.parameters(),lr=1e-4);ema=EMAModel(copy.deepcopy(model),power=.75)
    class Objective:
        gain=1.;weights={};last_geometry_metrics=None
        def components(self,prediction,truth,valid,scale,offset,context):
            self.prediction=prediction;prediction.retain_grad();self.valid=valid;self.truth=truth
            return dict(base=(prediction-truth)[valid].square().mean())
    obj=Objective()
    result=execute_update(model,opt,ema,obj,['x'],[r],[ep],torch.Generator().manual_seed(3),
        selected_anchors=[a],anchor_receipts=[receipt],geometry_generators=[torch.Generator().manual_seed(4)])
    row=a.tolist().index(108)
    assert not obj.valid[row,:2].any() and obj.valid[row,2]
    assert torch.equal(obj.truth[row,2],r['clean_sample'][110])
    assert not obj.prediction.grad[row,:2].any() and obj.prediction.grad[row,2].abs().sum()>0
    assert result['visits'][0]['anchor_sampling']['native_entry_anchor']==108
    assert result['visits'][0]['visual_frames']==137 and ema.optimization_step==1
    ledger=EntryLedger({'x':r});ledger.step(1,ledger.contributions({'x':r},[('x',a)]))
    assert all(row['optimizer_steps']==1 for row in ledger.rows())
