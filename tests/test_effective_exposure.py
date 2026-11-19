import torch
import pytest
from visuotactile.effective_exposure import Ledger, cover_anchors, dp_ticks
from visuotactile.anchor_sampling import coverage_sampling_profile, sample_training_anchors


def record():
    mask=torch.ones(91,dtype=torch.bool);mask[:7]=False;mask[36:40]=False
    return dict(time=torch.arange(91)/60,supervision_mask=mask,
        control_segment=torch.tensor([0]*40+[1]*51),command=torch.zeros(91,29))


def test_every_legal_tick_covered_without_crossing_mask_or_segment():
    r=record()
    for epoch in range(1,5):
        a,receipt=sample_training_anchors(r,coverage_sampling_profile(),identity='x',epoch=epoch,visit_index=1)
        assert set(dp_ticks(r,a))==set(r['supervision_mask'].nonzero().flatten().tolist())
        assert set(receipt['original_random_anchors'])<=set(a.tolist())
        assert r['supervision_mask'][a].all()


def test_step_dedup_and_mask_rejection():
    l=Ledger({'x':[False,True,True]})
    l.step(1,[('x',[1,1,2]),('x',[1,2])])
    assert l.counts['x'].tolist()==[0,1,1]
    with pytest.raises(ValueError):l.step(1,[('x',[1])])
    with pytest.raises(ValueError):l.step(2,[('x',[0])])


def test_selected_state_has_own_fifty_steps_and_restores_exactly():
    l=Ledger({'x':[True,True],'y':[True]})
    for step in range(1,50):l.step(step,[('x',[0,1]),('y',[0])])
    assert not l.summary({'x':50,'y':50})['eligible']
    state=l.state_dict()
    restored=Ledger(l.legal,state)
    restored.step(50,[('x',[0,1]),('y',[0])])
    assert restored.summary({'x':50,'y':50})['eligible']
    assert not restored.summary({'x':49,'y':50})['eligible']
    assert not l.summary({'x':50,'y':50})['eligible']
