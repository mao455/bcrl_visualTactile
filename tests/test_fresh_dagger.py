import torch

from tools import tight_fresh_train as trainer
from visuotactile.anchor_sampling import labels


def test_takeover_retains_real_prefix_but_never_supervises_it(monkeypatch):
    n=41
    record=dict(time=torch.arange(n),command=torch.arange(n)[:,None].expand(n,29).float(),
        supervision_mask=torch.arange(n)>=17,control_segment=(torch.arange(n)>=17).long(),
        target_window_contract='recovery_onset_commit12_v1')
    frames=torch.arange(n,dtype=torch.float32,requires_grad=True)
    def encode(model,rec):
        assert rec is record
        anchors=torch.arange(0,n,12)
        return frames.cumsum(0)[anchors,None],anchors
    monkeypatch.setattr(trainer,'encode_dense',encode)
    condition,anchors=trainer.supervised_conditions(None,record)
    assert anchors.tolist()==[12,24,36]
    truth,valid=labels(record,anchors,24)
    indices=anchors[:,None]+torch.arange(24)
    assert not valid[indices<17].any()
    assert valid[0,5] and truth[0,5,0]==17
    condition.sum().backward()
    assert (frames.grad[:17]>0).all(), 'Real student prefix must still influence the supervised condition'


def test_recovery_window_cannot_bridge_a_later_invalid_hole(monkeypatch):
    n=41;mask=torch.arange(n)>=17;mask[27]=False
    record=dict(time=torch.arange(n),command=torch.zeros(n,29),supervision_mask=mask,
        control_segment=torch.where(torch.arange(n)<28,1,2),target_window_contract='recovery_onset_commit12_v1')
    _,valid=labels(record,torch.tensor([12,24]),24)
    assert valid[0,5:15].all() and not valid[0,15:].any()
    assert valid[1,:3].all() and not valid[1,3:].any()
