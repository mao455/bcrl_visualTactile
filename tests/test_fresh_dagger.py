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


def test_aggregate_prepares_live_resource_links_for_both_cache_workers(tmp_path,monkeypatch):
    import json
    from tools.tight_fresh_dagger import prepare_aggregate
    from visuotactile import narrow_data
    root=tmp_path/'initial';root.mkdir();collection=tmp_path/'collection';collection.mkdir()
    output=tmp_path/'round'
    write=lambda path,value:path.write_text(json.dumps(value))
    write(root/'session.json',{'deadline':123});write(root/'metadata.json',{})
    write(root/'shared_resources.json',{'monitor':1});write(root/'gate.json',{'passed':True})
    write(collection/'result.json',{'software_complete':True})
    write(collection/'correction_entries.json',{'new':{}})
    pair=dict(frontend='actual.pt',frontend_sha256='source',backend='rear.pt',backend_sha256='rear')
    write(collection/'plan.json',{'pairs':{'A':pair,'B':pair}})
    def aggregate(source,entries,destination,supervision,**kwargs):
        assert kwargs['include_frontend_tactile'] is False
        destination.mkdir()
        (destination/'statistics.pt').write_bytes(b'preserved original affine')
        original=[f't{i}' for i in range(30)]
        return dict(visual_stride=1,supervision=supervision,normalization_identity=supervision+'_fresh_train_only_affine_v1',
            normalization_scope='unchanged original training affine; no aggregate refit',normalization_train_ids=original,
            splits=dict(train=original+['new'],validation=[f'v{i}' for i in range(6)],test=[]),
            aggregation=dict(version='expert_takeover_aggregate_v1',student_prefix_supervised=False),
            entries={'new':{'trajectory_kind':'expert_takeover_v1'}})
    monkeypatch.setattr(narrow_data,'build_dagger_aggregate',aggregate)
    prepare_aggregate(root,collection,output)
    # The monitor replaces this JSON atomically; workers must follow its live path.
    write(root/'shared_resources.json',{'monitor':2})
    for path in [output,output/'A',output/'B']:
        assert json.loads((path/'shared_resources.json').read_text())=={'monitor':2}
    assert (output/'prepared.json').exists()
