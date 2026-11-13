"""Random rear initialization must retain original affine and reject stale DP caches."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest
import torch

from visuotactile import anchor_continue, state_resources
from visuotactile.compact import sha256
from visuotactile.fresh_contracts import FRESH_AGGREGATE_CONTRACT
from visuotactile.models.native_dp import TargetResidual
from visuotactile.native_data import supervision_mask_sha256


def test_random_fresh_rear_accepts_original_affine_and_checks_all_cache_bindings(tmp_path,monkeypatch):
    root=tmp_path/'A';data_root=root/'state_data';data_root.mkdir(parents=True)
    nominal=root/'nominal';nominal.mkdir();common=tmp_path/'command';common.mkdir()
    write=lambda path,value:path.write_text(json.dumps(value))
    old=[f't{i}' for i in range(30)];validation=[f'v{i}' for i in range(6)]
    identity='u_expert_fresh_train_only_affine_v1'
    data=dict(visual_stride=1,supervision='q_reached',normalization_identity='q_reached_fresh_train_only_affine_v1',
        normalization_scope='unchanged original training affine; no aggregate refit',normalization_train_ids=old,
        splits=dict(train=old+['new'],validation=validation,test=[]),
        aggregation=dict(version='expert_takeover_aggregate_v1',student_prefix_supervised=False),
        entries={'new':{'trajectory_kind':'expert_takeover_v1'}})
    write(data_root/'data.json',data)
    write(common/'data.json',dict(data,supervision='u_expert',normalization_identity=identity))
    stats=dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29))
    torch.save(stats,common/'statistics.pt');torch.save(dict(stats,action_scale=torch.full((29,),2.)),data_root/'statistics.pt')
    base=tmp_path/'dp.pt';base.write_bytes(b'updated frontend')
    pair=dict(version=TargetResidual.version,fresh_data_contract=FRESH_AGGREGATE_CONTRACT,
        bases={'A':dict(checkpoint=str(base),sha256=sha256(base))},
        residual_statistics_path=str(common/'statistics.pt'),initialization_seed=3)
    write(root/'configuration.json',dict(residual_pair=pair))
    write(root/'session.json',dict(dino_sha256='fixed',resource_root=str(tmp_path),training_deadline=123))
    binding=dict(base_checkpoint_sha256=sha256(base),statistics_sha256=sha256(data_root/'statistics.pt'),
        data_sha256=sha256(data_root/'data.json'),residual_statistics_sha256=sha256(common/'statistics.pt'))
    rows={i:dict(time=torch.arange(3),supervision_mask=torch.tensor([False,True,True]),
        control_segment=torch.tensor([0,1,1]),cache_identity=dict(binding)) for i in old+['new']+validation}
    norm=dict(residual_statistics_sha256=binding['residual_statistics_sha256'],residual_normalization_identity=identity)
    index=dict(norm,base_sha256=sha256(base),data_sha256=binding['data_sha256'],
        supervision_mask_sha256=supervision_mask_sha256(rows),episodes={})
    for key,row in rows.items():
        torch.save(row,nominal/(key+'.pt'));index['episodes'][key]={'sha256':sha256(nominal/(key+'.pt'))}
    write(nominal/'index.json',index);write(nominal/'scales.json',dict(norm,scales=[1.]*29))
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext))
    monkeypatch.setattr(TargetResidual,'cuda',lambda self:self)
    for smoke in [False,True]:
        _,_,sample,records,model,configuration,_=anchor_continue.residual_setup(root,smoke=smoke)
        assert configuration['normalization_train_ids']==old
        assert torch.equal(model.action_scale,stats['action_scale'])
        assert set(records)==set(rows), 'Even a smoke run verifies every cache source'
        assert len(sample['splits']['train'])==(1 if smoke else 31)
    # Rehashing a stale cache must not make it acceptable.
    rows['new']['cache_identity']['base_checkpoint_sha256']='old DP'
    torch.save(rows['new'],nominal/'new.pt');index['episodes']['new']['sha256']=sha256(nominal/'new.pt')
    write(nominal/'index.json',index)
    with pytest.raises(ValueError,match='cache binding'):anchor_continue.residual_setup(root)
    rows['new']['cache_identity']=binding
    rows['new']['supervision_mask'][0]=True
    torch.save(rows['new'],nominal/'new.pt');index['episodes']['new']['sha256']=sha256(nominal/'new.pt')
    write(nominal/'index.json',index)
    with pytest.raises(ValueError,match='mask identity'):anchor_continue.residual_setup(root)
