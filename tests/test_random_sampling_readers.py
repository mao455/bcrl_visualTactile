"""Actual random starts survive both receipt formats and stage reconstruction."""
import copy

import pytest
import torch

from tools.tight_fresh_learning_report import sampled_visit_coverage,exposure_rows
from tools.tight_fresh_stage_exposure import sampled_stage_totals
from visuotactile.anchor_sampling import labels,sampling_profile,planning_anchors
from visuotactile.fresh_visit_sampling import visit,VERSION,Coverage
from visuotactile.geometry_sampling import profile as geometry_profile
from visuotactile.trajectory_training import profile as batch_profile
from visuotactile.training_receipts import iter_visits,BATCH_ROW_VERSION


def fixture_rows():
    n=37;record=dict(time=torch.arange(n)/60,command=torch.zeros(n,29),
        supervision_mask=torch.arange(n)>=3,control_segment=(torch.arange(n)>=3).long())
    profile=dict(version=VERSION,anchors=sampling_profile(42),geometry=geometry_profile())
    visits=[];coverage=Coverage()
    for index in (1,2):
        anchors,receipt,_=visit(record,profile,identity='train',epoch=1,visit_index=index)
        values=dict(episode='train',episode_index=index,anchors=len(anchors),target_ticks=int(labels(record,anchors,24)[1].sum()),
            visual_frames=n,anchor_sampling=receipt,geometry_sampling=None)
        visits.append(values);coverage.add('train',receipt,None)
    single=[];a=t=0
    for index,value in enumerate(visits,1):
        a+=value['anchors'];t+=value['target_ticks']
        single.append(dict(epoch=1,updates=index,episode='train',episode_index=index,visual_frames=n,
            anchors_seen=a,target_ticks=t,anchor_sampling=value['anchor_sampling'],geometry_sampling=None))
    batch=[dict(epoch=1,version=BATCH_ROW_VERSION,full_trajectory_batch=batch_profile(),updates=1,
        anchors_seen=a,target_ticks=t,visual_frames=2*n,visits_seen=2,
        visits=[dict(v,loss_weight=.5) for v in visits])]
    return record,profile,single,batch,coverage


def test_single_and_b2_reconstruct_actual_random_slots_and_coverage_without_grid_assumption():
    record,profile,single,batch,coverage=fixture_rows()
    mask=record['supervision_mask'];stage=torch.stack([mask&(torch.arange(len(mask))>=25),mask])
    results=[]
    for rows in (single,batch):
        normalized=list(iter_visits(rows))
        assert [v['anchor_sampling']['selected_anchors'] for v in normalized]==[
            v['anchor_sampling']['selected_anchors'] for v in iter_visits(single)]
        assert sampled_visit_coverage(rows,profile).state_dict()==coverage.state_dict()
        result=sampled_stage_totals({'train':record},{'train':stage},['late','legal_expert'],rows,len(rows),profile,labels)
        results.append(result)
        counts,buckets=exposure_rows(rows,{'train'},{'train'})
        assert counts=={'train':2} and buckets['correction']['target_ticks']==result['train']['target_slots']
    assert results[0]==results[1]
    old_slots=2*int(labels(record,planning_anchors(record),24)[1].sum())
    assert results[0]['train']['target_slots']!=old_slots
    first=sampled_stage_totals({'train':record},{'train':stage},['late','legal_expert'],single,1,profile,labels)
    assert first['train']['visits']==1


def test_random_receipts_cannot_silently_be_read_as_grid_or_bind_a_different_visit():
    _,profile,single,_,_=fixture_rows()
    with pytest.raises(ValueError,match='registered sampling profile'):sampled_visit_coverage(single,None)
    changed=copy.deepcopy(single);changed[0]['anchor_sampling']['visit_index']=2
    with pytest.raises(ValueError,match='training visit'):sampled_visit_coverage(changed,profile)


def test_root_random_frontend_and_real_b2_update_use_selected_times(monkeypatch):
    from test_full_trajectory_training import TinyNative,Objective,make_records,install_small_vision
    from tools import tight_fresh_train as trainer
    from visuotactile import trajectory_batch,continuous_vision,anchor_continue,trajectory_training
    from visuotactile.models.vendor.ema_model import EMAModel
    install_small_vision(monkeypatch)
    monkeypatch.setattr(continuous_vision,'frame_features',trajectory_batch.frame_features)
    monkeypatch.setattr(continuous_vision,'causal_visual_readouts',lambda vision,tokens,time,ct,cv,a,**kw:tokens[a])
    monkeypatch.setattr(anchor_continue,'projection_batch',lambda objective,selected,device,**kw:dict(selected=selected))
    records,episodes=make_records();_,profile,_,_,_=fixture_rows()
    model=TinyNative();opt=torch.optim.AdamW(model.parameters(),lr=1e-3);ema=EMAModel(copy.deepcopy(model),power=.75)
    selected=[visit(records[i],profile,identity=i,epoch=1,visit_index=j) for j,i in enumerate(('x','y'),1)]
    conditions,anchors=trainer.supervised_conditions(model,records['x'],episode=episodes['x'],selected_anchors=selected[0][0])
    assert torch.equal(anchors,selected[0][0]) and len(conditions)==len(anchors)
    result=trajectory_training.execute_update(model,opt,ema,Objective(),['x','y'],[records['x'],records['y']],
        [episodes['x'],episodes['y']],torch.Generator().manual_seed(99),
        selected_anchors=[v[0] for v in selected],anchor_receipts=[v[1] for v in selected],geometry_generators=[v[2] for v in selected])
    assert ema.optimization_step==1 and all(int(s['step'])==1 for s in opt.state.values())
    assert [r['anchor_sampling'] for r in result['visits']]==[v[1] for v in selected]
    assert [r['target_ticks'] for r in result['visits']]==[int(labels(records[i],v[0],24)[1].sum()) for i,v in zip(('x','y'),selected)]
