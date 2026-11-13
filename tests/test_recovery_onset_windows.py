from types import SimpleNamespace

import pytest
import torch

from visuotactile.anchor_sampling import AnchorSampler, labels, mask_totals, planning_anchors, supervision_population, weighted_parts
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW, with_target_window
from visuotactile.dp_objectives import masked_objectives


def record(first=62,n=110):
    ticks=torch.arange(n)
    return dict(time=ticks.float()/60,command=torch.randn(n,29),
        supervision_mask=ticks>=first,control_segment=(ticks>=first).long())


@pytest.mark.parametrize('phase',range(12))
def test_every_onset_phase_uses_original_clock_and_only_legal_targets(phase):
    old=record(60+phase);new=with_target_window({'e':old},RECOVERY_ONSET_WINDOW)['e']
    anchors=planning_anchors(new)
    assert anchors[0]==60
    y,m=labels(new,anchors,24)
    coverage=torch.zeros(len(old['time']),dtype=torch.bool)
    for a,row in zip(anchors,m):coverage[a+row.nonzero().flatten()]=True
    assert torch.equal(coverage,old['supervision_mask'])
    assert torch.equal(old['supervision_mask'],new['supervision_mask'])
    _,before=labels(old,torch.tensor([60]),24)
    assert before.any().item()==(phase==0)
    assert not m[0,:phase].any()
    assert torch.equal(y[0,phase:],old['command'][60+phase:84])


def test_no_bridge_from_older_plan_or_across_hole_or_second_segment():
    r=with_target_window({'e':record()},RECOVERY_ONSET_WINDOW)['e']
    r['supervision_mask'][70:74]=False
    r['control_segment'][70:]=2
    _,m=labels(r,torch.tensor([48,60,72,84]),24)
    assert not m[0].any() and not m[2].any()
    assert m[1].nonzero().flatten().tolist()==list(range(2,10))
    assert m[3].all()
    r['supervision_mask'][70:74]=True
    _,m=labels(r,torch.tensor([60]),24)
    assert m[0].nonzero().flatten().tolist()==list(range(2,10))


def test_tail_population_and_geometry_denominator_count_actual_slots():
    r=with_target_window({'e':record(n=80)},RECOVERY_ONSET_WINDOW)
    totals=mask_totals({'e':80},[('e',60),('e',72)],12,records=r)
    assert totals==(26,24,18)  # onset18 + tail8, geometry10 +8, not12+8
    assert supervision_population(r)['supervised_window_targets']==26
    _,m=labels(r['e'],torch.tensor([60,72]),24)
    prediction=torch.randn(2,24,29,requires_grad=True)
    y,_=labels(r['e'],torch.tensor([60,72]),24)
    whole=masked_objectives(prediction,y,m)['B']
    fragments=[]
    for j in range(2):
        value=masked_objectives(prediction[j:j+1],y[j:j+1],m[j:j+1])['B']
        scaled,_=weighted_parts({'base':value},m[j:j+1],totals,{})
        fragments.append(scaled)
    assert torch.allclose(sum(fragments),whole)
    whole.backward()
    assert (prediction.grad[~m]==0).all()
    assert (prediction.grad[m].abs().sum(-1)>0).all()


def test_new_onset_is_recovery_pool_and_old_sampler_cannot_resume():
    old={'core':record(0),'fix':record()};new=with_target_window(old,RECOVERY_ONSET_WINDOW)
    kwargs=dict(strata={'core':['core'],'historical_correction':['fix'],'current_correction':[]},
        recovery_prefix=dict(ticks=48,frontend_fraction=.5,backend_weight=3.))
    before=AnchorSampler({'core':110,'fix':110},torch.Generator(),records=old,**kwargs)
    after=AnchorSampler({'core':110,'fix':110},torch.Generator(),records=new,**kwargs)
    assert after.anchors['fix'][0]==60 and 0 in after.recovery_pools['fix']['initial']
    assert after.target_counts['fix'][0]==22
    assert after.recovery_target_counts['fix'][0]==22
    with pytest.raises(ValueError,match='population changed'):after.load_state_dict(before.state_dict())


def test_geometry_selection_never_chooses_transition_slots():
    from visuotactile.anchor_continue import geometry_selection
    records=with_target_window({'e':record()},RECOVERY_ONSET_WINDOW)
    objective=SimpleNamespace(supervision_records=records,loss_context={'e':dict(stage=torch.zeros(110,dtype=torch.long),contact_mask=torch.zeros(110,5,dtype=torch.bool))})
    mask,report=geometry_selection(objective,[('e',60)],{'e':110},42)
    assert not mask[0,:2].any() and not mask[0,12:].any()
    assert report['eligible_commit_count']==10 and int(mask.sum())==2


def test_default_core_and_unknown_contract():
    r=record(0)
    plain=labels(r,torch.tensor([0,96]),24)
    explicit=labels(with_target_window({'e':r},RECOVERY_ONSET_WINDOW)['e'],torch.tensor([0,96]),24)
    assert all(torch.equal(a,b) for a,b in zip(plain,explicit))
    with pytest.raises(ValueError,match='Unknown'):with_target_window({'e':r},'bad')
