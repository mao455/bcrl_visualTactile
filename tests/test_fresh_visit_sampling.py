import torch
import pytest
from visuotactile.fresh_visit_sampling import Coverage,VERSION,validate_profile,visit
from visuotactile.anchor_sampling import sampling_profile
from visuotactile.geometry_sampling import profile

def test_config_and_distinct_exposure_resume():
    value=dict(version=VERSION,anchors=sampling_profile(),geometry=profile(32))
    assert validate_profile(value)==value
    rec=dict(time=torch.arange(72),command=torch.zeros(72,29),supervision_mask=torch.arange(72)>=5,
        control_segment=torch.zeros(72,dtype=torch.long))
    a,ar,g=visit(rec,value,identity='real',epoch=1,visit_index=1)
    b,br,h=visit(rec,value,identity='real',epoch=2,visit_index=1)
    c,cr,k=visit(rec,value,identity='real',epoch=1,visit_index=1)
    assert torch.equal(a,c) and ar==cr and torch.equal(g.get_state(),k.get_state())
    assert not torch.equal(g.get_state(),h.get_state())
    ledger=Coverage();geo=dict(selected_ticks=[dict(identity='real',tick=7,stage='held_far')])
    ledger.add('real',ar,geo);restored=Coverage(ledger.state_dict());restored.add('real',br,geo)
    summary=restored.summary()
    assert summary['visits']==2 and summary['unique_anchor_ticks']==len(set(a.tolist()+b.tolist()))
    assert summary['geometry_draws']==2 and summary['unique_geometry_ticks']==1
    with pytest.raises(ValueError):restored.add('real',ar,dict(selected_ticks=[dict(identity='wrong',tick=7,stage='held_far')]))

def test_cycle_training_profile_cli_and_registration():
    import inspect
    from tools.tight_fresh_cycle import run,main
    assert 'training_sampling' in inspect.signature(run).parameters
    assert '--training-sampling-json' in inspect.getsource(main)


@pytest.mark.parametrize('first_legal',[60,180])
def test_grasp_balanced_visits_keep_legal_history_and_late_coverage(first_legal):
    from visuotactile.anchor_sampling import grasp_sampling_profile
    value=dict(version=VERSION,anchors=grasp_sampling_profile(),geometry=profile(32))
    rec=dict(time=torch.arange(360),command=torch.zeros(360,29),supervision_mask=torch.arange(360)>=first_legal,
        control_segment=(torch.arange(360)>=first_legal).long())
    a,receipt,_=visit(rec,value,identity='real',epoch=1,visit_index=1)
    b,_,_=visit(rec,value,identity='real',epoch=2,visit_index=1)
    same,same_receipt,_=visit(rec,value,identity='real',epoch=1,visit_index=1)
    assert torch.equal(a,same) and receipt==same_receipt and not torch.equal(a,b)
    assert len(a)==len(torch.arange(0,360,12)[torch.arange(0,360,12)>=first_legal])
    assert rec['supervision_mask'][a].all() and len(a.unique())==len(a)
    assert int((a<150).sum())==((len(a)+1)//2 if first_legal<150 else 0)
    assert (a>=150).any() and (a%12!=0).any()
