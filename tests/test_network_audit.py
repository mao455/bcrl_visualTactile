import torch
import pytest

from visuotactile.audit_network import intervene, parameter_rows
from visuotactile.audit_training_initials import first_difference


def test_force_interventions_preserve_nonforce_fields_and_past_support():
    tactile=torch.arange(4*2*5*20).reshape(4,2,5,20).float()
    recent=torch.arange(2*26*5*20).reshape(2,26,5,20).float()
    recent[...,-1]=1
    recent[:,0:4,:,-1]=0
    inputs=dict(tactile=tactile,recent_tactile=recent,base=torch.randn(4,29),plan_valid=torch.ones(2,64,dtype=torch.bool))
    for mode in ('zero','reverse_past'):
        result=intervene(inputs,mode)
        for key in ('base','plan_valid'):
            assert torch.equal(result[key],inputs[key])
        for key in ('tactile','recent_tactile'):
            assert torch.equal(result[key][...,6:],inputs[key][...,6:])
    reversed_input=intervene(inputs,'reverse_past')
    assert torch.equal(reversed_input['recent_tactile'][:,4:,:,:6],recent[:,4:,:,:6].flip(1))
    assert torch.equal(inputs['tactile'],tactile)


def test_parameter_audit_rejects_partial_optimizer_scope():
    model=torch.nn.Linear(3,2)
    optimizer=torch.optim.AdamW([model.weight])
    with pytest.raises(ValueError,match='tensor count'):
        parameter_rows(model,'probe',optimizer=optimizer.state_dict())


def test_first_deviation_uses_time_axis_and_tolerance():
    actual=[[0.,0.],[1e-8,0.],[0.,.02],[.4,0.]]
    expected=torch.zeros(4,2).numpy()
    assert first_difference(actual,expected,1e-3)==2
    assert first_difference(expected,expected,0.) is None
