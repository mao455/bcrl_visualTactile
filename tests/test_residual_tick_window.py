import copy

import pytest
import torch

from tools import tight_execution_centered as experiment
from tools.runtime import write
from visuotactile.residual_tick_window import (
    ResidualTickWindow, TICK_WINDOW_CONTRACT, validate_tick_window)


def test_accumulation_equals_fixed_tick_weighted_numerator():
    model=torch.nn.Linear(1,1,bias=False).double()
    with torch.no_grad():model.weight.fill_(.05)
    optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    window=ResidualTickWindow(TICK_WINDOW_CONTRACT)
    for count,weight in ((60,1.),(70,3.)):
        values=torch.ones(count,1,dtype=torch.double)
        mean=model(values).square().mean()
        window.backward(mean,optimizer,weight_mass=count*weight,supervised_ticks=count,
            contact_ticks=0,history_ticks=count,weight_report={})
    expected=torch.tensor([[2*.05*(60+210)/128]],dtype=torch.double)
    torch.testing.assert_close(model.weight.grad,expected)
    assert window.ready and window.report['window_chunks']==2
    report=window.step(model,optimizer)
    torch.testing.assert_close(model.weight,torch.tensor([[.05]],dtype=torch.double)-.01*expected)
    assert report['supervised_ticks']==130 and report['loss_weight_mass']==270
    assert not window.pending and not report['group_tail']


def test_short_tail_keeps_fixed_denominator_and_threefold_weight():
    gradients=[]
    for weight in (1.,3.):
        model=torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.fill_(.1)
        optimizer=torch.optim.SGD(model.parameters(),lr=.01)
        window=ResidualTickWindow(TICK_WINDOW_CONTRACT)
        loss=model(torch.ones(1,1)).square().mean()
        window.backward(loss,optimizer,weight_mass=weight,supervised_ticks=1,
            contact_ticks=0,history_ticks=32,weight_report={})
        gradients.append(float(model.weight.grad))
        assert not window.ready
        report=window.step(model,optimizer)
        assert report['group_tail'] and report['fixed_denominator']==128
        with pytest.raises(ValueError,match='empty'):window.step(model,optimizer)
    assert gradients[1]==pytest.approx(3*gradients[0])
    assert gradients[0]==pytest.approx(.2/128)


def test_experiment_changes_only_explicit_optimizer_measure(tmp_path):
    configuration={'residual_pair':{'continuation_from':{'path':'original.pt'},
        'sampling_weights':{'core':.5,'historical_correction':.2,'current_correction':.3}}}
    write(tmp_path/'configuration.json',configuration)
    old=experiment.configuration_for(tmp_path)
    new=experiment.configuration_for(tmp_path,tick_window=True)
    assert new['residual_pair'].pop('tick_window_contract')==TICK_WINDOW_CONTRACT
    assert old==new
    broken=copy.deepcopy(TICK_WINDOW_CONTRACT);broken['target_supervised_ticks']=0
    with pytest.raises(ValueError):validate_tick_window(broken)
    assert validate_tick_window(None) is None
