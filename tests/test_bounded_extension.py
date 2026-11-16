import copy
import pytest
from visuotactile.bounded_extension import Plateau,profile,split_audit

class Optimizer:
    def __init__(self):self.param_groups=[dict(lr=3e-5)]

def test_plateau_reduces_then_stops_only_after_minimum():
    opt=Optimizer();p=Plateau(profile(),baseline=1.)
    for epoch in range(9,13):p.observe(epoch,1.001,opt)
    assert opt.param_groups[0]['lr']==1.5e-5 and not p.state['stop']
    for epoch in range(13,16):p.observe(epoch,1.,opt)
    assert not p.state['stop']
    p.observe(16,1.,opt);assert p.state['stop']

def test_small_noise_does_not_reset_patience_and_real_improvement_does():
    opt=Optimizer();p=Plateau(profile(),baseline=1.)
    p.observe(9,.999,opt);assert p.state['bad']==1
    p.observe(10,.99,opt);assert p.state['bad']==0
    p.observe(11,1.,opt)
    q=Plateau(profile(),copy.deepcopy(p.state))
    assert p.observe(12,.99,opt)==q.observe(12,.99,Optimizer())

def test_nonfinite_validation_fails():
    with pytest.raises(FloatingPointError):Plateau(profile()).observe(1,float('nan'),Optimizer())

def test_family_leak_rejected():
    data=dict(splits=dict(train=['a'],validation=['b']),entries=dict(a=dict(family='x'),b=dict(family='x')))
    with pytest.raises(ValueError,match='family leak'):split_audit(data)
