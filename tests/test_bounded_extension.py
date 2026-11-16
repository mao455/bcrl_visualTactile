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


def test_long_budget_rejects_unregistered_change_and_restores_only_declared_fields(tmp_path):
    from visuotactile.bounded_extension import long_fixed_profile,phase_dagger_profile,admit_dp_restore
    from visuotactile.compact import sha256
    from visuotactile.trajectory_training import budget
    checkpoint=tmp_path/'last.pt';checkpoint.write_bytes(b'complete checkpoint identity')
    old=dict(data_sha256='fixed',bounded_extension=phase_dagger_profile(),**budget(56,40))
    new=dict(old,bounded_extension=long_fixed_profile(),**budget(56,192))
    registration=dict(version='fixed56_full_state_extension_v1',last_sha256=sha256(checkpoint),
        old_policy=old['bounded_extension'],new_policy=new['bounded_extension'],new_budget=budget(56,192))
    assert admit_dp_restore(old,new,registration,checkpoint)['optimizer_restored']
    with pytest.raises(ValueError,match='training semantics'):
        admit_dp_restore(old,dict(new,data_sha256='changed'),registration,checkpoint)
    with pytest.raises(ValueError,match='budget mismatch'):
        admit_dp_restore(old,dict(new,planned_optimizer_updates=9999),registration,checkpoint)
    checkpoint.write_bytes(b'other state')
    with pytest.raises(ValueError,match='registered complete last'):
        admit_dp_restore(old,new,registration,checkpoint)


def test_long_budget_does_not_stop_at_historical_plateau():
    from visuotactile.bounded_extension import long_fixed_profile
    opt=Optimizer();p=Plateau(long_fixed_profile(),baseline=1.)
    for epoch in range(27,192):
        p.observe(epoch,1.001,opt)
        assert not p.state['stop']
    p.observe(192,1.001,opt)
    assert p.state['stop'] and opt.param_groups[0]['lr']==1.5e-5
