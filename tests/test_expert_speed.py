import numpy as np
import pytest
import torch
from visuotactile.expert_speed import progress_target,retime_targets,reliability,cap_hand_target


def test_identity_is_bitwise_and_separate_arm_hand_factors():
    g=torch.Generator().manual_seed(7);p=torch.randn(9,29,generator=g);q=torch.randn(9,29,generator=g)
    assert progress_target(p,q) is q
    r=progress_target(p,q,.5,1.)
    assert torch.equal(r[:,7:],q[:,7:])
    torch.testing.assert_close(r[:,:7]-p[:,:7],.5*(q[:,:7]-p[:,:7]))
    assert ((r>=torch.minimum(p,q)) & (r<=torch.maximum(p,q))).all()


def test_existing_arm_cap_is_reduced_not_reapplied_and_hand_limits_hold():
    p=torch.zeros(1,29);q=p.clone();q[:,:7]=.15/60;q[:,7:]=.1
    r=progress_target(p,q,.67,.5)
    torch.testing.assert_close(r[:,:7]*60,torch.full((1,7),.15*.67))
    torch.testing.assert_close(r[:,7:],torch.full((1,22),.05))
    with pytest.raises(ValueError):progress_target(p,q,0,1)


def test_retime_uses_initial_target_and_no_one_tick_shift():
    before=np.zeros(29,dtype='f4');after=np.stack([np.ones(29),np.ones(29)*2]).astype('f4')
    assert np.array_equal(retime_targets(before,after,1),after)
    np.testing.assert_array_equal(retime_targets(before,after,2)[:,0],[.5,1,1.5,2])
    x=retime_targets(before,after,1.25)
    np.testing.assert_allclose(x[:,0],[.8,1.6,2]);assert x.dtype==after.dtype


def test_reliability_counts_failures_and_does_not_infer_time_from_factor():
    base=[dict(identity=str(i),success=True,duration_sim_s=3.4) for i in range(10)]
    rows=[dict(identity=str(i),legal=True,software_complete=True,success=i<8,duration_sim_s=5.8) for i in range(10)]
    gate=reliability(rows,base);assert gate['passed'] and gate['rate']==.8
    rows[-3]['success']=False;assert not reliability(rows,base)['passed']
    for row in rows:row.update(success=True,duration_sim_s=3.4)
    assert not reliability(rows,base)['passed']


def test_native_pipeline_matches_pinned_official_math_rng_and_delay():
    import copy
    import subprocess
    from pathlib import Path
    from types import SimpleNamespace as NS
    path='isaacsimenvs/tasks/play/utils/action_utils.py'
    upstream={};local={}
    exec(subprocess.check_output(['git','show','ff2fc62873f6c6cd93164123c8358e9ce2d65c9b:'+path]),upstream)
    exec(Path(path).read_text(),local)
    env=NS(cfg=NS(domain_randomization=NS(use_action_delay=True,action_delay_max=3),
        action=NS(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
        step_dt=1/60,device='cpu',num_envs=2,_perm_canon_to_lab=torch.arange(29),
        _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),
        _arm_lower=torch.full((7,),-2.),_arm_upper=torch.full((7,),2.),
        _hand_lower=torch.full((22,),-1.),_hand_upper=torch.full((22,),1.),
        episode_length_buf=torch.zeros(2,dtype=torch.long),_successes=torch.zeros(2),
        _action_queue=torch.zeros(2,4,29),_prev_targets=torch.zeros(2,29),_cur_targets=torch.zeros(2,29))
    other=copy.deepcopy(env);torch.manual_seed(193)
    for tick in range(100):
        command=torch.randn(2,29).clamp(-1,1);state=torch.get_rng_state()
        upstream['apply_action_pipeline'](env,command);after=torch.get_rng_state()
        torch.set_rng_state(state);local['apply_action_pipeline'](other,command)
        assert torch.equal(torch.get_rng_state(),after)
        assert torch.equal(env._cur_targets,other._cur_targets)
        assert torch.equal(env._action_queue,other._action_queue)
        assert progress_target(other._prev_targets,other._cur_targets) is other._cur_targets
        env.episode_length_buf+=1;other.episode_length_buf+=1


def test_speed_subclass_filters_once_and_feeds_actual_target_back(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType
    module=ModuleType('visuotactile.sim.native_diagnostics')
    class FakeNative:
        def __init__(self):
            self.step_dt=1/60
            self._prev_targets=torch.zeros(1,29);self._cur_targets=self._prev_targets.clone()
            self._perm_lab_to_canon=self._perm_canon_to_lab=torch.arange(29)
            self._successes=torch.zeros(1);self.action_control_mode='native_command_clipped_pre_delay';self.calls=0
        def _pre_physics_step(self,actions):
            self.calls+=1;self._cur_targets=self._prev_targets+actions
            self._prev_targets=self._cur_targets.clone();self.last_transition={}
    module.NativeDiagnosticEnv=FakeNative
    monkeypatch.setitem(sys.modules,module.__name__,module)
    spec=importlib.util.spec_from_file_location('visuotactile.sim._test_speed_env',Path('visuotactile/sim/expert_speed_environment.py'))
    loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded)
    env=loaded.SpeedLimitedDiagnosticEnv(speed_config=dict(kind='closed_loop',arm=.5,hand=.8,phase='all'))
    for _ in range(2):env._pre_physics_step(torch.ones(1,29)*.1)
    assert env.calls==2
    torch.testing.assert_close(env._cur_targets[:,:7],torch.full((1,7),.1))
    torch.testing.assert_close(env._cur_targets[:,7:],torch.full((1,22),.16))
    assert torch.equal(env._prev_targets,env.last_transition['target_after_rad'])
    capped=loaded.SpeedLimitedDiagnosticEnv(speed_config=dict(kind='closed_loop',arm=.5,hand=1.,phase='all',hand_target_cap_rad_s=2.))
    for _ in range(2):capped._pre_physics_step(torch.ones(1,29)*.1)
    assert capped.calls==2
    torch.testing.assert_close(capped._cur_targets[:,7:],torch.full((1,22),4/60))
    torch.testing.assert_close(capped._cur_targets[:,:7],torch.full((1,7),.1))
    assert torch.equal(capped._prev_targets,capped.last_transition['target_after_rad'])
    # Processed student targets bypass BOTH collection modifiers, including cap.
    capped.action_control_mode='absolute_joint_position_target_rad'
    prev=capped._prev_targets.clone();capped._pre_physics_step(torch.ones(1,29))
    assert torch.equal(capped._cur_targets,prev+1)
    assert 'speed_before_hand_cap_target_rad' not in capped.last_transition


def test_hand_cap_disabled_exact_and_per_joint_rate_preserves_arm_and_bounds():
    g=torch.Generator().manual_seed(15);p=torch.randn(8,29,generator=g);t=torch.randn(8,29,generator=g)
    assert cap_hand_target(p,t,None) is t
    limits=torch.linspace(.1,3,22)
    out=cap_hand_target(p,t,limits)
    assert torch.equal(out[:,:7],t[:,:7])
    assert ((out[:,7:]-p[:,7:]).abs()*60 <= limits+1e-5).all()
    assert ((out>=torch.minimum(p,t)) & (out<=torch.maximum(p,t))).all()
    t=p+1e-5
    assert torch.equal(cap_hand_target(p,t,2),t)
    for bad in (0,-1,float('nan'),float('inf'),[1]*21,[[1]*22]):
        with pytest.raises(ValueError):cap_hand_target(p,t,bad)
    for bad in (0,-1,float('nan')):
        with pytest.raises(ValueError):cap_hand_target(p,t,2,bad)
