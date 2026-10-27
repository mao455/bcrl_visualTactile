"""Experiment denominator/gating checks; no Isaac Sim or training required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from visuotactile.analyze_rl_trials import aggregate, exposure, wilson, episode_outcome
from visuotactile.run_rl_trials import make_trials
from visuotactile.run_rl_trials import prior_attempt
import json
import pytest


def native_wrench():
    spec=importlib.util.spec_from_file_location('native_action',Path(__file__).parents[1]/'isaacsimenvs/tasks/play/utils/action_utils.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod.apply_wrench_dr


def fake_env(scale):
    applied={}
    def setter(**values):
        applied.update({k:v.clone() if hasattr(v,'clone') else v for k,v in values.items()})
    dr=SimpleNamespace(force_scale=scale,torque_scale=2.,force_decay=0.,torque_decay=0.,
                       force_decay_interval=.08,torque_decay_interval=.08,
                       force_only_when_lifted=True,torque_only_when_lifted=True)
    env=SimpleNamespace(cfg=SimpleNamespace(domain_randomization=dr),num_envs=3,device='cpu',step_dt=1/60,
        _object_forces=torch.zeros(3,1,3),_object_torques=torch.zeros(3,1,3),_object_mass=torch.ones(3,1)*.0578,
        _random_force_prob=torch.ones(3),_random_torque_prob=torch.ones(3),
        _lifted_object=torch.tensor([True,False,True]),
        _wrench_dr_active_mask=lambda:torch.tensor([True,True,False]),
        object=SimpleNamespace(set_external_force_and_torque=setter))
    return env,applied


def test_native_scale_changes_only_force_with_same_rng_and_gates():
    apply=native_wrench();a,sa=fake_env(20);b,sb=fake_env(30)
    torch.manual_seed(194);apply(a);rng_a=torch.random.get_rng_state()
    torch.manual_seed(194);apply(b);rng_b=torch.random.get_rng_state()
    assert torch.equal(rng_a,rng_b)
    assert torch.allclose(sb['forces'],sa['forces']*1.5)
    assert torch.equal(sa['torques'],sb['torques'])
    assert sa['is_global'] is True
    assert sa['forces'][0].norm()>0 and not sa['forces'][1:].any()
    assert torch.equal(sa['forces'],a._object_forces)
    a._random_force_prob.zero_();a._random_torque_prob.zero_();apply(a)
    assert not sa['forces'].any() and not sa['torques'].any()


def test_censor_errors_remain_unknown_not_robot_failure():
    assert episode_outcome(True,False,True,False)=='failure'  # native timeout
    assert episode_outcome(False,False,False,False)=='censored'  # external budget
    assert episode_outcome(True,True,False,True)=='success'
    assert episode_outcome(True,True,False,False)=='failure'
    with pytest.raises(ValueError):
        episode_outcome(True,False,False,False)
    rows=[dict(condition='original',started=True,outcome=outcome,runtime_error=outcome=='runtime_error',
               audit_errors=['bad'] if outcome=='audit_error' else [])
          for outcome in ('success','failure','censored','runtime_error','audit_error')]
    rows.append(dict(condition='original',started=False,outcome='not_started'))
    stats=aggregate(rows)['original']
    assert stats['planned']==6 and stats['started']==5 and stats['completed']==2
    assert stats['failure']==1 and stats['censored']==1
    assert stats['runtime_errors']==1 and stats['audit_errors']==1
    assert stats['planned_success_bounds']==[1/6,5/6]
    assert stats['success_rate_completed']==.5
    assert np.allclose(wilson(2,3),[.2076596,.9385081])


def test_alternation_and_impulse_units():
    plan=make_trials([4,5,6],'acceptance')
    assert [row['condition'] for row in plan]==['original','force_1p5','force_1p5','original','original','force_1p5']
    assert [row['seed'] for row in plan]==[4,4,5,5,6,6]
    force=np.zeros((5,1,3));force[[1,2,4],0,0]=[2,3,4]
    out=exposure(force,.02)
    assert out['nonzero_steps']==3 and out['nonzero_runs']==2
    assert np.isclose(out['integral_norm'],.18) and np.isclose(out['duration_s'],.06)


def test_startup_retry_retains_failed_attempt_and_rejects_partial_data(tmp_path):
    name='pair01_original'
    (tmp_path/(name+'.status.json')).write_text(json.dumps({'returncode':-6,'episode_started':False}))
    (tmp_path/name).mkdir()
    assert prior_attempt(tmp_path,name)==(name+'_attempt02',None)
    (tmp_path/name/'trajectories.h5').touch()
    with pytest.raises(RuntimeError,match='Unsafe'):
        prior_attempt(tmp_path,name)
