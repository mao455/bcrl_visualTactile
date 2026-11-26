"""Wait for frozen trainers to close, then add explicit per-case noise admission."""
from pathlib import Path
import json,time,subprocess,sys
D=Path('outputs/diagnostics/tight_correction_20261127_01');R=Path('outputs/visuotactile/run/tight_correction_20261127_01');read=lambda p:json.loads(p.read_text());deadline=read(R/'short_old_only/session.json')['deadline']
while True:
 if (R/'short_old_only/short_complete.json').exists() and (R/'short_with_corrections/short_complete.json').exists():
  exits=[R/'owners/short_old_backend_retry03/project_exit.json',R/'short_with_corrections/project_exit.json']
  if all(p.exists() and read(p)['children_reaped'] for p in exits):break
 if time.time()>=deadline:raise TimeoutError('Original round deadline expired before sampler option')
 time.sleep(10)
p=Path('visuotactile/paired_execution.py');s=p.read_text();anchor='def install(output):\n';assert anchor in s
helper='''def reset_registered_sampler(base, plan, directory, *, identity):
    """Explicit paired-case noise opt-in; default persistent queue state is unchanged."""
    if 'sampler_episode_seed' not in plan:
        return None
    seed = plan['sampler_episode_seed']
    if type(seed) is not int or seed != base.model.contract.get('sampler_seed'):
        raise ValueError('Per-case sampler seed must match the frozen model contract')
    base.generator.manual_seed(seed)
    receipt = dict(identity=identity, seed=seed, extra_random_draws=0,
                   phase='case admission before first planning action',
                   scope='Diffusion generator only; environment random streams unchanged')
    atomic_json(directory/'sampler_episode_reset.json', receipt)
    return receipt

'''
assert 'def reset_registered_sampler' not in s;p.write_text(s.replace(anchor,helper+anchor))
p=Path('visuotactile/queued_rollout.py');s=p.read_text();anchor="            slots[lane]=case;controller.assign(lane,case['role']);lane_started[lane]=time.perf_counter()\n";assert anchor in s
s=s.replace(anchor,anchor+"            from .paired_execution import reset_registered_sampler\n            reset_registered_sampler(controller.lanes[lane].base,plan,directory,identity=case['name'])\n");p.write_text(s)
p=Path('tests/test_sampler_episode_admission.py');assert not p.exists();p.write_text('''from types import SimpleNamespace
import torch
import pytest
from visuotactile.paired_execution import reset_registered_sampler

def base():
    return SimpleNamespace(generator=torch.Generator().manual_seed(7091),model=SimpleNamespace(contract={'sampler_seed':7091}))

def test_default_retains_persistent_queue_stream(tmp_path):
    b=base();torch.randn(27,generator=b.generator);state=b.generator.get_state().clone()
    assert reset_registered_sampler(b,{},tmp_path,identity='case') is None
    assert torch.equal(state,b.generator.get_state()) and not list(tmp_path.iterdir())

def test_explicit_paired_case_restarts_before_any_draw(tmp_path):
    b=base();torch.randn(27,generator=b.generator)
    receipt=reset_registered_sampler(b,{'sampler_episode_seed':7091},tmp_path,identity='case')
    assert torch.equal(b.generator.get_state(),base().generator.get_state())
    assert receipt['extra_random_draws']==0 and (tmp_path/'sampler_episode_reset.json').exists()

@pytest.mark.parametrize('seed',[7092,True,'7091'])
def test_invalid_seed_rejects_without_mutating_stream(tmp_path,seed):
    b=base();state=b.generator.get_state().clone()
    with pytest.raises(ValueError):reset_registered_sampler(b,{'sampler_episode_seed':seed},tmp_path,identity='case')
    assert torch.equal(state,b.generator.get_state()) and not list(tmp_path.iterdir())
''')
with (D/'queue_noise_tests.log').open('x') as stream:
 result=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_sampler_episode_admission.py','tests/test_reset_admission.py','tests/test_machine_review.py'],stdout=stream,stderr=subprocess.STDOUT)
assert result.returncode==0
(D/'queue_noise_ready.json').write_text(json.dumps(dict(utc=time.time(),sampler_episode_seed=7091,default_changed=False,tests_log='queue_noise_tests.log',all_frozen_training_children_reaped=True,matched_before_start=True),indent=2)+'\n')
print('Per-case diffusion schedule ready; original production default unchanged')
