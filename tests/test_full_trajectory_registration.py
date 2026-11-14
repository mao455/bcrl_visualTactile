"""Formal B2 registration separates visit dose from optimizer/EMA steps."""
import json
import pytest

from tools import tight_fresh_cycle as cycle
from visuotactile.trajectory_training import profile
from visuotactile.training_resources import atomic_json
from test_fresh_parallel_backends import CycleFixture


@pytest.mark.parametrize('parallel',[False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_cycle_dispatches_explicit_method_and_binds_actual_odd_deck(tmp_path,monkeypatch,parallel,enabled):
    fixture=CycleFixture(tmp_path,monkeypatch,frontend_parallel=parallel)
    launch=cycle.subprocess.Popen
    def with_manifest(command,**kwargs):
        proc=launch(command,**kwargs)
        if proc.stage=='aggregate':
            for role in 'AB':
                (fixture.output/role/'state_data').mkdir(parents=True)
                atomic_json(fixture.output/role/'state_data/data.json',dict(splits=dict(train=['x','y','z'])))
                atomic_json(fixture.output/role/'continuation.json',{})
        return proc
    monkeypatch.setattr(cycle.subprocess,'Popen',with_manifest)
    fixture.run(parallel_frontends=parallel,trajectory_batch_profile=profile() if enabled else None)
    registration=json.loads((fixture.ledger/'registration.json').read_text())
    assert ('full_trajectory_batch' in registration)==enabled
    assert (fixture.output/'full_trajectory_batch.json').exists()==enabled
    if enabled:
        planned=json.loads((fixture.output/'full_trajectory_batch.json').read_text())
        assert planned==json.loads((fixture.ledger/'frontend_batch_budget.json').read_text())
        for role in 'AB':
            row=planned['roles'][role]
            assert row['planned_episode_visits']==30
            assert row['planned_optimizer_updates']==row['planned_ema_updates']==20
            assert row['tail_singletons_per_epoch']==1
            assert row['episode_multiplicities']==dict(x=1,y=1,z=1)
    for name,command in fixture.calls:
        assert ('--full-trajectory-batch' in command)==(enabled and name.startswith('frontend_'))


@pytest.mark.parametrize('incompatible',[dict(paired_episode_io=True),dict(episode_io={}),
    dict(episode_io=dict(version='bounded_episode_prefetch_v1',budget_mib=512))])
def test_cycle_rejects_explicit_or_implicit_incompatible_io_before_any_files(tmp_path,incompatible):
    with pytest.raises(ValueError,match='B2 rejects'):
        cycle.run(tmp_path,tmp_path,tmp_path/'out',1,1,1,1,trajectory_batch_profile=profile(),**incompatible)
    assert not (tmp_path/'out_stages').exists()
