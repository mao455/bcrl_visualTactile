import json
from pathlib import Path
import pytest
from tools import tight_project


def job(operation='evaluate', entry='evaluate', args=None):
    return dict(version='tight_project_job_v1', operation=operation,
                stages=[dict(entry=entry, args=args or ['--output', '/tmp/run with spaces'])])


def test_registered_arguments_remain_literal_and_ordered():
    command = tight_project.commands(job(args=['--output', '/tmp/$(touch should_not_exist)']), 'evaluate')[0]
    assert command[1:3] == ['-m', 'tools.tight_evaluate']
    assert command[-1] == '/tmp/$(touch should_not_exist)'
    assert not Path('should_not_exist').exists()


@pytest.mark.parametrize('operation,entry', [('train','evaluate'),('dagger','frontend'),('prepare','backend_train')])
def test_no_implicit_cross_stage_work(operation, entry):
    with pytest.raises(ValueError, match='Separate'):
        tight_project.commands(job(operation,entry),operation)


def test_no_arbitrary_date_module_or_empty_work():
    with pytest.raises(KeyError):
        tight_project.commands(job(entry='scripts.old_experiment.run'),'evaluate')
    with pytest.raises(ValueError,match='finite'):
        tight_project.commands(dict(version='tight_project_job_v1',operation='train',stages=[]),'train')


def test_expired_historical_job_cannot_start_workers(tmp_path, monkeypatch):
    (tmp_path/'session.json').write_text(json.dumps(dict(deadline=0)))
    registered=job();registered['resource_root']=str(tmp_path)
    def forbidden(*args,**kwargs):
        pytest.fail('Expired historical job must not create a process')
    monkeypatch.setattr(tight_project.subprocess,'Popen',forbidden)
    with pytest.raises(ValueError,match='Expired'):
        tight_project.execute(registered,'evaluate')
    assert not (tmp_path/'project_job_started.json').exists()


def test_historical_pairing_import_is_the_canonical_function():
    from scripts.tight_fulltask_20261116.paired_eval import install as historical
    from visuotactile.paired_execution import install
    assert historical is install


def residual_snapshot(order):
    import torch
    model=torch.nn.Linear(2,2)
    model.loss_contract=model.history_contract=model.output_contract=None
    optimizer=torch.optim.AdamW(model.parameters())
    generator=torch.Generator().manual_seed(4)
    configuration={'episode_sampling':{'version':'latest_takeover_repeat_v1','latest_repeat':3}}
    saved=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),schedule_rng=generator.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=[],updates=0,epoch=1,next_episode_group=2,
        episode_order=order,supervised_ticks=0,control_ticks=0,contact_anchors=0,best=1.,history=[],
        configuration=configuration,train_ids=['original','correction'])
    return saved,model,optimizer,generator,configuration


def test_repeated_backend_resume_keeps_saved_order_and_cursor():
    from visuotactile.anchor_continue import restore_residual_training
    args=residual_snapshot(['correction','original','correction','correction'])
    before=list(args[0]['episode_order'])
    report=restore_residual_training(*args,['original','correction'],
        episode_multiplicities={'original':1,'correction':3})
    assert report['passed'] and report['next_episode_group']==2
    assert args[0]['episode_order']==before


@pytest.mark.parametrize('counts', [None, {'original':2,'correction':2}, {'original':1}, {'original':1,'correction':True}])
def test_repeated_backend_restore_rejects_wrong_or_missing_deck(counts):
    from visuotactile.anchor_continue import restore_residual_training
    args=residual_snapshot(['correction','original','correction','correction'])
    with pytest.raises(ValueError):
        restore_residual_training(*args,['original','correction'],episode_multiplicities=counts)


def test_owner_termination_unwinds_and_restores_signal_handlers():
    import os
    import signal
    previous = signal.getsignal(signal.SIGTERM)
    cleaned = []
    with pytest.raises(KeyboardInterrupt, match='Owner received'):
        with tight_project.termination_signals():
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            finally:
                cleaned.append(True)
    assert cleaned == [True]
    assert signal.getsignal(signal.SIGTERM) is previous
