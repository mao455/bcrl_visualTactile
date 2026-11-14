import pytest

from visuotactile.state_resources import owned_command_root


def command(module, *args, stage='run'):
    return ['/frozen/.venv_isaacsim/bin/python','-m','tools.'+module,stage,*args]


def test_actual_formal_batch_and_probe_owner_argv():
    # Matches tight_batch_owner and tight_batch_probe_owner launch shapes.
    assert owned_command_root(command('tight_fresh_batch_run','--output','/runs/session06/new_family_08',''),
                              ['/runs/session06'],cwd='/frozen')
    assert owned_command_root(command('tight_fresh_batch_eval','--root','/runs/session05',
        '--output','/runs/session05/scaling_gpu_tiled_n8_raw_diagnostic_03'),['/runs/session05'],cwd='/frozen')
    assert owned_command_root(command('tight_fresh_batch_eval','--output=session05/probe',
        '--root=session05'),['/runs/session05'],cwd='/runs')


@pytest.mark.parametrize('module',['tight_fresh_batch_run','tight_fresh_batch_eval'])
@pytest.mark.parametrize('stage',['prepare','prepare_run','run_extra'])
def test_only_run_is_owned_for_these_registered_modules(module,stage):
    assert not owned_command_root(command(module,'--root','/runs/session','--output','/runs/session/case',
        stage=stage),['/runs/session'],cwd='/runs')


@pytest.mark.parametrize('module',['tight_fresh_batch_run_other','tight_fresh_batch_eval_extra',
    'other.tight_fresh_batch_run','tight_fresh_batch_runs'])
def test_similar_modules_cannot_claim_output_ownership(module):
    assert not owned_command_root(command(module,'--output','/runs/session/case'),['/runs/session'],cwd='/runs')


@pytest.mark.parametrize('args',[
    ['--output','/runs/session/case'],  # batch_eval CLI requires --root too.
    ['--root','/runs/session'],
    ['--root','/runs/session','--root=/runs/session','--output','/runs/session/case'],
    ['--root','/runs/session','--output=/runs/session/case','--output','/runs/session/case'],
    ['--root','/runs/session','--output','/runs/other'],
    ['--root','/runs/other','--output','/runs/session/case'],
    ['--root','/runs/session','--output','/runs/session/../other'],
    ['--root','/runs/session','--output='],
    ['--root','/runs/session','--output'],
])
def test_probe_requires_unique_resource_root_and_unique_bound_output(args):
    assert not owned_command_root(command('tight_fresh_batch_eval',*args),['/runs/session'],cwd='/runs')


@pytest.mark.parametrize('args',[
    ['--root','/runs/session'],
    ['--output','/runs/session/case','--root','/runs/session'],
    ['--output','/runs/session/case','--output=/runs/session/case'],
    ['--output','/runs/session_other/case'],
    ['--output','/runs/session/../other'],
    ['--output','--root'],
])
def test_batch_run_rejects_wrong_or_ambiguous_root_shape(args):
    assert not owned_command_root(command('tight_fresh_batch_run',*args),['/runs/session'],cwd='/runs')


def test_real_symlink_output_escapes_and_relative_worker_cwd(tmp_path):
    root=tmp_path/'session';root.mkdir();outside=tmp_path/'other';outside.mkdir()
    (root/'escape').symlink_to(outside,target_is_directory=True)
    for module in ('tight_fresh_batch_run','tight_fresh_batch_eval'):
        args=['--output','session/escape']
        if module.endswith('eval'):args.extend(['--root','session'])
        assert not owned_command_root(command(module,*args),[root],cwd=tmp_path)
    alias=tmp_path/'alias';alias.symlink_to(root,target_is_directory=True)
    assert owned_command_root(command('tight_fresh_batch_eval','--root','alias','--output','session/probe'),
                              [root],cwd=tmp_path)
    assert not owned_command_root(command('tight_fresh_batch_run','--output','session/probe'),
                                  [root],cwd=outside)
