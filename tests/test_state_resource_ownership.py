from pathlib import Path

import pytest

from visuotactile.state_resources import owned_command_root


@pytest.mark.parametrize('tail, expected', [
    (['--root', '/runs/task'], True),
    (['--root', '/runs/task/round_04/A/backend/A'], True),
    (['--root=/runs/task/round_04'], True),
    (['--root', '/runs/task_other'], False),
    (['--root', '/runs/task/../unrelated'], False),
    (['--root', '/runs'], False),
    (['--root', 'task/round_04'], True),
    (['--root', 'unrelated'], False),
    ([], False),
    (['--root'], False),
    (['--root', ''], False),
    (['--root='], False),
    (['--root', '--other'], False),
    (['--root', '/runs/task', '--root=/runs/task'], False),
])
def test_exact_descendant_sibling_and_argument_boundaries(tail, expected):
    assert owned_command_root(['python', '-m', 'worker', *tail], ['/runs/task'], cwd='/runs') is expected


def test_relative_root_uses_worker_cwd():
    assert owned_command_root(['python', '--root', 'task'], ['/runs/task'], cwd='/runs')
    assert not owned_command_root(['python', '--root', 'task'], ['/runs/task'], cwd='/other')


def test_symlink_escape_and_alias(tmp_path):
    root = tmp_path / 'task'
    root.mkdir()
    other = tmp_path / 'external'
    other.mkdir()
    (root / 'escape').symlink_to(other, target_is_directory=True)
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    assert not owned_command_root(['python', '--root', str(root / 'escape')], [root], cwd=tmp_path)
    assert owned_command_root(['python', '--root', str(alias / 'round_04')], [root], cwd=tmp_path)


def test_additional_declared_root(tmp_path):
    first, second = tmp_path / 'one', tmp_path / 'two'
    assert owned_command_root(['python', '--root', str(second / 'A')], [first, second], cwd=tmp_path)


def test_bad_foreign_symlink_cannot_stop_monitor(tmp_path):
    loop = tmp_path / 'loop'
    loop.symlink_to(loop)
    assert not owned_command_root(['python', '--root', str(loop)], [tmp_path], cwd=tmp_path)


@pytest.mark.parametrize('tail, expected', [
    (['--output', '/runs/task/collection'], True),
    (['--output=task/collection'], True),
    (['--output', '/runs/task_other/collection'], False),
    (['--output', '/runs/task/../other'], False),
    (['--output', '/runs/task', '--output=/runs/task'], False),
    (['--output', '/runs/task', '--root=/runs/task'], False),
    (['--output'], False),
])
def test_only_explicit_stage_worker_can_declare_output_as_ownership(tail, expected):
    command = ['python', '-m', 'tools.tight_fresh_stage_takeover', 'run', *tail]
    assert owned_command_root(command, ['/runs/task'], cwd='/runs') is expected
    command[2] = 'tools.unrelated'
    if not any(x.startswith('--root') for x in tail):
        assert not owned_command_root(command, ['/runs/task'], cwd='/runs')


def test_stage_prepare_and_unrelated_output_are_not_owned():
    for command in (["python", '-m', 'tools.tight_fresh_stage_takeover', 'prepare'],
                    ['python', '-c', 'tools.tight_fresh_stage_takeover run']):
        assert not owned_command_root([*command, '--output', '/runs/task'], ['/runs/task'], cwd='/runs')
