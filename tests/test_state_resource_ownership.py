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
