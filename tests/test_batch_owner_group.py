"""Owner-created session cleanup after leader exit; no unrelated PID signals."""
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest

from tools.tight_batch_owner import bind_worker_group, cleanup_owned_group, owned_group_members, process_identity


def test_exited_leader_does_not_leave_live_descendant(tmp_path):
    code = """import os, subprocess, sys
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
print(child.pid,flush=True)
sys.stdin.readline()
os._exit(7)
"""
    leader = subprocess.Popen([sys.executable, '-c', code], start_new_session=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    child_pid = None
    try:
        child_pid = int(leader.stdout.readline())
        binding = bind_worker_group(leader.pid)
        assert any(item['pid'] == child_pid for item in owned_group_members(binding))
        leader.stdin.write('\n'); leader.stdin.flush()
        assert leader.wait(timeout=5) == 7
        assert process_identity(leader.pid) is None  # Normal stop_owned would return here.
        receipt = cleanup_owned_group(binding, grace_seconds=1.)
        assert receipt['live_descendants_exited']
        assert any(item['pid'] == child_pid for item in receipt['before'])
        assert int(signal.SIGTERM) in receipt['signals']
        remaining = process_identity(child_pid)
        assert remaining is None or remaining['state'] in ('Z', 'X')
    finally:
        if leader.poll() is None:
            os.killpg(leader.pid, signal.SIGKILL)
            leader.wait(timeout=5)
        if child_pid is not None:
            remaining = process_identity(child_pid)
            if remaining is not None and remaining['state'] not in ('Z', 'X'):
                os.kill(child_pid, signal.SIGKILL)


def fake_process(proc, pid, pgid, sid, start):
    directory = proc/str(pid)
    directory.mkdir(parents=True)
    fields = ['S', '1', str(pgid), str(sid), *(['0']*15), str(start)]
    (directory/'stat').write_text(str(pid)+' (owned fixture) '+' '.join(fields))


@pytest.mark.parametrize('mutation', ['reused_leader', 'changed_session', 'older_member', 'unbound_group'])
def test_cleanup_refuses_reused_or_unbound_group_before_signal(tmp_path, monkeypatch, mutation):
    proc = tmp_path/'proc'; proc.mkdir()
    fake_process(proc, 199999991, 199999991, 199999991, 100)
    binding = bind_worker_group(199999991, proc)
    if mutation == 'reused_leader':
        (proc/'199999991/stat').unlink(); (proc/'199999991').rmdir()
        fake_process(proc, 199999991, 199999991, 199999991, 200)
    elif mutation == 'changed_session': fake_process(proc, 199999992, 199999991, 999, 110)
    elif mutation == 'older_member': fake_process(proc, 199999992, 199999991, 199999991, 90)
    elif mutation == 'unbound_group': binding['pgid'] = 888
    signals = []
    monkeypatch.setattr(os, 'killpg', lambda *args:signals.append(args))
    with pytest.raises(ValueError):
        cleanup_owned_group(binding, grace_seconds=.01, proc=proc)
    assert signals == []


def test_formal_module_spawn_import_does_not_run_kit(tmp_path):
    # multiprocessing.spawn uses runpy with __mp_main__ for the actual -m entry.
    code = """import runpy,sys,json
sys.argv=['tools.tight_fresh_batch_run','run','--output','/must-not-run']
runpy.run_module('tools.tight_fresh_batch_run',run_name='__mp_main__')
assert not any(name.startswith(('omni.kit','isaaclab','isaacsim')) for name in sys.modules)
assert 'torch' not in sys.modules
print(json.dumps({'spawn_main_imported':True,'kit_imported':False,'torch_imported':False}))
"""
    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True,
        cwd=Path(__file__).resolve().parents[1], timeout=10)
    assert '"kit_imported": false' in result.stdout
