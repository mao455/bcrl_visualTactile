import subprocess
import sys
import time
from pathlib import Path
import pytest
from tools.tight_fresh_eval_recovery import run,stop_owned,wait_owned


def test_recovery_rejects_existing_control_initialization(tmp_path):
    previous=tmp_path/'previous';previous.mkdir();(previous/'initialization.json').write_text('{}')
    with pytest.raises(ValueError,match='pre-control'):
        run(tmp_path/'root',previous,tmp_path/'next',tmp_path,180)


def test_owned_startup_process_is_reaped():
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    try:
        stop_owned(process)
        assert process.returncode is not None
        stop_owned(process)
    finally:
        if process.poll() is None:process.kill();process.wait()


def test_missing_startup_marker_stops_owned_group(tmp_path):
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    reason=wait_owned(process,time.time()+5,startup_marker=tmp_path/'init',startup_seconds=.1)
    assert reason=='pre_control_startup_timeout'
    assert process.returncode is not None


def test_initialized_worker_can_outlive_startup_window(tmp_path):
    marker=tmp_path/'init';marker.write_text('{}')
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(.15)'],start_new_session=True)
    reason=wait_owned(process,time.time()+5,startup_marker=marker,startup_seconds=.05)
    assert reason is None
    assert process.returncode==0


def test_absolute_deadline_applies_after_initialization(tmp_path):
    marker=tmp_path/'init';marker.write_text('{}')
    process=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],start_new_session=True)
    reason=wait_owned(process,time.time()+.1,startup_marker=marker,startup_seconds=5)
    assert reason=='original_session_deadline'
    assert process.returncode is not None
