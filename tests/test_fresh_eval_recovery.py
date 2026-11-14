import subprocess
import sys
from pathlib import Path
import pytest
from tools.tight_fresh_eval_recovery import run,stop_owned


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
