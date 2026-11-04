from pathlib import Path
import pytest
from tools.runtime import classify








def test_fault_phase_and_fast_exit(tmp_path):
    assert classify(tmp_path,-6,None,'FileThread',False)=='startup_FileThread_before_scene'
    (tmp_path/'environment.json').write_text('{}')
    assert classify(tmp_path,-6,None,'FileThread',False)=='runtime_FileThread'
    assert classify(tmp_path,0,None,'',True)=='sdk_fast_exit_zero'
    assert classify(tmp_path,-11,None,'',True)=='shutdown_fault_after_complete_runtime'
