"""Source-review regression cases. Mock all process/affinity I/O; CPU only."""
import importlib.util
from pathlib import Path
from unittest.mock import patch
import pytest

spec=importlib.util.spec_from_file_location('independent_affinity16',Path(__file__).with_name('startup_affinity16.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class P:
    pid=123
    def create_time(self):return 200.
    def is_running(self):return True
    def cmdline(self):return [m.PYTHON,'-m',m.MODULE,'worker','--output',str(m.R/'evaluation/test/wave33')]
    def cwd(self):return str(m.B)
    def nice(self):return 10


def state():return dict(command=P().cmdline(),pid=123,stage=16,utc=100.)


def test_reused_pid_newer_than_status_cannot_establish_ownership():
    with pytest.raises(ValueError):
        m.identity(state(),P(),200.,m.R/'evaluation/test/wave33')


def test_readiness_race_no_change_makes_no_affinity_write():
    changes=[]
    with patch.object(m.psutil,'Process',return_value=P()),patch.object(m,'identity'), \
         patch.object(m,'read',return_value={'current_stage':'environment_constructor_started'}), \
         patch.object(m,'ready',side_effect=[False,True]), \
         patch.object(m,'masks',return_value=[(123,m.OLD)]), \
         patch.object(m,'change',side_effect=lambda p,b,cpus:changes.append(set(cpus)) or [123]), \
         patch.object(m,'write'),patch.object(m.time,'time',return_value=300.):
        row=m.probe(state())
    assert row['readiness_race_no_change'] is True
    assert changes==[]


def test_failed_restore_still_writes_wave_exit():
    writes=[];calls=[]
    def mask(p,b):
        return [(123,m.OLD if not calls else m.WIDE)]
    with patch.object(m.psutil,'Process',return_value=P()),patch.object(m,'identity'), \
         patch.object(m,'read',return_value={'current_stage':'environment_constructor_started'}), \
         patch.object(m,'ready',return_value=False),patch.object(m,'masks',side_effect=mask), \
         patch.object(m,'change',side_effect=lambda p,b,cpus:calls.append(set(cpus)) or [123]), \
         patch.object(m,'write',side_effect=lambda p,row:writes.append(p.name)), \
         patch.object(m.time,'time',return_value=300.), \
         patch.object(m.time,'monotonic',side_effect=[1.,100.]):
        with pytest.raises(ValueError,match='Failed original thread restoration'):
            m.probe(state())
    assert writes==['wave33_intent.json','wave33_exit.json']
