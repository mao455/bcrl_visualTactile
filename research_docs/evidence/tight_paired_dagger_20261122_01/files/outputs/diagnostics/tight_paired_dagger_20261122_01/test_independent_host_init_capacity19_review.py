"""Independent19 fixtures: all process/clock/lifecycle I/O is synthetic."""
from contextlib import contextmanager
from copy import deepcopy
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


spec = importlib.util.spec_from_file_location('independent_capacity19', Path(__file__).with_name('host_init_capacity19.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Proc:
    pid = 123
    def create_time(self): return 100.
    def is_running(self): return True
    def status(self): return 'running'
    def cmdline(self): return [m.PYTHON, '-m', m.MODULE, 'worker', '--output', str(m.TEST/'wave45')]
    def cwd(self): return str(m.B)
    def nice(self): return 10


@contextmanager
def mapped_identity_fixture():
    wave = m.TEST/'wave45'; command = Proc().cmdline()
    status = dict(stage=28, pid=123, utc=100.899, command=command)
    job = dict(resource_root=str(m.O), operation='evaluate', stages=[{} for _ in range(47)])
    job['stages'][28] = dict(entry='paired_evaluate_continuation15', args=command[3:])
    commands = [[] for _ in range(47)]; commands[28] = command
    rows = {
        m.REGISTRY: dict(owner_identity=dict(pid=10, created=90.)),
        m.REGISTRY15: dict(job_sha256='job', original_test_registration_sha256='freeze',
            eligible_indices=list(range(17, 64)), frozen_pairs={'A':'a','B':'b'}, original13_registry_sha256='reg13'),
        m.REVIEW15: dict(status='source_and_actual_missing47_review_passed_inactive', registration_sha256='reg15'),
        m.ACTIVATED15: dict(registry_sha256='reg15', root_review_sha256='review15'),
        m.JOB: job,
        m.O/'project_job_started.json': dict(pid=10, utc=90.899, job=job, commands=commands),
        m.O/'project_status.json': status,
        m.O/'launches/wave45.json': dict(stage=28, index=45, command=command, registration_sha256='reg15',
            dispatch_attempt=1, rerun=False, utc=100.895),
        m.O/'dispatches/wave45.json': dict(pid=123, index=45, wave=str(wave), dispatch_attempt=1,
            rerun=False, registration_sha256='reg15', original13_registration_sha256='reg13',
            launch_sha256='launch', plan_sha256='plan', utc=101.15),
        m.TEST/'registration.json': dict(wave_plan_sha256={str(wave):'plan'}),
        wave/'plan.json': dict(pairs={'A':'a','B':'b'}, resource_root=str(m.TEST)),
    }
    hashes = {m.REGISTRY15:'reg15',m.REVIEW15:'review15',m.JOB:'job',
        m.TEST/'registration.json':'freeze',wave/'plan.json':'plan',
        m.O/'launches/wave45.json':'launch',m.O/'dispatches/wave45.json':'dispatch'}
    processes = {
        10:dict(pid=10,created=90.,ppid=1,command=[m.PYTHON,'-m',m.MODULE,'project','evaluate','--job',str(m.JOB)],cwd=str(m.B)),
        123:dict(pid=123,created=100.,ppid=10,command=command,cwd=str(m.B)),
    }
    mappings = {123:dict(lower=100.893,upper=100.903),10:dict(lower=90.893,upper=90.903)}
    with patch.object(m, 'read', side_effect=lambda p:deepcopy(rows[Path(p)])), \
         patch.object(m, 'digest', side_effect=lambda p:hashes[Path(p)]), \
         patch.object(m, 'process_row', side_effect=lambda pid:deepcopy(processes[pid])), \
         patch.object(m.psutil, 'Process', side_effect=lambda pid:SimpleNamespace(pid=pid)), \
         patch.object(m, 'mapped_birth', side_effect=lambda proc,birth:deepcopy(mappings[proc.pid])):
        yield status,rows,processes,mappings,wave


def test_actual_style_fractional_boot_epoch_passes_exact_identity():
    with mapped_identity_fixture() as (status,rows,_,__,wave):
        result = m.identity(status,Proc(),100.,wave)
        assert 100.-rows[m.O/'launches/wave45.json']['utc'] < -.05
        assert result['worker_birth_clock'] == dict(lower=100.893,upper=100.903)
        assert result['index'] == 45


@pytest.mark.parametrize('case', ['worker_newer_than_status','worker_before_launch','owner_newer_than_started',
    'owner_raw_birth_reused','launch_sha','dispatch_pid','dispatch_attempt','dispatch_rerun',
    'launch_command','job_stage','status_changed'])
def test_clock_mapping_does_not_relax_other_exact_identity_guards(case):
    with mapped_identity_fixture() as (status,rows,processes,mappings,wave):
        if case=='worker_newer_than_status': mappings[123]=dict(lower=101.,upper=101.01)
        elif case=='worker_before_launch': mappings[123]=dict(lower=100.8,upper=100.81)
        elif case=='owner_newer_than_started': mappings[10]=dict(lower=91.,upper=91.01)
        elif case=='owner_raw_birth_reused': processes[10]['created']=91.
        elif case=='launch_sha': rows[m.O/'dispatches/wave45.json']['launch_sha256']='other'
        elif case=='dispatch_pid': rows[m.O/'dispatches/wave45.json']['pid']=124
        elif case=='dispatch_attempt': rows[m.O/'dispatches/wave45.json']['dispatch_attempt']=2
        elif case=='dispatch_rerun': rows[m.O/'dispatches/wave45.json']['rerun']=True
        elif case=='launch_command': rows[m.O/'launches/wave45.json']['command']=['other']
        elif case=='job_stage': rows[m.JOB]['stages'][28]={}
        elif case=='status_changed': rows[m.O/'project_status.json']=dict(status,utc=100.9)
        with pytest.raises(ValueError):m.identity(status,Proc(),100.,wave)


def test_kernel_birth_parses_starttick_after_embedded_parentheses():
    fields=['0']*20;fields[19]='256493441'
    with patch.object(Path,'read_text',side_effect=['123 (worker (child)) '+' '.join(fields),'cpu 0\nbtime 1792940224\n']), \
         patch.object(m.os,'sysconf',return_value=100):
        assert m.kernel_birth(123)==dict(pid=123,starttick=256493441,hz=100,integer_btime=1792940224)


def test_mapped_birth_accepts_integer_btime_offset_and_records_tick_uncertainty():
    kernel=dict(pid=123,starttick=256493441,hz=100,integer_btime=1792940224)
    sample=dict(wall_before=1795505939.199095,wall_after=1795505939.1990957,boottime=2565714.305639852,
        offset_lower=1792940224.8934553,offset_upper=1792940224.893456)
    with patch.object(m,'same',return_value=True) as same,patch.object(m,'kernel_birth',return_value=kernel), \
         patch.object(m,'clock_sample',return_value=sample):
        result=m.mapped_birth(Proc(),1795505158.41)
    assert result['lower']==pytest.approx(1795505159.3034554,rel=0,abs=1e-6)
    assert result['upper']==pytest.approx(1795505159.313456,rel=0,abs=1e-6)
    assert result['quantization_seconds']==.01
    assert result['kernel']==kernel and result['clock_sample']==sample
    assert same.call_count==2


@pytest.mark.parametrize('field,value', [('pid',124),('starttick',True),('starttick',0),('hz',True),('hz',0),('integer_btime',True)])
def test_mapped_birth_rejects_kernel_identity_or_schema_drift(field,value):
    kernel=dict(pid=123,starttick=1000,hz=100,integer_btime=90)
    kernel[field]=value
    sample=dict(offset_lower=90.893,offset_upper=90.893001)
    with patch.object(m,'same',return_value=True),patch.object(m,'kernel_birth',return_value=kernel), \
         patch.object(m,'clock_sample',return_value=sample):
        with pytest.raises(ValueError):m.mapped_birth(Proc(),100.)


@pytest.mark.parametrize('values', [[False],[True,False]])
def test_mapped_birth_rejects_raw_pid_reuse_before_or_after_sample(values):
    with patch.object(m,'same',side_effect=values), \
         patch.object(m,'kernel_birth',return_value=dict(pid=123,starttick=1000,hz=100,integer_btime=90)), \
         patch.object(m,'clock_sample',return_value=dict(offset_lower=90.893,offset_upper=90.893001)):
        with pytest.raises(ValueError,match='Raw PID/birth changed'):m.mapped_birth(Proc(),100.)


def test_mapped_birth_rejects_raw_epoch_mismatch_without_affinity_writes():
    with patch.object(m,'same',return_value=True), \
         patch.object(m,'kernel_birth',return_value=dict(pid=123,starttick=1000,hz=100,integer_btime=90)), \
         patch.object(m,'clock_sample',return_value=dict(offset_lower=90.893,offset_upper=90.893001)), \
         patch.object(m.os,'sched_setaffinity') as change:
        with pytest.raises(ValueError,match='does not match'):m.mapped_birth(Proc(),101.)
    change.assert_not_called()


def test_forward_or_backward_clock_mapping_step_rejects_tight_launch_status_pair():
    with mapped_identity_fixture() as (status,_,__,mappings,wave):
        for shift in (-1.,1.):
            mappings[123]=dict(lower=100.893+shift,upper=100.903+shift)
            with pytest.raises(ValueError):m.identity(status,Proc(),100.,wave)


def test_narrow19_fix_preserves_all_nonclock_runtime_logic_from_frozen18():
    source19=ast.parse(Path(m.__file__).read_bytes())
    source18=ast.parse(Path(m.__file__).with_name('host_init_capacity18.py').read_bytes())
    old={node.name:ast.dump(node,include_attributes=False) for node in source18.body if isinstance(node,ast.FunctionDef)}
    new={node.name:ast.dump(node,include_attributes=False) for node in source19.body if isinstance(node,ast.FunctionDef)}
    changed={name for name in old if old[name]!=new.get(name)}
    assert changed=={'source_guard','receipt_birth','identity'}
    assert set(new)-set(old)=={'clock_sample','kernel_birth','mapped_birth','valid_utc','launch_birth'}
