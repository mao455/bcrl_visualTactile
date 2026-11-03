"""Explicit recorder/lifecycle recovery around the unchanged StateDP experiment."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import h5py
import numpy as np
import torch

from visuotactile.compact import sha256
from visuotactile.recording import flatten, numpy_tree
from visuotactile.state_full_run import Runner, recover_preaction_start, startup_failure_evidence
from visuotactile.training_resources import atomic_json


RECOVERY = 'runtime_recovery_02'
FAILED_STAGE = 'single_takeover_12_single_00'
STARTUP_FAILED_STAGE = 'single_takeover_24_single_00'
VALIDITY = 'recorded_field_valid'


def append_mode_numeric(group, values, step):
    """Keep mode-specific fields separate, with explicit missing-value masks."""
    arrays = {name: np.asarray(value)[0] for name, value in flatten(numpy_tree(values)).items()}
    if any(name.startswith(VALIDITY+'/') for name in arrays):
        raise ValueError('Reserved numeric validity namespace')
    if step != group.attrs.get('completed_steps', 0):
        raise ValueError('Numeric recording must append exactly one consecutive tick')
    fields = set(json.loads(group.attrs.get('recorded_fields', '[]')))
    for name, array in arrays.items():
        if name in fields and (group[name].shape[1:] != array.shape or group[name].dtype != array.dtype):
            raise ValueError('Numeric field shape/dtype changed: '+name)
        if array.dtype.kind not in 'biuf':
            raise ValueError('Unsupported numeric field: '+name)
    for name in sorted(fields | arrays.keys()):
        if name not in fields:
            array = arrays[name]
            parent, _, leaf = name.rpartition('/')
            group.require_group(parent).create_dataset(leaf, shape=(step, *array.shape),
                maxshape=(None, *array.shape), dtype=array.dtype, chunks=(1, *array.shape),
                compression='lzf', fillvalue=np.nan if array.dtype.kind == 'f' else 0)
            group.require_group(VALIDITY+'/'+parent).create_dataset(leaf, shape=(step,),
                maxshape=(None,), dtype=bool, chunks=True, fillvalue=False)
        dataset = group[name]; valid = group[VALIDITY+'/'+name]
        dataset.resize(step+1, axis=0); valid.resize(step+1, axis=0)
        valid[step] = name in arrays
        if name in arrays:
            dataset[step] = arrays[name]
    group.attrs['recorded_fields'] = json.dumps(sorted(fields | arrays.keys()))
    group.attrs['numeric_recording_version'] = 2
    group.attrs['missing_field_semantics'] = 'Use recorded_field_valid; NaN/zero fill is not an observation or action'
    group.attrs['completed_steps'] = step+1


def correct_takeover_result(path, result):
    """Report mixed command units and position error only on position-mode ticks."""
    if path.name != 'result.json' or 'takeover' not in result:
        return result
    with h5py.File(path.parent/'numeric.h5', 'r') as stream:
        group = stream['episode']
        mask = group[VALIDITY+'/transition/position_requested_rad'][:]
        requested = group['transition/position_requested_rad'][:]
        target = group['transition/target_after_rad'][:]
        native_path = VALIDITY+'/transition/command_clipped_pre_delay'
        native = group[native_path][:] if native_path in group else np.zeros(len(mask), dtype=bool)
        assert np.array_equal(native, ~mask)
        expected = np.arange(len(mask)) < result['takeover']['student_ticks']
        assert np.array_equal(mask, expected)
    trace_path = path.parent/'trace.npz'
    with np.load(trace_path) as trace:
        arrays = {name: trace[name] for name in trace.files}
    arrays['command_is_position'] = mask
    np.savez_compressed(trace_path, **arrays)
    result.update(action_control_mode='position_prefix_then_native_expert',
        submitted_target_max_error=float(np.abs(requested[mask]-target[mask]).max()) if mask.any() else None,
        submitted_target_error_scope='Position prefix only; native expert commands are dimensionless, not radians',
        recording_version=2, command_units='trace.command_is_position: true=rad, false=native dimensionless command')
    return result


def verify_registration(root):
    folder = root/RECOVERY
    registration = json.loads((folder/'registration.json').read_text())
    if registration['launcher_sha256'] != sha256(Path(__file__)):
        raise ValueError('Recovery launcher source changed')
    if registration['session_sha256'] != sha256(root/'session.json'):
        raise ValueError('Frozen session changed')
    session = json.loads((root/'session.json').read_text())
    for name, digest in session['implementation_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Frozen runtime source changed: '+name)
    for name, digest in registration['preserved_sha256'].items():
        if sha256(root/name) != digest:
            raise ValueError('Recovery protected artifact changed: '+name)
    for name, digest in json.loads((folder/'failed_attempt_inventory.json').read_text()).items():
        if sha256(root/name) != digest:
            raise ValueError('Failed attempt changed: '+name)
    return registration


def shutdown_stall(output, now):
    error = output/'worker_error.json'
    if error.exists() and now-error.stat().st_mtime >= 5:
        return 'Durable worker_error; error cleanup exceeded five seconds'
    lifecycle = output/'lifecycle.json'
    if lifecycle.exists():
        state = json.loads(lifecycle.read_text())
        if state.get('state') == 'closing' and now-state['utc'] >= 30:
            return 'Kit closing exceeded thirty seconds'
    return None


class RecoveryRunner(Runner):
    def __init__(self, root):
        super().__init__(root)
        self.registration = verify_registration(root)

    def evaluate(self, name, mode, checkpoint=None, specs=None, **kwargs):
        if name not in ('single_takeover_12', 'single_takeover_24'):
            return super().evaluate(name, mode, checkpoint, specs, **kwargs)
        verify_registration(self.root)
        stage = FAILED_STAGE if name == 'single_takeover_12' else STARTUP_FAILED_STAGE
        ticks = 12 if stage == FAILED_STAGE else 24
        job = json.loads((self.root/(stage+'_job.json')).read_text())
        if (specs != job['specs'] or sha256(checkpoint) != job['checkpoint_sha256']
                or mode != 'dp' or kwargs.get('takeover_steps') != ticks
                or kwargs.get('inference') != job['inference']):
            raise ValueError('Failed-attempt identity changed')
        if stage == STARTUP_FAILED_STAGE:
            evidence = startup_failure_evidence(self.root, stage)
            exited = json.loads((self.root/(stage+'_process_exit.json')).read_text())
            if evidence is None or not exited['reaped'] or Path('/proc', str(exited['pid'])).exists():
                raise ValueError('Missing pre-scene failure or process release evidence')
            classification = 'pre_scene_native_fault'
            evidence_path = self.root/(stage+'_exit.json')
        else:
            classification = 'post_action_recorder_schema_error'
            evidence_path = self.root/stage/'worker_error.json'
        result = dict(software_complete=False, mode=mode, trials=[], registered=1,
            missing=[dict(spec=specs[0], stage=stage, native_outcome=None,
                classification=classification, replayed=False, evidence=str(evidence_path))])
        path = self.root/name/'result.json'
        path.parent.mkdir(exist_ok=True)
        if path.exists() and json.loads(path.read_text()) != result:
            raise ValueError('Preserved missing outcome changed')
        atomic_json(path, result)
        return result

    def run_motion(self):
        from visuotactile import state_full_run
        original_write = state_full_run.atomic_json
        def guarded_write(path, value):
            if path == self.root/'motion_selection.json':
                incomplete = []
                for name in ('single_takeover_12', 'single_takeover_24', 'h24_selected', 'temporal_h32_selected'):
                    result_path = self.root/name/'result.json'
                    if result_path.exists() and not json.loads(result_path.read_text()).get('software_complete'):
                        incomplete.append(name)
                if incomplete:
                    original_write(self.root/'motion_selection_withheld.json', dict(incomplete=incomplete,
                        reason='Preserved software-missing evaluations; no final selection or held-out test', utc=time.time()))
                    raise RuntimeError('Final selection/test withheld: '+', '.join(incomplete))
            return original_write(path, value)
        state_full_run.atomic_json = guarded_write
        try:
            return super().run_motion()
        finally:
            state_full_run.atomic_json = original_write

    def stage(self, name, module='visuotactile.state_benchmark', extra=None):
        verify_registration(self.root)
        if module != 'visuotactile.state_benchmark' or (self.root/(name+'_committed.json')).exists():
            return super().stage(name, module, extra)
        self.check()
        if name in (FAILED_STAGE, STARTUP_FAILED_STAGE):
            raise RuntimeError('Never rerun the preserved failed attempt')
        output = self.root/name
        if output.exists():
            raise RuntimeError('Preserve existing attempt, no blind recovery: '+name)
        args = ['bash', 'visuotactile/run_cpu.sh', '-m', 'tools.state_motion_recovery',
                'benchmark', '--root', str(self.root), '--output', str(output)]
        args += extra if extra is not None else ['--mode', name]
        atomic_json(self.root/(name+'_launch.json'), dict(command=args, utc=time.time(),
            recovery_registration_sha256=sha256(self.root/RECOVERY/'registration.json'),
            frozen_benchmark_sha256=sha256('visuotactile/state_benchmark.py'),
            recorder_and_lifecycle_override_only=True))
        with (self.root/(name+'.log')).open('a') as log:
            self.worker = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, start_new_session=True)
            atomic_json(self.root/'status.json', dict(state='running', stage=name,
                pid=os.getpid(), worker_pid=self.worker.pid, utc=time.time(), recovery=RECOVERY))
            while self.worker.poll() is None:
                self.check()
                reason = shutdown_stall(output, time.time())
                if reason:
                    atomic_json(self.root/(name+'_cleanup_watchdog.json'),
                        dict(pid=self.worker.pid, reason=reason, utc=time.time(), retry=False))
                    os.killpg(self.worker.pid, signal.SIGKILL)
                    self.worker.wait(timeout=15)
                    break
                time.sleep(1)
            code = self.worker.returncode
            exited_pid = self.worker.pid
            atomic_json(self.root/(name+'_process_exit.json'),
                dict(pid=exited_pid, code=code, utc=time.time(), reaped=True))
            self.worker = None
        if code != 0 or not (output/'result.json').exists() or (output/'worker_error.json').exists():
            atomic_json(self.root/(name+'_exit.json'), dict(error=f'{name} failed: exit={code}', utc=time.time()))
            if recover_preaction_start(self.root, name, self.session['deadline']):
                return self.stage(name, module, extra)
            raise RuntimeError(f'{name} failed: exit={code}')
        result = json.loads((output/'result.json').read_text())
        if result.get('software_complete') is not True:
            raise RuntimeError(name+' incomplete')
        self.verify_worker_exit(name)
        atomic_json(self.root/(name+'_committed.json'), dict(sha256=sha256(output/'result.json'), utc=time.time()))
        return result


def benchmark(root, output, mode, job):
    verify_registration(root)
    from visuotactile import state_benchmark
    state_benchmark.append_numeric = append_mode_numeric
    original_write = state_benchmark.atomic_json
    def write_result(path, value):
        return original_write(path, correct_takeover_result(path, value))
    state_benchmark.atomic_json = write_result
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    state_benchmark.run(root, output, mode, job)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('run', 'benchmark'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--mode')
    parser.add_argument('--job', type=Path)
    args = parser.parse_args()
    if args.command == 'benchmark':
        benchmark(args.root.resolve(), args.output.resolve(), args.mode, args.job)
    else:
        with (args.root/'owner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            RecoveryRunner(args.root.resolve()).run_motion()
