"""Persistent whole-card telemetry for native DP; only owned workers may stop."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import time

from .study_resources import StudyLease, violation_reasons
from .training_resources import ResourceGuard, atomic_json


class NativeLease(StudyLease):
    def check(self, deadline=None):
        while shutil.disk_usage(self.root).free < 12_000_000_000:
            session = json.loads((self.root/'session.json').read_text())
            if time.time() >= (deadline or self.deadline or session['training_deadline']):
                raise TimeoutError('Deadline while waiting for disk reserve')
            if (self.root/'STOP').exists():
                raise InterruptedError('Stop while waiting for disk reserve')
            atomic_json(self.root/'disk_wait.json', dict(state='waiting_resources', resource='disk', utc_epoch=time.time()))
            time.sleep(2)
        super().check(deadline)


def restore_telemetry(guard):
    path = guard.output/'resources.jsonl'
    if not path.exists():
        return
    previous = None
    first_utc = None
    for line in path.read_text().splitlines():
        row = json.loads(line)
        first_utc = row['utc_epoch'] if first_utc is None else first_utc
        if previous is not None:
            duration = row['elapsed_s']-previous['elapsed_s']
            value = previous['gpu_util_percent'] if duration <= 15 else None
            if value is None:
                guard.missing_s += duration
            else:
                guard.area += value*duration
                guard.measured_s += duration
            guard.window.append((previous['elapsed_s'], row['elapsed_s'], value))
            while guard.window and guard.window[0][1] <= row['elapsed_s']-60:
                guard.window.popleft()
            guard.max_gap = max(guard.max_gap, duration)
        guard.samples += 1
        guard.missing += row['gpu_util_percent'] is None
        guard.max_rolling = max(guard.max_rolling, row.get('rolling_60s_percent') or 0)
        guard.max_used = max(guard.max_used, row['gpu_used_mib'] or 0)
        guard.min_ram = min(guard.min_ram, row['ram_available_bytes'])
        previous = row
    if previous is not None:
        guard.started = time.monotonic()-(time.time()-first_utc)
        guard.last = previous


def monitor(root):
    guard = ResourceGuard(root/'resources', poll_s=1, cuda_metrics=False)
    restore_telemetry(guard)
    atomic_json(root/'monitor.json', dict(pid=os.getpid(), started=time.time()))
    recovery = None
    while not (root/'STOP_MONITOR').exists():
        began = time.monotonic()
        row = guard.sample()
        extra = dict(utc_epoch=row['utc_epoch'], power_w=None, gpu_processes=None)
        try:
            extra['power_w'] = float(subprocess.check_output(['nvidia-smi', '--query-gpu=power.draw',
                '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines()[0])
            processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True, timeout=3)
            extra['gpu_processes'] = processes.splitlines()
        except (OSError, ValueError, subprocess.SubprocessError, IndexError) as error:
            extra['telemetry_error'] = repr(error)
        with (root/'resources/power_processes.jsonl').open('a') as stream:
            stream.write(json.dumps(extra)+'\n')
        reasons = violation_reasons(row, guard.summary())
        if reasons:
            atomic_json(root/'RESOURCE_PAUSE', dict(reasons=reasons, utc_epoch=time.time()))
            with (root/'resource_incidents.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(reasons=reasons, sample=row))+'\n')
            recovery = None
        elif (root/'RESOURCE_PAUSE').exists():
            safe = (row['rolling_60s_percent'] is not None and row['rolling_60s_percent'] < 40
                    and guard.summary()['gpu_time_weighted_mean_percent'] < 45
                    and row['gpu_free_mib'] >= 6144 and row['ram_available_bytes'] >= 14*1024**3)
            recovery = (recovery or time.monotonic()) if safe else None
            if recovery is not None and time.monotonic()-recovery >= 60:
                (root/'RESOURCE_PAUSE').unlink()
        time.sleep(max(.05, 1-(time.monotonic()-began)))
    guard.sample()
    atomic_json(root/'monitor_exit.json', dict(pid=os.getpid(), ended=time.time(), summary=guard.summary()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    monitor(parser.parse_args().root)
