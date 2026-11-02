"""Opt-in shared-card scheduling: memory admission, recent-load pacing, no GPU % veto."""
import argparse
from collections import deque
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from .long_resources import Lease
from .native_resources import restore_telemetry
from .training_resources import ResourceGuard, atomic_json


def policy(row, now):
    reasons = []
    if now-row.get('utc_epoch', 0) > 8 or row.get('gpu_free_mib') is None:
        reasons.append('missing_or_stale_memory_telemetry')
    elif row['gpu_free_mib'] < 6144:
        reasons.append('free_vram_below_6GiB')
    if row.get('ram_available_bytes', 0) < 12*1024**3:
        reasons.append('available_ram_below_12GiB')
    one, five = row.get('rolling_60s_percent'), row.get('rolling_300s_percent')
    ratio = 0.
    if row.get('external_gpu_processes'):
        ratio = .25
        if one is not None and one > 85:
            ratio = 1.
        if one is not None and five is not None and one > 92 and five > 85:
            ratio = 2.
    return reasons, ratio


def lease_for(root, deadline=None):
    session = json.loads((Path(root)/'session.json').read_text())
    if session.get('resource_policy') == 'shared_recent_v1':
        return SharedLease(root, deadline=deadline)
    from .native_v2_resources import LeaseV2
    return LeaseV2(root, deadline=deadline)


class SharedLease(Lease):
    @contextmanager
    def block(self, *, deadline=None, cuda=True):
        began = time.monotonic()
        with (self.root/'gpu.lock').open('a') as lock:
            while True:
                self.check(deadline)
                try:
                    row = json.loads((self.root/'shared_resources.json').read_text())
                    reasons, ratio = policy(row, time.time())
                except (OSError, ValueError):
                    reasons, ratio = ['telemetry_not_ready'], 0.
                if not reasons:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        reasons = ['owned_gpu_worker_busy']
                atomic_json(self.root/f'wait_{os.getpid()}.json', dict(utc=time.time(), reasons=reasons))
                time.sleep(.25)
            self.wait_s += time.monotonic()-began
            (self.root/f'wait_{os.getpid()}.json').unlink(missing_ok=True)
            started = time.monotonic()
            try:
                yield
            finally:
                try:
                    if cuda:
                        import torch
                        if torch.cuda.is_initialized():
                            torch.cuda.synchronize()
                finally:
                    duration = time.monotonic()-started
                    self.work_s += duration
                    fcntl.flock(lock, fcntl.LOCK_UN)
        pacing = time.monotonic()
        until = pacing+duration*ratio
        while time.monotonic() < until:
            self.check(deadline)
            time.sleep(min(.25, max(0, until-time.monotonic())))
        self.pacing_s += time.monotonic()-pacing
        with (self.root/'work.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(pid=os.getpid(), utc=time.time(), work_s=duration,
                total_work_s=self.work_s, total_wait_s=self.wait_s, total_pacing_s=self.pacing_s,
                pacing_ratio=ratio, memory=self.memory(cuda)))+'\n')


def monitor(root):
    guard = ResourceGuard(root/'resources', poll_s=1, cuda_metrics=False,
        min_gpu_free_mib=6144, min_ram_available_gib=12, max_gpu_mean_percent=None)
    restore_telemetry(guard)
    atomic_json(root/'monitor.json', dict(pid=os.getpid(), started=time.time()))
    window = deque(); previous = None
    while not (root/'STOP_MONITOR').exists():
        began = time.monotonic(); row = guard.sample(); now = row['utc_epoch']
        window.append((now, row['gpu_util_percent']))
        while len(window) > 1 and window[1][0] < now-300:
            window.popleft()
        intervals = [(b[0]-max(a[0], now-300), a[1]) for a, b in zip(window, list(window)[1:])]
        valid = bool(intervals) and all(v is not None and 0 <= dt < 8 for dt, v in intervals)
        row['rolling_300s_percent'] = (sum(dt*v for dt, v in intervals)/sum(dt for dt, _ in intervals)
            if valid and now-window[0][0] >= 299 else None)
        processes = []; errors = []
        try:
            lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines()
            for line in lines:
                pid, name, memory = [v.strip() for v in line.split(',', 2)]
                try:
                    command = Path('/proc', pid, 'cmdline').read_bytes().decode().split('\0')
                    owned = '--root' in command and Path(command[command.index('--root')+1]).resolve() == root.resolve()
                except (OSError, IndexError):
                    owned = False
                processes.append(dict(pid=int(pid), name=name, vram_mib=int(memory), owned=owned))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(repr(error))
        row.update(gpu_processes=processes, external_gpu_processes=[p for p in processes if not p['owned']],
            process_telemetry_errors=errors, external_throughput_available=False)
        reasons, ratio = policy(row, time.time())
        row.update(pause_reasons=reasons, pacing_ratio=ratio)
        atomic_json(root/'shared_resources.json', row)
        with (root/'resources/shared.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if (reasons, ratio) != previous:
            with (root/'resource_decisions.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(utc=now, reasons=reasons, pacing_ratio=ratio))+'\n')
            previous = reasons, ratio
        time.sleep(max(.05, 1-(time.monotonic()-began)))
    guard.sample()
    atomic_json(root/'monitor_exit.json', dict(pid=os.getpid(), ended=time.time(), summary=guard.summary()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', required=True, type=Path)
    monitor(parser.parse_args().root)
