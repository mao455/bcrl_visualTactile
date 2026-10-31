"""One monitored GPU lease shared by all workers of a long-horizon session."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import time

from .training_resources import ResourceGuard, atomic_json


def update_pacing(state, row, target=40.):
    if not math.isfinite(target) or not 10 <= target <= 40:
        raise ValueError('GPU target must be between 10 and 40 percent, below the 50 percent cap')
    state = dict(state)
    now = row['utc_epoch']
    observations = list(state.get('observations',[]))
    if not observations or now > observations[-1][0]:
        observations.append([now,row['gpu_util_percent']])
    while len(observations)>1 and observations[1][0] <= now-10:
        observations.pop(0)
    area = duration = 0.
    valid = bool(observations and now-observations[0][0]>=10)
    for before,after in zip(observations,observations[1:]):
        interval = after[0]-max(before[0],now-10)
        if before[1] is None or after[0]-before[0]>4:
            valid = False
        else:
            area += before[1]*interval
            duration += interval
    short_mean = area/duration if valid and duration>=9.999 else None
    ratio = float(state.get('sleep_work_ratio',1.))
    adjusted = state.get('last_adjusted_utc',now)
    if now-adjusted>=5 and short_mean is not None:
        rolling = row.get('rolling_60s_percent')
        if short_mean>target+2 or (rolling is not None and rolling>target+2):
            ratio = min(6.,ratio+.25)
        elif short_mean<target-2 and rolling is not None and rolling<target+2:
            ratio = max(0.,ratio-.1)
        adjusted = now
    return dict(observations=observations,short_mean_percent=short_mean,sleep_work_ratio=ratio,
                last_adjusted_utc=adjusted,target_percent=target,updated_utc=now)


class Lease:
    def __init__(self, root, *, deadline=None):
        self.root = Path(root)
        self.deadline = deadline
        self.work_s = self.wait_s = self.pacing_s = 0.
        self.peak_rss_bytes = 0
        self.peak_cuda_allocated_bytes = self.peak_cuda_reserved_bytes = 0

    def memory(self, cuda=True):
        status = dict(line.split(':',1) for line in Path('/proc/self/status').read_text().splitlines() if ':' in line)
        self.peak_rss_bytes = max(self.peak_rss_bytes,int(status['VmHWM'].split()[0])*1024)
        if cuda:
            import torch
            self.peak_cuda_allocated_bytes = max(self.peak_cuda_allocated_bytes,torch.cuda.max_memory_allocated())
            self.peak_cuda_reserved_bytes = max(self.peak_cuda_reserved_bytes,torch.cuda.max_memory_reserved())
        return dict(peak_rss_bytes=self.peak_rss_bytes,peak_cuda_allocated_bytes=self.peak_cuda_allocated_bytes,
                    peak_cuda_reserved_bytes=self.peak_cuda_reserved_bytes,
                    scope='this worker; CUDA allocator excludes renderer and driver allocations')

    def check(self, deadline=None):
        plan = json.loads((self.root / 'session.json').read_text())
        if time.time() >= (deadline or self.deadline or plan['training_deadline']):
            raise TimeoutError('Session deadline reached; preserve resumable checkpoint')
        if (self.root / 'STOP').exists():
            raise InterruptedError('Owned session STOP requested')
        if shutil.disk_usage(self.root).free < 12_000_000_000:
            raise RuntimeError('Disk reserve below 12 decimal GB')

    @contextmanager
    def block(self, *, deadline=None, cuda=True):
        tick = time.monotonic()
        with (self.root / 'gpu.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            while True:
                self.check(deadline)
                try:
                    report = json.loads((self.root / 'resources/resources.json').read_text())
                    row = report['latest']
                    settings_path = self.root/'gpu_schedule.json'
                    settings = json.loads(settings_path.read_text()) if settings_path.exists() else {}
                    target = float(settings.get('target_percent',40.))
                    state_path = self.root/'gpu_pacing_state.json'
                    state = json.loads(state_path.read_text()) if state_path.exists() else {}
                    state = update_pacing(state,row,target)
                    atomic_json(state_path,state)
                    admission = min(45.,float(settings.get('admission_percent',45.)))
                    free_mib = max(6144.,float(settings.get('min_gpu_free_mib',6144.)))
                    ram_bytes = max(14.,float(settings.get('min_ram_gib',14.)))*1024**3
                    safe = (time.time()-row['utc_epoch'] < 4 and row['gpu_free_mib'] is not None
                            and row['gpu_free_mib'] >= free_mib and row['ram_available_bytes'] >= ram_bytes
                            and report['gpu_time_weighted_mean_percent'] is not None
                            and report['gpu_time_weighted_mean_percent'] <= admission
                            and row['rolling_60s_percent'] is not None and row['rolling_60s_percent'] <= admission
                            and (state['short_mean_percent'] is None or state['short_mean_percent']<=admission))
                except (FileNotFoundError, ValueError, KeyError, TypeError):
                    safe = False
                if safe:
                    break
                time.sleep(1)
            self.wait_s += time.monotonic()-tick
            began = time.monotonic()
            try:
                yield
            finally:
                if cuda:
                    import torch
                    torch.cuda.synchronize()
                duration = time.monotonic()-began
                self.work_s += duration
                pacing = time.monotonic()
                ratio = max(2.5,state['sleep_work_ratio']) if duration>=2 else state['sleep_work_ratio']
                remaining = ratio*duration
                while remaining > 0:
                    interval = min(remaining, 1.)
                    time.sleep(interval)
                    remaining -= interval
                self.pacing_s += time.monotonic()-pacing
                with (self.root / 'work.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(pid=os.getpid(),utc=time.time(),work_s=duration,
                        total_work_s=self.work_s,total_wait_s=self.wait_s,total_pacing_s=self.pacing_s,
                        sleep_work_ratio=ratio,target_percent=target,short_mean_percent=state['short_mean_percent'],
                        memory=self.memory(cuda)))+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    guard = ResourceGuard(args.root / 'resources', poll_s=1, cuda_metrics=False)
    atomic_json(args.root / 'monitor.json', dict(pid=os.getpid(),started=time.time()))
    while not (args.root / 'STOP_MONITOR').exists():
        started = time.monotonic()
        guard.sample()
        time.sleep(max(.05, 1.-(time.monotonic()-started)))
    guard.sample()


if __name__ == '__main__':
    main()
