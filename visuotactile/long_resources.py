"""One monitored GPU lease shared by all workers of a long-horizon session."""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import shutil
import time

from .training_resources import ResourceGuard, atomic_json


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
                    safe = (time.time()-row['utc_epoch'] < 4 and row['gpu_free_mib'] is not None
                            and row['gpu_free_mib'] >= 6144 and row['ram_available_bytes'] >= 14*1024**3
                            and report['gpu_time_weighted_mean_percent'] is not None
                            and report['gpu_time_weighted_mean_percent'] <= 38
                            and row['rolling_60s_percent'] is not None and row['rolling_60s_percent'] <= 35)
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
                remaining = 2.5*duration
                while remaining > 0:
                    interval = min(remaining, 1.)
                    time.sleep(interval)
                    remaining -= interval
                self.pacing_s += time.monotonic()-pacing
                with (self.root / 'work.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(pid=os.getpid(),utc=time.time(),work_s=duration,
                        total_work_s=self.work_s,total_wait_s=self.wait_s,total_pacing_s=self.pacing_s,
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
