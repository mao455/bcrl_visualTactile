"""Memory/GPU-average admission with bounded telemetry; no external signals."""
from collections import deque
import json
import os
from pathlib import Path
import subprocess
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(temporary, path)


class ResourceGuard:
    def __init__(self, output, *, min_gpu_free_mib=4096, min_ram_available_gib=10,
                 poll_s=2, wait_timeout_s=3600, stop_requested=lambda: False,
                 max_gpu_mean_percent=50):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.min_gpu_free_mib = min_gpu_free_mib
        self.min_ram_bytes = min_ram_available_gib * 1024**3
        self.poll_s, self.timeout = poll_s, wait_timeout_s
        self.stop_requested = stop_requested
        self.max_gpu_mean = max_gpu_mean_percent
        # Start waiting below the cap to leave room for sampling/admission lag.
        self.gpu_admission_mean = None if self.max_gpu_mean is None else self.max_gpu_mean-5
        self.started = time.monotonic()
        self.last = None
        self.window = deque()
        self.samples = self.missing = 0
        self.area = self.measured_s = self.missing_s = self.wait_seconds = 0.
        self.max_rolling = self.max_used = self.max_rss = 0.
        self.min_ram = float('inf')
        self.max_gap = 0.

    def sample(self):
        import torch
        try:
            line = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.free',
                '--format=csv,noheader,nounits'], text=True, timeout=3).splitlines()[0]
            util, used, free = map(float, line.split(','))
        except (OSError, ValueError, subprocess.SubprocessError, IndexError):
            util = used = free = None
        mem = {line.split(':')[0]: int(line.split()[1])*1024
               for line in Path('/proc/meminfo').read_text().splitlines()}
        rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        now = time.monotonic()
        row = dict(pid=os.getpid(), utc_epoch=time.time(), elapsed_s=now-self.started, gpu_util_percent=util,
                   gpu_used_mib=used, gpu_free_mib=free, ram_available_bytes=mem['MemAvailable'],
                   process_rss_bytes=rss, cuda_allocated_bytes=torch.cuda.memory_allocated(),
                   cuda_reserved_bytes=torch.cuda.memory_reserved())
        if self.last:
            dt = row['elapsed_s']-self.last['elapsed_s']
            self.max_gap = max(self.max_gap, dt)
            prev = self.last['gpu_util_percent']
            if prev is None or dt > max(15., 3*self.poll_s):
                self.missing_s += dt
                value = None
            else:
                self.area += prev*dt
                self.measured_s += dt
                value = prev
            self.window.append((self.last['elapsed_s'], row['elapsed_s'], value))
        cutoff = row['elapsed_s']-60.
        while self.window and self.window[0][1] <= cutoff:
            self.window.popleft()
        area = duration = 0.
        missing = False
        for lo, hi, value in self.window:
            dt = hi-max(lo,cutoff)
            if value is None:
                missing = True
            else:
                area += value*dt
                duration += dt
        rolling = area/duration if duration >= 59.999 and not missing else None
        self.max_rolling = max(self.max_rolling, rolling or 0.)
        self.samples += 1
        self.missing += util is None
        self.max_used = max(self.max_used, used or 0)
        self.max_rss = max(self.max_rss, rss)
        self.min_ram = min(self.min_ram, mem['MemAvailable'])
        self.last = row
        row['rolling_60s_percent'] = rolling
        with (self.output/'resources.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        atomic_json(self.output/'resources.json', self.summary())
        return row

    def memory_safe(self, row):
        return (row['gpu_free_mib'] is not None and row['gpu_free_mib'] >= self.min_gpu_free_mib
                and row['ram_available_bytes'] >= self.min_ram_bytes)

    def utilization_safe(self, row):
        if self.max_gpu_mean is None:
            return True
        if row['gpu_util_percent'] is None or not self.measured_s:
            return False  # require an actual measured interval before admission
        means=[self.area/self.measured_s]
        if row['rolling_60s_percent'] is not None:
            means.append(row['rolling_60s_percent'])
        # Historical peak is evidence, not a condition that can never recover.
        return max(means) <= self.gpu_admission_mean

    def safe(self, row):
        return self.memory_safe(row) and self.utilization_safe(row)

    def before_chunk(self, log=None):
        import torch
        if self.last and time.monotonic()-self.started-self.last['elapsed_s'] < self.poll_s and self.safe(self.last):
            return
        row = self.sample()
        if not self.memory_safe(row):
            # Cached blocks are not live tensors. Release them before deciding
            # that our own previous peak has consumed the admission reserve.
            torch.cuda.empty_cache()
            row = self.sample()
        began = last_log = time.monotonic()
        while not self.safe(row):
            if self.stop_requested():
                raise InterruptedError('Stop while waiting for resources; use last safe group checkpoint')
            if time.monotonic()-began > self.timeout:
                raise RuntimeError('Resource budget unavailable for bounded wait; last safe checkpoint retained')
            if log and (last_log == began or time.monotonic()-last_log > 30):
                log('resource_wait '+json.dumps(row)); last_log = time.monotonic()
            tick = time.monotonic()
            time.sleep(self.poll_s)
            self.wait_seconds += time.monotonic()-tick
            row = self.sample()

    def summary(self):
        return dict(pid=os.getpid(), latest=self.last, samples=self.samples,
            gpu_time_weighted_mean_percent=self.area/self.measured_s if self.measured_s else None,
            max_complete_60s_mean_percent=self.max_rolling or None,
            missing_samples=self.missing, missing_seconds=self.missing_s, max_sample_gap_s=self.max_gap,
            wait_seconds=self.wait_seconds, peak_gpu_used_mib=self.max_used,
            peak_process_rss_bytes=self.max_rss, min_ram_available_bytes=self.min_ram if self.samples else None,
            thresholds=dict(gpu_free_mib=self.min_gpu_free_mib, ram_available_bytes=self.min_ram_bytes,
                            max_gpu_mean_percent=self.max_gpu_mean,gpu_admission_mean_percent=self.gpu_admission_mean),
            scope='whole card, time-weighted batch and complete 60-second means; current process lifetime; other jobs untouched')
