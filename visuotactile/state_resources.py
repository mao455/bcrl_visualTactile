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

import math
import shutil

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
        if shutil.disk_usage(self.root).free < plan.get('reserve_bytes',12_000_000_000):
            raise RuntimeError('Configured disk reserve reached')

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
                    self.check(deadline)
                    interval = min(remaining, 1.)
                    time.sleep(interval)
                    remaining -= interval
                self.pacing_s += time.monotonic()-pacing
                with (self.root / 'work.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(pid=os.getpid(),utc=time.time(),work_s=duration,
                        total_work_s=self.work_s,total_wait_s=self.wait_s,total_pacing_s=self.pacing_s,
                        sleep_work_ratio=ratio,target_percent=target,short_mean_percent=state['short_mean_percent'],
                        memory=self.memory(cuda)))+'\n')
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
    if session.get('resource_policy') == 'full_gpu_v1':
        return FullLease(root, deadline=deadline)
    if session.get('resource_policy') == 'bounded_mean_v1':
        return Lease(root, deadline=deadline)
    if session.get('resource_policy') == 'shared_recent_v1':
        return SharedLease(root, deadline=deadline)
    raise ValueError('Current experiments require explicit shared_recent_v1 resource policy')


class FullLease(Lease):
    """Explicit full-card authorization; memory/disk/deadline remain enforced."""
    @contextmanager
    def block(self, *, deadline=None, cuda=True):
        import sys
        began=time.monotonic()
        worker_root=Path(sys.argv[sys.argv.index('--root')+1]) if '--root' in sys.argv else Path('.')
        group=next((p for p in worker_root.parts if p in ('G00','G10','G01','G11')),None)
        while True:
            self.check(deadline)
            schedule=self.root/'concurrency.json'
            paused=json.loads(schedule.read_text()).get('pause_groups',[]) if schedule.exists() else []
            if group not in paused:break
            time.sleep(.25)
        self.wait_s+=time.monotonic()-began
        row=json.loads((self.root/'shared_resources.json').read_text())
        limits=json.loads((self.root/'session.json').read_text())['resources']
        if time.time()-row['utc_epoch']>8:raise RuntimeError('Stale resource telemetry')
        if row['ram_available_bytes']<limits['min_ram_gib']*1024**3:raise RuntimeError('RAM reserve')
        if cuda and (row['gpu_free_mib'] is None or row['gpu_free_mib']<limits['min_gpu_free_mib']):
            raise RuntimeError('VRAM reserve; no OOM retry')
        started=time.monotonic()
        try:yield
        finally:self.work_s+=time.monotonic()-started


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
                    ratio=max(ratio,float(row.get('target_pacing_ratio',0.)))
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
    session=json.loads((root/'session.json').read_text())
    target=session.get('resource_target_percent')
    owned_roots={root.resolve(),*(Path(path).resolve() for path in session.get('owned_worker_roots',[]))}
    target_ratio=0.; last_adjustment=time.time()
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
                    owned = '--root' in command and Path(command[command.index('--root')+1]).resolve() in owned_roots
                except (OSError, IndexError):
                    owned = False
                processes.append(dict(pid=int(pid), name=name, vram_mib=int(memory), owned=owned))
        except (OSError, ValueError, subprocess.SubprocessError) as error:
            errors.append(repr(error))
        row.update(gpu_processes=processes, external_gpu_processes=[p for p in processes if not p['owned']],
            process_telemetry_errors=errors, external_throughput_available=False)
        reasons, ratio = policy(row, time.time())
        if session.get('resource_policy')=='full_gpu_v1':
            limits=session['resources'];ratio=0.
            reasons=[]
            if row.get('gpu_free_mib') is None or row['gpu_free_mib']<limits['min_gpu_free_mib']:reasons.append('VRAM_reserve')
            if row.get('ram_available_bytes',0)<limits['min_ram_gib']*1024**3:reasons.append('RAM_reserve')
        if target is not None and now-last_adjustment>=5:
            recent=row.get('rolling_60s_percent')
            if recent is not None:
                target_ratio=max(0.,min(4.,target_ratio+(.15 if recent>target+2 else -.1 if recent<target-2 else 0.)))
            last_adjustment=now
        row['target_pacing_ratio']=target_ratio
        ratio=max(ratio,target_ratio)
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
