"""Wall-clock supervision of ONLY owned children; no simulation-time changes."""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import stat
import subprocess
import time

from .trial_utils import write_json

LIMIT_BYTES = 60_000_000_000
# Data and runtime products, not source/assets/dependencies/released weights.
OUTPUT_ROOTS = ('outputs', 'train_dir', 'wandb', 'videos', 'evals', 'runs', 'build', 'dist')


def output_inventory(repo):
    """Apparent bytes, no symlink traversal, including nested temporary files."""
    result = {}
    for name in OUTPUT_ROOTS:
        root = Path(repo)/name
        if root.is_symlink() or not root.exists():
            continue
        count = 0
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(directory)/d).is_symlink()]
            for filename in files:
                try:
                    s = (Path(directory)/filename).lstat()
                    if stat.S_ISREG(s.st_mode): count += s.st_size
                except FileNotFoundError: pass  # An owned atomic publication.
        result[name] = count
    return result


def time_mean(samples, now, window=None):
    """Time-weighted previous observation, not a biased irregular sample mean.

    Missing telemetry is charged at 100% for the guard, never reported as a
    measured GPU value. All time is monotonic within one supervising process.
    """
    if not samples: return None
    start = max(samples[0][0], now-window) if window else samples[0][0]
    area = duration = 0.
    for i, (t, value) in enumerate(samples):
        stop = samples[i+1][0] if i+1 < len(samples) else now
        dt = max(0., min(stop, now)-max(t,start))
        area += dt*(100. if value is None else value); duration += dt
    return area/duration if duration else samples[-1][1]


def gpu_sample():
    try:
        v = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
            '--format=csv,noheader,nounits'],text=True,timeout=3).splitlines()[0].split(',')
        return float(v[0]), float(v[1])
    except (OSError, ValueError, subprocess.SubprocessError, IndexError):
        return None, None


class Supervisor:
    """Continuous one-second loop across child startup, execution and waiting.

    The same object covers a whole batch, including conversion/audit children.
    Parent work must be short; sampling gaps are explicitly reported. Signals
    are limited to a new session/process group created by this instance.
    """
    def __init__(self, root, deadline_epoch, repo=None, *, disk_reserve_bytes=None, average_only=False):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.repo=Path(repo or Path(__file__).resolve().parents[1])
        self.deadline=float(deadline_epoch);self.samples=[];self.peak_bytes=0;self.peak_mib=0
        self.started=time.monotonic();self.max_gap=0;self.missing=0;self.paused_s=0
        self.child=None;self.paused=False;self.stop_reason=None
        self.process_samples=[];self.last_process_sample=0
        self.disk_reserve_bytes=disk_reserve_bytes;self.average_only=average_only
        self._inventory={};self._inventory_time=-1e10
        self.sample()

    def sample(self):
        value,mib=gpu_sample();now=time.monotonic()
        if self.child and now-self.last_process_sample>=15:
            self.last_process_sample=now
            try:
                output=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True,timeout=3)
                for line in output.splitlines():
                    fields=line.split()
                    if len(fields)<7 or not fields[0].isdigit():continue
                    try:
                        pid=int(fields[1])
                        if os.getpgid(pid)!=self.child.pid:continue
                    except (ValueError,ProcessLookupError):continue
                    self.process_samples.append(dict(utc_epoch=time.time(),pid=pid,type=fields[2],
                        sm_percent=None if fields[3]=='-' else float(fields[3]),
                        memory_util_percent=None if fields[4]=='-' else float(fields[4]),raw=line))
                write_json(self.root/'owned_gpu_process_samples.json',self.process_samples)
            except (OSError,subprocess.SubprocessError):pass
        if self.samples:self.max_gap=max(self.max_gap,now-self.samples[-1][0])
        self.samples.append((now,value));self.missing+=int(value is None)
        if now-self._inventory_time>=30:
            self._inventory=output_inventory(self.repo);self._inventory_time=now
        inv=self._inventory;size=sum(inv.values())
        free=shutil.disk_usage(self.repo).free
        self.peak_bytes=max(self.peak_bytes,size);self.peak_mib=max(self.peak_mib,mib or 0)
        result=dict(utc_epoch=time.time(),elapsed_s=now-self.started,whole_card_percent=value,
            whole_card_rolling_60s_percent=time_mean(self.samples,now,60) if now-self.started>=60 else None,
            rolling_window_ready=now-self.started>=60,
            whole_card_batch_percent=time_mean(self.samples,now),gpu_memory_mib=mib,
            peak_gpu_memory_mib=self.peak_mib,project_output_bytes=size,peak_output_bytes=self.peak_bytes,
            inventory_bytes=inv,missing_samples=self.missing,max_sample_gap_s=self.max_gap,
            disk_free_bytes=free,disk_reserve_bytes=self.disk_reserve_bytes,
            owned_child_pid=self.child.pid if self.child else None,owned_group_paused=self.paused,
            deadline_epoch=self.deadline,wall_throttled_s=self.paused_s,
            coverage='Only measured interval from supervisor start; earlier probes unmonitored',
            attribution='Whole card is authoritative; pmon snapshots diagnostic, missing RTX/SM values never assumed zero')
        with (self.root/'gpu_samples.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        write_json(self.root/'resources.json',result)
        self.latest=result
        if time.time()>=self.deadline-5:self.stop_reason='absolute delivery deadline (5 s shutdown reserve)'
        # Keep 512 MB emergency margin for at most one sampling interval of writes.
        if self.disk_reserve_bytes is not None:
            if free<self.disk_reserve_bytes:self.stop_reason='filesystem free-space reserve'
        elif size>=LIMIT_BYTES-512_000_000:self.stop_reason='global output capacity margin'
        return result

    def check(self, reserve=0):
        if self.stop_reason:raise RuntimeError(self.stop_reason)
        if time.time()>=self.deadline:raise RuntimeError('absolute delivery deadline')
        if self.disk_reserve_bytes is not None:
            if shutil.disk_usage(self.repo).free<self.disk_reserve_bytes+reserve:
                raise RuntimeError('Insufficient free disk plus bounded temporary reserve')
        elif sum(self._inventory.values())+reserve>=LIMIT_BYTES-512_000_000:
            raise RuntimeError('Insufficient global capacity for next raw+compact attempt')

    def wait(self, seconds):
        until=time.monotonic()+seconds
        while time.monotonic()<until:
            self.sample();self.check();time.sleep(min(1,max(0,until-time.monotonic())))

    def run(self, cmd, log, timeout=900, reserve=0, stop_file=None, heartbeat=None):
        self.sample();self.check(reserve)
        # Do not start against a busy/unknown GPU. Include this wait in deadline.
        while self.latest['whole_card_percent'] is None or self.latest['whole_card_percent']>45:
            self.wait(1)
        start=time.monotonic();stop_requested=None
        heartbeat_value=None;stale_active_s=0.;last_tick=start
        with Path(log).open('x') as f:
            self.child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                while self.child.poll() is None:
                    self.sample();self.check()
                    tick=time.monotonic()
                    if heartbeat:
                        path=Path(heartbeat)
                        value=path.stat().st_mtime_ns if path.exists() else None
                        if value!=heartbeat_value:stale_active_s=0.;heartbeat_value=value
                        elif not self.paused:stale_active_s+=tick-last_tick
                        if stale_active_s>180:raise RuntimeError('Owned collector heartbeat stalled for 180 active seconds')
                    last_tick=tick
                    if stop_file and Path(stop_file).exists():
                        stop_requested=stop_requested or time.monotonic()
                        if time.monotonic()-stop_requested>60:raise RuntimeError('Owned collector graceful stop timeout')
                    if time.monotonic()-start>timeout:raise RuntimeError('Owned attempt wall limit')
                    now=time.monotonic()
                    if self.average_only:
                        threshold=40 if self.paused else 45
                        busy=(self.latest['whole_card_percent'] is None or
                              (self.latest['whole_card_rolling_60s_percent'] or 0)>threshold or
                              (now-self.started>=15 and (self.latest['whole_card_batch_percent'] or 0)>threshold))
                    else:
                        busy=(self.latest['whole_card_percent'] is None or self.latest['whole_card_percent']>45 or
                              (self.latest['whole_card_rolling_60s_percent'] or 0)>45)
                    # Signal only the isolated process group we created. Pausing
                    # elapsed wall time adds no physics/render/control step.
                    if busy!=self.paused and self.child.poll() is None:
                        os.killpg(self.child.pid,signal.SIGSTOP if busy else signal.SIGCONT)
                        self.paused=busy
                    time.sleep(1)
                    if self.paused:self.paused_s+=time.monotonic()-now
                return self.child.returncode
            finally:
                if self.child.poll() is None:
                    os.killpg(self.child.pid,signal.SIGCONT)
                    os.killpg(self.child.pid,signal.SIGTERM)
                    try:self.child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(self.child.pid,signal.SIGKILL);self.child.wait()
                self.child=None;self.paused=False;self.sample()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--deadline-epoch',type=float,required=True);p.add_argument('--reserve-bytes',type=int,default=12_000_000_000)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    s=Supervisor(a.output,a.deadline_epoch)
    cmd=a.command[1:] if a.command[0]=='--' else a.command
    try:
        rc=s.run(cmd,a.output/'child.log',reserve=a.reserve_bytes)
        write_json(a.output/'result.json',dict(returncode=rc,resources=s.latest))
    except BaseException as e:
        write_json(a.output/'result.json',dict(error=repr(e),resources=s.latest));raise
    raise SystemExit(rc if rc>=0 else 128-rc)

if __name__=='__main__':main()
