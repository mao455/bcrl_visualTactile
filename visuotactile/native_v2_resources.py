"""V2 lease: CPU loading and resource/pacing waits never own the GPU lock."""
from contextlib import contextmanager
import fcntl
import json
import os
import time
from .native_resources import NativeLease
from .long_resources import update_pacing
from .training_resources import atomic_json


class LeaseV2(NativeLease):
    def admission(self):
        self.check()
        try:
            report=json.loads((self.root/'resources/resources.json').read_text());row=report['latest']
            safe=(time.time()-row['utc_epoch']<4 and row['gpu_free_mib'] is not None and row['gpu_free_mib']>=6144
                  and row['ram_available_bytes']>=14*1024**3 and report['gpu_time_weighted_mean_percent'] is not None
                  and report['gpu_time_weighted_mean_percent']<=45 and row['rolling_60s_percent'] is not None
                  and row['rolling_60s_percent']<=45 and not (self.root/'RESOURCE_PAUSE').exists())
            return safe,row
        except (FileNotFoundError,KeyError,ValueError,TypeError):
            return False,None

    @contextmanager
    def block(self, *, deadline=None, cuda=True):
        began=time.monotonic()
        with (self.root/'gpu.lock').open('a') as lock:
            while True:
                safe,row=self.admission()
                if safe:
                    try:
                        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        pass
                    else:
                        break
                time.sleep(.25)
            self.wait_s+=time.monotonic()-began
            work_start=time.monotonic()
            ratio=1.
            try:
                state_path=self.root/'gpu_pacing_state.json'
                state=json.loads(state_path.read_text()) if state_path.exists() else {}
                state=update_pacing(state,row,40.);atomic_json(state_path,state)
                ratio=state['sleep_work_ratio']
                yield
            finally:
                try:
                    if cuda:
                        import torch
                        torch.cuda.synchronize()
                finally:
                    duration=time.monotonic()-work_start
                    self.work_s+=duration
                    fcntl.flock(lock,fcntl.LOCK_UN)
        pace_start=time.monotonic();remaining=duration*(max(2.5,ratio) if duration>=2 else ratio)
        while remaining>0:
            self.check(deadline)
            delay=min(.25,remaining);time.sleep(delay);remaining-=delay
        self.pacing_s+=time.monotonic()-pace_start
        with (self.root/'work.jsonl').open('a') as f:
            f.write(json.dumps(dict(pid=os.getpid(),utc=time.time(),work_s=duration,total_work_s=self.work_s,
                total_wait_s=self.wait_s,total_pacing_s=self.pacing_s,lock_includes_wait=False,
                memory=self.memory(cuda)))+'\n')
