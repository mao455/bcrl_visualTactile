"""Independent resource watchdog for the bounded loss/residual study."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from .long_resources import Lease
from .training_resources import ResourceGuard, atomic_json


class StudyLease(Lease):
    def check(self, deadline=None):
        super().check(deadline)
        while (self.root/'RESOURCE_PAUSE').exists():
            super().check(deadline)
            time.sleep(1.)


def owned_workers(root):
    result = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            arguments = path.read_bytes().decode().split('\0')
            if '--root' not in arguments:
                continue
            supplied = Path(arguments[arguments.index('--root')+1]).resolve()
            modules = ('visuotactile.train_loss_study','visuotactile.study_audit',
                       'visuotactile.train_transformer_residual','visuotactile.evaluate_study',
                       'visuotactile.transformer_data')
            if supplied == root.resolve() and any(module in arguments for module in modules):
                result.append(int(path.parent.name))
        except (OSError,ValueError,IndexError):
            pass
    return result


def violation_reasons(row, summary):
    reasons = []
    if row['gpu_util_percent'] is None:
        reasons.append('missing_gpu_telemetry')
    if row['gpu_free_mib'] is not None and row['gpu_free_mib'] < 4096:
        reasons.append('vram_below_4gib')
    if row['ram_available_bytes'] < 10*1024**3:
        reasons.append('ram_below_10gib')
    for label,value in (('complete60',row['rolling_60s_percent']),
                        ('batch',summary['gpu_time_weighted_mean_percent'])):
        if value is not None and value > 50:
            reasons.append(label+'_above_50')
    return reasons


def monitor(root):
    session = json.loads((root/'session.json').read_text())
    guard = ResourceGuard(root/'resources',poll_s=1,cuda_metrics=False)
    atomic_json(root/'monitor.json',dict(pid=os.getpid(),started_epoch=time.time()))
    recovery_since = None
    signalled = set()
    while not (root/'STOP_MONITOR').exists():
        started = time.monotonic()
        row = guard.sample()
        summary = guard.summary()
        reasons = violation_reasons(row,summary)
        if reasons:
            event = dict(utc_epoch=time.time(),reasons=reasons,sample=row,
                         action='pause all owned GPU leases; lower target to35; preserve existing checkpoints')
            atomic_json(root/'RESOURCE_PAUSE',event)
            with (root/'resource_incidents.jsonl').open('a') as stream:
                stream.write(json.dumps(event)+'\n')
            settings = json.loads((root/'gpu_schedule.json').read_text())
            settings['target_percent'] = 35.
            atomic_json(root/'gpu_schedule.json',settings)
            recovery_since = None
        elif (root/'RESOURCE_PAUSE').exists():
            recovered = (row['rolling_60s_percent'] is not None and row['rolling_60s_percent'] < 40
                         and row['gpu_free_mib'] >= 6144 and row['ram_available_bytes'] >= 16*1024**3)
            recovery_since = (recovery_since or time.monotonic()) if recovered else None
            if recovery_since is not None and time.monotonic()-recovery_since >= 60:
                (root/'RESOURCE_PAUSE').unlink()
        if time.time() >= session['deadline']:
            atomic_json(root/'STOP',dict(reason='original six-hour deadline',utc_epoch=time.time()))
            workers = owned_workers(root)
            for pid in workers:
                if pid not in signalled:
                    os.kill(pid,signal.SIGTERM)
                    signalled.add(pid)
            if not workers:
                break
        time.sleep(max(.05,1.-(time.monotonic()-started)))
    guard.sample()
    atomic_json(root/'monitor_exit.json',dict(pid=os.getpid(),utc_epoch=time.time(),summary=guard.summary()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    monitor(parser.parse_args().root)
