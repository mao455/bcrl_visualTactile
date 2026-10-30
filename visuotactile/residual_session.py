"""Continuous external telemetry and immutable deadlines for residual experiments."""
import argparse
import atexit
import json
import os
from pathlib import Path
import shutil
import time

from .training_resources import ResourceGuard, atomic_json


class SessionGate:
    def __init__(self, root, *, gpu_stage=False):
        self.root = Path(root)
        self.plan = json.loads((self.root / 'session.json').read_text())
        self.gpu_stage = gpu_stage
        self.wait_s = 0.
        self.admission_wait_s = 0.
        self.pacing_wait_s = 0.
        atexit.register(self.save_waits)

    def save_waits(self):
        atomic_json(self.root / ('waits_'+str(os.getpid())+'.json'), dict(pid=os.getpid(),
            admission_wait_s=self.admission_wait_s, pacing_wait_s=self.pacing_wait_s,
            total_wait_s=self.wait_s, recorded_epoch=time.time()))

    def remaining(self):
        deadline = self.plan['deadline_epoch']
        stage = self.root / 'gpu_stage.json'
        if self.gpu_stage and stage.exists():
            deadline = min(deadline, json.loads(stage.read_text())['deadline_epoch'])
        return deadline - time.time()

    def before(self, reserve_s=0):
        began = time.monotonic()
        try:
            while True:
                if self.remaining() <= reserve_s or (self.root / 'STOP').exists():
                    raise InterruptedError('Immutable session budget/STOP reached')
                status = json.loads((self.root / 'admission.json').read_text())
                if time.time() - status['utc_epoch'] > 5:
                    raise RuntimeError('Continuous resource monitor is stale')
                if shutil.disk_usage(self.root).free < 12_000_000_000:
                    raise RuntimeError('Disk reserve below 12 GB')
                if status['safe']:
                    break
                time.sleep(.25)
        finally:
            waited = time.monotonic() - began
            self.wait_s += waited
            self.admission_wait_s += waited
        if self.gpu_stage and not (self.root / 'gpu_stage.json').exists():
            atomic_json(self.root / 'gpu_stage.json', dict(started_epoch=time.time(),
                        deadline_epoch=min(time.time()+5400, self.plan['deadline_epoch'])))

    def pace(self, began):
        delay = min(2 * (time.monotonic() - began), max(0., self.remaining()))
        started = time.monotonic()
        time.sleep(delay)
        waited = time.monotonic() - started
        self.wait_s += waited
        self.pacing_wait_s += waited


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--viewing', action='store_true')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=False)
    started = time.time()
    duration = 7200 if args.viewing else 21600
    seeds = [981002] if args.viewing else [981001, 981002]
    atomic_json(args.root / 'session.json', dict(session_version=1, started_epoch=started,
        deadline_epoch=started+duration, first_gpu_stage_seconds=5400, monitor_pid=os.getpid(),
        purpose='viewing_rerun' if args.viewing else 'residual_training_diagnostics',
        pilot_train_count=16, pilot_validation_count=8, full_epochs=5, extra_epochs_max=5,
        diagnostic_seeds=seeds, diagnostics_max=6 if args.viewing else 12,
        diagnostics=[dict(mode=mode, seed=seed, variant=variant)
            for mode in ('autonomous', 'expert_rollin') for seed in seeds
            for variant in ('a', 'b', 'c')],
        rollin_condition='first observation with valid max finger force >0.1 N for 3 consecutive control ticks',
        rollin_limit_steps=240, missing_contact='no switch; report unavailable, no replacement',
        memory='causal warmup from reset, no clearing at switch',
        gate='real current observed max finger force >0.1 N and valid',
        physics_hz=120, control_hz=60))
    guard = ResourceGuard(args.root / 'resources', poll_s=1, cuda_metrics=False)
    try:
        while True:
            row = guard.sample()
            safe = guard.safe(row) and row['rolling_60s_percent'] is not None
            atomic_json(args.root / 'admission.json', dict(utc_epoch=time.time(), safe=safe,
                        resources=guard.summary()))
            if (args.root / 'MONITOR_STOP').exists() or time.time() >= started+duration:
                break
            time.sleep(1)
    finally:
        guard.sample()
        atomic_json(args.root / 'monitor_final.json', dict(ended_epoch=time.time(),
                    resources=guard.summary(), owned_monitor_exited=True))


if __name__ == '__main__':
    main()
