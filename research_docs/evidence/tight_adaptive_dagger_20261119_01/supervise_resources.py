"""Finite read-only supervision; distinguish actual work, resource wait, and failed coordinator."""
import datetime
import json
import os
from pathlib import Path
import shutil
import time
import psutil

ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def snapshot(registration):
    now = time.time()
    owner = None
    children = []
    try:
        owner = psutil.Process(registration['pid'])
        if owner.create_time() != registration['create_time'] or owner.status() == psutil.STATUS_ZOMBIE:
            owner = None
        else:
            for process in owner.children(recursive=True):
                try:
                    children.append(dict(pid=process.pid, create_time=process.create_time(),
                                         name=process.name(), rss_gib=process.memory_info().rss / 2**30))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
    except psutil.NoSuchProcess:
        pass
    failures = [dict(path=str(p), receipt=read(p)) for p in ROOT.glob('round2*failure.json')
                if p.stat().st_mtime >= registration['utc']]
    admissions = list((ROOT / 'owners').glob('*/resource_admission.json'))
    latest_admission = max(admissions, key=lambda p: p.stat().st_mtime) if admissions else None
    progress = read(ROOT / 'progress.json')
    complete = read(ROOT / 'cycle_complete.json')
    closed_batches = [dict(path=str(p.parent), result=read(p))
                      for p in sorted((ROOT / 'round02').glob('ram_recovery03_micro*/result.json'))]
    if complete:
        state = 'complete'
    elif failures or owner is None:
        state = 'failed_or_coordinator_exited'
    elif children:
        state = 'owned_workers_active'
    else:
        state = 'coordinator_waiting_without_worker'
    return dict(utc=now, local_time=datetime.datetime.now().astimezone().isoformat(),
                state=state, progress=progress, coordinator_alive=owner is not None, children=children,
                disk_free_gib=shutil.disk_usage(ROOT).free / 2**30,
                ram_available_gib=psutil.virtual_memory().available / 2**30,
                latest_admission_path=str(latest_admission),
                latest_admission=read(latest_admission) if latest_admission else None,
                closed_recovery_batches=closed_batches, failures=failures,
                final_test_opened=(ROOT / 'final_freeze.json').exists(),
                deadline=read(ROOT / 'session.json')['deadline'], other_project_mutations=0,
                limitation='Worker existence is not a completed physical case; native worker monitors enforce runtime guards. This supervisor reads state and does not kill or alter any process.')


def run():
    registration = read(ROOT / 'round2_ram_recovery03_controller.json')
    with (ROOT / 'supervision_started.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                       utc=time.time(), coordinator=registration, poll_seconds=30), stream)
    with (ROOT / 'supervision.jsonl').open('x') as stream:
        while True:
            row = snapshot(registration)
            stream.write(json.dumps(row) + '\n')
            stream.flush()
            write(ROOT / 'supervision_latest.json', row)
            if row['state'] in ('complete', 'failed_or_coordinator_exited') or time.time() >= row['deadline']:
                write(ROOT / 'supervision_stopped.json', row)
                return
            time.sleep(min(30, max(.1, row['deadline'] - time.time())))


if __name__ == '__main__':
    run()
