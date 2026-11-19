"""One registered recovery, gated on disk capacity under the original deadline.

No cleanup, resource-limit relaxation, deadline extension or repeated GPU retry.
The supervisor waits without loading a model; the unified job restores last.pt.
"""
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
PUBLIC = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def main():
    registration = read(PUBLIC / 'recovery_registration.json')
    root = Path(registration['original_root'])
    owner = Path(registration['owner_root'])
    deadline = registration['deadline']
    assert Path.cwd().resolve() == PROJECT
    assert read(root / 'session.json')['deadline'] == deadline
    assert read(owner / 'session.json')['deadline'] == deadline
    with (owner / 'recovery_supervisor.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not (owner / 'recovery_started.json').exists()
        write(owner / 'recovery_started.json', dict(pid=os.getpid(), utc=time.time()))
        try:
            while True:
                now = time.time()
                if now >= deadline:
                    raise TimeoutError('Original deadline reached while awaiting storage')
                if (root / 'STOP').exists() or (owner / 'STOP').exists():
                    raise InterruptedError('Explicit STOP')
                free = shutil.disk_usage(root).free
                write(owner / 'recovery_status.json', dict(
                    state='waiting_disk', utc=now, free_bytes=free,
                    minimum_start_free_bytes=registration['minimum_start_free_bytes'],
                    deadline=deadline, gpu_training_started=False))
                if free >= registration['minimum_start_free_bytes']:
                    break
                time.sleep(min(30, deadline-now))

            # Verify the durable identity before replacing only the telemetry route.
            from visuotactile.compact import sha256
            audit = read(PUBLIC / 'interruption_checkpoint_audit.json')
            for row in audit['checkpoints']:
                assert sha256(root / 'A/training' / row['file']) == row['sha256']
            assert (root / 'project_exit.json').is_file()
            assert read(root / 'project_exit.json')['children_reaped']
            assert not (root / 'project_complete.json').exists()
            assert not (owner / 'project_job_started.json').exists()
            # Original workers read root/shared_resources.json. Keep the old sample
            # as evidence and route the new monitor without changing trainer inputs.
            telemetry = root / 'shared_resources.json'
            previous = owner / 'original_shared_resources.json'
            assert not telemetry.is_symlink() and not previous.exists()
            telemetry.rename(previous)
            telemetry.symlink_to(owner / 'shared_resources.json')
            write(owner / 'recovery_status.json', dict(
                state='training', utc=time.time(), gpu_training_started=True,
                resumed_from_epoch=52, resumed_from_updates=4940))
            with (owner / 'recovery_train.log').open('x') as log:
                subprocess.run([sys.executable, '-m', 'tools.tight_project', 'train',
                    '--job', registration['training_job']], cwd=PROJECT,
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            completed = read(owner / 'project_complete.json')
            # Explicit aggregate receipt; retain original failure and exit records.
            write(root / 'project_complete.json', dict(completed,
                completion_owner=str(owner), recovered_from='original disk reserve failure'))
            write(owner / 'recovery_status.json', dict(state='post_training', utc=time.time()))
            with (owner / 'recovery_finish.log').open('x') as log:
                subprocess.run([sys.executable, str(PUBLIC / 'finish.py')],
                    cwd=PROJECT, env=dict(os.environ, TIGHT_POST_ATTEMPT_ROOT=str(owner)),
                    stdout=log, stderr=subprocess.STDOUT, check=True)
            write(owner / 'recovery_complete.json', dict(utc=time.time(), review_required=True))
        except BaseException as exc:
            write(owner / 'recovery_failure.json', dict(utc=time.time(), error=repr(exc)))
            raise


if __name__ == '__main__':
    main()
