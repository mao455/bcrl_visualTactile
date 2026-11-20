"""Finite actual-job resource admission with queued single-lane collection."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

PUBLIC = Path(__file__).resolve().parent
ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_registration():
    session = read(ROOT / 'session.json')
    registration = read(ROOT / 'round02/ram_recovery03.json')
    if time.time() >= session['deadline']:
        raise TimeoutError('Original session deadline; recovery cannot extend it')
    assert registration['original_deadline'] == session['deadline']
    assert not (ROOT / 'round02/learning').exists()
    assert not (ROOT / 'final_freeze.json').exists()
    predecessor = Path(registration['previous_failure']).resolve()
    assert predecessor.is_relative_to(ROOT.resolve()) and predecessor.is_file()
    assert digest(predecessor) == registration['previous_failure_sha256']
    assert set(registration['verified_batches']) == set(registration['verified_batch_result_sha256'])
    for folder, expected in registration['verified_batch_result_sha256'].items():
        folder = Path(folder).resolve()
        assert folder.is_relative_to(ROOT.resolve())
        assert digest(folder / 'result.json') == expected
    assert digest(PUBLIC / 'round02/strategy.json') == registration['source_strategy_sha256']
    assert registration['stable_resource_admission']['stable_seconds'] >= 60
    return session, registration


class StableWindow:
    def __init__(self, seconds, poll):
        self.seconds = seconds
        self.poll = poll
        self.since = None
        self.last = None

    def observe(self, now, ready):
        if not ready or (self.last is not None and now - self.last > 2 * self.poll + 5):
            self.since = None
        if ready and self.since is None:
            self.since = now
        self.last = now
        return bool(ready and now - self.since >= self.seconds)


def stable_admission(path, *, simulation=False, lanes=4):
    import psutil
    session = read(ROOT / 'session.json')
    registration = read(ROOT / 'round02/ram_recovery03.json')
    policy = registration['stable_resource_admission']
    window = StableWindow(policy['stable_seconds'], policy['poll_seconds'])
    required_ram = policy['collection_min_ram_gib' if simulation and lanes == 1 else 'simulation_min_ram_gib' if simulation else 'training_min_ram_gib']
    required_vram = policy['simulation_min_vram_mib' if simulation else 'training_min_vram_mib']
    required_disk = session['resources']['min_disk_gib'] + 4
    job_path = PUBLIC.parents[2] / 'configs/experiments/adaptive_dagger' / (Path(path).name + '.json')
    if simulation and job_path.is_file():
        for stage in read(job_path)['stages']:
            if '--output' in stage['args']:
                plan_path = Path(stage['args'][stage['args'].index('--output') + 1]) / 'plan.json'
                required_disk = max(required_disk, read(plan_path).get('minimum_start_disk_gib', 0))
    while True:
        now = time.time()
        if now >= session['deadline']:
            raise TimeoutError('Original session deadline during stable resource admission')
        disk = shutil.disk_usage(ROOT).free / 2**30
        ram = psutil.virtual_memory().available / 2**30
        gpu_error = None
        try:
            gpu = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                text=True, timeout=15)
            vram = int(gpu.strip().splitlines()[0])
        except (subprocess.SubprocessError, OSError, ValueError, IndexError) as error:
            vram = 0
            gpu_error = repr(error)
        stamp = time.monotonic()
        instant = (disk >= required_disk
                   and ram >= required_ram and vram >= required_vram)
        ready = window.observe(stamp, instant)
        write(Path(path) / 'resource_admission.json', dict(
            ready=ready, instantaneous_ready=instant, disk_gib=disk, required_disk_gib=required_disk, ram_gib=ram,
            vram_free_mib=vram, required_ram_gib=required_ram, required_vram_mib=required_vram,
            stable_seconds_required=policy['stable_seconds'],
            stable_seconds_observed=0 if window.since is None else stamp - window.since,
            simulation=simulation, lanes=lanes, utc=time.time(), deadline=session['deadline'],
            gpu_query_error=gpu_error, other_project_mutations=0))
        if ready and time.time() < session['deadline']:
            return
        time.sleep(min(policy['poll_seconds'], max(.1, session['deadline'] - time.time())))


def run():
    validate_registration()
    receipt = ROOT / 'round02/ram_recovery03_bootstrap_started.json'
    with receipt.open('x') as stream:
        json.dump(dict(pid=os.getpid(), utc=time.time()), stream)
    write(ROOT / 'progress.json', dict(
        round=2, stage='prepare_single_lane_collection', utc=time.time(), verified_corrections=17))
    # Each actual job performs stable admission after its plan exists; avoid duplicate full windows.
    spec = importlib.util.spec_from_file_location('round2_cycle', PUBLIC / 'round2_cycle.py')
    cycle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cycle)
    cycle.experiment.resource_wait = stable_admission
    cycle.run(resume_ram=True, ram_attempt=3)


if __name__ == '__main__':
    try:
        run()
    except BaseException as error:
        write(ROOT / 'round2_ram_recovery03_failure.json', dict(error=repr(error), utc=time.time()))
        raise
