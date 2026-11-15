"""Retire one exact reviewed list of completed nonbest optimizer endpoints."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

RUN = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
MANIFEST = Path(__file__).resolve().parents[2] / 'research_docs/evidence/tight_fresh_20261114/grasp_followup05/retirement_manifest.json'
EXPECTED = 'c7c13e4e27a780020c2fdb492979d3ddaaabc920f1e5d967c91c253998a3e8fd'


def identity(path):
    stat = path.stat()
    return dict(dev=stat.st_dev, ino=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, blocks=stat.st_blocks)


def run(output, apply):
    payload = MANIFEST.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == EXPECTED
    rows = json.loads(payload)['files']
    assert len(rows) == 15 and len({row['path'] for row in rows}) == 15
    for row in rows:
        path = Path(row['path'])
        assert path.resolve() == path and (path.name == 'last.pt' or path.name.startswith('comparison_')) and path.parent.name == 'training'
        assert 'tight_fresh_20261114_02/round_05/' not in str(path)
        assert not path.is_symlink() and path.stat().st_nlink == 1
        assert identity(path) == {key: row[key] for key in identity(path)}
        assert Path(row['retained_best']) == path.with_name('best.pt')
        assert identity(path.with_name('best.pt')) == row['retained_best_stat']
        raw = Path(row['finished']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == row['finished_sha256']
        done = json.loads(raw)
        assert done.get('stop_reason') in ('validation_plateau_or_overfit','update_budget') and done.get('updates',0)>0
    selected = {(row['dev'], row['ino']) for row in rows}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            for fd in (process / 'fd').iterdir():
                try:
                    stat = fd.stat()
                except (FileNotFoundError, PermissionError):
                    continue
                assert (stat.st_dev, stat.st_ino) not in selected, 'Candidate is open by PID ' + process.name
        except (FileNotFoundError, PermissionError):
            continue
    output.mkdir(parents=True, exist_ok=False)
    report = dict(manifest=str(MANIFEST), manifest_sha256=EXPECTED, apply=apply, started=time.time(),
                  selected_files=len(rows), selected_allocated_bytes=sum(row['blocks'] * 512 for row in rows),
                  free_before=shutil.disk_usage(RUN).free, deleted=0,
                  preserved='All raw trajectories, all bests, original round05, all currently referenced visual features; retired optimizer endpoints no longer resumable.')
    (output / 'started.json').write_text(json.dumps(report, indent=2) + '\n')
    try:
        if apply:
            with (output / 'deleted.jsonl').open('x') as log:
                for row in rows:
                    path = Path(row['path'])
                    assert identity(path) == {key: row[key] for key in identity(path)}
                    path.unlink()
                    report['deleted'] += 1
                    log.write(json.dumps(row) + '\n')
                    log.flush()
                    os.fsync(log.fileno())
        assert all(identity(Path(row['retained_best'])) == row['retained_best_stat'] for row in rows)
        report['complete'] = True
    finally:
        report.update(ended=time.time(), free_after=shutil.disk_usage(RUN).free)
        report['free_delta'] = report['free_after'] - report['free_before']
        (output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    run(args.output, args.apply)
