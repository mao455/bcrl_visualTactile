"""Retire four byte-container duplicates whose complete saved payloads match.

Nov12 is closed. Its validation_best and retained best containers have different
SHA256s, but every tensor, Adam state, RNG and metadata field was compared equal.
Bind that comparison to both original hashes before removing only the duplicate.
"""
import json
import os
from pathlib import Path
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

BASE = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
CLOSED = BASE / 'tight_dagger_matched_20261112_01'
OUT = BASE / 'tight_fresh_20261116_01/closed_checkpoint_dedup'


def run():
    status = json.loads((CLOSED / 'status.json').read_text())
    assert status['state'] == 'COMPLETE' and status['owned_workers_exited']
    assert (CLOSED / 'monitor_exit.json').exists()
    proof = json.loads((OUT / 'payload_comparison.json').read_text())
    expected = {str(CLOSED / f'round_{i:02d}' / role / 'dp/training/validation_best.pt')
                for i in (1, 2) for role in ('G01', 'QG01')}
    assert {r['duplicate'] for r in proof['rows']} == expected
    assert not (OUT / 'finished.json').exists()
    rows = []
    for row in proof['rows']:
        duplicate = Path(row['duplicate']); retained = Path(row['retained'])
        assert row['entire_payload_equal'] and not row['different_fields']
        assert retained == duplicate.with_name('best.pt')
        assert not duplicate.is_symlink() and not retained.is_symlink()
        assert sha256(retained) == row['retained_sha256']
        assert sha256(duplicate) == row['duplicate_sha256']
        s = duplicate.stat()
        assert s.st_ino == row['duplicate_inode'] and s.st_size == row['bytes']
        assert s.st_nlink == 1
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit(): continue
            try:
                for fd in (proc / 'fd').iterdir():
                    try: opened = fd.stat()
                    except OSError: continue
                    assert (opened.st_dev, opened.st_ino) != (s.st_dev, s.st_ino), 'Duplicate still open'
            except OSError: continue
        evidence = dict(**row, retired_allocated_bytes=s.st_blocks * 512, utc=time.time(),
            scope='Retained best has exactly the entire payload, including optimizer and RNG; no unique training state or observations removed.')
        atomic_json(duplicate.with_name('validation_best_retired.json'), evidence)
        duplicate.unlink(); rows.append(evidence)
        atomic_json(OUT / 'progress.json', dict(rows=rows))
    atomic_json(OUT / 'finished.json', dict(complete=True, rows=rows, utc=time.time(),
        retired_allocated_bytes=sum(r['retired_allocated_bytes'] for r in rows)))


if __name__ == '__main__': run()
