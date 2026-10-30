"""Explicit one-session cleanup of invalid caches and redundant failed videos."""
import json
from pathlib import Path

from .compact import sha256
from .training_resources import atomic_json


def main():
    invalid = Path('outputs/visuotactile/run/long_horizon_20261030')
    videos = Path('outputs/visuotactile/demos/residual_20261030_01/viewing')
    ledger = Path('research_docs/evidence/long_horizon_cleanup_20261030.json')
    if ledger.exists():
        raise ValueError('Cleanup already classified; do not repeat or overwrite ledger')
    paths = sorted(invalid.rglob('*.pt'))+sorted(videos.glob('*.mp4'))
    rows = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError('Refuse unknown/symlink cleanup object')
        rows.append(dict(path=path.as_posix(),bytes=path.stat().st_size,sha256=sha256(path),
            type='invalid_feature_or_checkpoint' if path.suffix == '.pt' else 'redundant_failed_viewing_video',
            experiment='long_horizon_20261030_invalid' if path.suffix == '.pt' else 'residual_20261030_01',
            reason='Current-command leakage and wrong split; never eligible for recovery/deployment' if path.suffix == '.pt'
                else 'Completed failure analysis; same old weights; numeric traces and result summaries retained',deleted=False))
    atomic_json(ledger,dict(authorization='2026-10-30 user request section10',files=rows,deleted_bytes=0))
    for row in rows:
        path = Path(row['path'])
        if path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            raise RuntimeError('Cleanup candidate changed after classification')
        path.unlink()
        row['deleted'] = True
        atomic_json(ledger,dict(authorization='2026-10-30 user request section10',files=rows,
            deleted_bytes=sum(item['bytes'] for item in rows if item['deleted'])))
    print(json.dumps(dict(files=len(rows),deleted_bytes=sum(row['bytes'] for row in rows)),indent=2))


if __name__ == '__main__':
    main()
