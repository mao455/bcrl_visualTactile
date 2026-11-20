"""Losslessly archive old, unused failed capture files; retain all failure bytes."""
import importlib.util
import json
from pathlib import Path
import time

PUBLIC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('failure_record_archive', PUBLIC / 'archive_closed_records.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)
SCOPES = ('tight_fresh_20261115_07', 'tight_recovery_20261118_01')
a.LEDGER = a.ROOT / 'resource_cleanup_20261120/attempt08_failure_archive'


def safe(path):
    path = Path(path)
    assert path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
    assert path.is_relative_to(a.BASE) and path.relative_to(a.BASE).parts[0] in SCOPES
    assert path.name == 'data.h5.partial' and path.stat().st_nlink == 1
    assert not path.with_name('data.h5').exists()
    return path


a.safe = safe


def plan():
    a.LEDGER.mkdir(parents=True, exist_ok=False)
    protected = a.helper.protection()
    _, opened, census = a.helper.process_references()
    candidates = []
    for name in SCOPES:
        for path in sorted((a.BASE / name).rglob('data.h5.partial')):
            safe(path)
            info = a.helper.identity(path)
            if info['size'] < 2**20:
                continue
            assert time.time() - info['mtime_ns'] / 1e9 >= 86400
            assert str(path) not in protected and (info['dev'], info['ino']) not in opened
            candidates.append(dict(path=str(path), sha256=a.helper.digest(path), identity=info))
    manifest = dict(files=candidates, protected_before=protected, census=census,
                    utc=time.time(), scope=list(SCOPES),
                    evidence_policy='Unique failed capture bytes preserved in verified Zstandard archives; restore before inspecting historical HDF5 path')
    a.helper.dump(a.LEDGER / 'manifest.json', manifest)
    summary = dict(files=len(candidates), original_bytes=sum(x['identity']['size'] for x in candidates),
                   manifest_sha256=a.helper.digest(a.LEDGER / 'manifest.json'))
    a.helper.dump(a.LEDGER / 'plan.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-sha256')
    args = parser.parse_args()
    if args.apply_sha256:
        a.apply(args.apply_sha256)
    else:
        plan()
