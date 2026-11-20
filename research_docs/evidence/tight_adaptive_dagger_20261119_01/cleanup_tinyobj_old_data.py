"""Explicitly authorized old generated grasp data; clean dataset is never modified."""
import argparse
import collections
import datetime
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

PUBLIC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tiny_cleanup_helpers', PUBLIC / 'cleanup_stale_assets.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
BASE = Path('/home/CNS2025331827/docs/grasp_small_obj/FingerEyeLab/outputs')
LEDGER = helper.ROOT / 'resource_cleanup_20261121/tinyobj_old_data'
CUTOFF = datetime.datetime(2026, 11, 6, tzinfo=datetime.timezone(datetime.timedelta(hours=8))).timestamp()
EXCLUDED = {'workspace_archive', 'workspace_consolidation_20261026', 'source_cleanup_20261015',
            'rax_dexbin_tinyobj', 'config', 'configs', 'reports', '.git', 'third_party', 'vendor'}
CODE = {'.py', '.pyc', '.sh', '.bash', '.md', '.rst', '.toml', '.ini', '.cfg', '.ipynb', '.cpp', '.h', '.so'}


def blocked(path):
    parts = Path(path).relative_to(BASE).parts
    return any('cleandataset' in part.lower().replace('_', '').replace('-', '')
               or part in EXCLUDED for part in parts)


def safe(path):
    path = Path(path)
    assert path.is_absolute() and path.resolve(strict=True) == path
    assert path.is_relative_to(BASE) and not blocked(path)
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and s.st_nlink == 1
    assert s.st_mtime < CUTOFF and path.suffix.lower() not in CODE
    return path


def process_guard():
    _, opened, census = helper.process_references()
    protected = set()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            cwd = Path(os.readlink(proc / 'cwd'))
            for raw in (proc / 'cmdline').read_bytes().split(b'\0'):
                arg = raw.decode(errors='replace')
                if not arg or len(arg) > 1000:
                    continue
                arg = arg.split('=', 1)[-1] if arg.startswith('--') and '=' in arg else arg
                p = Path(arg)
                p = p if p.is_absolute() else cwd / p
                p = p.resolve()
                if p != BASE and p.is_relative_to(BASE) and p.exists():
                    protected.add(p)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return opened, protected, census


def clean_snapshot():
    roots = []
    rows = []
    for root, dirs, names in os.walk(BASE, followlinks=False):
        cur = Path(root)
        for name in list(dirs):
            p = cur / name
            if 'cleandataset' in name.lower().replace('_', '').replace('-', ''):
                assert p.resolve() == p and not p.is_symlink()
                roots.append(p)
            if p.is_symlink() or name in EXCLUDED or p in roots:
                dirs.remove(name)
    for root in roots:
        for current, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not Path(current, d).is_symlink()]
            for name in names:
                p = Path(current, name)
                rows.append(dict(path=str(p), identity=helper.identity(p)))
    return roots, rows


def plan():
    LEDGER.mkdir(parents=True, exist_ok=False)
    opened, active, census = process_guard()
    roots, clean = clean_snapshot()
    assert roots, 'Expected protected clean dataset must exist'
    with gzip.open(LEDGER / 'clean_before.json.gz', 'wt') as stream:
        json.dump(clean, stream)
    total = allocated = 0
    groups = collections.Counter()
    with gzip.open(LEDGER / 'manifest.jsonl.gz', 'wt') as stream:
        for root, dirs, names in os.walk(BASE, followlinks=False):
            cur = Path(root)
            dirs[:] = [d for d in dirs if not (cur / d).is_symlink() and not blocked(cur / d)]
            # A new experiment may vendor old files; its date overrides their mtime.
            dates = re.findall(r'(20\d{2})-?(\d{2})-?(\d{2})', str(cur.relative_to(BASE)))
            if any(''.join(d) >= '20261106' for d in dates):
                dirs[:] = []
                continue
            for name in sorted(names):
                p = cur / name
                try:
                    safe(p)
                    before = helper.identity(p)
                    if (before['dev'], before['ino']) in opened or any(p == a or p.is_relative_to(a) for a in active):
                        continue
                    digest = helper.digest(p)
                    assert helper.identity(p) == before
                    row = dict(path=str(p), identity=before, sha256=digest)
                    stream.write(json.dumps(row) + '\n')
                    total += 1
                    allocated += before['blocks'] * 512
                    groups[str(p.relative_to(BASE).parts[0])] += before['blocks'] * 512
                except (AssertionError, ValueError, FileNotFoundError):
                    continue
    result = dict(files=total, allocated_bytes=allocated, groups=dict(groups),
                  cutoff='2026-11-06T00:00:00+08:00', scope=str(BASE),
                  protected_clean_roots=[str(p) for p in roots], protected_clean_files=len(clean),
                  clean_snapshot_sha256=helper.digest(LEDGER / 'clean_before.json.gz'),
                  manifest_sha256=helper.digest(LEDGER / 'manifest.jsonl.gz'),
                  active_paths=sorted(str(p) for p in active), census=census, utc=time.time(),
                  authorization='User permits roughly half-month-old and older script-generated TinyObjGrasp data except cleandataset',
                  limitation='Deleted old generated data is not archived; old generation/replay may require regeneration. Code, operational config/reports, recent TinyObj experiments, clean datasets and active paths are excluded.')
    helper.dump(LEDGER / 'plan.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('census', 'active_paths')}), flush=True)


def apply(expected):
    assert helper.digest(LEDGER / 'manifest.jsonl.gz') == expected
    assert not (LEDGER / 'operations.jsonl').exists()
    plan = json.loads((LEDGER / 'plan.json').read_text())
    assert plan['manifest_sha256'] == expected
    assert helper.digest(LEDGER / 'clean_before.json.gz') == plan['clean_snapshot_sha256']
    with gzip.open(LEDGER / 'clean_before.json.gz', 'rt') as stream:
        clean = json.load(stream)
    assert all(helper.identity(Path(r['path'])) == r['identity'] for r in clean)
    opened, active, census = process_guard()
    free_before = shutil.disk_usage(BASE).free
    count = allocated = 0
    with gzip.open(LEDGER / 'manifest.jsonl.gz', 'rt') as stream, (LEDGER / 'operations.jsonl').open('x') as journal:
        for line in stream:
            row = json.loads(line)
            p = safe(row['path'])
            info = row['identity']
            assert helper.identity(p) == info
            assert (info['dev'], info['ino']) not in opened
            assert not any(p == a or p.is_relative_to(a) for a in active)
            journal.write(json.dumps(dict(event='intent', path=str(p), sha256=row['sha256'])) + '\n')
            journal.flush()
            if count % 500 == 0:
                os.fsync(journal.fileno())
            fd = os.open(p.parent, os.O_DIRECTORY | os.O_RDONLY | os.O_NOFOLLOW)
            try:
                s = os.stat(p.name, dir_fd=fd, follow_symlinks=False)
                assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink) == (
                    info['dev'], info['ino'], info['size'], info['mtime_ns'], info['ctime_ns'], 1)
                os.unlink(p.name, dir_fd=fd)
            finally:
                os.close(fd)
            count += 1
            allocated += info['blocks'] * 512
            journal.write(json.dumps(dict(event='deleted', path=str(p))) + '\n')
            if count % 10000 == 0:
                print(json.dumps(dict(deleted=count, released_gib=allocated / 2**30)), flush=True)
        journal.flush()
        os.fsync(journal.fileno())
    unchanged = all(helper.identity(Path(r['path'])) == r['identity'] for r in clean)
    result = dict(complete=True, files_deleted=count, allocated_bytes_released=allocated,
                  free_before=free_before, free_after=shutil.disk_usage(BASE).free,
                  clean_files=len(clean), clean_stat_identities_unchanged=unchanged,
                  cutoff=plan['cutoff'], other_project_roots_modified=[], utc=time.time(), census=census,
                  limitation=plan['limitation'])
    helper.dump(LEDGER / 'result.json', result)
    assert unchanged
    print(json.dumps({k: v for k, v in result.items() if k != 'census'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-sha256')
    args = parser.parse_args()
    apply(args.apply_sha256) if args.apply_sha256 else plan()
