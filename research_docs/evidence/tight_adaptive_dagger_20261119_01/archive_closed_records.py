"""Archive explicitly named inactive project records; verify every byte before unlink."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import time
import zstandard as zstd

PUBLIC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('record_cleanup_helper', PUBLIC / 'cleanup_stale_assets.py')
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
ROOT = helper.ROOT
BASE = ROOT.parent
LEDGER = ROOT / 'resource_cleanup_20261120/attempt07_records'
NAMES = {'learning.jsonl', 'shared.jsonl', 'resources.jsonl', 'work.jsonl',
         'native_inference_timing.json', 'sampled_ticks.csv', 'base_sampled_ticks.csv'}


def safe(path):
    path = Path(path)
    assert path.resolve(strict=True) == path and path.is_file() and not path.is_symlink()
    assert path.is_relative_to(BASE) and path.name in NAMES
    assert path.stat().st_nlink == 1
    if path.is_relative_to(ROOT):
        relative = path.relative_to(ROOT)
        assert relative.parts[:2] in [('round01', 'learning'), ('owners', 'r1_train')]
    return path


def digest_stream(stream):
    value = hashlib.sha256()
    while block := stream.read(4 * 2**20):
        value.update(block)
    return value.hexdigest()


def archive(path, expected, *, opened):
    path = safe(path)
    before = helper.identity(path)
    assert (before['dev'], before['ino']) not in opened
    assert helper.digest(path) == expected
    target = path.with_name(path.name + '.zst')
    receipt = path.with_name(path.name + '.archive.json')
    assert not receipt.exists()
    assert shutil.disk_usage(ROOT).free > 12 * 2**30 + before['size']
    if not target.exists():
        with path.open('rb') as source, target.open('xb') as output:
            zstd.ZstdCompressor(level=3, threads=0).copy_stream(source, output)
            output.flush()
            os.fsync(output.fileno())
    with target.open('rb') as source, zstd.ZstdDecompressor().stream_reader(source) as decoded:
        assert digest_stream(decoded) == expected
    assert helper.identity(path) == before
    _, fresh_opened, _ = helper.process_references()
    assert (before['dev'], before['ino']) not in fresh_opened
    saved = before['blocks'] * 512 - target.stat().st_blocks * 512
    row = dict(path=str(path), archive=str(target), sha256=expected,
               archive_sha256=helper.digest(target), original_bytes=before['size'],
               compressed_bytes=target.stat().st_size, saved_allocated_bytes=saved,
               mode=stat.S_IMODE(before['mode']), mtime_ns=before['mtime_ns'],
               byte_roundtrip_verified=True, original_removed=False, utc=time.time(),
               restore_command=f'python -m tools.tight_cache_archive restore --manifest {LEDGER / "restore_manifest.json"}')
    helper.dump(receipt, row)
    if saved <= 0:
        target.unlink()
        receipt.unlink()
        return dict(path=str(path), skipped='Archive saves no blocks')
    path.unlink()
    row['original_removed'] = True
    helper.dump(receipt, row)
    return row


def plan():
    LEDGER.mkdir(parents=True, exist_ok=False)
    protected = helper.protection()
    protected_paths = {Path(p).resolve() for p in protected}
    _, opened, census = helper.process_references()
    candidates = []
    for run in BASE.iterdir():
        if not run.is_dir() or run.is_symlink():
            continue
        for root, dirs, names in os.walk(run, followlinks=False):
            dirs[:] = [d for d in dirs if not Path(root, d).is_symlink()]
            for name in sorted(set(names) & NAMES):
                path = Path(root, name)
                try:
                    safe(path)
                    info = helper.identity(path)
                    if info['size'] < 2**20 or time.time() - info['mtime_ns'] / 1e9 < 1800:
                        continue
                    assert path not in protected_paths
                    assert (info['dev'], info['ino']) not in opened
                    assert not path.with_name(name + '.archive.json').exists()
                    if name == 'learning.jsonl':
                        finished = path.parent / 'finished.json'
                        assert finished.is_file() and json.loads(finished.read_text()).get('complete') is True
                    candidates.append(dict(path=str(path), sha256=helper.digest(path), identity=info))
                except (AssertionError, FileNotFoundError):
                    continue
    manifest = dict(files=candidates, protected_before=protected, census=census,
                    utc=time.time(), scope='Only closed, named project records; raw/model/config/failed captures excluded')
    helper.dump(LEDGER / 'manifest.json', manifest)
    helper.dump(LEDGER / 'restore_manifest.json', dict(files=[dict(path=x['path']) for x in candidates]))
    result = dict(files=len(candidates), original_bytes=sum(x['identity']['size'] for x in candidates),
                  manifest_sha256=helper.digest(LEDGER / 'manifest.json'))
    helper.dump(LEDGER / 'plan.json', result)
    print(json.dumps(result), flush=True)


def apply(expected):
    assert helper.digest(LEDGER / 'manifest.json') == expected
    manifest = json.loads((LEDGER / 'manifest.json').read_text())
    assert not (LEDGER / 'archive.jsonl').exists()
    for path, info in manifest['protected_before'].items():
        assert helper.identity(Path(path)) == info
    _, opened, census = helper.process_references()
    before = shutil.disk_usage(ROOT).free
    rows = []
    with (LEDGER / 'archive.jsonl').open('x') as journal:
        for item in manifest['files']:
            assert helper.identity(Path(item['path'])) == item['identity']
            row = archive(item['path'], item['sha256'], opened=opened)
            rows.append(row)
            helper.dump(LEDGER / 'restore_manifest.json', dict(files=[dict(path=r['path']) for r in rows if r.get('original_removed')]))
            journal.write(json.dumps(row) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
            print(json.dumps(dict(files=len(rows), total=len(manifest['files']),
                                  saved_gib=sum(r.get('saved_allocated_bytes', 0) for r in rows) / 2**30)), flush=True)
    unchanged = all(helper.identity(Path(p)) == s for p, s in manifest['protected_before'].items())
    result = dict(complete=True, files_archived=sum(r.get('original_removed', False) for r in rows),
                  saved_allocated_bytes=sum(r.get('saved_allocated_bytes', 0) for r in rows),
                  free_before=before, free_after=shutil.disk_usage(ROOT).free,
                  protected_paths=len(manifest['protected_before']), protected_stat_unchanged=unchanged,
                  byte_roundtrip_verified=True, evidence_discarded=False, other_project_mutations=0,
                  utc=time.time(), census=census)
    helper.dump(LEDGER / 'result.json', result)
    assert unchanged
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply-sha256')
    args = parser.parse_args()
    if args.apply_sha256:
        apply(args.apply_sha256)
    else:
        plan()
