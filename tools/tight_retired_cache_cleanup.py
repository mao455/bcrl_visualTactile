"""Exact, opt-in retirement of the reviewed pre-fresh derived payloads.

Only the small candidate manifest is hashed. No payload loads, payload hashes,
recursive deletion, automatic recovery, or dependency rescans are performed.
Every invocation needs a new ledger directory; the default is a dry run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time


RUN = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
RETIRED_ROOTS = frozenset((
    'tight_vision_20261109_01', 'tight_transformer_20261110_02',
    'tight_dagger_20261111_01', 'tight_dagger_pool_20261111_01',
    'tight_dagger_matched_20261112_01'))
CACHE_SELECTION = 'direct child regular *.pt only; preserve all JSON/metadata/statistics and directory'


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def checked_path(value):
    path = Path(value)
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise ValueError('Absolute, existing paths without symlink aliases are required: ' + str(path))
    relative = path.relative_to(RUN)
    if not relative.parts or relative.parts[0] not in RETIRED_ROOTS:
        raise ValueError('Only explicitly retired pre-fresh roots are allowed: ' + str(path))
    if any(part.lower() in {'raw', 'episodes', 'data', 'demos', 'expert', 'experts', 'pretrained'}
           or part.startswith('tight_fresh_') for part in relative.parts):
        raise ValueError('Protected raw, expert, pretrained or fresh path: ' + str(path))
    return path


def fingerprint(info):
    return dict(dev=info.st_dev, ino=info.st_ino, size=info.st_size,
                mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns,
                blocks=info.st_blocks, nlink=info.st_nlink, mode=info.st_mode)


def preflight(manifest):
    if (manifest.get('version') != 'retired_derived_payload_candidates_v1'
            or manifest.get('mode') != 'proposal_only_no_deletion'
            or manifest.get('active_candidate_cmdline_matches')):
        raise ValueError('Wrong or active cleanup proposal')
    candidates = manifest['candidates']
    if not candidates: raise ValueError('No candidates')
    files, groups, inodes, paths, parents = [], {}, set(), set(), {}
    for index, candidate in enumerate(candidates):
        kind = candidate['kind']
        if (candidate.get('active_direct_path_or_inode_matches') != 0
                or candidate.get('all_nlink_one') is not True):
            raise ValueError('Candidate has not passed the recorded dependency/link classification')
        if kind in ('frozen_visual_cache', 'nominal_backend_dependency_cache'):
            directory = checked_path(candidate['directory'])
            expected_name = 'perception' if kind == 'frozen_visual_cache' else 'nominal'
            if (directory.name != expected_name or not directory.is_dir()
                    or candidate.get('selection') != CACHE_SELECTION
                    or candidate.get('raw_preserved') is not True
                    or candidate.get('raw_paths_missing') != 0):
                raise ValueError('Only classified direct-child derived cache payloads are allowed')
            if kind == 'frozen_visual_cache' and directory.parent.name != 'state_data':
                raise ValueError('Unexpected frozen perception directory')
            # The reviewed selection is direct regular payloads, not symlink
            # entries referring to earlier cache pools. Preserve those aliases.
            # A symlink replacing a formerly selected payload changes the hash.
            selected = sorted(path for path in directory.glob('*.pt') if not path.is_symlink())
            names = '\n'.join(path.name for path in selected)
            if hashlib.sha256(names.encode()).hexdigest() != candidate['filename_list_sha256']:
                raise ValueError('Candidate filename list changed: ' + str(directory))
            if not Path(candidate['source_manifest']).is_file():
                raise ValueError('Retained source manifest is missing')
            if kind == 'nominal_backend_dependency_cache':
                if (not Path(candidate['retained_index']).is_file()
                        or not candidate.get('retained_base_checkpoints')
                        or not all(Path(path).is_file() for path in candidate['retained_base_checkpoints'])):
                    raise ValueError('Retained nominal index or selected DP is missing')
        elif kind == 'retired_nonbest_last':
            selected = [checked_path(candidate['path'])]
            path = selected[0]
            if (path.name != 'last.pt' or path.parent.name != 'training'
                    or candidate.get('selection') != 'this exact last.pt only'
                    or candidate.get('target_reached') is not True):
                raise ValueError('Only the exact completed nonbest last snapshot is allowed')
            best = checked_path(candidate['best_preserved'])
            receipt = checked_path(candidate['finished_receipt'])
            if (best != path.with_name('best.pt') or not best.is_file()
                    or receipt != path.with_name('finished.json')
                    or json.loads(receipt.read_text()).get('target_reached') is not True):
                raise ValueError('Selected best or completed-stage receipt is missing/changed')
            if (best.stat().st_dev, best.stat().st_ino) == (path.stat().st_dev, path.stat().st_ino):
                raise ValueError('last and best share an inode')
        else:
            raise ValueError('Unknown candidate kind')
        rows = []
        for path in selected:
            path = checked_path(path)
            if kind != 'retired_nonbest_last' and any(word in path.stem.lower() for word in
                    ('best', 'last', 'statistic', 'normaliz', 'config', 'manifest', 'index')):
                raise ValueError('Protected checkpoint or metadata payload: ' + str(path))
            info = path.lstat()
            inode = (info.st_dev, info.st_ino)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Every deletion must be a regular singly linked file: ' + str(path))
            if path in paths or inode in inodes:
                raise ValueError('Duplicate candidate path or inode')
            paths.add(path); inodes.add(inode)
            if str(path.parent) not in parents:
                parent_stat = path.parent.stat()
                parents[str(path.parent)] = dict(dev=parent_stat.st_dev, ino=parent_stat.st_ino)
            rows.append(dict(path=str(path), candidate=index, kind=kind, **fingerprint(info)))
        if (len(rows) != candidate['file_count']
                or sum(row['size'] for row in rows) != candidate['logical_bytes']
                or sum(row['blocks']*512 for row in rows) != candidate['allocated_bytes']):
            raise ValueError('Candidate count, logical bytes or allocated blocks changed')
        group = groups.setdefault(kind, dict(directories_or_files=0, allocated_bytes=0))
        group['directories_or_files'] += 1
        group['allocated_bytes'] += candidate['allocated_bytes']
        files.extend(rows)
    allocated = sum(row['blocks']*512 for row in files)
    if allocated != manifest['total_allocated_bytes'] or groups != manifest['groups']:
        raise ValueError('Candidate totals changed')
    return dict(files=files, parents=parents, groups=groups,
                file_count=len(files), allocated_bytes=allocated,
                logical_bytes=sum(row['size'] for row in files))


def run(manifest_path, expected_sha256, ledger, *, apply=False):
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256):
        raise ValueError('An explicit lowercase manifest SHA256 is required')
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Reviewed manifest SHA256 mismatch')
    manifest = json.loads(raw)
    ledger = ledger.absolute()
    ledger.mkdir(parents=True, exist_ok=False)
    state = dict(version='retired_payload_cleanup_v1', apply=apply, complete=False,
                 manifest=str(manifest_path.resolve()), manifest_sha256=expected_sha256,
                 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 started_utc=time.time(), deleted_files=0, deleted_allocated_bytes=0,
                 free_bytes_before=shutil.disk_usage(RUN).free,
                 scope='Retired derived .pt and exact nonbest last only; no payload hashing; all parents and metadata retained')
    atomic_json(ledger/'status.json', state)
    try:
        inventory = preflight(manifest)
        for parent in inventory['parents']:
            if ledger.resolve().is_relative_to(Path(parent)):
                raise ValueError('Ledger must be outside every deletion directory')
        atomic_json(ledger/'inventory.json', dict(manifest_sha256=expected_sha256, **inventory))
        state.update(preflight_complete=True, candidate_count=len(manifest['candidates']),
                     file_count=inventory['file_count'], allocated_bytes=inventory['allocated_bytes'])
        atomic_json(ledger/'status.json', state)
        if apply:
            with (ledger/'deleted.jsonl').open('x') as journal:
                for row in inventory['files']:
                    path = Path(row['path'])
                    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
                    try:
                        parent_stat = os.fstat(fd)
                        if dict(dev=parent_stat.st_dev, ino=parent_stat.st_ino) != inventory['parents'][str(path.parent)]:
                            raise ValueError('Parent identity changed before unlink')
                        # A bounded per-file stat at unlink prevents using stale
                        # preflight metadata; it reads no file payload.
                        current = fingerprint(os.stat(path.name, dir_fd=fd, follow_symlinks=False))
                        if current != {key:row[key] for key in current}:
                            raise ValueError('File changed after preflight: ' + str(path))
                        os.unlink(path.name, dir_fd=fd)
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    state['deleted_files'] += 1
                    state['deleted_allocated_bytes'] += row['blocks']*512
                    journal.write(json.dumps(dict(path=str(path), utc=time.time(), **current))+'\n')
                    journal.flush(); os.fsync(journal.fileno())
        state.update(complete=True, dry_run=not apply, ended_utc=time.time(),
                     free_bytes_after=shutil.disk_usage(RUN).free,
                     free_space_note='Concurrent job writes can change observed free space; stat blocks are the selected allocation')
    except BaseException as error:
        state.update(error=repr(error), ended_utc=time.time(),
                     recovery='No automatic retry. Review inventory and journal before any new identity.')
        raise
    finally:
        atomic_json(ledger/'status.json', state)
    print(json.dumps(state, sort_keys=True))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--expected-sha256', required=True)
    parser.add_argument('--ledger', type=Path, required=True, help='New directory; never resume or overwrite a ledger')
    parser.add_argument('--apply', action='store_true', help='Delete the exact fully checked candidates; default only inventories them')
    args = parser.parse_args()
    run(args.manifest, args.expected_sha256, args.ledger, apply=args.apply)


if __name__ == '__main__':
    main()
