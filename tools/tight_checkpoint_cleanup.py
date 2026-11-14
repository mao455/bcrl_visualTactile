"""Pinned, opt-in retirement of 135 reviewed intermediate checkpoints.

Default is read-only. No tensor storage is loaded and no large SHA is computed.
Deleting these files intentionally discards their intermediate optimizer/RNG
restart points; selected models, original evidence and all data remain intact.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pickletools
import re
import stat
import subprocess
import sys
import uuid
import zipfile


APPROVED_ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile')
AUDIT_REL = 'run/tight_fresh_20261114_05/cpu_audits'
APPROVED_MANIFEST = APPROVED_ROOT/AUDIT_REL/'space_inventory_20261114_final_manifest.json'
APPROVED_SHA256 = 'a0b8441fc44756eed0b285631021d3a480d71e8ee8eb3120c81d42c9a9fd2302'
DEFAULT_LEDGER_ROOT = APPROVED_ROOT/AUDIT_REL/'checkpoint_cleanup_ledgers'
ALLOWED_SESSION = APPROVED_ROOT/'run/tight_fresh_20261114_05'
PROTECTED_PREFIXES = ('data/', 'run/tight_fresh_20261114_02/round_05/',
                      'run/tight_fresh_20261114_05/')
PROTECTED_NAMES = {'best.pt', 'validation_best.pt', 'update0_reference.pt',
                   'initial_export.pt', 'environment_rng_initial.pt'}
IDENTITY_KEYS = ('dev', 'ino', 'nlink', 'size', 'mtime_ns', 'blocks_bytes')
CANDIDATE_NAME = re.compile(r'(?:last|comparison_(?:epoch_)?[0-9]+)\.pt\Z')
REFERENCE_PATH = re.compile(r'(?:outputs/visuotactile/)?(run/[^\s\"\'`<>]+?\.pt)')
INVENTORY_AUDITS = {
    f'{AUDIT_REL}/space_inventory_20261114_{name}.json'
    for name in ('raw', 'references', 'candidates_draft', 'review_manifest',
                 'deep_references', 'final_manifest')
}
# Explicitly reviewed non-research daemons. PID alone never grants an exception.
VISIBILITY_EXCEPTIONS = (
    dict(pid=2184, uid=1001, command='/usr/lib/systemd/systemd --user', comm='systemd',
         start_ticks=1935, boot_id='9dca45ac-9c70-43f7-8cdd-1a44c8c4fd40'),
    dict(pid=2194, uid=1001, command='(sd-pam)', comm='(sd-pam)', start_ticks=1937,
         boot_id='9dca45ac-9c70-43f7-8cdd-1a44c8c4fd40'),
    dict(pid=430847, uid=1001, command='/usr/bin/ssh-agent -D -a /run/user/1001/keyring/.ssh',
         comm='ssh-agent', start_ticks=3972545, boot_id='9dca45ac-9c70-43f7-8cdd-1a44c8c4fd40'),
)


class CleanupRefused(RuntimeError):
    pass


def identity(path):
    value = Path(path).lstat()
    if not stat.S_ISREG(value.st_mode):
        raise CleanupRefused(f'Not a regular file: {path}')
    return dict(dev=value.st_dev, ino=value.st_ino, nlink=value.st_nlink,
                size=value.st_size, mtime_ns=value.st_mtime_ns,
                blocks_bytes=value.st_blocks*512)


def safe_path(root, relative):
    rel = PurePosixPath(relative)
    if rel.is_absolute() or '..' in rel.parts or str(rel) != relative:
        raise CleanupRefused(f'Noncanonical manifest path: {relative}')
    path = root/relative
    if path.resolve() != path:
        raise CleanupRefused(f'Symlink or path escape: {relative}')
    return path


def require_identity(path, expected):
    try:
        actual = identity(path)
    except OSError as exc:
        raise CleanupRefused(f'Cannot stat {path}: {exc}') from exc
    if any(actual[key] != expected[key] for key in IDENTITY_KEYS):
        raise CleanupRefused(f'File identity/link/block drift: {path}')
    return actual


def load_manifest(path, *, approved_sha256=APPROVED_SHA256,
                  approved_root=APPROVED_ROOT, expected_counts=(135, 124)):
    """Only the CLI's compiled approval identity authorizes the real files."""
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != approved_sha256:
        raise CleanupRefused('Manifest differs from the reviewed immutable whitelist')
    manifest = json.loads(raw)
    root = Path(manifest['root'])
    if root != Path(approved_root).resolve() or root.resolve() != root:
        raise CleanupRefused('Manifest root differs from approved project outputs')
    candidates = manifest['candidate_files']
    protected = manifest['protected_models']
    if (len(candidates), len(protected)) != expected_counts:
        raise CleanupRefused('Manifest candidate/protected counts changed')
    names = set()
    protected_names = {row['path'] for row in protected}
    protected_inodes = {(row['dev'], row['ino']) for row in protected}
    for row in candidates:
        relative = row['path']
        path = safe_path(root, relative)
        if (relative in names or relative in protected_names
                or relative.startswith(PROTECTED_PREFIXES)
                or '/G11/' in relative or not relative.startswith('run/')
                or '/training/' not in relative
                or not CANDIDATE_NAME.fullmatch(path.name)
                or path.name in PROTECTED_NAMES
                or row['nlink'] != 1 or row.get('aliases') != [relative]
                or (row['dev'], row['ino']) in protected_inodes
                or row.get('non_receipt_references')
                or row.get('retained_references')):
            raise CleanupRefused(f'Whitelist/protected/source-reference rejection: {relative}')
        if row.get('finished_receipt') != str(PurePosixPath(relative).parent/'finished.json'):
            raise CleanupRefused(f'Unbound completion receipt: {relative}')
        names.add(relative)
    for row in protected:
        safe_path(root, row['path'])
    return manifest, raw


def stat_set(root, rows):
    return [dict(path=row['path'], **require_identity(safe_path(root, row['path']), row))
            for row in rows]


def verify_completion_receipts(root, candidates):
    seen = {}
    for row in candidates:
        path = safe_path(root, row['finished_receipt'])
        if path not in seen:
            seen[path] = json.loads(path.read_text())
        evidence = row.get('completion_evidence')
        if not evidence or any(seen[path].get(key) != value for key, value in evidence.items()):
            raise CleanupRefused(f'Completed-training evidence changed: {path}')
    return len(seen)


def gpu_compute_pids():
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader,nounits'], check=True, capture_output=True,
            text=True, timeout=15)
        return {int(line.strip()) for line in result.stdout.splitlines() if line.strip()}
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise CleanupRefused(f'GPU process census unavailable: {exc}') from exc


def inspect_processes(inodes, *, uid, allowed_session, proc=Path('/proc'),
                      gpu_pids=None, self_pid=None, visibility_exceptions=()):
    """Match device+inode; fail on inaccessible processes except pinned daemons."""
    self_pid = os.getpid() if self_pid is None else self_pid
    gpu_pids = gpu_compute_pids() if gpu_pids is None else set(gpu_pids)
    rows = {}
    consumers = []
    unreadable = []
    exceptions_used = []
    boot_id = ((proc/'sys/kernel/random/boot_id').read_text().strip()
               if visibility_exceptions else None)
    def process_row(directory):
        args = (directory/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        procstat = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(directory.name), uid=directory.stat().st_uid,
            ppid=int(procstat[1]), start_ticks=int(procstat[19]), command=args,
            comm=(directory/'comm').read_text().strip())
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            pid = int(directory.name)
            rows[pid] = process_row(directory)
            for fd in (directory/'fd').iterdir():
                try:
                    value = fd.stat()
                except FileNotFoundError:
                    continue  # Ordinary close racing the read-only fd census.
                if (value.st_dev, value.st_ino) in inodes:
                    consumers.append(dict(pid=pid, kind='fd', fd=fd.name,
                        dev=value.st_dev, ino=value.st_ino))
            for line in (directory/'maps').read_text().splitlines():
                fields = line.split(None, 5)
                if len(fields) < 5:
                    continue
                major, minor = (int(v, 16) for v in fields[3].split(':'))
                key = os.makedev(major, minor), int(fields[4])
                if key in inodes:
                    consumers.append(dict(pid=pid, kind='mmap', dev=key[0], ino=key[1]))
        except (FileNotFoundError, ProcessLookupError):
            if directory.exists():
                unreadable.append(dict(pid=int(directory.name), reason='live process census incomplete'))
        except (OSError, ValueError, IndexError) as exc:
            if directory.exists():
                permitted = None
                if isinstance(exc, PermissionError) and int(directory.name) not in gpu_pids:
                    try:
                        current = dict(process_row(directory), boot_id=boot_id)
                        permitted = next((binding for binding in visibility_exceptions
                            if all(current.get(key) == value for key, value in binding.items())), None)
                    except (OSError, ValueError, IndexError):
                        pass
                if permitted:
                    exceptions_used.append(dict(binding=permitted, reason=str(exc),
                        limitation='Reviewed resident non-research daemon; fd/maps are not visible'))
                else:
                    unreadable.append(dict(pid=int(directory.name), reason=str(exc)))

    session = str(allowed_session)
    def allowed(row):
        seen = set()
        while row and row['pid'] not in seen:
            seen.add(row['pid'])
            if re.search(re.escape(session)+r'(?=[/\s]|$)', row['command']):
                return True
            row = rows.get(row['ppid'])
        return False
    research = []
    for row in rows.values():
        if row['pid'] == self_pid:
            continue
        command = row['command']
        is_research = ('-m tools.tight_checkpoint_cleanup' not in command
            and ('-m tools.tight_' in command or '-m visuotactile.' in command
            or ('bc+rl_RSS' in command and ('torchrun ' in command or 'train.py' in command))))
        if is_research or row['pid'] in gpu_pids:
            row = dict(row, gpu_compute=row['pid'] in gpu_pids, allowed_current_session=allowed(row))
            research.append(row)
    blocked = [row for row in research if not row['allowed_current_session']]
    report = dict(same_uid=uid, same_uid_processes=len(rows), consumers=consumers,
                  unreadable=unreadable, research_and_gpu_processes=research,
                  disallowed_processes=blocked, visibility_exceptions=exceptions_used,
                  all_same_uid_processes_fully_visible=not exceptions_used and not unreadable)
    if consumers or unreadable or blocked:
        raise CleanupRefused('Process ownership/consumer check failed: '+json.dumps(report))
    return report


def reference_scan(root, candidates, *, worktrees, ledger_root):
    """Rescan current source references; no trust in an old absence-of-reference."""
    paths = {row['path'] for row in candidates}
    own_receipts = {row['path']:root/row['finished_receipt'] for row in candidates}
    hits = collections.defaultdict(list)
    counts = collections.Counter()
    seen_pt = set()
    skip = {'.git', 'outputs', '__pycache__', '.pytest_cache', 'node_modules', 'logs', 'wandb'}
    def inspect(strings, source, kind):
        found = set()
        for value in strings:
            found.update(match.group(1) for match in REFERENCE_PATH.finditer(value)
                         if match.group(1) in paths)
        for relative in found:
            if source != own_receipts[relative]:
                hits[relative].append(dict(source=str(source), kind=kind))
    def files(tree, is_outputs):
        for directory, dirs, names in os.walk(tree, followlinks=False):
            dirs[:] = [name for name in dirs
                if not (Path(directory)/name).is_symlink()
                and not name.startswith('.venv')
                and (is_outputs or name not in skip)
                and not (Path(directory)/name).is_relative_to(ledger_root)]
            for name in names:
                path = Path(directory)/name
                if path.is_symlink():
                    target = path.resolve()
                    if target.is_relative_to(root) and str(target.relative_to(root)) in paths:
                        hits[str(target.relative_to(root))].append(dict(source=str(path), kind='symlink'))
                    continue
                if is_outputs and str(path.relative_to(root)) in INVENTORY_AUDITS:
                    continue
                yield path
    for tree, is_outputs in [(root, True)]+[(Path(tree), False) for tree in worktrees]:
        for path in files(tree, is_outputs):
            try:
                if path.suffix == '.pt' and is_outputs:
                    value = path.stat(); key = value.st_dev, value.st_ino
                    if key in seen_pt:
                        continue
                    seen_pt.add(key)
                    with zipfile.ZipFile(path) as archive:
                        entries = [i for i in archive.infolist()
                            if i.filename == 'data.pkl' or i.filename.endswith('/data.pkl')]
                        if len(entries) != 1 or entries[0].file_size > 2*1024*1024:
                            raise CleanupRefused(f'Uninspectable PT metadata: {path}')
                        data = archive.read(entries[0])
                        values = [arg for op, arg, _ in pickletools.genops(data)
                            if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE',
                                           'BINUNICODE8', 'STRING') and isinstance(arg, str)]
                        inspect(values, path, 'pt_data_pickle_only')
                        counts['pt_metadata_files'] += 1
                        counts['pt_metadata_bytes'] += len(data)
                elif path.suffix in ('.json', '.md', '.py', '.sh', '.yaml', '.toml'):
                    if path.stat().st_size > 64*1024*1024:
                        raise CleanupRefused(f'Uninspectable large source metadata: {path}')
                    inspect([path.read_text()], path, 'output_text' if is_outputs else 'worktree_text')
                    counts['output_text' if is_outputs else 'worktree_text'] += 1
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise CleanupRefused(f'Cannot inspect source metadata {path}: {exc}') from exc
    report = dict(counts=dict(counts), references=dict(hits), worktrees=[str(x) for x in worktrees])
    if hits:
        raise CleanupRefused('A retained source now references a cleanup candidate: '+json.dumps(report))
    return report


def filesystem_snapshot(root):
    fs = os.statvfs(root)
    return dict(block_size=fs.f_frsize, free_blocks=fs.f_bfree,
        available_blocks=fs.f_bavail, free_bytes=fs.f_bfree*fs.f_frsize,
        available_bytes=fs.f_bavail*fs.f_frsize)


class Ledger:
    def __init__(self, root):
        root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = root/(stamp+'_'+uuid.uuid4().hex)
        self.path.mkdir()
        self.events = (self.path/'events.jsonl').open('x')

    def event(self, kind, **fields):
        value = dict(kind=kind, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
        self.events.write(json.dumps(value)+'\n'); self.events.flush(); os.fsync(self.events.fileno())

    def artifact(self, name, value):
        with (self.path/name).open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())

    def close(self):
        self.events.close()


def remove_one(root, row, ledger):
    """Directory fd binds the unlink to the just-checked parent directory."""
    path = safe_path(root, row['path'])
    parent_fd = os.open(path.parent, os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        actual = require_identity(path, row)
        fs_before = filesystem_snapshot(root)
        ledger.event('delete_intent', path=row['path'], identity=actual, filesystem=fs_before)
        # Check again after the durable intent; no recursive/glob deletion exists.
        current = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        if (current.st_dev, current.st_ino, current.st_nlink, current.st_size, current.st_mtime_ns,
                current.st_blocks*512) != tuple(row[k] for k in IDENTITY_KEYS):
            raise CleanupRefused(f'Identity changed after delete intent: {path}')
        os.unlink(path.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
        ledger.event('deleted', path=row['path'], unlinked_allocated_bytes=actual['blocks_bytes'],
                     filesystem_before=fs_before, filesystem_after=filesystem_snapshot(root))
        return actual['blocks_bytes']
    finally:
        os.close(parent_fd)


def operate(manifest, *, execute=False, ledger_root=DEFAULT_LEDGER_ROOT,
            process_check, source_check):
    """Internal fixture-friendly operation; real CLI reaches this only via pin check."""
    root = Path(manifest['root']); candidates = manifest['candidate_files']
    ledger = None; lock = None; deleted = []; before = filesystem_snapshot(root)
    try:
        if execute:
            ledger_root.mkdir(parents=True, exist_ok=True)
            lock = (ledger_root/'.cleanup.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            ledger = Ledger(ledger_root)
            ledger.artifact('approved_manifest.json', manifest)
            ledger.event('started', candidate_count=len(candidates), filesystem_before=before,
                discarded_state='Intermediate optimizer/EMA/RNG restart checkpoints are intentionally retired')
        protected_before = stat_set(root, manifest['protected_models'])
        stat_set(root, candidates)
        completion_count = verify_completion_receipts(root, candidates)
        processes = process_check()
        sources = source_check()
        # The potentially long metadata scan must not leave a stale process/stat gate.
        processes = process_check()
        stat_set(root, candidates)
        stat_set(root, manifest['protected_models'])
        preflight = dict(protected_before=protected_before, process_check=processes,
            source_check=sources, completion_receipts_verified=completion_count)
        if ledger:
            ledger.artifact('preflight.json', preflight)
        allocated = 0
        if execute:
            for row in candidates:
                process_check()  # Includes mmap consumers, not merely open fd paths.
                allocated += remove_one(root, row, ledger)
                deleted.append(row['path'])
        protected_after = stat_set(root, manifest['protected_models'])
        report = dict(mode='executed' if execute else 'verified_read_only', candidate_count=len(candidates),
            deleted=deleted, unlinked_allocated_bytes=allocated,
            candidate_allocated_bytes=sum(row['blocks_bytes'] for row in candidates),
            filesystem_before=before, filesystem_after=filesystem_snapshot(root),
            protected_models_unchanged=protected_before == protected_after,
            protected_before=protected_before, protected_after=protected_after,
            preflight=preflight, ledger=str(ledger.path) if ledger else None,
            warning='Filesystem free-byte delta includes concurrent research writes and ledger metadata; '
                    'unlinked_allocated_bytes counts the verified single-link files only.')
        if ledger:
            ledger.artifact('complete.json', report); ledger.event('complete', deleted_count=len(deleted))
        return report
    except BaseException as exc:
        if ledger:
            try:
                protected_after = stat_set(root, manifest['protected_models'])
                protected_error = None
            except BaseException as check_error:
                protected_after = None; protected_error = repr(check_error)
            ledger.artifact('failed.json', dict(error=repr(exc), deleted=deleted,
                filesystem_before=before, filesystem_after=filesystem_snapshot(root),
                protected_after=protected_after, protected_check_error=protected_error,
                remaining=[row['path'] for row in candidates if row['path'] not in deleted],
                candidate_paths_currently_absent=[row['path'] for row in candidates
                    if not (root/row['path']).exists()],
                note='If unlink happened before its deleted event, durable delete_intent plus '
                     'candidate_paths_currently_absent preserves that partial state; no automatic resume.'))
            ledger.event('failed', error=repr(exc), deleted_count=len(deleted))
        raise
    finally:
        if ledger:
            ledger.close()
        if lock:
            lock.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=APPROVED_MANIFEST)
    parser.add_argument('--execute', action='store_true', help='Explicitly retire only the pinned 135 files')
    parser.add_argument('--ledger-root', type=Path, default=DEFAULT_LEDGER_ROOT)
    parser.add_argument('--uid', type=int, default=APPROVED_ROOT.stat().st_uid,
        help='Research UID to inspect; root may inspect this UID without bypassing /proc gates')
    args = parser.parse_args(argv)
    try:
        manifest, _ = load_manifest(args.manifest)
        if args.ledger_root.resolve() != DEFAULT_LEDGER_ROOT:
            raise CleanupRefused('Real cleanup ledger location is fixed to this approved session')
        if args.uid != APPROVED_ROOT.stat().st_uid:
            raise CleanupRefused('Process census UID must match the approved research owner')
        worktrees = [Path(line.split(' ', 1)[1]) for line in subprocess.check_output(
            ['git', 'worktree', 'list', '--porcelain'], text=True).splitlines()
            if line.startswith('worktree ')]
        inodes = {(row['dev'], row['ino']) for row in manifest['candidate_files']}
        report = operate(manifest, execute=args.execute, ledger_root=args.ledger_root,
            process_check=lambda:inspect_processes(inodes, uid=args.uid,
                allowed_session=ALLOWED_SESSION, visibility_exceptions=VISIBILITY_EXCEPTIONS),
            source_check=lambda:reference_scan(APPROVED_ROOT, manifest['candidate_files'],
                worktrees=worktrees, ledger_root=args.ledger_root))
        print(json.dumps(report, indent=2))
        return 0
    except (CleanupRefused, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps(dict(status='refused', error=str(exc))), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
