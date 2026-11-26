"""Run-local, exact-command resource waits before FullLease scientific bodies.

This module changes no production file. It retries only a new original context
manager's __enter__, never a scientific body, worker process or __exit__.
Importing it starts no process, opens no weights and writes no experiment data.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import runpy
import stat
import sys
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ORIGINAL_DEADLINE = 1795684571.2261844
VERSION = 'scoped_preentry_resource_wait09_atomicfd_v1'
RETRY_MESSAGES = frozenset((
    'Configured disk reserve reached', 'RAM reserve', 'VRAM reserve; no OOM retry'))
TARGETS = {
    ('tools.tight_paired_learning', 'rear-prepare'),
    ('tools.tight_paired_learning', 'rear-train'),
    ('tools.tight_finish', 'residual_cache'),
    ('visuotactile.anchor_continue', 'residual-offline'),
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def finite(value, *, positive=False):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and (value > 0 if positive else value >= 0))


def stat_identity(stat):
    return dict(dev=stat.st_dev, ino=stat.st_ino, size=stat.st_size,
                mtime_ns=stat.st_mtime_ns, ctime_ns=stat.st_ctime_ns)


def small_bytes(path, limit=16 * 2**20):
    path = Path(path)
    before = path.stat()
    require(before.st_size <= limit, 'Oversized resource wrapper metadata')
    raw = path.read_bytes()
    after = path.stat()
    require(stat_identity(before) == stat_identity(after) and len(raw) == before.st_size,
            'Resource wrapper metadata changed during read')
    return raw


def read(path):
    return json.loads(small_bytes(path))


def mutable_read(path, limit=16 * 2**20):
    """Read one bounded regular JSON snapshot through a single open FD.

    Owned monitors replace these files atomically. The FD remains a consistent
    old snapshot if replacement unlinks its inode; ctime/nlink can then change
    while size and mtime remain stable. Immutable metadata still uses read().
    In-place writes, truncation, malformed JSON and nonregular files fail.
    """
    require(isinstance(limit, int) and not isinstance(limit, bool) and limit > 0,
            'Invalid mutable resource metadata bound')
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC)
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), 'Nonregular mutable resource metadata')
        require(before.st_size <= limit, 'Oversized mutable resource metadata')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream:
            raw = stream.read(limit + 1)
        after = os.fstat(descriptor)
        identity = lambda row: (row.st_dev, row.st_ino, row.st_mode,
                                row.st_size, row.st_mtime_ns)
        # Atomic replacement may unlink this opened inode. Linux can complete
        # its ctime update after nlink has already reached zero, so both a
        # falling link count and a still-unlinked inode permit that transition.
        # An in-place write with restored mtime on a linked inode still fails.
        link_transition = (before.st_ctime_ns == after.st_ctime_ns
                           and before.st_nlink == after.st_nlink
                           or after.st_nlink < before.st_nlink
                           or before.st_nlink == after.st_nlink == 0)
        require(identity(before) == identity(after) and link_transition
                and len(raw) == before.st_size,
                'Mutable resource metadata changed in place during read')
        return json.loads(raw)
    finally:
        os.close(descriptor)


def digest(path):
    return hashlib.sha256(small_bytes(path)).hexdigest()


def under(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    require(path.is_relative_to(parent), 'Resource wrapper path escapes project')
    return path


def worker_root(argv):
    require(argv.count('--root') == 1, 'Exactly one scientific --root required')
    at = argv.index('--root')
    require(at + 1 < len(argv), 'Missing scientific worker root')
    return Path(argv[at + 1]).resolve()


class LaunchPolicy:
    """Frozen exact-command scope plus strict real, finite telemetry validation."""

    def __init__(self, registration, command_key, command, *, project=PROJECT):
        self.project = Path(project).resolve()
        self.registration_path = under(registration, self.project)
        registration_raw = small_bytes(self.registration_path)
        registration_row = json.loads(registration_raw)
        cfg = registration_row['scoped_resource_wait09']
        require(cfg['version'] == VERSION, 'Unknown scoped resource wait version')
        require(cfg['wrapper_sha256'] == digest(__file__), 'Scoped wrapper SHA differs')
        require(cfg['state_resources_sha256'] == digest(self.project/'visuotactile/state_resources.py'),
                'Original state_resources SHA differs')
        require(finite(cfg['original_deadline'], positive=True)
                and cfg['original_deadline'] == ORIGINAL_DEADLINE,
                'Original finite deadline changed')
        self.deadline = cfg['original_deadline']
        spec = cfg['commands'][command_key]
        require(isinstance(command, list) and command == spec['argv']
                and len(command) >= 4 and command[0] == '-m'
                and (command[1], command[2]) in TARGETS,
                'Exact registered scientific command required')
        self.command = list(command)
        self.command_key = command_key
        self.worker_root = under(spec['worker_root'], self.project)
        require(worker_root(command) == self.worker_root, 'Scientific worker root differs')
        roots = spec['lease_roots']
        require(isinstance(roots, list) and bool(roots), 'Exact lease roots required')
        self.lease_roots = frozenset(under(x, self.project) for x in roots)
        require(len(self.lease_roots) == len(roots), 'Repeated lease roots')
        self.telemetry_path = under(spec['telemetry_path'], self.project)
        stops = cfg['stop_paths']
        require(isinstance(stops, list) and bool(stops), 'Owned STOP paths required')
        self.stop_paths = tuple(under(x, self.project) for x in stops)
        self.registration_sha256 = hashlib.sha256(registration_raw).hexdigest()
        self.session_bindings = {root: self.session_binding(root) for root in self.lease_roots}
        self.used_lease_roots = set()

    def session_binding(self, root):
        path = root/'session.json'
        before = stat_identity(path.stat())
        row = read(path)
        require(before == stat_identity(path.stat()), 'Lease session changed while binding')
        require(row.get('resource_policy') == 'full_gpu_v1', 'Only exact FullLease policy is scoped')
        limits = row['resources']
        require(all(finite(limits.get(key), positive=True) for key in (
            'min_gpu_free_mib', 'min_ram_gib', 'min_disk_gib')),
            'Invalid real resource limits')
        require(limits['min_gpu_free_mib'] >= 4096 and limits['min_ram_gib'] >= 10
                and limits['min_disk_gib'] >= 12, 'Resource limits were relaxed')
        reserve = row.get('reserve_bytes', 12_000_000_000)
        require(finite(reserve, positive=True) and reserve >= 12 * 2**30,
                'Disk reserve was relaxed or invalid')
        require(all(finite(row.get(name), positive=True) and row[name] <= self.deadline
                    for name in ('deadline', 'training_deadline')),
                'Lease session deadline is invalid or expanded')
        groups = row.get('residual_pair', row.get('vision_pair', {})).get(
            'groups', ('G00', 'G10', 'G01', 'G11'))
        require(isinstance(groups, (list, tuple)) and all(isinstance(x, str) for x in groups),
                'Invalid owned resource groups')
        return dict(resource_policy=row['resource_policy'], resources=limits,
                    reserve_bytes=reserve, deadline=row['deadline'],
                    training_deadline=row['training_deadline'],groups=tuple(groups),
                    stat_identity=before)

    def check_controls(self, lease, deadline, now):
        if now >= deadline:
            raise TimeoutError('Original finite deadline reached during pre-entry resource wait')
        paths = self.stop_paths + (self.worker_root/'STOP', Path(lease.root)/'STOP')
        if any(path.exists() for path in paths):
            raise InterruptedError('Owned STOP requested during pre-entry resource wait')

    def validate(self, lease, now, cuda):
        root = Path(lease.root).resolve()
        require(stat_identity((root/'session.json').stat())
                == self.session_bindings[root]['stat_identity'],
                'Real resource limits/session changed during scoped wait')
        path = root/'shared_resources.json'
        require(path.resolve() == self.telemetry_path, 'Owned telemetry attachment differs')
        row = mutable_read(path)
        require(all(finite(row.get(key)) for key in (
            'utc_epoch', 'ram_available_bytes', 'gpu_free_mib')),
            'Invalid real resource telemetry')
        require(-2 <= now-row['utc_epoch'] <= 8, 'Stale or future resource telemetry')
        return row

    def paused(self, lease):
        root = Path(lease.root).resolve()
        require(stat_identity((root/'session.json').stat())
                == self.session_bindings[root]['stat_identity'],
                'Real resource limits/session changed during scoped wait')
        groups = self.session_bindings[root]['groups']
        group = next((part for part in self.worker_root.parts if part in groups), None)
        path = root/'concurrency.json'
        try:
            paused = mutable_read(path).get('pause_groups', [])
        except FileNotFoundError:
            # Native semantics: the optional schedule's absence means no pause.
            # A watcher may restore the original absent state by unlinking it.
            paused = []
        require(isinstance(paused, list) and all(isinstance(x, str) for x in paused),
                'Invalid owned pause schedule')
        return group is not None and group in paused


@contextmanager
def wait_before_entry(original, lease, policy, emit, *, deadline=None, cuda=True,
                      now=time.time, monotonic=time.monotonic, sleep=time.sleep,
                      interval=.25):
    """Retry __enter__ only; after it succeeds no retry catch surrounds yield."""
    require(finite(interval, positive=True) and interval <= .5, 'Invalid finite poll interval')
    deadlines = [policy.deadline]
    bound = getattr(policy, 'session_bindings', {}).get(Path(lease.root).resolve(), {})
    for value in (deadline, getattr(lease, 'deadline', None),
                  bound.get('deadline'), bound.get('training_deadline')):
        if value is not None:
            require(finite(value, positive=True) and value <= policy.deadline,
                    'Lease deadline override expands original deadline')
            deadlines.append(value)
    effective = min(deadlines)
    tries = waits = 0
    wait_seconds = 0.
    last_reason = None
    last_emit = float('-inf')

    def controls_and_telemetry():
        stamp = now()
        require(finite(stamp, positive=True), 'Invalid resource wait clock')
        policy.check_controls(lease, effective, stamp)
        policy.validate(lease, stamp, cuda)

    def wait(reason):
        nonlocal waits, wait_seconds, last_reason, last_emit
        controls_and_telemetry()
        stamp = now()
        if reason != last_reason or stamp-last_emit >= 5:
            emit(dict(event='preentry_wait', utc=stamp, reason=reason,
                      attempts=tries, waits=waits, deadline=effective,
                      lease_root=str(Path(lease.root).resolve())))
            last_reason, last_emit = reason, stamp
        before = monotonic()
        sleep(min(interval, max(0., effective-stamp)))
        elapsed = monotonic()-before
        require(finite(elapsed), 'Invalid resource wait duration')
        lease.wait_s += elapsed
        wait_seconds += elapsed
        waits += 1
        controls_and_telemetry()

    while True:
        controls_and_telemetry()
        if policy.paused(lease):
            wait('owned_group_paused')
            continue
        # A new manager per attempt. Failed __enter__ has never yielded a body.
        manager = original(lease, deadline=effective, cuda=cuda)
        tries += 1
        # FullLease can itself wait on a cooperative group pause inside enter.
        # Its existing check runs on every iteration. During enter only, add
        # strict controls/telemetry to that exact instance's check; restore it
        # before any scientific body or original exit can execute.
        original_check = lease.check
        own_check = lease.__dict__.get('check')
        had_own_check = 'check' in lease.__dict__
        def strict_check(checked_deadline=None):
            controls_and_telemetry()
            return original_check(checked_deadline)
        lease.check = strict_check
        try:
            try:
                entered = manager.__enter__()
            finally:
                if had_own_check:
                    lease.check = own_check
                else:
                    del lease.check
        except RuntimeError as error:
            if type(error) is not RuntimeError or str(error) not in RETRY_MESSAGES:
                raise
            wait(str(error))
            continue
        break
    # Nothing below is covered by a reserve retry catch. ExitStack forwards the
    # exact body exception triple and invokes the original __exit__ once.
    with ExitStack() as stack:
        stack.push(manager.__exit__)
        controls_and_telemetry()
        if waits:
            emit(dict(event='preentry_resumed', utc=now(), attempts=tries, waits=waits,
                      wait_seconds=wait_seconds, deadline=effective,
                      lease_root=str(Path(lease.root).resolve())))
        yield entered


@contextmanager
def installed(policy, emit, *, lease_class=None, **clock):
    if lease_class is None:
        from visuotactile.state_resources import FullLease
        lease_class = FullLease
    original = lease_class.block

    def scoped(lease, *, deadline=None, cuda=True):
        root = Path(lease.root).resolve()
        if type(lease) is not lease_class or root not in policy.lease_roots:
            return original(lease, deadline=deadline, cuda=cuda)
        try:
            current_worker = worker_root(sys.argv)
        except ValueError:
            return original(lease, deadline=deadline, cuda=cuda)
        if current_worker != policy.worker_root:
            return original(lease, deadline=deadline, cuda=cuda)
        policy.used_lease_roots.add(str(root))
        return wait_before_entry(original, lease, policy, emit,
                                 deadline=deadline, cuda=cuda, **clock)

    lease_class.block = scoped
    try:
        yield
    finally:
        lease_class.block = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--command-key', required=True)
    parser.add_argument('--receipt-dir', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    policy = LaunchPolicy(args.registration, args.command_key, command)
    output = under(args.receipt_dir, HERE)
    require(output != HERE, 'Receipt directory must be a new audit descendant')
    output.mkdir(parents=False, exist_ok=False)
    def write_new(name, row):
        with (output/name).open('x') as stream:
            json.dump(row, stream, indent=2, allow_nan=False)
            stream.write('\n')
    def emit(row):
        with (output/'waits.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    write_new('registration.json', dict(version=VERSION, command_key=args.command_key,
        original_command=command, wrapper_sha256=digest(__file__),
        registration=str(policy.registration_path), registration_sha256=policy.registration_sha256,
        original_deadline=policy.deadline, worker_root=str(policy.worker_root),
        lease_roots=sorted(map(str, policy.lease_roots)), telemetry_path=str(policy.telemetry_path),
        resource_limits={str(k): v for k, v in policy.session_bindings.items()},
        retry_scope='original_context_manager_enter_only', source_files_modified=0))
    previous_argv = sys.argv
    began = time.time()
    success = False
    try:
        sys.argv = [command[1]]+command[2:]
        with installed(policy, emit):
            runpy.run_module(command[1], run_name='__main__', alter_sys=True)
        success = True
    except SystemExit as error:
        success = error.code in (None, 0)
        raise
    finally:
        sys.argv = previous_argv
        write_new('exit.json', dict(version=VERSION, started_utc=began, utc=time.time(),
            success=success, original_deadline=policy.deadline,
            used_lease_roots=sorted(policy.used_lease_roots),
            original_command_replayed=False, body_retries=0,
            method_restored=True, source_files_modified=0))


if __name__ == '__main__':
    main()
