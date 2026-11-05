"""Bounded Linux inotify admission probe. Never launches Kit or signals processes.

Counts are FD-reference upper bounds, not deduplicated kernel instances. A small
real allocation probe catches ENOSPC even when /proc is incomplete. Passing this
resource check does NOT certify simulator correctness or successful episodes.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import resource
import shutil
import tempfile
import time


LIMIT_NAMES = ('max_user_watches', 'max_user_instances', 'max_queued_events')


def census(proc=Path('/proc'), uid=None):
    uid = os.getuid() if uid is None else uid
    rows, denied, raced = [], [], 0
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        try:
            if directory.stat().st_uid != uid:
                continue
            instances = watches = 0
            fds = list((directory/'fd').iterdir())
            for fd in fds:
                try:
                    if 'inotify' in os.readlink(fd):
                        instances += 1
                        watches += sum(line.startswith('inotify wd:') for line in
                            (directory/'fdinfo'/fd.name).read_text().splitlines())
                except (FileNotFoundError, ProcessLookupError):
                    raced += 1
            status = dict(line.split(':', 1) for line in
                (directory/'status').read_text().splitlines() if ':' in line)
            rows.append(dict(pid=int(directory.name), comm=(directory/'comm').read_text().strip(),
                fd_references=instances, watch_references=watches, fds=len(fds),
                threads=int(status.get('Threads', '0')), rss_kib=int(status.get('VmRSS', '0 kB').split()[0])))
        except (FileNotFoundError, ProcessLookupError):
            raced += 1
        except PermissionError:
            denied.append(int(directory.name))
    return dict(uid=uid, utc=time.time(),
        watch_references=sum(row['watch_references'] for row in rows),
        instance_fd_references=sum(row['fd_references'] for row in rows),
        processes=sorted(rows, key=lambda row: -row['watch_references']),
        unreadable_pids=denied, proc_races=raced,
        counting='FD-reference upper bound among readable same-UID processes; inherited/shared descriptions may duplicate, denied PIDs may hide usage')


def allocate_probe(count=256, parent=None):
    """Allocate <=1024 private directory watches, then close/remove all of them."""
    if not 1 <= count <= 1024:
        raise ValueError('Probe budget must be in [1, 1024]')
    libc = ctypes.CDLL(None, use_errno=True)
    libc.inotify_init1.argtypes = [ctypes.c_int]
    libc.inotify_init1.restype = ctypes.c_int
    libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    libc.inotify_add_watch.restype = ctypes.c_int
    result = dict(requested=count, allocated=0, errno=None, error=None, cleanup_complete=False)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    if fd < 0:
        code = ctypes.get_errno()
        return dict(result, errno=code, error=os.strerror(code), phase='inotify_init1', cleanup_complete=True)
    try:
        with tempfile.TemporaryDirectory(prefix='kit-watch-probe-', dir=parent) as temporary:
            for index in range(count):
                path = Path(temporary)/str(index)
                path.mkdir()
                if libc.inotify_add_watch(fd, os.fsencode(path), 0x00000100 | 0x01000000) < 0:
                    code = ctypes.get_errno()
                    result.update(errno=code, error=os.strerror(code), phase='inotify_add_watch')
                    break
                result['allocated'] += 1
            # Drop the kernel references before deleting the temporary paths.
            os.close(fd)
            fd = -1
        result['cleanup_complete'] = True
    finally:
        if fd >= 0:
            os.close(fd)
    return result


def decision(snapshot, limits, probe):
    reasons = []
    # Conservative admission headroom is a resource rule, not a Kit reliability
    # estimate. A failed probe is never excused by a favorable estimated count.
    if snapshot['watch_references'] > .80*limits['max_user_watches']:
        reasons.append('inotify_watch_headroom_below_20_percent')
    if snapshot['instance_fd_references'] > .80*limits['max_user_instances']:
        reasons.append('inotify_instance_headroom_below_20_percent')
    if probe['allocated'] != probe['requested'] or probe['errno'] is not None:
        reasons.append('bounded_inotify_allocation_failed')
    if not probe['cleanup_complete']:
        reasons.append('probe_cleanup_unverified')
    return dict(state='blocked' if reasons else 'preflight', resource_admitted=not reasons,
        kit_reliability_passed=False, reasons=reasons,
        limitation='Resource admission only; the versioned evaluation must validate complete trajectories, durable outcomes and process reclamation; SDK teardown is separate')


def inspect(probe_watches=256, parent=None):
    limits = {name: int((Path('/proc/sys/fs/inotify')/name).read_text()) for name in LIMIT_NAMES}
    before = census()
    probe = allocate_probe(probe_watches, parent)
    after = census()
    available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    return dict(schema=1, utc=time.time(), limits=limits, before=before, probe=probe, after=after,
        rlimit_nofile=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        rlimit_nproc=list(resource.getrlimit(resource.RLIMIT_NPROC)),
        available_ram_bytes=available, free_disk_bytes=shutil.disk_usage(parent or '.').free,
        **decision(after, limits, probe))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--probe-watches', type=int, default=256)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite an existing preflight evidence file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = inspect(args.probe_watches, args.output.parent)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(dict(state=result['state'], watches=result['after']['watch_references'],
        watch_limit=result['limits']['max_user_watches'], probe=result['probe'],
        reasons=result['reasons'], output=str(args.output)), ensure_ascii=False))
    raise SystemExit(2 if result['state'] == 'blocked' else 0)


if __name__ == '__main__':
    main()
