"""Bounded, process-local SHA reuse for already hashed frozen payload files.

This is not a persistent checksum cache. Every digest originates from a full
read in this process, and every lookup opens and checks the requested file.
Callers remain responsible for comparison with their registered expected SHA.
"""
from collections import OrderedDict
import hashlib
import os
import stat
import time


VERSION = 'process_local_frozen_payload_sha_v1'


def _identity(value):
    return (value.st_dev, value.st_ino, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)


class FrozenPayloadAudit:
    """Single-process, sequential audit; retains metadata only, never files."""

    def __init__(self, *, max_entries=4096):
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError('Positive integer payload-audit entry bound required')
        self.max_entries = max_entries
        self.pid = os.getpid()
        self._memo = OrderedDict()
        self._counts = dict(calls=0, full_hash_attempts=0, full_hashes=0,
            actual_bytes_read=0, memo_hits=0, avoided_full_hash_bytes=0,
            requested_regular_file_bytes=0, evictions=0, failures=0,
            full_hash_seconds=0., open_and_validation_seconds=0.)

    def _read_digest(self, stream):
        digest = hashlib.sha256()
        buffer = bytearray(1024 * 1024)
        view = memoryview(buffer)
        while (length := stream.readinto(buffer)):
            self._counts['actual_bytes_read'] += length
            digest.update(view[:length])
        return digest.hexdigest()

    def sha256(self, path):
        if os.getpid() != self.pid:
            raise RuntimeError('Payload SHA memo cannot cross a process boundary')
        started = time.monotonic()
        hash_seconds = 0.
        key = None
        self._counts['calls'] += 1
        try:
            # Nonblocking open prevents a named pipe from blocking before the
            # regular-file check. It has no effect on regular file reads.
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
            try:
                stream = os.fdopen(descriptor, 'rb', buffering=0)
            except BaseException:
                os.close(descriptor)
                raise
            with stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError('Frozen payload must be a regular file')
                key = _identity(before)
                self._counts['requested_regular_file_bytes'] += before.st_size
                cached = self._memo.get(key)
                if cached is None:
                    self._counts['full_hash_attempts'] += 1
                    hash_started = time.monotonic()
                    try:
                        digest = self._read_digest(stream)
                    finally:
                        hash_seconds = time.monotonic() - hash_started
                        self._counts['full_hash_seconds'] += hash_seconds
                else:
                    digest = cached
                after = os.fstat(stream.fileno())
                named = os.stat(path)
                if (not stat.S_ISREG(after.st_mode) or not stat.S_ISREG(named.st_mode)
                        or _identity(after) != key or _identity(named) != key):
                    raise ValueError('Frozen payload changed or was replaced during SHA audit')
                if cached is None:
                    self._counts['full_hashes'] += 1
                    self._memo[key] = digest
                    if len(self._memo) > self.max_entries:
                        self._memo.popitem(last=False)
                        self._counts['evictions'] += 1
                else:
                    self._memo.move_to_end(key)
                    self._counts['memo_hits'] += 1
                    self._counts['avoided_full_hash_bytes'] += before.st_size
                return digest
        except BaseException:
            self._counts['failures'] += 1
            if key is not None:
                self._memo.pop(key, None)
            raise
        finally:
            self._counts['open_and_validation_seconds'] += max(
                0., time.monotonic() - started - hash_seconds)

    def statistics(self):
        return dict(version=VERSION, pid=self.pid, max_entries=self.max_entries,
            retained_entries=len(self._memo), **self._counts,
            identity_fields=['st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'],
            persistent_digest_trust=False, file_descriptors_retained=0,
            expected_sha_comparison='Still required at every caller',
            byte_semantics='Logical bytes read by SHA; OS physical disk traffic is not measured')
