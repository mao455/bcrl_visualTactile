"""Bounded, lossless multi-lane capture with one HDF5 writer thread.

Only detached CPU arrays cross this boundary. ``submit`` snapshots them before
returning; no simulator, policy, CUDA, or reset operation runs in the worker.
The byte cap includes copying, queued/in-flight payloads and retained writer
arrays. HDF5/compressor allocations and bounded NumPy flush temporaries are
outside that payload cap; measure process RAM when choosing block sizes.
Optional spawn workers verify already closed files with independent HDF5 locks;
their bounded decode buffers are separate from the write-payload byte budget.

Worker time overlaps simulation and submission. Component times must therefore
not be added together as elapsed rollout time. A returned capture has passed
the existing complete source-field verification; unfinished/error captures
remain on disk and are never reported as verified.
"""
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import threading
import time

import h5py
import numpy as np

from .recording import BufferedNumericWriter, CompactEpisodeStream, verify_stream_fields


class CaptureWorkerError(RuntimeError):
    """The shared writer failed; all further submissions are rejected."""


class CaptureFinishTicket:
    """Wait for a queued finalizer without pausing other simulator lanes."""
    def __init__(self, future, queued_at):
        self._future = future
        self._queued_at = queued_at

    def done(self):
        return self._future.done()

    def result(self, timeout=None):
        """Return the verified receipt; timeout does not cancel disk work."""
        started = time.perf_counter()
        raw = self._future.result(timeout=timeout)
        receipt = {key: value for key, value in raw.items() if key != '_verified_at'}
        receipt['profile'] = dict(raw['profile'],
            drain_and_finalize_wait_s=time.perf_counter() - started,
            enqueue_to_verified_s=raw['_verified_at'] - self._queued_at,
            timing_scope='main submission and worker times overlap; do not sum into rollout wall time')
        return receipt


def _arrays(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _arrays(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _arrays(child)
    elif isinstance(value, np.ndarray):
        yield value


def _size(value):
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values())
    if hasattr(value, 'detach'):
        raise TypeError('Capture submission requires detached CPU NumPy arrays, not tensors')
    if not isinstance(value, (np.ndarray, np.generic, int, float, complex, bool)):
        raise TypeError('Capture fields must be CPU arrays or numeric scalars')
    array = np.asarray(value)
    if array.dtype.kind not in 'buifc':
        raise TypeError('Capture fields must have numeric or boolean dtype')
    return array.nbytes


def _copy(value):
    if isinstance(value, dict):
        return {key: _copy(child) for key, child in value.items()}
    return np.array(value, copy=True, order='C')


def _json_copy(value):
    return json.loads(json.dumps(value, default=str))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _publish_receipt(path, value):
    """Atomic no-overwrite publication after the underlying files are closed."""
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, path)
    temporary.unlink()


def validate_verification_parameters(workers, max_pending):
    """Explicit process/task bounds; zero workers preserves serial verification."""
    if type(workers) is not int or workers < 0:
        raise ValueError('verification_workers must be a nonnegative integer')
    if type(max_pending) is not int or max_pending < 1 or max_pending < workers:
        raise ValueError('verification_max_pending must be positive and at least verification_workers')


def _verify_closed_capture(receipt, process_worker=False):
    """Only closed file paths and small metadata cross the spawn boundary.

    A process owns its read handles independently of the writer's HDF5 lock.
    Every original field digest and the complete compact-file SHA are retained.
    Publish in the verifier, so completed evidence survives parent interruption.
    """
    if process_worker:
        receipt['profile']['verification_queue_wait_s'] = (
            time.perf_counter() - receipt.pop('_verification_enqueued_at'))
        _publish_receipt(Path(receipt['numeric_path']).parent/'verification_started.json', dict(
            pid=os.getpid(), utc=time.time(), process_start_method='spawn',
            status='reading_closed_capture', compact_path=receipt['dataset_capture']['path']))
    binding = receipt['dataset_capture']
    if binding is not None:
        started = time.perf_counter()
        binding['verified_source_fields'] = verify_stream_fields(binding['path'])
        receipt['profile']['worker_verify_s'] += time.perf_counter() - started
        started = time.perf_counter()
        binding['sha256'] = _sha256(binding['path'])
        receipt['profile']['worker_hash_s'] += time.perf_counter() - started
    if process_worker:
        receipt.update(verification_process_pid=os.getpid(), verification_execution='spawn_process_v1')
    receipt.update(_verified_at=time.perf_counter(),
        status='source_verified' if binding is not None else 'numeric_complete')
    _publish_receipt(Path(receipt['numeric_path']).parent / 'capture_complete.json', receipt)
    return receipt


@dataclass
class _Job:
    kind: str
    lane: object = None
    payload: object = None
    reserved: int = 0
    future: object = None


class SharedCaptureWriter:
    """One worker shared by all lanes; backpressure waits without dropping rows.

    ``max_queue`` also counts submissions currently copying. The byte budget
    accounts retained blocks across lanes, and pressure flushes those blocks.
    If mandatory previous-frame state plus one incoming row cannot fit, fail
    explicitly instead of waiting forever for a frame that cannot be admitted.
    """
    def __init__(self, max_pending_bytes=512 << 20, max_queue=64,
                 verification_workers=0, verification_max_pending=4):
        if type(max_pending_bytes) is not int or max_pending_bytes < 1:
            raise ValueError('max_pending_bytes must be a positive integer')
        if type(max_queue) is not int or max_queue < 1:
            raise ValueError('max_queue must be a positive integer')
        validate_verification_parameters(verification_workers, verification_max_pending)
        self.max_pending_bytes = max_pending_bytes
        self.max_queue = max_queue
        self.verification_workers = verification_workers
        self.verification_max_pending = verification_max_pending
        self._condition = threading.Condition()
        self._queue = deque()
        self._lanes = []
        self._reserved = self._peak_bytes = self._peak_queue = 0
        self._copying = 0
        self._inflight = False
        self._flush_requested = False
        self._flush_generation = 0
        self._error = None
        self._closed = False
        self._closing = False
        self._verification_jobs = {}
        self._verification_pending = self._verification_peak = 0
        self._verification_submitted = self._verification_completed = self._verification_failed = 0
        self._verification_slot_wait_s = 0.
        self._verification_pids = set()
        self._verification_executor = (ProcessPoolExecutor(max_workers=verification_workers,
            mp_context=multiprocessing.get_context('spawn')) if verification_workers else None)
        self._thread = threading.Thread(target=self._run, name='tight-capture-writer', daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise CaptureWorkerError(str(self._error)) from self._error
        if self._closed or self._closing:
            raise RuntimeError('Capture writer is closed')

    def check(self):
        """Surface an asynchronous failure even while no lane is submitting."""
        with self._condition:
            self._check()

    def statistics(self):
        with self._condition:
            return dict(max_pending_bytes=self.max_pending_bytes, max_queue=self.max_queue,
                reserved_bytes=self._reserved, peak_reserved_bytes=self._peak_bytes,
                peak_queued_or_copying=self._peak_queue,
                byte_scope='owned payload arrays, including retained buffers; excludes bounded flush/compressor scratch',
                worker_alive=self._thread.is_alive(), failed=self._error is not None,
                verification_workers=self.verification_workers,
                verification_max_pending=self.verification_max_pending,
                verification_pending=self._verification_pending,
                verification_peak_pending=self._verification_peak,
                verification_submitted=self._verification_submitted,
                verification_completed=self._verification_completed,
                verification_failed=self._verification_failed,
                verification_slot_wait_s=self._verification_slot_wait_s,
                verification_process_pids=sorted(self._verification_pids),
                verification_scope='Bound includes queued and running closed-file tasks; only paths/metadata sent. '
                    'Spawned verifier decode buffers/RSS are separate from payload byte budget.')

    def _reserve(self, size):
        if size > self.max_pending_bytes:
            raise BufferError(f'Capture row copy needs {size} bytes; budget is {self.max_pending_bytes}')
        with self._condition:
            observed_flush = self._flush_generation
            while True:
                self._check()
                slots = len(self._queue) + self._copying
                if self._reserved + size <= self.max_pending_bytes and slots < self.max_queue:
                    self._reserved += size
                    self._copying += 1
                    self._peak_bytes = max(self._peak_bytes, self._reserved)
                    self._peak_queue = max(self._peak_queue, slots + 1)
                    return
                if (self._flush_generation > observed_flush and not self._inflight
                        and not self._queue and not self._copying
                        and self._reserved + size > self.max_pending_bytes):
                    raise BufferError('Retained previous frames plus the next row exceed capture byte budget; increase the shared budget or reduce lanes')
                if self._reserved + size > self.max_pending_bytes:
                    self._flush_requested = True
                    self._condition.notify_all()
                self._condition.wait(.1)

    def _submit(self, lane, numeric, compact, step):
        numeric_bytes = _size(numeric)
        # BufferedNumericWriter makes its own per-row numeric copy in-worker.
        size = 2 * numeric_bytes + (_size(compact) if compact is not None else 0)
        started = time.perf_counter()
        self._reserve(size)
        waited = time.perf_counter() - started
        started = time.perf_counter()
        try:
            payload = (_copy(numeric), _copy(compact) if compact is not None else None, step)
        except BaseException:
            with self._condition:
                self._reserved -= size
                self._copying -= 1
                self._condition.notify_all()
            raise
        copied = time.perf_counter() - started
        with self._condition:
            self._copying -= 1
            try:
                self._check()
            except BaseException:
                self._reserved -= size
                self._condition.notify_all()
                raise
            self._queue.append(_Job('append', lane, payload, size))
            self._condition.notify_all()
        return waited, copied

    def _request_async(self, kind, lane=None, payload=None):
        future = Future()
        with self._condition:
            while len(self._queue) + self._copying >= self.max_queue:
                self._check()
                self._condition.wait(.1)
            self._check()
            self._queue.append(_Job(kind, lane, payload, future=future))
            self._peak_queue = max(self._peak_queue, len(self._queue) + self._copying)
            self._condition.notify_all()
        return future

    def _request(self, kind, lane=None, payload=None):
        return self._request_async(kind, lane, payload).result()

    def _reaccount(self, lane, released=0):
        seen = set()
        size = 0
        objects = [lane._numeric.pending] if lane._numeric is not None else []
        if lane._stream is not None:
            objects.extend((lane._stream.rows, lane._stream.previous))
        for array in _arrays(objects):
            if id(array) not in seen:
                seen.add(id(array))
                size += array.nbytes
        with self._condition:
            self._reserved += size - lane._retained - released
            lane._retained = size
            assert 0 <= self._reserved <= self.max_pending_bytes
            self._condition.notify_all()

    def _flush(self, lane):
        started = time.perf_counter()
        if lane._numeric is not None:
            lane._numeric.flush()
        if lane._stream is not None:
            lane._stream.flush()
            # append/finish need only these fields after a block is persisted.
            # Keep exact arrays (not rounded/hash substitutes) for legacy checks.
            previous = lane._stream.previous
            if previous is not None:
                lane._stream.previous = {key: value for key, value in previous.items()
                    if key.startswith('transition/observation_after/') or key in ('terminated', 'truncated')}
        lane._profile['worker_flush_s'] += time.perf_counter() - started
        self._reaccount(lane)

    def _verification_done(self, future):
        """Executor callback never accesses open writer handles."""
        with self._condition:
            lane, ticket = self._verification_jobs[future]
        failure = None
        try:
            receipt = future.result()
        except BaseException as error:
            # This lane has already closed every handle; preserve its failed
            # verification input and wake producers/full-queue waiters promptly.
            lane._abort_worker(repr(error))
            failure = error
        with self._condition:
            del self._verification_jobs[future]
            self._verification_pending -= 1
            if failure is not None:
                self._verification_failed += 1
                if self._error is None:
                    self._error = failure
                if not ticket.done():
                    ticket.set_exception(CaptureWorkerError(str(failure)))
            else:
                self._verification_completed += 1
                self._verification_pids.add(receipt['verification_process_pid'])
                if not ticket.done():
                    ticket.set_result(receipt)
            self._condition.notify_all()

    def _dispatch_verification(self, job, receipt):
        started = time.perf_counter()
        with self._condition:
            while self._verification_pending >= self.verification_max_pending:
                if self._error is not None:
                    raise CaptureWorkerError(str(self._error)) from self._error
                self._condition.wait(.1)
            if self._error is not None:
                raise CaptureWorkerError(str(self._error)) from self._error
            waited = time.perf_counter() - started
            self._verification_slot_wait_s += waited
            self._verification_pending += 1
            self._verification_submitted += 1
            self._verification_peak = max(self._verification_peak, self._verification_pending)
        receipt['profile']['verification_slot_wait_s'] = waited
        receipt['_verification_enqueued_at'] = time.perf_counter()
        try:
            _publish_receipt(job.lane.directory/'verification_queued.json', dict(
                status='closed_pending_source_verification', numeric_path=receipt['numeric_path'],
                compact_path=receipt['dataset_capture']['path'], steps=receipt['steps'],
                process_start_method='spawn', verification_workers=self.verification_workers,
                verification_max_pending=self.verification_max_pending, utc=time.time()))
            future = self._verification_executor.submit(_verify_closed_capture, receipt, True)
        except BaseException as error:
            with self._condition:
                self._verification_pending -= 1
                self._condition.notify_all()
            job.lane._abort_worker(repr(error))
            raise
        with self._condition:
            self._verification_jobs[future] = (job.lane, job.future)
        future.add_done_callback(self._verification_done)

    def _run(self):
        job = None
        try:
            while True:
                with self._condition:
                    while not self._queue and not self._flush_requested and self._error is None:
                        self._condition.wait()
                    if self._error is not None:
                        raise CaptureWorkerError(str(self._error)) from self._error
                    if self._queue:
                        job = self._queue.popleft()
                    else:
                        self._flush_requested = False
                        job = _Job('flush')
                    self._inflight = True
                    self._condition.notify_all()
                if job.kind == 'stop':
                    for lane in self._lanes:
                        if not lane._worker_finished:
                            lane._abort_worker('writer closed before lane.finish')
                            self._reaccount(lane)
                    job.future.set_result(None)
                    return
                if job.kind == 'flush':
                    for lane in self._lanes:
                        if not lane._worker_finished:
                            self._flush(lane)
                    with self._condition:
                        self._flush_generation += 1
                elif job.kind == 'open':
                    self._lanes.append(job.lane)
                    job.lane._open_worker()
                    job.future.set_result(None)
                elif job.kind == 'append':
                    numeric, compact, step = job.payload
                    lane = job.lane
                    started = time.perf_counter()
                    lane._numeric.append(numeric, step)
                    lane._profile['worker_numeric_append_s'] += time.perf_counter() - started
                    if lane._stream is not None:
                        started = time.perf_counter()
                        lane._stream.append(compact)
                        lane._profile['worker_compact_append_s'] += time.perf_counter() - started
                    lane._accepted += 1
                    self._reaccount(lane, job.reserved)
                    job.reserved = 0
                    job.payload = None
                    del numeric, compact
                elif job.kind == 'finish':
                    deferred = self._verification_executor is not None and job.lane.capture
                    result = job.lane._finish_worker(job.payload, defer_verification=deferred)
                    self._reaccount(job.lane)
                    if deferred:
                        self._dispatch_verification(job, result)
                    else:
                        job.future.set_result(result)
                with self._condition:
                    self._inflight = False
                    self._condition.notify_all()
        except BaseException as error:
            with self._condition:
                self._error = error
                if job is not None and job.future is not None and not job.future.done():
                    job.future.set_exception(CaptureWorkerError(str(error)))
                for pending in self._queue:
                    self._reserved -= pending.reserved
                    if pending.future is not None and not pending.future.done():
                        pending.future.set_exception(CaptureWorkerError(str(error)))
                self._queue.clear()
                if job is not None:
                    self._reserved -= job.reserved
                self._inflight = False
                self._condition.notify_all()
            for lane in self._lanes:
                closed_current_finalizer = (job is not None and job.kind == 'finish' and job.lane is lane
                    and not (lane.directory/'capture_complete.json').exists())
                if not lane._worker_finished or closed_current_finalizer:
                    lane._abort_worker(repr(error))
                    self._reaccount(lane)

    def close(self):
        """Drain accepted work; preserve unfinished captures without publishing."""
        with self._condition:
            if self._closed or self._closing:
                return
            failed = self._error is not None
            self._closing = True
            self._condition.notify_all()
        try:
            if not failed:
                future = Future()
                with self._condition:
                    # Stop is a control sentinel, not a payload reservation.
                    # Closing rejects new producers; accepted rows remain FIFO.
                    if self._error is None:
                        self._queue.append(_Job('stop', future=future))
                        self._condition.notify_all()
                    else:
                        future = None
                if future is not None:
                    future.result()
        finally:
            self._thread.join()
            if self._verification_executor is not None:
                # Running read-only jobs finish or surface child failures;
                # owner process-group termination also reaches spawned workers.
                self._verification_executor.shutdown(wait=True, cancel_futures=False)
            with self._condition:
                self._closed = True
                self._condition.notify_all()
        if self._error is not None:
            raise CaptureWorkerError(str(self._error)) from self._error

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        try:
            self.close()
        except CaptureWorkerError:
            if kind is None:
                raise


class AsyncLaneRecorder:
    """Independent consecutive lane, sharing a bounded writer thread.

    ``numeric_tree`` has the existing leading singleton environment axis;
    ``compact_row`` is flattened and unbatched. ``capture=False`` records only
    numeric diagnostics, suitable for full autonomous evaluation.
    """
    def __init__(self, directory, metadata, episode, *, writer,
                 capture=True, block_size=8, numeric_buffer_steps=32,
                 compression='lzf', action_source='student_then_expert_position_bridge_v1'):
        if type(block_size) is not int or block_size < 1:
            raise ValueError('block_size must be a positive integer')
        if type(numeric_buffer_steps) is not int or numeric_buffer_steps < 1:
            raise ValueError('numeric_buffer_steps must be a positive integer')
        self.directory = Path(directory)
        self.metadata = _json_copy(metadata)
        self.episode = str(episode)
        self.writer = writer
        self.capture = bool(capture)
        self.block_size = block_size
        self.numeric_buffer_steps = numeric_buffer_steps
        self.compression = compression
        self.action_source = action_source
        self._file = self._numeric = self._stream = None
        self._retained = self._accepted = self._submitted = 0
        self._worker_finished = self._finished = False
        self._finish_ticket = None
        self._lock = threading.Lock()
        self._profile = dict(copy_s=0., submission_wait_s=0., worker_numeric_append_s=0.,
            worker_compact_append_s=0., worker_flush_s=0., worker_finish_s=0.,
            worker_verify_s=0., worker_hash_s=0.)
        self.writer._request('open', self)

    def _open_worker(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.directory / 'numeric.h5', 'x')
        self._file.attrs['metadata_json'] = json.dumps(self.metadata)
        self._file.attrs['async_recording_status'] = 'partial'
        self._numeric = BufferedNumericWriter(self._file.create_group('episode'), self.numeric_buffer_steps)
        if self.capture:
            self._stream = CompactEpisodeStream(self.directory / 'demonstration', self.metadata,
                self.episode, block_size=self.block_size, compression=self.compression)
            self._stream.group.attrs['action_source'] = self.action_source

    def submit(self, numeric_tree, compact_row, step):
        with self._lock:
            self.writer.check()
            if self._finished:
                raise RuntimeError('Lane has already finished')
            if type(step) is not int or step != self._submitted:
                raise ValueError('Lane requires consecutive zero-based steps')
            if self.capture != (compact_row is not None):
                raise ValueError('compact_row presence must match capture flag')
            def validate(value):
                if isinstance(value, dict):
                    for child in value.values():
                        validate(child)
                else:
                    array = np.asarray(value)
                    if array.ndim < 1 or array.shape[0] != 1:
                        raise ValueError('numeric_tree fields require one leading lane axis')
            validate(numeric_tree)
            waited, copied = self.writer._submit(self, numeric_tree, compact_row, step)
            self._profile['submission_wait_s'] += waited
            self._profile['copy_s'] += copied
            self._submitted += 1

    def _finish_worker(self, result, *, defer_verification=False):
        if self._accepted != self._submitted or not self._accepted:
            raise ValueError('Cannot finish empty or incomplete submission sequence')
        self.writer._flush(self)
        self._file.attrs['async_recording_status'] = 'complete'
        self._file.flush()
        self._file.close()
        self._file = None
        self._numeric = None
        binding = None
        if self._stream is not None:
            started = time.perf_counter()
            self._stream.finish(result)
            self._profile['worker_finish_s'] += time.perf_counter() - started
            path = self.directory / 'demonstration/data.h5'
            binding = dict(path=str(path))
            if 'checkpoint_sha256' in result:
                binding['checkpoint_sha256'] = result['checkpoint_sha256']
            if 'dagger' in result:
                dagger = result['dagger']
                binding.update(trajectory_kind='expert_takeover_v1', accepted_success=bool(
                    dagger.get('expert_takeover_success') and dagger.get('expert_valid_ticks', 0) > 0))
            else:
                binding['accepted_success'] = bool(result.get('success'))
            self._stream = None
        receipt = dict(numeric_path=str(self.directory / 'numeric.h5'), dataset_capture=binding,
            steps=self._accepted, profile=dict(self._profile),
            recording_contract='shared_async_capture_v2' if defer_verification else 'shared_async_capture_v1')
        if not defer_verification:
            receipt = _verify_closed_capture(receipt)
        self._worker_finished = True
        return receipt

    def finish_async(self, result):
        """Queue finalization once; a ticket returns only after source verification.

        Queue admission may briefly apply normal bounded backpressure. The
        simulator may then continue other lanes while this one finishes.
        """
        with self._lock:
            self.writer.check()
            if self._finished:
                raise RuntimeError('Lane has already finished')
            started = time.perf_counter()
            future = self.writer._request_async('finish', self, _json_copy(result))
            self._finish_ticket = CaptureFinishTicket(future, started)
            self._finished = True
            return self._finish_ticket

    def wait_finish(self, timeout=None):
        if self._finish_ticket is None:
            raise RuntimeError('Call finish_async before waiting for a lane')
        return self._finish_ticket.result(timeout=timeout)

    def finish(self, result):
        return self.finish_async(result).result()

    def _abort_worker(self, error):
        cleanup_errors = []
        for writer in (self._numeric, self._stream):
            if writer is not None:
                try:
                    writer.flush()
                except BaseException as failure:
                    cleanup_errors.append(repr(failure))
        for handle in (self._file, self._stream.file if self._stream is not None else None):
            if handle is not None:
                try:
                    if handle.id.valid:
                        handle.close()
                except BaseException as failure:
                    cleanup_errors.append(repr(failure))
        try:
            if self.directory.exists():
                path = self.directory / 'capture_error.json'
                with path.open('x') as stream:
                    json.dump(dict(error=error, accepted_rows=self._accepted, submitted_rows=self._submitted,
                        verified=False, cleanup_errors=cleanup_errors), stream, indent=2)
        except BaseException:
            pass  # The original error must still reach the producer on disk failure.
        self._file = self._numeric = self._stream = None
        self._worker_finished = True
