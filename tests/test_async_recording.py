"""Real HDF5 equivalence, bounded backpressure and preserved capture failures."""
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from concurrent.futures import TimeoutError

import h5py
import numpy as np
import pytest

from visuotactile.async_recording import AsyncLaneRecorder, CaptureWorkerError, SharedCaptureWriter
from visuotactile.async_recording import _verify_closed_capture as verify_closed_capture_original
from visuotactile.compact import EpisodeFile
from visuotactile.recording import BufferedNumericWriter, CompactEpisodeStream, verify_stream_fields


def rows(n=11, identity=17):
    rng = np.random.default_rng(identity)
    rgb = rng.integers(0, 256, (n+1, 3, 3, 4, 5), dtype=np.uint8).astype(np.float32) / np.float32(255)
    depth = rng.random((n+1, 3, 1, 4, 5), dtype=np.float32)
    image = np.concatenate((rgb, depth), axis=2)
    valid = rng.random(depth.shape) > .2
    pose = rng.random((n+1, 3, 7), dtype=np.float32)
    for step in range(n):
        available = step < n-1
        row = {'obs/image': image[step], 'obs/depth_m': depth[step], 'obs/depth_valid': valid[step],
            'obs/step_id': np.asarray(step), 'obs/episode_id': np.asarray(identity),
            'transition/step_id': np.asarray(step), 'transition/episode_id': np.asarray(identity),
            'transition/observation_after/available': np.asarray(available),
            'transition/observation_after/image': image[step+1] if available else np.zeros_like(image[step]),
            'transition/observation_after/depth_valid': valid[step+1] if available else np.zeros_like(valid[step]),
            'transition/observation_after/step_id': np.asarray(step+1),
            'transition/observation_after/episode_id': np.asarray(identity),
            'terminated': np.asarray(not available), 'truncated': np.asarray(False),
            'transition/target_after_rad': rng.random(29, dtype=np.float32)}
        for key, sequence in {
            'camera_time_s': np.arange(n+1)[:, None].repeat(3, 1) / 60,
            'camera_frame': np.arange(n+1)[:, None].repeat(3, 1),
            'camera_pose_w': pose, 'camera_parent_pose_w': pose * 2,
        }.items():
            row['obs/'+key] = sequence[step]
            row['transition/observation_after/'+key] = sequence[step+1] if available else np.zeros_like(sequence[step])
        numeric = {'obs': {'step': np.asarray([step]), 'state': np.asarray([[step, -0., np.nan]], dtype=np.float32)},
            'transition': {'target': row['transition/target_after_rad'][None], 'valid': np.asarray([available])}}
        # A fresh owner prevents accidental fixture mutation in callers.
        yield numeric, {key: value.copy() for key, value in row.items()}


def contents(path, compact=False):
    result = {}
    with (EpisodeFile(path) if compact else h5py.File(path)) as stream:
        group = stream['episodes/episode'] if compact else stream['episode']
        def visit(key, obj):
            if hasattr(obj, 'dtype'):
                array = obj[:]
                result[key] = (array.dtype.str, array.shape, array.tobytes())
        group.visititems(visit)
    return result


@pytest.mark.parametrize('compression', ['lzf', 'gzip4_lossless_v1'])
@pytest.mark.parametrize('verification_workers', [0, 2])
def test_two_lanes_match_sync_source_fields_and_numeric(tmp_path, compression, verification_workers):
    metadata = {'camera_rig': {'version': 3}, 'trace_version': 2}
    result = {'success': True, 'checkpoint_sha256': 'fixture',
        'dagger': {'expert_takeover_success': True, 'expert_valid_ticks': 7}}
    with SharedCaptureWriter(max_pending_bytes=50_000, max_queue=3,
            verification_workers=verification_workers, verification_max_pending=4) as pool:
        lanes = [AsyncLaneRecorder(tmp_path / f'async_{i}', metadata, 'episode', writer=pool,
            compression=compression, numeric_buffer_steps=4, block_size=3) for i in range(2)]
        material = [list(rows(identity=17+i)) for i in range(2)]
        for step in range(11):
            for i, lane in enumerate(lanes):
                numeric, compact = material[i][step]
                lane.submit(numeric, compact, step)
        receipts = [lane.finish(result) for lane in lanes]
        stats = pool.statistics()
        assert stats['peak_reserved_bytes'] <= 50_000
        assert stats['peak_queued_or_copying'] <= 3
        assert stats['reserved_bytes'] == 0
        if verification_workers:
            assert stats['verification_completed'] == 2
            assert stats['verification_pending'] == 0
            assert stats['verification_peak_pending'] <= 4
    for i in range(2):
        directory = tmp_path / f'sync_{i}'
        directory.mkdir()
        stream = CompactEpisodeStream(directory/'demonstration', metadata, 'episode',
            block_size=3, compression=compression)
        stream.group.attrs['action_source'] = 'student_then_expert_position_bridge_v1'
        with h5py.File(directory/'numeric.h5', 'x') as file:
            writer = BufferedNumericWriter(file.create_group('episode'), 4)
            for step, (numeric, compact) in enumerate(material[i]):
                writer.append(numeric, step)
                stream.append(compact)
            writer.flush()
        stream.finish(result)
        assert contents(directory/'numeric.h5') == contents(tmp_path/f'async_{i}/numeric.h5')
        assert contents(directory/'demonstration/data.h5', True) == contents(tmp_path/f'async_{i}/demonstration/data.h5', True)
        with h5py.File(directory/'demonstration/data.h5') as sync, h5py.File(tmp_path/f'async_{i}/demonstration/data.h5') as asynchronous:
            assert sync.attrs['source_field_digests_json'] == asynchronous.attrs['source_field_digests_json']
        receipt = receipts[i]
        binding = receipt['dataset_capture']
        assert binding['verified_source_fields'] == verify_stream_fields(binding['path'])
        assert binding['sha256'] == hashlib.sha256(open(binding['path'], 'rb').read()).hexdigest()
        assert binding['accepted_success'] and binding['trajectory_kind'] == 'expert_takeover_v1'
        assert receipt['steps'] == 11 and receipt['profile']['worker_verify_s'] > 0
        persisted = json.loads((tmp_path/f'async_{i}/capture_complete.json').read_text())
        assert persisted['dataset_capture'] == binding and persisted['status'] == 'source_verified'
        if verification_workers:
            assert persisted['verification_process_pid'] != os.getpid()
            assert persisted['verification_execution'] == 'spawn_process_v1'


def test_caller_array_mutation_cannot_change_accepted_snapshot(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = BufferedNumericWriter.append
    def blocked(self, values, step):
        entered.set()
        assert release.wait(3)
        return original(self, values, step)
    monkeypatch.setattr(BufferedNumericWriter, 'append', blocked)
    pool = SharedCaptureWriter()
    try:
        lane = AsyncLaneRecorder(tmp_path/'async', {}, 'episode', writer=pool)
        numeric, compact = next(rows(n=1))
        expected_image = compact['obs/image'].copy()
        expected_target = numeric['transition']['target'].copy()
        lane.submit(numeric, compact, 0)
        assert entered.wait(2)
        numeric['transition']['target'][:] = 999
        compact['obs/image'][:] = 999
        release.set()
        receipt = lane.finish({'success': False})
        with EpisodeFile(receipt['dataset_capture']['path']) as stream:
            np.testing.assert_array_equal(stream['episodes/episode']['obs/image'][0], expected_image)
        with h5py.File(receipt['numeric_path']) as stream:
            np.testing.assert_array_equal(stream['episode/transition/target'][:], expected_target)
    finally:
        release.set()
        pool.close()


def test_numeric_only_pressure_flushes_without_hanging(tmp_path, monkeypatch):
    entered, release, completed = threading.Event(), threading.Event(), threading.Event()
    original = BufferedNumericWriter.append
    def blocked(self, values, step):
        if step == 0:
            entered.set()
            assert release.wait(3)
        return original(self, values, step)
    monkeypatch.setattr(BufferedNumericWriter, 'append', blocked)
    pool = SharedCaptureWriter(max_pending_bytes=768, max_queue=1)
    lane = AsyncLaneRecorder(tmp_path/'numeric', {}, 'episode', writer=pool, capture=False)
    errors = []
    try:
        lane.submit({'state': np.zeros((1, 64), np.float32)}, None, 0)
        assert entered.wait(2)
        def submit_rest():
            try:
                for step in range(1, 5):
                    lane.submit({'state': np.full((1, 64), step, np.float32)}, None, step)
            except BaseException as error:
                errors.append(error)
            finally:
                completed.set()
        producer = threading.Thread(target=submit_rest, daemon=True)
        producer.start()
        assert not completed.wait(.1)
        release.set()
        assert completed.wait(3)
        producer.join()
        assert not errors
        receipt = lane.finish({'success': False})
        assert receipt['dataset_capture'] is None
        assert receipt['profile']['submission_wait_s'] >= .09
        assert pool.statistics()['peak_reserved_bytes'] <= 768
        with h5py.File(receipt['numeric_path']) as stream:
            np.testing.assert_array_equal(stream['episode/state'][:, 0], np.arange(5))
    finally:
        release.set()
        pool.close()


def test_worker_failure_releases_blocked_producer_and_preserves_partial(tmp_path, monkeypatch):
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def broken(self, values, step):
        entered.set()
        assert release.wait(3)
        raise OSError('injected disk failure')
    monkeypatch.setattr(BufferedNumericWriter, 'append', broken)
    pool = SharedCaptureWriter(max_pending_bytes=512, max_queue=1)
    lane = AsyncLaneRecorder(tmp_path/'broken', {}, 'episode', writer=pool, capture=False)
    errors = []
    lane.submit({'state': np.zeros((1, 64), np.float32)}, None, 0)
    assert entered.wait(2)
    def submit_second():
        try:
            lane.submit({'state': np.ones((1, 64), np.float32)}, None, 1)
        except BaseException as error:
            errors.append(error)
        finally:
            finished.set()
    producer = threading.Thread(target=submit_second, daemon=True)
    producer.start()
    assert not finished.wait(.1)
    release.set()
    assert finished.wait(3)
    producer.join()
    assert len(errors) == 1 and isinstance(errors[0], CaptureWorkerError)
    with pytest.raises(CaptureWorkerError, match='injected disk failure'):
        lane.finish({'success': True})
    with pytest.raises(CaptureWorkerError, match='injected disk failure'):
        pool.close()
    assert not pool.statistics()['worker_alive']
    evidence = json.loads((tmp_path/'broken/capture_error.json').read_text())
    assert not evidence['verified']
    assert (tmp_path/'broken/numeric.h5').exists()
    assert not (tmp_path/'broken/result.json').exists()
    assert not (tmp_path/'broken/capture_complete.json').exists()


def test_close_preserves_unfinished_compact_without_publication(tmp_path):
    pool = SharedCaptureWriter()
    lane = AsyncLaneRecorder(tmp_path/'partial', {}, 'episode', writer=pool)
    numeric, compact = next(rows())
    lane.submit(numeric, compact, 0)
    pool.close()
    assert (tmp_path/'partial/demonstration/data.h5.partial').exists()
    assert not (tmp_path/'partial/demonstration/data.h5').exists()
    assert json.loads((tmp_path/'partial/capture_error.json').read_text())['accepted_rows'] == 1
    assert pool.statistics()['reserved_bytes'] == 0


def test_reject_too_small_budget_and_invalid_lane_axis_without_enqueuing(tmp_path):
    with SharedCaptureWriter(max_pending_bytes=100) as pool:
        lane = AsyncLaneRecorder(tmp_path/'numeric', {}, 'episode', writer=pool, capture=False)
        with pytest.raises(BufferError, match='budget'):
            lane.submit({'state': np.zeros((1, 64), np.float32)}, None, 0)
        with pytest.raises(ValueError, match='leading lane axis'):
            lane.submit({'state': np.zeros((2, 2), np.float32)}, None, 0)
        with pytest.raises(ValueError, match='consecutive'):
            lane.submit({'state': np.zeros((1, 2), np.float32)}, None, 1)
        assert pool.statistics()['reserved_bytes'] == 0


def test_retained_previous_frame_capacity_fails_instead_of_waiting_forever(tmp_path):
    material = list(rows(n=2))
    row_bytes = sum(value.nbytes for value in material[0][1].values())
    numeric_bytes = sum(value.nbytes for fields in material[0][0].values() for value in fields.values())
    # Exactly one incoming reservation fits; after step 0 the necessary post
    # frame occupies positive capacity even after every block has been flushed.
    with SharedCaptureWriter(max_pending_bytes=row_bytes + 2 * numeric_bytes) as pool:
        lane = AsyncLaneRecorder(tmp_path/'capacity', {}, 'episode', writer=pool)
        lane.submit(*material[0], 0)
        started = time.perf_counter()
        with pytest.raises(BufferError, match='Retained previous frames'):
            lane.submit(*material[1], 1)
        assert time.perf_counter() - started < 2
    assert (tmp_path/'capacity/demonstration/data.h5.partial').exists()


def test_every_hdf5_handle_opens_in_shared_worker(tmp_path, monkeypatch):
    owners = []
    original = h5py.File
    def tracked(*args, **kwargs):
        owners.append(threading.current_thread().name)
        return original(*args, **kwargs)
    monkeypatch.setattr(h5py, 'File', tracked)
    with SharedCaptureWriter() as pool:
        lane = AsyncLaneRecorder(tmp_path/'ownership', {}, 'episode', writer=pool)
        lane.submit(*next(rows(n=1)), 0)
        lane.finish({'success': True})
    assert owners and set(owners) == {'tight-capture-writer'}


def test_async_finalization_allows_other_lane_submission_and_timeout_keeps_work(tmp_path, monkeypatch):
    entered, release = threading.Event(), threading.Event()
    original = verify_stream_fields
    def paused(path):
        if 'first' in str(path):
            entered.set()
            assert release.wait(3)
        return original(path)
    monkeypatch.setattr('visuotactile.async_recording.verify_stream_fields', paused)
    pool = SharedCaptureWriter(max_pending_bytes=1 << 20, max_queue=4)
    try:
        first = AsyncLaneRecorder(tmp_path/'first', {}, 'episode', writer=pool)
        second = AsyncLaneRecorder(tmp_path/'second', {}, 'episode', writer=pool)
        first.submit(*next(rows(n=1)), 0)
        ticket = first.finish_async({'success': True})
        assert entered.wait(2)
        assert not ticket.done()
        with pytest.raises(TimeoutError):
            ticket.result(timeout=.01)
        with pytest.raises(RuntimeError, match='finished'):
            first.submit(*next(rows(n=1)), 1)
        # The writer is verifying first; simulation can submit another lane.
        second.submit(*next(rows(n=1, identity=19)), 0)
        second_ticket = second.finish_async({'success': False})
        assert not second_ticket.done()
        release.set()
        receipt = first.wait_finish(timeout=2)
        assert receipt['dataset_capture']['accepted_success']
        assert not second_ticket.result(timeout=2)['dataset_capture']['accepted_success']
        assert ticket.done()
        assert receipt['profile']['enqueue_to_verified_s'] >= .01
    finally:
        release.set()
        pool.close()


def test_compact_verification_failure_never_returns_success_binding(tmp_path, monkeypatch):
    def broken(path):
        raise ValueError('injected source digest mismatch')
    monkeypatch.setattr('visuotactile.async_recording.verify_stream_fields', broken)
    pool = SharedCaptureWriter()
    lane = AsyncLaneRecorder(tmp_path/'verify', {}, 'episode', writer=pool)
    numeric, compact = next(rows(n=1))
    lane.submit(numeric, compact, 0)
    with pytest.raises(CaptureWorkerError, match='source digest mismatch'):
        lane.finish({'success': True})
    with pytest.raises(CaptureWorkerError):
        pool.close()
    evidence = json.loads((tmp_path/'verify/capture_error.json').read_text())
    assert not evidence['verified']
    assert (tmp_path/'verify/demonstration/data.h5').exists()  # preserved failed verification input
    assert not (tmp_path/'verify/capture_complete.json').exists()


def _paused_spawn_verifier(receipt, process_worker):
    """Spawn-importable test fault, with filesystem synchronization only."""
    root = Path(receipt['numeric_path']).parent.parent
    (root/('entered_'+Path(receipt['numeric_path']).parent.name)).touch()
    deadline = time.monotonic()+10
    while not (root/'release_verifiers').exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Test did not release spawned verifier')
        time.sleep(.01)
    if (root/'crash_verifier').exists():
        os._exit(23)
    return verify_closed_capture_original(receipt, process_worker)


def wait_until(predicate, seconds=5):
    deadline = time.monotonic()+seconds
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for a bounded test condition')
        time.sleep(.01)


def test_spawn_verification_does_not_block_other_lane_append_and_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr('visuotactile.async_recording._verify_closed_capture', _paused_spawn_verifier)
    pool = SharedCaptureWriter(verification_workers=1, verification_max_pending=1)
    try:
        first = AsyncLaneRecorder(tmp_path/'first', {}, 'episode', writer=pool)
        second = AsyncLaneRecorder(tmp_path/'second', {}, 'episode', writer=pool)
        first.submit(*next(rows(n=1)), 0)
        a = first.finish_async({'success': True})
        wait_until(lambda:(tmp_path/'entered_first').exists())
        assert not a.done() and not (tmp_path/'first/capture_complete.json').exists()
        second.submit(*next(rows(n=1, identity=19)), 0)
        # The shared writer actually appends, rather than merely accepting a
        # queued copy while HDF verification monopolizes its thread/PHIL lock.
        wait_until(lambda:second._accepted == 1)
        b = second.finish_async({'success': False})
        wait_until(lambda:(tmp_path/'second/demonstration/data.h5').exists())
        assert pool.statistics()['verification_pending'] == 1
        assert pool.statistics()['verification_peak_pending'] == 1
        assert not (tmp_path/'entered_second').exists()
        (tmp_path/'release_verifiers').touch()
        assert a.result(timeout=5)['dataset_capture']['accepted_success']
        assert not b.result(timeout=5)['dataset_capture']['accepted_success']
        processes = list(pool._verification_executor._processes.values())
        pool.close()
        assert all(not process.is_alive() for process in processes)
        assert pool.statistics()['verification_completed'] == 2
        assert pool.statistics()['verification_pending'] == 0
    finally:
        (tmp_path/'release_verifiers').touch(exist_ok=True)
        pool.close()


def test_spawn_digest_failure_never_publishes_and_reaches_check_ticket_close(tmp_path, monkeypatch):
    original = AsyncLaneRecorder._finish_worker
    def corrupt_after_close(self, result, **kwargs):
        receipt = original(self, result, **kwargs)
        with h5py.File(receipt['dataset_capture']['path'], 'r+') as stream:
            stream['episodes/episode/transition/target_after_rad'][0, 0] += 1
        return receipt
    monkeypatch.setattr(AsyncLaneRecorder, '_finish_worker', corrupt_after_close)
    pool = SharedCaptureWriter(verification_workers=2, verification_max_pending=4)
    lane = AsyncLaneRecorder(tmp_path/'corrupt', {}, 'episode', writer=pool)
    lane.submit(*next(rows(n=1)), 0)
    ticket = lane.finish_async({'success': True})
    with pytest.raises(CaptureWorkerError, match='content differs'):
        ticket.result(timeout=5)
    with pytest.raises(CaptureWorkerError, match='content differs'):
        pool.check()
    with pytest.raises(CaptureWorkerError, match='content differs'):
        pool.close()
    assert not (tmp_path/'corrupt/capture_complete.json').exists()
    assert (tmp_path/'corrupt/demonstration/data.h5').exists()
    assert not json.loads((tmp_path/'corrupt/capture_error.json').read_text())['verified']
    assert pool.statistics()['verification_failed'] == 1
    assert pool.statistics()['verification_pending'] == 0


def test_spawn_child_crash_unblocks_full_writer_queue_and_preserves_files(tmp_path, monkeypatch):
    monkeypatch.setattr('visuotactile.async_recording._verify_closed_capture', _paused_spawn_verifier)
    pool = SharedCaptureWriter(max_queue=1, verification_workers=1, verification_max_pending=1)
    first = AsyncLaneRecorder(tmp_path/'first', {}, 'episode', writer=pool)
    second = AsyncLaneRecorder(tmp_path/'second', {}, 'episode', writer=pool)
    third = AsyncLaneRecorder(tmp_path/'third', {}, 'episode', writer=pool, capture=False)
    first.submit(*next(rows(n=1)), 0)
    a = first.finish_async({'success': True})
    wait_until(lambda:(tmp_path/'entered_first').exists())
    second.submit(*next(rows(n=1, identity=19)), 0)
    b = second.finish_async({'success': True})
    wait_until(lambda:(tmp_path/'second/demonstration/data.h5').exists())
    third.submit({'state':np.zeros((1, 4), np.float32)}, None, 0)
    errors, done = [], threading.Event()
    def blocked_submit():
        try: third.submit({'state':np.ones((1, 4), np.float32)}, None, 1)
        except BaseException as error: errors.append(error)
        finally: done.set()
    producer = threading.Thread(target=blocked_submit, daemon=True)
    producer.start()
    assert not done.wait(.05)
    (tmp_path/'crash_verifier').touch()
    (tmp_path/'release_verifiers').touch()
    assert done.wait(5)
    producer.join()
    assert len(errors) == 1 and isinstance(errors[0], CaptureWorkerError)
    for ticket in (a, b):
        with pytest.raises(CaptureWorkerError): ticket.result(timeout=5)
    with pytest.raises(CaptureWorkerError): pool.check()
    with pytest.raises(CaptureWorkerError): pool.close()
    for name in ('first', 'second', 'third'):
        assert (tmp_path/name/'capture_error.json').exists()
        assert not (tmp_path/name/'capture_complete.json').exists()
    assert pool.statistics()['verification_pending'] == 0
    assert not pool.statistics()['worker_alive']


def test_numeric_only_mode_never_starts_verifier_processes(tmp_path):
    with SharedCaptureWriter(verification_workers=2, verification_max_pending=4) as pool:
        lane = AsyncLaneRecorder(tmp_path/'numeric', {}, 'episode', writer=pool, capture=False)
        lane.submit({'state':np.zeros((1, 4), np.float32)}, None, 0)
        receipt = lane.finish({'success': False})
        assert receipt['dataset_capture'] is None
        assert pool.statistics()['verification_submitted'] == 0
        assert pool._verification_executor._processes == {}
