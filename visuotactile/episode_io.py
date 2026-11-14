"""Bounded exact CPU staging of frozen episode features, without learned caches.

One worker prefetches in the already chosen visit order. Current and next clones
share one byte budget. A large next episode waits until the current visit exits;
insufficient RAM or an oversized episode uses the original mmap representation.
"""
from concurrent.futures import ThreadPoolExecutor, CancelledError
from contextlib import contextmanager
from pathlib import Path
import resource
import threading
import time


VERSION = 'frozen_episode_prefetch_v1'
GIB = 1 << 30


def normalize_io(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {'version', 'max_bytes', 'minimum_available_bytes'}:
        raise ValueError('Explicit episode I/O configuration required')
    config = dict(version=VERSION, max_bytes=GIB, minimum_available_bytes=10*GIB)
    config.update(value)
    if config['version'] != VERSION or type(config['max_bytes']) is not int or not 0 < config['max_bytes'] <= GIB:
        raise ValueError('Frozen episode prefetch requires a positive budget no larger than 1 GiB')
    if type(config['minimum_available_bytes']) is not int or config['minimum_available_bytes'] < 10*GIB:
        raise ValueError('Episode prefetch retains at least 10 GiB available RAM')
    return config


def available_memory():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1])*1024
    raise RuntimeError('Available RAM telemetry missing')


def io_snapshot():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    result = dict(utc=time.time(), minor_faults=usage.ru_minflt, major_faults=usage.ru_majflt)
    for line in Path('/proc/self/io').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in ('read_bytes', 'rchar', 'syscr'):
            result[key] = int(value)
    return result


def io_delta(before, after):
    return {key:after[key]-before[key] for key in ('minor_faults', 'major_faults', 'read_bytes', 'rchar', 'syscr')}


class EpisodePrefetch:
    """Call ``visit(index)`` in order and consume tensors only inside its scope.

    Scope exit clears the staged episode's tensor dictionaries. The caller must
    finish backward before exit and must not retain feature tensors afterwards.
    The byte cap covers clones owned by this loader, not OS page-cache residency.
    Process I/O counters include overlapping prefetch, never per-lane attribution.
    """
    def __init__(self, identities, factory, config, *, memory_available=available_memory):
        self.identities = list(identities)
        self.factory = factory
        self.config = normalize_io(config)
        if self.config is None:
            raise ValueError('EpisodePrefetch must be explicitly enabled')
        self.memory_available = memory_available
        self.condition = threading.Condition()
        self.owned_bytes = self.peak_bytes = 0
        self.closed = False
        self.next_index = 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='frozen-episode-io')
        self.future = self.pool.submit(self._load, 0) if self.identities else None
        self.receipts = []

    def _load(self, index):
        started = time.perf_counter()
        episode = self.factory(self.identities[index])
        receipt = dict(identity=self.identities[index], visit_index=index,
            mode='mmap', clone_bytes=0, budget_wait_s=0., load_s=time.perf_counter()-started)
        if episode.visual_stride != 1 or episode.perception is None:
            raise ValueError('Prefetch only supports the exact dense frozen visual representation')
        dictionaries = (episode.record, episode.perception)
        if any(not hasattr(value, 'device') or value.device.type != 'cpu'
               for dictionary in dictionaries for value in dictionary.values()):
            raise ValueError('Prefetch accepts CPU frozen tensors only')
        size = sum(value.numel()*value.element_size() for dictionary in dictionaries for value in dictionary.values())
        receipt['feature_bytes'] = size
        if size > self.config['max_bytes']:
            receipt['fallback'] = 'episode_exceeds_clone_budget'
            return episode, receipt, 0
        waiting = time.perf_counter()
        with self.condition:
            while self.owned_bytes+size > self.config['max_bytes'] and not self.closed:
                self.condition.wait()
            if self.closed:
                raise CancelledError('Closed episode prefetch')
            if self.memory_available()-size < self.config['minimum_available_bytes']:
                receipt['fallback'] = 'available_ram_reserve'
                return episode, receipt, 0
            self.owned_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.owned_bytes)
        receipt['budget_wait_s'] = time.perf_counter()-waiting
        cloning = time.perf_counter()
        try:
            episode.record = {key:value.clone() for key, value in episode.record.items()}
            episode.perception = {key:value.clone() for key, value in episode.perception.items()}
        except BaseException:
            with self.condition:
                self.owned_bytes -= size
                self.condition.notify_all()
            raise
        receipt.update(mode='cpu_clone', clone_bytes=size, clone_s=time.perf_counter()-cloning,
                       load_s=time.perf_counter()-started)
        return episode, receipt, size

    @contextmanager
    def visit(self, index):
        if self.closed or index != self.next_index or self.future is None:
            raise ValueError('Prefetch visits must retain the exact registered order')
        waited = time.perf_counter()
        episode, receipt, size = self.future.result()
        receipt['consumer_wait_s'] = time.perf_counter()-waited
        self.next_index += 1
        self.future = (self.pool.submit(self._load, self.next_index)
                       if self.next_index < len(self.identities) else None)
        try:
            yield episode, receipt
        finally:
            # Do this before releasing the allocation reservation. No future
            # visit may count memory as free while this loader owns the tensors.
            episode.record.clear()
            episode.perception.clear()
            with self.condition:
                self.owned_bytes -= size
                self.condition.notify_all()
            self.receipts.append(dict(receipt))

    def summary(self):
        return dict(version=VERSION, visits=len(self.receipts), peak_owned_clone_bytes=self.peak_bytes,
            clone_bytes=sum(r['clone_bytes'] for r in self.receipts),
            consumer_wait_s=sum(r['consumer_wait_s'] for r in self.receipts),
            load_s=sum(r['load_s'] for r in self.receipts),
            budget_wait_s=sum(r['budget_wait_s'] for r in self.receipts),
            mmap_fallbacks=sum(r['mode']=='mmap' for r in self.receipts),
            scope='Exact frozen CPU features only; no pooled representation or learned hidden state cache')

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if self.future is not None:
            self.future.cancel()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.future is not None and not self.future.cancelled():
            try:
                episode, _, size = self.future.result()
            except BaseException:
                pass  # A worker error is propagated by visit; cleanup adds none.
            else:
                episode.record.clear(); episode.perception.clear()
                self.owned_bytes -= size
        self.future = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()
