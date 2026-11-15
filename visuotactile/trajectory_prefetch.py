"""One bounded CPU read-ahead batch; no RNG, CUDA or learned-state caching."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

from .episode_io import available_memory
from .frozen_visual_io import prefetch_profile


class TrajectoryPrefetch:
    def __init__(self, groups, factory, root, configuration, *, memory=available_memory):
        if configuration != prefetch_profile() or not groups:
            raise ValueError('Explicit nonempty B2 prefetch registration required')
        self.groups=[list(group) for group in groups]
        if any(not 1 <= len(group) <= 2 for group in self.groups):
            raise ValueError('Only existing one/two-trajectory batches are supported')
        self.factory=factory; self.root=Path(root); self.config=configuration; self.memory=memory
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='frozen-B2-read')
        self.future=None; self.index=0; self.closed=False; self.current=None; self.receipts=[]

    @staticmethod
    def release(episodes):
        for episode in episodes:
            episode.record.clear(); episode.perception.clear()
        episodes.clear()

    def _load(self,index,background):
        identities=self.groups[index]
        size=sum((self.root/'perception'/(identity+'.pt')).stat().st_size for identity in identities)
        # Admission is before allocation. No waiting on a current batch's release,
        # no growing queue and no mmap copy. Foreground retains its original guard.
        if background and (size>self.config['max_prefetch_bytes'] or
                self.memory()-size-256*1024**2<self.config['minimum_available_bytes']):
            return None,dict(index=index,mode='foreground_fallback',estimated_bytes=size)
        started=time.perf_counter();episodes=[]
        try:
            for identity in identities:episodes.append(self.factory(identity))
        except BaseException:
            self.release(episodes);raise
        return episodes,dict(index=index,mode='prefetched' if background else 'foreground',
            estimated_bytes=size,load_s=time.perf_counter()-started,
            loaded_storage_bytes=sum(e.frozen_visual_io_receipt['loaded_storage_bytes'] for e in episodes))

    def take(self,index):
        if self.closed or self.current is not None or index!=self.index:
            raise ValueError('Read-ahead must follow the original batch order')
        started=time.perf_counter();fallback=None
        if self.future is None:episodes,row=self._load(index,False)
        else:
            future=self.future;self.future=None
            episodes,row=future.result();del future
            if episodes is None:
                fallback=row;episodes,row=self._load(index,False)
        row['consumer_wait_s']=time.perf_counter()-started
        if fallback is not None:row['fallback']=fallback
        self.current=episodes;self.receipts.append(row);self.index+=1
        if self.index<len(self.groups):self.future=self.pool.submit(self._load,self.index,True)
        return episodes

    def finish(self):
        if self.current is not None:self.release(self.current);self.current=None

    def summary(self):
        return dict(version=self.config['version'],batches=len(self.receipts),
            prefetch_hits=sum(r['mode']=='prefetched' for r in self.receipts),
            admission_fallbacks=sum('fallback' in r for r in self.receipts),
            consumer_wait_s=sum(r['consumer_wait_s'] for r in self.receipts),
            load_s=sum(r['load_s'] for r in self.receipts),
            maximum_next_batch_bytes=max((r['loaded_storage_bytes'] for r in self.receipts if r['mode']=='prefetched'),default=0),
            scope='One current B2 plus one bounded CPU frozen-feature batch; actual load time overlaps compute, not added to wall time')

    def close(self):
        if self.closed:return
        self.closed=True;self.finish()
        future=self.future;self.future=None
        if future is not None:future.cancel()
        self.pool.shutdown(wait=True,cancel_futures=True)
        if future is not None and not future.cancelled():
            episodes,_=future.result()
            if episodes is not None:self.release(episodes)
