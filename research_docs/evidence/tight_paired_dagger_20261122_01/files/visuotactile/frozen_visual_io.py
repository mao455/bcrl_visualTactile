"""Explicit eager/mapped immutable visual reads; no learned history cache."""
import json
from pathlib import Path

VERSION = 'frozen_visual_eager_v1'


def profile():
    return dict(version=VERSION,torch_load_mmap=False,
        lifetime='current_episode_or_current_B2_visits_only',learned_state_cache=False)


def prefetch_profile():
    return dict(version='frozen_visual_eager_prefetch_v2',torch_load_mmap=False,
        lifetime='current_B2_plus_at_most_one_next_B2',learned_state_cache=False,
        max_prefetch_bytes=3*1024**3,minimum_available_bytes=12*1024**3)


def mapped_profile():
    """Explicit zero-private-copy alternative; immutable pages share the OS cache.

    No asynchronous materialization: only the current one/two episodes are
    mapped. Learned history is still recomputed for every optimizer update.
    Old eager profiles and absent-profile behavior are unchanged.
    """
    return dict(version='frozen_visual_mapped_current_v3',torch_load_mmap=True,
        lifetime='current_episode_or_current_B2_visits_only',learned_state_cache=False)


def validate_profile(value):
    if value is None:return None
    if (not isinstance(value,dict) or value not in (profile(),prefetch_profile(),mapped_profile())
            or type(value.get('torch_load_mmap')) is not bool
            or value.get('learned_state_cache') is not False):
        raise ValueError('Exact frozen visual I/O profile required')
    return dict(value)


def load_profile(root):
    path=Path(root)/'frozen_visual_io.json'
    return validate_profile(json.loads(path.read_text())) if path.exists() else None


class FrozenVisualHotset:
    """Opt-in bounded CUDA residency for immutable, already verified features.

    The priority list affects residency only, never visit/anchor order. Entries
    are filled lazily and never contain trainable projections or recurrent state.
    The caller owns one loader for a fixed data/cache version, and must retain
    model/activation headroom in addition to this explicit byte budget.
    """
    def __init__(self, root, priority, *, max_bytes, device='cuda:0', minimum_free_bytes=8*1024**3):
        import torch
        from .compact import sha256
        self.root=Path(root).resolve();self.device=torch.device(device)
        if type(max_bytes) is not int or not 0<max_bytes<=4*1024**3:
            raise ValueError('Hotset must be explicitly bounded to at most 4 GiB')
        if self.device.type!='cuda' or minimum_free_bytes<8*1024**3:
            raise ValueError('CUDA hotset requires at least 8 GiB admission headroom')
        self.max_bytes=max_bytes;self.minimum_free_bytes=minimum_free_bytes
        self.binding=dict(data_sha256=sha256(self.root/'data.json'),
            cache_index_sha256=sha256(self.root.parent/'cache_index.json'))
        self.metadata={p:(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns)
            for p in (self.root/'data.json',self.root.parent/'cache_index.json')}
        self.selected={};self.cached={};self.bytes=0;self.hits=self.misses=self.fallbacks=0
        self.avoided_payload_bytes=0;self.first_fill_seconds=0.;allocated=0
        for identity in dict.fromkeys(priority):
            path=self.root/'perception'/(identity+'.pt');s=path.stat()
            if allocated+s.st_size<=max_bytes:
                self.selected[identity]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
                allocated+=s.st_size
        self.planned_bytes=allocated

    def __call__(self, identity):
        import time,torch
        from .native_data import Episode
        for path,expected in self.metadata.items():
            s=path.stat()
            if (s.st_ino,s.st_size,s.st_mtime_ns)!=expected:
                raise ValueError('Frozen visual data/cache version changed during hotset lifetime')
        if identity in self.selected:
            path=self.root/'perception'/(identity+'.pt');s=path.stat()
            if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)!=self.selected[identity]:
                raise ValueError('Immutable visual source changed during hotset lifetime')
        if identity in self.cached:
            self.hits+=1;self.avoided_payload_bytes+=self.selected[identity][2]
            return self.cached[identity]
        self.misses+=1
        episode=Episode(self.root,identity,'I-HistoryDP',frozen_visual_io=mapped_profile())
        if identity not in self.selected:return episode
        tensors=[x for group in (episode.record,episode.perception) for x in group.values()]
        size=sum(x.numel()*x.element_size() for x in tensors)
        if self.bytes+size>self.max_bytes or torch.cuda.mem_get_info(self.device)[0]-size<self.minimum_free_bytes:
            self.fallbacks+=1;return episode
        start=time.perf_counter()
        for group in (episode.record,episode.perception):
            for key,value in group.items():group[key]=value.to(self.device)
        torch.cuda.synchronize(self.device)
        self.first_fill_seconds+=time.perf_counter()-start
        self.bytes+=size;self.cached[identity]=episode
        return episode

    def audit_cached(self):
        """Diagnostic only: exact round-trip check after the measured stage."""
        import torch
        from .native_data import Episode
        checks=[]
        for identity,episode in self.cached.items():
            reference=Episode(self.root,identity,'I-HistoryDP',frozen_visual_io=mapped_profile())
            for name in ('record','perception'):
                for key,value in getattr(episode,name).items():
                    original=getattr(reference,name)[key]
                    equal=(value.shape==original.shape and value.dtype==original.dtype
                        and not value.requires_grad)
                    for a in range(0,len(value),32):
                        equal=equal and torch.equal(value[a:a+32].cpu().view(torch.uint8),
                            original[a:a+32].view(torch.uint8))
                    if not equal:raise ValueError('Frozen visual bytewise round trip failed')
                    checks.append(dict(identity=identity,array=key,bytes=value.numel()*value.element_size(),equal=equal))
        return dict(passed=True,checks=checks,binding=self.binding)

    def summary(self):
        return dict(version='frozen_visual_cuda_hotset_diagnostic_v1',max_bytes=self.max_bytes,
            resident_bytes=self.bytes,planned_file_bytes=self.planned_bytes,priority_ids=list(self.selected),
            cached_ids=list(self.cached),hits=self.hits,misses=self.misses,admission_fallbacks=self.fallbacks,
            avoided_logical_payload_bytes=self.avoided_payload_bytes,first_fill_seconds=self.first_fill_seconds,
            minimum_free_bytes=self.minimum_free_bytes,binding=self.binding,
            scope='Immutable patches/depth/camera timestamps only; no learned features, labels, history or backend inputs')
