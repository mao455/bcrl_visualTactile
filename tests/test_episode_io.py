import json
import time
from pathlib import Path

import pytest
import torch

from visuotactile.episode_io import EpisodePrefetch, normalize_io, GIB
from visuotactile.native_data import Episode


def dataset(root):
    (root/'perception').mkdir()
    torch.save(dict(patches=torch.arange(384, dtype=torch.float32).reshape(2,3,4,4,4),
                    depth=torch.ones(2,3,2,4,4),camera_time=torch.arange(6,dtype=torch.float64).reshape(2,3),
                    camera_valid=torch.ones(2,3,dtype=torch.bool)),root/'perception/a.pt')
    torch.save(dict(patches=torch.arange(384, dtype=torch.float32).reshape(2,3,4,4,4)+1,
                    depth=torch.zeros(2,3,2,4,4),camera_time=torch.ones(2,3,dtype=torch.float64),
                    camera_valid=torch.ones(2,3,dtype=torch.bool)),root/'perception/b.pt')
    (root/'data.json').write_text(json.dumps(dict(visual_pair=True,visual_stride=1,entries={'a':{},'b':{}})))
    return lambda identity:Episode(root,identity,'I-HistoryDP')


def test_exact_features_operation_gradients_and_rng(tmp_path):
    factory=dataset(tmp_path)
    original=factory('a');model=torch.nn.Sequential(torch.nn.Conv2d(4,8,1),torch.nn.GroupNorm(2,8),
        torch.nn.SiLU(),torch.nn.AdaptiveAvgPool2d((2,2)))
    state=torch.get_rng_state().clone()
    expected=model(original.perception['patches'].reshape(-1,4,4,4))
    expected_grads=torch.autograd.grad(expected.sum(),tuple(model.parameters()))
    with EpisodePrefetch(['a','b','a'],factory,{},memory_available=lambda:100*GIB) as prefetch:
        for index,identity in enumerate(['a','b','a']):
            with prefetch.visit(index) as (episode,receipt):
                reference=factory(identity)
                for field in ('record','perception'):
                    for key,value in getattr(reference,field).items():
                        assert torch.equal(value,getattr(episode,field)[key])
                if identity=='a':
                    result=model(episode.perception['patches'].reshape(-1,4,4,4))
                    assert torch.equal(result,expected)
                    grads=torch.autograd.grad(result.sum(),tuple(model.parameters()))
                    assert all(torch.equal(a,b) for a,b in zip(grads,expected_grads))
                assert receipt['mode']=='cpu_clone'
            assert not episode.record and not episode.perception
        assert prefetch.summary()['visits']==3
    assert torch.equal(state,torch.get_rng_state())


def test_current_and_pending_share_byte_budget_and_close_unblocks(tmp_path):
    factory=dataset(tmp_path);sample=factory('a')
    size=sum(t.numel()*t.element_size() for d in (sample.record,sample.perception) for t in d.values())
    with EpisodePrefetch(['a','b'],factory,dict(max_bytes=size),memory_available=lambda:100*GIB) as prefetch:
        with prefetch.visit(0) as (episode,_):
            time.sleep(.03)
            assert prefetch.owned_bytes==size
            assert prefetch.peak_bytes==size
        with prefetch.visit(1) as (_,receipt):
            assert receipt['budget_wait_s']>0
        assert prefetch.owned_bytes==0
    prefetch=EpisodePrefetch(['a','b'],factory,dict(max_bytes=size),memory_available=lambda:100*GIB)
    with prefetch.visit(0):
        pass
    prefetch.close()
    assert prefetch.owned_bytes==0


@pytest.mark.parametrize('kind',['size','reserve'])
def test_fallback_preserves_original_mmap_values(tmp_path,kind):
    factory=dataset(tmp_path)
    config=dict(max_bytes=1) if kind=='size' else {}
    with EpisodePrefetch(['a'],factory,config,memory_available=lambda:10*GIB) as prefetch:
        with prefetch.visit(0) as (episode,receipt):
            assert receipt['mode']=='mmap' and receipt['clone_bytes']==0
            assert torch.equal(episode.perception['patches'],factory('a').perception['patches'])
        assert prefetch.peak_bytes==0


def test_no_out_of_order_visits_or_worker_error_suppression(tmp_path):
    factory=dataset(tmp_path)
    with EpisodePrefetch(['a','missing'],factory,{},memory_available=lambda:100*GIB) as prefetch:
        with pytest.raises(ValueError,match='order'):
            with prefetch.visit(1):pass
        with prefetch.visit(0):pass
        with pytest.raises(KeyError):
            with prefetch.visit(1):pass
    for config in [dict(max_bytes=GIB+1),dict(max_bytes=True),dict(minimum_available_bytes=1),dict(mode='pool')]:
        with pytest.raises(ValueError):normalize_io(config)
