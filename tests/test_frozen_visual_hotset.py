"""CPU lifecycle guards; actual CUDA/data/gradient parity is measured separately."""
import pytest
import torch
from visuotactile.frozen_visual_io import FrozenVisualHotset

class Value:
    def numel(self):return 2
    def element_size(self):return 1
    def to(self,device):return self

@pytest.fixture
def sources(tmp_path,monkeypatch):
    root=tmp_path/'state_data';root.mkdir();(root/'perception').mkdir()
    (root/'data.json').write_text('{}');(tmp_path/'cache_index.json').write_text('{}')
    for identity in ['one','two']:(root/'perception'/(identity+'.pt')).write_bytes(b'x'*12)
    calls=[]
    def episode(root,identity,group,**kwargs):
        from types import SimpleNamespace
        calls.append(identity)
        return SimpleNamespace(record={'camera_time':Value(),'camera_valid':Value()},perception={'patches':Value(),'depth':Value()})
    monkeypatch.setattr('visuotactile.native_data.Episode',episode)
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *args:(32*1024**3,48*1024**3))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda *args:None)
    return root,calls

def test_bounded_identity_and_no_rng(sources):
    root,calls=sources;before=torch.get_rng_state().clone()
    cache=FrozenVisualHotset(root,['one','one','two'],max_bytes=20)
    first=cache('one');assert cache('one') is first
    cache('two');cache('two')
    assert calls==['one','two','two']
    assert cache.summary()['resident_bytes']==8 and cache.summary()['priority_ids']==['one']
    assert cache.summary()['hits']==1 and cache.summary()['avoided_logical_payload_bytes']==12
    assert torch.equal(before,torch.get_rng_state())

def test_source_mutation_rejected(sources):
    root,_=sources;cache=FrozenVisualHotset(root,['one'],max_bytes=20);cache('one')
    (root/'perception/one.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='source changed'):cache('one')

def test_headroom_falls_back_without_residency(sources,monkeypatch):
    root,calls=sources;monkeypatch.setattr(torch.cuda,'mem_get_info',lambda *args:(7*1024**3,24*1024**3))
    cache=FrozenVisualHotset(root,['one'],max_bytes=20);cache('one');cache('one')
    assert cache.summary()['resident_bytes']==0 and cache.summary()['admission_fallbacks']==2
    assert calls==['one','one']

@pytest.mark.parametrize('size',[0,-1,4*1024**3+1])
def test_budget_rejected(sources,size):
    with pytest.raises(ValueError,match='bounded'):FrozenVisualHotset(sources[0],['one'],max_bytes=size)


def test_data_version_change_rejected(sources):
    root,_=sources;cache=FrozenVisualHotset(root,['one'],max_bytes=20);cache('one')
    (root/'data.json').write_text('{"changed": true}')
    with pytest.raises(ValueError,match='version changed'):cache('one')
