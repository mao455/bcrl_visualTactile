"""Eager immutable visual reads reach the backend without changing replay."""
from contextlib import contextmanager
import json
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

from visuotactile.frozen_visual_io import profile
from visuotactile import native_data
from tools import tight_finish, tight_fresh_backend


def equal_tree(left,right):
    if isinstance(left,torch.Tensor):assert torch.equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:equal_tree(left[key],right[key])
    else:assert left==right


def test_eager_backend_replay_same_values_lease_and_current_episode_lifetime(tmp_path,monkeypatch):
    from test_residual_cache_planning import cache_fixture,Base
    row,binding,entry,_=cache_fixture(tmp_path,monkeypatch,length=25)
    state={'admitted':False};references=[];calls=[]
    @contextmanager
    def block():
        assert not state['admitted']
        state['admitted']=True
        try:yield
        finally:state['admitted']=False
    class Episode:
        def __init__(self,*args,**kwargs):
            calls.append(kwargs)
            if kwargs:
                assert state['admitted'] and kwargs==dict(frozen_visual_io=profile())
                self.frozen_visual_io_receipt=dict(profile=profile(),identity='recovery',
                    tensor_load_s=.01,file_bytes=100,loaded_storage_bytes=80,extra_clone_bytes=0)
            references.append(weakref.ref(self))
        def frame(self,tick,device):return dict(tick=tick)
    monkeypatch.setattr(native_data,'Episode',Episode)
    lease=SimpleNamespace(block=block)
    common=dict(cache_identity=binding,transfer_mode='episode',scheduler_index_mode='host')
    expected=native_data.cache_residual_episode(Base('u_expert'),row,entry,lease,**common)
    progress=[]
    actual=native_data.cache_residual_episode(Base('u_expert'),row,entry,lease,
        frozen_visual_io=profile(),progress=lambda **kw:progress.append(kw),**common)
    equal_tree(expected,actual)
    assert calls==[{},dict(frozen_visual_io=profile())]
    assert all(reference() is None for reference in references)
    assert progress[0]['phase']=='frozen_visual_loaded'
    assert progress[0]['frozen_visual_io_receipt']['extra_clone_bytes']==0


def test_backend_cli_profile_and_preparation_binding(tmp_path,monkeypatch):
    from test_residual_cache_planning import test_backend_prepare_explicit_profile_propagates_and_default_remains_absent
    # Reuse the existing real CPU preparation fixture and its default checks.
    original=tight_fresh_backend.prepare
    def prepare(*args,**kwargs):return original(*args,**kwargs,frozen_visual_io=profile())
    monkeypatch.setattr(tight_fresh_backend,'prepare',prepare)
    test_backend_prepare_explicit_profile_propagates_and_default_remains_absent(tmp_path,monkeypatch,32)
    backend=tmp_path/'round/A/backend'
    assert tight_finish.read(backend/'session.json')['residual_pair']['frozen_visual_io']==profile()
    assert tight_finish.read(backend/'prepared.json')['frozen_visual_io']==profile()
    path=tmp_path/'profile.json';path.write_text(json.dumps(profile()));seen=[]
    monkeypatch.setattr(tight_fresh_backend,'prepare',lambda *a,**kw:seen.append((a,kw)))
    monkeypatch.setattr(sys,'argv',['backend','--root',str(tmp_path),'--role','B','--frozen-visual-io-json',str(path)])
    tight_fresh_backend.main()
    assert seen[0][1]['frozen_visual_io']==profile()
    with pytest.raises(ValueError,match='frozen visual'):
        original(tmp_path/'invalid','A',1,frozen_visual_io=dict(profile(),torch_load_mmap=True))
    assert not (tmp_path/'invalid').exists()


def test_backend_cache_worker_propagates_profile_and_persists_read_receipt(tmp_path,monkeypatch):
    from test_dagger_evaluation import cache_fixture
    from visuotactile.models import native_dp
    records,_,_=cache_fixture(tmp_path,monkeypatch)
    path=tmp_path/'G11/nominal/core.pt';real_load=torch.load
    rebuilt=real_load(path,weights_only=False);path.unlink()
    visual=tmp_path/'state_data/perception/core.pt';visual.parent.mkdir();visual.write_bytes(b'pinned visual')
    tight_finish.write(tmp_path/'source/cache_index.json',dict(entries={'core':dict(sha256=tight_finish.sha(visual))}))
    session=tight_finish.read(tmp_path/'session.json');session['residual_pair']['frozen_visual_io']=profile()
    tight_finish.write(tmp_path/'session.json',session)
    monkeypatch.setattr(torch,'load',lambda path,*a,**kw:{} if str(path)==str(tmp_path/'base.pt') else real_load(path,*a,**kw))
    monkeypatch.setattr(native_dp,'load_policy',lambda *a,**kw:torch.nn.Linear(1,1))
    receipt=dict(profile=profile(),identity='core',tensor_load_s=.01,file_bytes=13,loaded_storage_bytes=8,extra_clone_bytes=0)
    def cache(model,record,entry,lease,**kwargs):
        assert kwargs['frozen_visual_io']==profile() and record is records['core']
        kwargs['progress'](phase='frozen_visual_loaded',frozen_visual_io_receipt=receipt)
        return rebuilt
    monkeypatch.setattr(native_data,'cache_residual_episode',cache)
    tight_finish.residual_cache(tmp_path,acceptance=True)
    index=tight_finish.read(tmp_path/'G11/nominal/index.json')
    assert index['frozen_visual_io']==profile()
    assert index['episodes']['core']['implementation']['frozen_visual_io_receipt']==receipt
    assert index['episodes']['val']['recovered_orphan']  # No invented eager-read receipt for old payloads.
    assert 'implementation' not in index['episodes']['val']
