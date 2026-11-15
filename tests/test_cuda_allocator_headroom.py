"""Explicit reclamation uses live free bytes, and never relaxes the reserve."""
import json,time
import pytest,torch
from visuotactile.state_resources import FullLease

def test_reclaim_only_unused_memory_and_recheck_same_reserve(tmp_path,monkeypatch):
    mib=1024**2;free=[5000*mib];calls=[]
    (tmp_path/'session.json').write_text(json.dumps(dict(deadline=time.time()+60,
        resources=dict(min_ram_gib=12,min_gpu_free_mib=6144))))
    (tmp_path/'shared_resources.json').write_text(json.dumps(dict(utc_epoch=time.time(),
        ram_available_bytes=30*1024**3,gpu_free_mib=5000)))
    (tmp_path/'cuda_allocator_policy.json').write_text(json.dumps(dict(version='reclaim_unused_on_low_headroom_v1')))
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda:(free[0],24*1024**3))
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda:8*1024**3)
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda:5*1024**3)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:5*1024**3)
    def release():calls.append(True);free[0]=8000*mib
    monkeypatch.setattr(torch.cuda,'empty_cache',release)
    with FullLease(tmp_path,deadline=time.time()+60).block():pass
    assert calls==[True]
    free[0]=5000*mib;monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    with pytest.raises(RuntimeError,match='VRAM reserve'):
        with FullLease(tmp_path,deadline=time.time()+60).block():pass
