from types import SimpleNamespace
import torch
from visuotactile.models.native_dp import NativePolicy,encode_segment
from visuotactile.position_control import continuous_image_contract
from visuotactile.continuous_vision import encode_dense


def test_batched_visual_path_matches_tick_path_and_future_causality():
    torch.manual_seed(8);m=NativePolicy('I-HistoryDP',24,modality='G00').eval();m.contract=continuous_image_contract('q_reached')
    n=26;t=torch.arange(n,dtype=torch.float64)/60;r=torch.randn(n,178)*.1;r[:,176]=1
    p=torch.randn(n,3,384,18,32)*.01;d=torch.rand(n,3,2,18,32)
    rec=dict(robot=r,time=t);ep=SimpleNamespace(record=dict(rec,camera_time=t[:,None].expand(-1,3),camera_valid=torch.ones(n,3,dtype=torch.bool)),
        perception=dict(patches=p,depth=d),visual_stride=1)
    ep.frame=lambda k,device:dict(patches=p[k],depth=d[k],time=ep.record['camera_time'][k],valid=ep.record['camera_valid'][k])
    with torch.no_grad():
        old,_,_=encode_segment(m,ep,0,n)
        batched,a=encode_dense(m,rec,episode=ep)
        assert torch.allclose(old,batched,atol=2e-5,rtol=1e-5),(old-batched).abs().max()
        with_visual,visual_anchors,visual=encode_dense(m,rec,episode=ep,return_visual=True)
        assert torch.equal(with_visual,batched) and torch.equal(visual_anchors,a)
        assert visual['ticks'].tolist()==list(range(n))
        from visuotactile.native_data import residual_visual
        memory=None;online_visual=[]
        for tick in range(n):
            memory=m.observe(r[tick],t[tick],tick,ep.frame(tick,'cpu'),memory)
            online_visual.append(residual_visual(m,memory,t[tick]))
        torch.testing.assert_close(visual['readout'],torch.cat(online_visual),atol=2e-5,rtol=1e-5)
        suffix,_=encode_dense(m,rec,episode=ep,start=12)
        assert torch.allclose(suffix,batched[1:],atol=2e-5,rtol=1e-5)
        p[13:]+=1
        changed,_=encode_dense(m,rec,episode=ep)
        assert torch.equal(changed[:2],batched[:2])
