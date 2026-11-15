"""Retired camera products must not be copied; refilled lanes read fresh data."""
import sys
from types import SimpleNamespace
import torch
import pytest
from visuotactile.queued_rollout import RenderLanes

def test_retire_read_and_refill(monkeypatch):
    textures=[SimpleNamespace(enabled=True) for _ in range(2)]
    for t in textures:
        t.set_updates_enabled=lambda enabled,t=t:setattr(t,'enabled',enabled)
        t.get_updates_enabled=lambda t=t:t.enabled
    monkeypatch.setitem(sys.modules,'omni.replicator.core.scripts.utils',SimpleNamespace(
        viewport_manager=SimpleNamespace(ViewportManager=lambda:None)))
    import visuotactile.inactive_render_products as products
    monkeypatch.setattr(products,'existing_product',lambda manager,p:SimpleNamespace(hydra_texture=textures[p]))
    reads=[]
    def update(ids):
        assert all(textures[i].enabled for i in ids),'Attempt to copy a disabled product'
        reads.append(list(ids))
    camera=SimpleNamespace(render_product_paths=[0,1],_data=SimpleNamespace(output={'rgb':True}),_update_buffers_impl=update)
    counts=[0,0];ann=[]
    for i in range(2):
        def pose(i=i):
            assert textures[i].enabled
            counts[i]+=1
            return {'frame':counts[i]}
        ann.append(SimpleNamespace(get_data=pose))
    env=SimpleNamespace(num_envs=2,cfg=SimpleNamespace(student_obs=SimpleNamespace(camera_backend='standard')),
        research_cameras=[camera],_camera_pose_annotators=[ann])
    lanes=RenderLanes(env)
    camera._update_buffers_impl(torch.tensor([0,1]))
    lanes.set(1,False);camera._update_buffers_impl(torch.tensor([0,1]))
    assert ann[1].get_data()=={'frame':1} and counts[1]==1
    lanes.set(1,True);camera._update_buffers_impl(torch.tensor([1]))
    assert ann[1].get_data()=={'frame':2}
    lanes.close()
    assert reads==[[0,1],[0],[1]] and camera._update_buffers_impl is update
    assert all(t.enabled for t in textures)


@pytest.mark.parametrize('becomes_ready',[True,False])
def test_empty_reactivated_buffer_gets_bounded_render_only_warmup(becomes_ready):
    from visuotactile.queued_rollout import ready_reactivated_cameras
    renders=[];sim=SimpleNamespace(current_time_step_index=77)
    sim.render=lambda:renders.append(77)
    def get():
        return torch.ones(2,3,3) if becomes_ready and renders else torch.empty(0)
    camera=SimpleNamespace(_rep_registry={'rgb':[SimpleNamespace(get_data=get)]},
        _process_annotator_output=lambda name,value:(value,None),
        _data=SimpleNamespace(output={'rgb':torch.zeros(1,2,3,3)}),_is_outdated=torch.zeros(1,dtype=torch.bool))
    env=SimpleNamespace(sim=sim,research_cameras=[camera])
    if becomes_ready:
        result=ready_reactivated_cameras(env,[0]);assert result['extra_renders']==1
        assert camera._is_outdated[0] and sim.current_time_step_index==77
    else:
        with pytest.raises(RuntimeError,match='remained unavailable'):ready_reactivated_cameras(env,[0])
        assert len(renders)==4 and sim.current_time_step_index==77
