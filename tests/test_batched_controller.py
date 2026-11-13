import torch
import pytest
from visuotactile.batched_controller import lane_tree, CurrentFrame, BatchedResidualController


def test_recursive_acknowledgement_keeps_lane_dimension():
    x={'a':torch.arange(6).reshape(2,3),'nested':{'b':torch.arange(2)},'scalar':torch.tensor(1)}
    y=lane_tree(x,1,2)
    assert y['a'].tolist()==[[3,4,5]] and y['nested']['b'].tolist()==[1]
    assert x['a'].shape==(2,3)


def test_consumed_frame_cannot_be_reused():
    dino=CurrentFrame();frame=torch.ones(3,2);dino.set(frame)
    assert dino(None) is frame
    with pytest.raises(ValueError):dino(None)
    dino.set(frame+1);assert torch.equal(dino(None),frame+1)


def test_batched_visuals_preserve_active_lane_and_episode_routing():
    from types import SimpleNamespace
    class Dino:
        def forward_lanes(self,rgb):return rgb[:,0,0,0,0,None]
    class Lane:
        def __init__(self):self.base=SimpleNamespace(dino=CurrentFrame());self.history=[]
        def act(self,p,d):
            self.history.append((int(p['episode_id'][0]),int(p['step_id'][0]),float(self.base.dino(None)[0])))
            return torch.full((1,29),float(p['episode_id'][0]))
        def observe_execution(self,t):self.ack=t['position_applied_rad'].clone()
    c=object.__new__(BatchedResidualController)
    c.count=3;c.device='cpu';c.dino=Dino();c.lanes=[Lane() for _ in range(3)];c.visual_frames=[0,0,0]
    p={'image':torch.arange(3.)[:,None,None,None,None].expand(3,3,5,2,2),
       'episode_id':torch.tensor([2,4,8]),'step_id':torch.tensor([0,23,7]),'actuator_target_rad':torch.ones(3,29)}
    out=c.act(p,{},[True,False,True]);c.observe_execution({'position_applied_rad':out},[True,False,True])
    assert c.lanes[0].history==[(2,0,0.)] and c.lanes[1].history==[] and c.lanes[2].history==[(8,7,2.)]
    assert c.visual_frames==[1,0,1] and torch.equal(out[1],p['actuator_target_rad'][1])
    assert torch.equal(c.lanes[2].ack,out[2:3])


def test_lane_reset_calls_native_with_local_config_and_restores_shared_fields():
    from types import SimpleNamespace
    from visuotactile.narrow_data import configure_direct_reset_lanes
    calls=[]
    env=SimpleNamespace(num_envs=2,device='cpu',cfg=SimpleNamespace(
        reset=SimpleNamespace(table_reset_z=1.),precise_assembly=SimpleNamespace(hole_x_range=[0.,0.])))
    def native(ids):calls.append((ids.tolist(),env.cfg.reset.table_reset_z,env.cfg.precise_assembly.hole_x_range))
    env._reset_idx=native
    configs=[dict(table_reset_z=2.,hole_x_range=[3.,3.]),dict(table_reset_z=4.,hole_x_range=[5.,5.])]
    configure_direct_reset_lanes(env,configs);configs[0]['table_reset_z']=99
    env._reset_idx(torch.tensor([1,0]))
    assert calls==[([1],4.,[5.,5.]),([0],2.,[3.,3.])]
    assert env.cfg.reset.table_reset_z==1. and env.cfg.precise_assembly.hole_x_range==[0.,0.]
    with pytest.raises(ValueError):configure_direct_reset_lanes(env,configs)
