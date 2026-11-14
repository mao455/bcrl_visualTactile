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


@pytest.mark.parametrize('microbatch',[2,8])
def test_batched_visuals_preserve_active_lane_and_episode_routing(microbatch):
    from types import SimpleNamespace
    class Dino:
        def forward_lanes(self,rgb,**kwargs):
            assert kwargs==({} if microbatch==2 else dict(vision_microbatch=microbatch))
            return rgb[:,0,0,0,0,None]
    class Lane:
        def __init__(self):self.base=SimpleNamespace(dino=CurrentFrame());self.history=[]
        def act(self,p,d):
            self.history.append((int(p['episode_id'][0]),int(p['step_id'][0]),float(self.base.dino(None)[0])))
            return torch.full((1,29),float(p['episode_id'][0]))
        def observe_execution(self,t):self.ack=t['position_applied_rad'].clone()
    c=object.__new__(BatchedResidualController)
    c.count=3;c.device='cpu';c.dino=Dino();c.lanes=[Lane() for _ in range(3)];c.visual_frames=[0,0,0]
    c.vision_microbatch=microbatch
    p={'image':torch.arange(3.)[:,None,None,None,None].expand(3,3,5,2,2),
       'episode_id':torch.tensor([2,4,8]),'step_id':torch.tensor([0,23,7]),'actuator_target_rad':torch.ones(3,29)}
    out=c.act(p,{},[True,False,True]);c.observe_execution({'position_applied_rad':out},[True,False,True])
    assert c.lanes[0].history==[(2,0,0.)] and c.lanes[1].history==[] and c.lanes[2].history==[(8,7,2.)]
    assert c.visual_frames==[1,0,1] and torch.equal(out[1],p['actuator_target_rad'][1])
    assert torch.equal(c.lanes[2].ack,out[2:3])
    # One call with two real lanes; all-inactive steps must do no visual work.
    c.act(p,{},[False]*3)
    report=c.vision_execution_report()
    assert report['controller_calls']==1 and report['real_lanes']==2
    assert report['consumed_observations_per_lane']==[1,0,1]
    assert report['padding_lanes']==microbatch-2 and report['processed_images']==3*microbatch
    c.visual_frames[0]+=1
    with pytest.raises(ValueError,match='consumed'):c.vision_execution_report()


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


def test_backend_batch_keeps_reset_lane_and_continuing_lane_independent():
    from test_target_residual import observations, statistics, segment
    from visuotactile.models.native_dp import TargetResidual
    from visuotactile.batched_controller import stack_lane_trees
    torch.manual_seed(981)
    model=TargetResidual(statistics(),[.1]*29).eval()
    with torch.no_grad():model.policy[-1].weight.normal_(0.,.02)
    source=observations(steps=17,batch=3)
    serial=[None]*3;batched=[None]*3
    with torch.no_grad():
        for tick in range(17):
            active=[0,1,2] if tick<15 else [0,2]
            rows=[]
            for lane in active:
                local=tick if lane!=0 or tick<13 else tick-13
                if lane==0 and tick==13:serial[lane]=None;batched[lane]=None
                rows.append(lane_tree(segment(source,local,local+1),lane,3))
            raw,state=model(stack_lane_trees(rows),stack_lane_trees([batched[l] for l in active]))
            for i,lane in enumerate(active):
                expected,serial[lane]=model(rows[i],serial[lane])
                batched[lane]=lane_tree(state,i,len(active))
                torch.testing.assert_close(raw[i:i+1],expected,atol=2e-6,rtol=2e-5)
                torch.testing.assert_close(batched[lane]['history'],serial[lane]['history'],atol=2e-6,rtol=2e-5)
                assert torch.equal(batched[lane]['history_valid'],serial[lane]['history_valid'])
    assert batched[0]['history_valid'].sum()==4
    assert batched[1]['history_valid'].sum()==15
    assert batched[2]['history_valid'].sum()==17


def test_batched_sampler_preserves_each_lane_random_stream():
    from types import SimpleNamespace
    from diffusers import DDPMScheduler
    from visuotactile.models.native_dp import NativePolicy
    stub=SimpleNamespace(horizon=24,denoiser=lambda sample,timestep,global_cond: sample*.01+global_cond[:,:1,None]*.1,
        noise=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='sample'),
        action_scale=torch.ones(29),action_offset=torch.zeros(29))
    batch_generators=[torch.Generator().manual_seed(7091+i) for i in range(3)]
    serial_generators=[torch.Generator().manual_seed(7091+i) for i in range(3)]
    for lanes in ([0,1,2],[1,2],[0,2]):
        conditions=torch.tensor(lanes)[:,None].float().expand(-1,512)
        result=NativePolicy.predict(stub,conditions,[batch_generators[i] for i in lanes],scheduler_index_mode='host')
        for row,lane in enumerate(lanes):
            expected=NativePolicy.predict(stub,conditions[row:row+1],serial_generators[lane],scheduler_index_mode='host')
            assert torch.equal(result[row:row+1],expected)
            assert torch.equal(batch_generators[lane].get_state(),serial_generators[lane].get_state())
    with pytest.raises(ValueError,match='own generator'):
        NativePolicy.predict(stub,conditions,[batch_generators[0]])
