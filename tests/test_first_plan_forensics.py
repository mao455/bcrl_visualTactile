from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from visuotactile.first_plan_forensics import FirstPlanRecorder, tensor_snapshot
from visuotactile.native_controller import NativeController
from visuotactile.position_control import continuous_image_contract


class MockPolicy:
    group='I-HistoryDP';modality='G00'
    def __init__(self):
        self.contract=continuous_image_contract('u_expert');self.observed=[];self.predicted=0
    def observe(self,robot,time,tick,frame,memory,tactile=None):
        normalized=robot*2+.1
        visual=frame['patches'].flatten()[:512][None]+frame['depth'].float().mean()
        self.observed.append((robot.clone(),time.clone(),tick))
        return SimpleNamespace(robot=[(normalized,time)],frames=[(frame['patches'],frame['time'],frame['valid'])],
            visual=visual,native=normalized[None])
    def condition(self,memory,time):
        return memory.visual+memory.robot[-1][0].mean()
    def predict(self,condition,generator,**kwargs):
        self.predicted+=1
        return torch.randn((1,24,29),generator=generator)*.01+condition.mean()


class MockDino:
    def __call__(self,rgb):
        return rgb[:,0,:18,:32,None].expand(-1,-1,-1,2).permute(0,3,1,2).contiguous()


def packet(episode,tick):
    time=torch.tensor([episode*20+tick/60],dtype=torch.float64)
    image=torch.arange(3*4*252*448,dtype=torch.float32).reshape(1,3,4,252,448)%255/255
    return dict(episode_id=torch.tensor([episode]),step_id=torch.tensor([tick]),time_s=time,
        actuator_target_rad=torch.linspace(-.1,.1,29)[None],task_id=torch.zeros(1,dtype=torch.int64),
        goal_valid=torch.zeros(1,dtype=torch.bool),camera_time_s=time[:,None].expand(1,3).clone(),
        camera_valid=torch.ones(1,3,dtype=torch.bool),camera_frame=torch.full((1,3),tick+1,dtype=torch.int64),
        image=image,depth_valid=torch.ones(1,3,1,252,448,dtype=torch.bool))


def test_enabled_controller_matches_disabled_output_inputs_clocks_and_rng(tmp_path,monkeypatch):
    import visuotactile.native_controller as module
    monkeypatch.setattr(module,'load_policy',lambda saved,device:MockPolicy())
    meta={'joint_limits_rad':[[[-2.,2.]]*29]};saved={'updates':1}
    recorder=FirstPlanRecorder(tmp_path/'probe',binding={'model_sha256':'test_identity'})
    baseline=NativeController(saved,meta,dino=MockDino(),device='cpu')
    recorded=NativeController(saved,meta,dino=MockDino(),device='cpu',first_plan_recorder=recorder)
    diagnostics=dict(joint_position_rad=torch.linspace(-.05,.05,29)[None],joint_velocity_rad_s=torch.zeros(1,29))
    global_before=torch.random.get_rng_state().clone()
    generator_before=recorded.generator.get_state().clone()
    first_condition=None
    for episode,ticks in [(0,range(13)),(1,range(2))]:
        for tick in ticks:
            p=packet(episode,tick);original={k:v.clone() for k,v in p.items()}
            expected=baseline.act(p,diagnostics);actual=recorded.act(p,diagnostics)
            if episode==0 and tick==0:
                first_condition=recorded.last_condition.detach().clone()
            assert torch.equal(actual,expected)
            assert torch.equal(recorded.generator.get_state(),baseline.generator.get_state())
            assert all(torch.equal(p[k],v) for k,v in original.items())
            baseline.observe_execution({'position_applied_rad':expected})
            recorded.observe_execution({'position_applied_rad':actual})
    assert torch.equal(torch.random.get_rng_state(),global_before)
    assert baseline.model.predicted==recorded.model.predicted==3
    assert len(baseline.model.observed)==len(recorded.model.observed)==15
    assert recorder.summary()['episodes']==2
    assert len(list((tmp_path/'probe').glob('*.json')))==2
    assert len(list((tmp_path/'probe').glob('*.npz')))==2
    receipt=json.loads(next((tmp_path/'probe').glob('*00000000*.json')).read_text())
    assert receipt['sampler_seed']==7091
    for key,shape in [('rgb_input',[3,3,252,448]),('depth_input',[3,1,252,448]),('robot178',[178]),
            ('dp_condition512',[1,512]),('plan',[1,24,29])]:
        assert receipt['tensors'][key]['shape']==shape
    assert 'npz_key' not in receipt['tensors']['rgb_input']
    with np.load(tmp_path/'probe'/receipt['small_tensors']['path']) as z:
        assert np.array_equal(z['generator_state_before'],generator_before.numpy())
        assert np.array_equal(z['robot178'],baseline.model.observed[0][0].view(torch.uint8).numpy())
        assert np.array_equal(z['dp_condition512'],tensor_snapshot(first_condition)[0].view(torch.uint8).reshape(-1).numpy())
    assert receipt['tensors']['observation_time_s']['shape']==[]


def test_tensor_hash_binds_dtype_shape_and_exact_bytes_without_rng():
    state=torch.random.get_rng_state().clone();x=torch.arange(8,dtype=torch.float32).reshape(2,4)
    clone,first=tensor_snapshot(x);assert torch.equal(clone,x)
    x[0,0]+=1
    assert not torch.equal(clone,x)
    assert first['sha256']!=tensor_snapshot(x)[1]['sha256']
    assert first['sha256']!=tensor_snapshot(clone.reshape(4,2))[1]['sha256']
    assert first['sha256']!=tensor_snapshot(clone.double())[1]['sha256']
    assert tensor_snapshot(torch.tensor(7,dtype=torch.int64))[1]['bytes']==8
    assert tensor_snapshot(torch.ones(2,dtype=torch.bfloat16))[1]['dtype']=='torch.bfloat16'
    assert torch.equal(state,torch.random.get_rng_state())


def test_preserve_identity_pending_and_existing_files(tmp_path):
    recorder=FirstPlanRecorder(tmp_path);g=torch.Generator().manual_seed(7091)
    with pytest.raises(ValueError,match='first episode tick'):
        recorder.before_sampling(3,12,{'value':torch.ones(1)},g)
    recorder.before_sampling(3,0,{'value':torch.ones(1)},g)
    with pytest.raises(ValueError,match='duplicate'):
        recorder.before_sampling(3,0,{'value':torch.ones(1)},g)
    with pytest.raises(ValueError,match='pending'):
        recorder.after_sampling(4,torch.ones(1,24,29),g)
    recorder.after_sampling(3,torch.ones(1,24,29),g)
    original=next(tmp_path.glob('*.json')).read_bytes()
    duplicate=FirstPlanRecorder(tmp_path);duplicate.before_sampling(3,0,{'value':torch.ones(1)},g)
    with pytest.raises(FileExistsError,match='Preserve'):
        duplicate.after_sampling(3,torch.ones(1,24,29),g)
    assert next(tmp_path.glob('*.json')).read_bytes()==original


def test_unsupported_controller_contract_rejects_opt_in(monkeypatch,tmp_path):
    import visuotactile.native_controller as module
    policy=MockPolicy();policy.contract['visual_stride']=12
    monkeypatch.setattr(module,'load_policy',lambda saved,device:policy)
    with pytest.raises(ValueError,match='every-frame G00'):
        NativeController({'updates':1},{},device='cpu',first_plan_recorder=FirstPlanRecorder(tmp_path))


def test_rollout_prepare_registers_opt_in_only_for_new_plan(tmp_path,monkeypatch):
    import tools.tight_fresh_rollouts as module
    import visuotactile.narrow_data as data
    root=tmp_path/'source';root.mkdir();(root/'session.json').write_text('{"domain":"mock"}')
    monkeypatch.setattr(module,'completed_pair',lambda root,role:{'role':role})
    monkeypatch.setattr(data,'direct_configuration',lambda domain,seed:{'domain':domain,'seed':seed})
    default=tmp_path/'default';enabled=tmp_path/'enabled'
    module.prepare(root,default,1,4)
    module.prepare(root,enabled,1,4,first_plan_forensics=True)
    ordinary=json.loads((default/'plan.json').read_text());recorded=json.loads((enabled/'plan.json').read_text())
    assert ordinary['first_plan_forensics'] is False
    assert recorded['first_plan_forensics'] is True
    assert ordinary['cases']==recorded['cases']
    assert ordinary['pairs']==recorded['pairs']
    with pytest.raises(FileExistsError):
        module.prepare(root,enabled,1,4,first_plan_forensics=True)
    with pytest.raises(ValueError,match='boolean'):
        module.prepare(root,tmp_path/'bad',1,4,first_plan_forensics='true')
