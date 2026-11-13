"""Residual I/O consolidation must not alter causal recurrence or labels."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from visuotactile import native_data


@pytest.mark.parametrize('stride',[1,12])
@pytest.mark.parametrize('length,modality,supervision', [(25,'G00','u_expert'),(47,'G10','q_reached')])
def test_episode_transfer_preserves_all_outputs_and_causal_calls(tmp_path,monkeypatch,length,modality,supervision,stride):
    class Base(torch.nn.Module):
        group='I-HistoryDP'
        def __init__(self):
            super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.1),requires_grad=False)
            self.modality=modality;self.contract={'supervision':supervision,'visual_stride':stride};self.calls=[];self.eval()
        def observe(self,robot,stamp,tick,frame,memory,*,tactile):
            assert (memory is None)==(tick==0)
            assert (frame is not None)==(tick%stride==0)
            assert (tactile is not None)==(modality=='G10')
            total=(0 if memory is None else memory.total)+robot[0]*self.weight
            self.calls.append(('observe',tick,float(total),float(robot[116])))
            return SimpleNamespace(total=total)
        def condition(self,memory,stamp):return memory.total
        def predict(self,condition,generator):
            self.calls.append(('predict',float(condition)))
            return torch.randn(1,24,29,generator=generator)*.01+condition
    monkeypatch.setattr(native_data,'Episode',lambda *a:SimpleNamespace(frame=lambda tick,device:dict(tick=tick)))
    monkeypatch.setattr(native_data,'residual_observations',lambda e,r,t:dict(robot=r.clone()))
    monkeypatch.setattr(native_data,'residual_visual',lambda m,h,t:torch.arange(512)[None].float()+h.total)
    row=dict(source_root=str(tmp_path),identity='recovery',robot=torch.arange(length*178).reshape(length,178).float()*.0001,
        time=torch.arange(length)/60,command=torch.ones(length,29)*.4,expert_command=torch.ones(length,29)*.6,
        joint_limits=torch.tensor([[-2.,2.]]*29),clean_sample=torch.ones(length,29)*.3,
        supervision_mask=torch.arange(length)>=12,control_segment=(torch.arange(length)>=12).long())
    if modality=='G10':row['tactile']=torch.zeros(length,2,32)
    before=row['robot'].clone();one_model=Base();two_model=Base();lease=SimpleNamespace(block=nullcontext)
    one=native_data.cache_residual_episode(one_model,row,{},lease,transfer_mode='tick')
    two=native_data.cache_residual_episode(two_model,row,{},lease,transfer_mode='episode')
    assert one.keys()==two.keys()
    for key in one:
        if isinstance(one[key],torch.Tensor):assert torch.equal(one[key],two[key]),key
        else:assert one[key]==two[key],key
    assert one_model.calls==two_model.calls
    assert torch.equal(row['robot'],before)
    assert torch.equal(two['target'],row['expert_command'])
    assert torch.equal(two['previous_nominal'][1:],two['base'][:-1])
    assert torch.equal(two['plan_valid'],torch.arange(24)[None]+two['plan_cursor'][:,None]<24)
    assert torch.equal(two['base'],two['plans'][:,0])
    if supervision=='q_reached':assert torch.equal(two['q_reached'],row['clean_sample'])
    with pytest.raises(ValueError,match='transfer implementation'):
        native_data.cache_residual_episode(Base(),row,{},lease,transfer_mode='unsupported')


@pytest.mark.parametrize('steps,clip',[(16,False),(16,True),(32,False)])
def test_host_scheduler_indices_preserve_sampler_and_denoiser_timesteps(steps,clip):
    from diffusers import DDPMScheduler
    from visuotactile.models.native_dp import NativePolicy
    calls=[]
    def denoise(sample,timestep,global_cond):
        assert isinstance(timestep,torch.Tensor)
        calls.append((int(timestep),timestep.device))
        return sample*.01+global_cond[:,:1,None]*.1
    stub=SimpleNamespace(horizon=24,denoiser=denoise,
        noise=DDPMScheduler(num_train_timesteps=100,beta_schedule='squaredcos_cap_v2',clip_sample=False,prediction_type='sample'),
        action_scale=torch.ones(29),action_offset=torch.zeros(29))
    condition=torch.ones(1,512)
    original=NativePolicy.predict(stub,condition,torch.Generator().manual_seed(7091),inference_steps=steps,clip_sample=clip)
    original_calls=list(calls);calls.clear()
    optimized=NativePolicy.predict(stub,condition,torch.Generator().manual_seed(7091),inference_steps=steps,clip_sample=clip,scheduler_index_mode='host')
    assert torch.equal(original,optimized)
    assert calls==original_calls and len(calls)==steps
