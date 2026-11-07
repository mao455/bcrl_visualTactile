"""Deterministic indices and radians; no simulator or new model experiment."""
import json
from types import SimpleNamespace as N

import h5py
import numpy as np
import pytest
import torch

from visuotactile.anchor_sampling import labels,anchor_batch
from visuotactile.narrow_data import record
from visuotactile.native_data import limits_affine
from visuotactile.position_control import apply_position_targets
from visuotactile.state_history import HistoryController,short_windows
from visuotactile.models.native_dp import NativePolicy


def diagnostic(n):
    pose=torch.zeros(n,7);pose[:,3]=1
    d={k:pose.clone() for k in ('base_pose_w','object_pose_w','goal_pose_w','hole_pose_w','table_pose_w','palm_body_pose_w')}
    d.update(object_velocity_w_m_s_rad_s=torch.zeros(n,6),goal_velocity_w_m_s_rad_s=torch.zeros(n,6),
        finger_position_w=torch.zeros(n,5,3),finger_quat_wxyz=pose[:,None,3:].repeat(1,5,1),
        joint_position_rad=torch.full((n,29),-.25),joint_velocity_rad_s=torch.zeros(n,29))
    return d


def target_ticks(n):
    # Distinct canonical Arm7/Hand22 columns and distinct physical ticks.
    return torch.arange(n)[:,None]*.001+torch.arange(29)[None,:]*.013-.4


def synthetic_hdf(tmp_path,n=25):
    urdf=tmp_path/'synthetic_box.urdf'
    urdf.write_text('<robot><link><collision><geometry><box size=".02 .03 .04"/></geometry></collision></link></robot>')
    metadata=dict(config=dict(assets=dict(object_urdf=str(urdf))),joint_limits_rad=[[[-2.,2.]]*29],physics_dt=1/120,control_dt=1/60,
        joint_order=[f'arm_{i}' for i in range(7)]+[f'hand_{i}' for i in range(22)])
    y=target_ticks(n);before=torch.cat((torch.zeros(1,29),y[:-1]));t=torch.arange(n,dtype=torch.float64)/60
    fields={'obs/time_s':t,'obs/step_id':torch.arange(n),'obs/actuator_target_rad':before,
        'transition/target_after_rad':y,'transition/target_before_rad':before,'transition/time_after_s':t+1/60,
        'obs/camera_time_s':t[:,None].repeat(1,3),'obs/camera_valid':torch.ones(n,3,dtype=torch.bool),
        'obs/contact_valid':torch.arange(n)>0,'transition/contact_substeps_after_w_N':torch.zeros(n,2,5,3),
        'frames/post_index':torch.cat((torch.arange(1,n),torch.tensor([-1]))),
        'transition/expert_actor_input_diagnostic':torch.zeros(n,141),'transition/actuator_substep_target_rad':y[:,None].repeat(1,2,1),
        'transition/q_after_rad':torch.full((n,29),1.75)}
    fields.update({'transition/diagnostics_before/'+k:v for k,v in diagnostic(n).items()})
    path=tmp_path/'synthetic.h5'
    with h5py.File(path,'w') as f:
        f.attrs['metadata_json']=json.dumps(metadata);g=f.create_group('episodes/synthetic');g.attrs['complete']=True
        for k,v in fields.items():g.create_dataset(k,data=v.numpy())
        # Sparse shape fixtures, never accepted as real RGB-D demonstrations.
        g.create_dataset('frames/rgb_u8',shape=(n,3,3,252,448),dtype='u1')
        g.create_dataset('frames/depth_m',shape=(n,3,1,252,448),dtype='f4')
    return path,y


def test_hdf_label_normalization_prediction_and_actuator_roundtrip(tmp_path,monkeypatch):
    path,y=synthetic_hdf(tmp_path)
    _,_,r=record(path)
    assert torch.equal(r['command'],y) and not torch.equal(r['robot'][:,:29],y)
    scale,offset=limits_affine(y)
    batch,valid=labels(r,torch.tensor([0]),24)
    normalized=batch*scale+offset
    seen=[]
    noise=N(config=N(prediction_type='sample'),add_noise=lambda target,*args:seen.append(target.clone()) or target)
    model=N(horizon=24,action_scale=scale,action_offset=offset,noise=noise,
        denoiser=lambda sample,*args,**kwargs:normalized)
    assert NativePolicy.loss(model,torch.zeros(1,512),batch,valid,torch.Generator().manual_seed(7))==0
    torch.testing.assert_close(seen[0],normalized,rtol=0,atol=0)
    class Sampler:
        def set_timesteps(self,n,device):self.timesteps=list(range(n))
        def step(self,prediction,*args,**kwargs):return N(prev_sample=prediction)
    monkeypatch.setattr('visuotactile.models.native_dp.DDIMScheduler.from_config',lambda *args,**kwargs:Sampler())
    predicted=NativePolicy.predict(model,torch.zeros(1,512),torch.Generator().manual_seed(7))
    torch.testing.assert_close(predicted,batch,atol=1e-6,rtol=0)
    permutation=torch.tensor(list(range(7,29))+list(range(7)));inverse=permutation.argsort()
    env=N(device='cpu',num_envs=1,_perm_canon_to_lab=permutation,_perm_lab_to_canon=inverse,
        robot=N(data=N(joint_pos_limits=torch.tensor([[[-2.,2.]]*29]))),_cur_targets=torch.zeros(1,29),_prev_targets=torch.zeros(1,29),
        cfg=N(domain_randomization=N(use_action_delay=False)))
    for tick in (0,7,23):
        applied,_=apply_position_targets(env,predicted[:,tick])
        torch.testing.assert_close(applied,y[tick:tick+1],atol=1e-6,rtol=0)
        torch.testing.assert_close(env._cur_targets[:,inverse],y[tick:tick+1],atol=1e-6,rtol=0)


@pytest.mark.parametrize('length',[1,11,12,13,23,24,25,63,64,65,73,97])
def test_future_labels_anchor_start_tail_and_no_history_offset(length):
    r=dict(command=target_ticks(length),time=torch.arange(length,dtype=torch.float64)/60)
    class Encoder:
        def encode(self,record,start,stop):
            assert start==(stop-1)//96*96
            return torch.tensor([[stop-1.]]),torch.tensor([stop-1]),None
    anchors=list(range(0,length,12))
    c,y,mask=anchor_batch(Encoder(),{'e':r},[('e',a) for a in anchors])
    for k,a in enumerate(anchors):
        available=min(24,length-a)
        assert c[k,0]==a and mask[k].sum()==available
        assert torch.equal(y[k,:available],r['command'][a:a+available])
        assert not mask[k,available:].any() and not y[k,available:].any()
        assert float(r['time'][a])==pytest.approx(a/60)


@pytest.mark.parametrize('anchor',[0,12,24,60,64,72])
def test_observation_short_windows_have_physical_ages_and_left_padding(anchor):
    n=anchor+1
    r=dict(robot=torch.arange(n).float()[:,None].repeat(1,178),geometry=torch.arange(n).float()[:,None].repeat(1,85),time=torch.arange(n,dtype=torch.float64)/60)
    model=N(normalized_robot=lambda x:x,geometry_mean=torch.zeros(85),geometry_std=torch.ones(85))
    robot,geo=short_windows(r,torch.tensor([anchor]),model)
    for values,offsets,width in [(robot.reshape(13,180),range(-12,1),178),(geo.reshape(3,87),[-24,-12,0],85)]:
        for value,offset in zip(values,offsets):
            if anchor+offset<0:assert not value.any()
            else:
                assert (value[:width]==anchor+offset).all() and value[-1]==1
                assert value[-2].item()==pytest.approx(-offset/60,abs=1e-7)


def test_real_online_action_slicing_first_commit_replan_and_episode_reset():
    class Bank:
        def __init__(self):self.rows=[]
        def observe(self,lane,episode,tick,model,actor,robot,geo,stamp):
            self.rows.append((episode,tick,robot.clone(),float(stamp)))
            return torch.tensor([[tick]]) if tick%12==0 else None
    plans=[]
    def predict(condition,*args,**kwargs):
        anchor=int(condition[0,0]);plans.append(anchor)
        return target_ticks(anchor+24)[anchor:][None]
    c=object.__new__(HistoryController)
    c.episode=None;c.position_mode=True;c.device='cpu';c.input_builder='canonical_joint_state_v1'
    c.limits=torch.tensor([[-2.,2.]]*29);c.dimensions=torch.tensor([.02,.03,.04]);c.bank=Bank()
    c.model=N(predict=predict);c.inference={};c.overlap_steps=0;c.generator=torch.Generator().manual_seed(7)
    for episode,length in [(9,75),(10,14)]:
        for tick in range(length):
            packet=dict(episode_id=torch.tensor([episode]),step_id=torch.tensor([tick]),time_s=torch.tensor([tick/60],dtype=torch.float64),
                actuator_target_rad=torch.zeros(1,29) if tick==0 else target_ticks(tick)[-1:])
            actual=c.act(torch.zeros(1,141),packet,diagnostic(1))
            assert torch.equal(actual,target_ticks(tick+1)[-1:])
            c.observe_execution(dict(position_applied_rad=actual.clone()))
            assert c.plan_tick==tick//12*12
            if tick:assert torch.equal(c.last_inputs['robot'][116:145],target_ticks(tick)[-1])
            else:assert not c.last_inputs['robot'][116:177].any()
    assert plans==[0,12,24,36,48,60,72,0,12]
    assert [r[1] for r in c.bank.rows[:75]]==list(range(75))
