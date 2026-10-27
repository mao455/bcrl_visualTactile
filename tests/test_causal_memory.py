"""Fixed-parameter causal tests; fixtures never used for training."""
from dataclasses import asdict
import copy
import importlib.util
from pathlib import Path
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_perception_v2 import packet
from visuotactile.history import detach_state, reset_state
from visuotactile.models.nominal import NominalDiffusionPolicy, PolicyConfig
from visuotactile.models.tactile import FingerTactileEncoder
from visuotactile.models.correction import CorrectionPolicy
from visuotactile.rollout import NominalController
from visuotactile.correction import CorrectionContext, CorrectionShadow, latest_correction_inputs
from visuotactile.dataset import EpisodeSequenceDataset, encode_episode_chunks
from visuotactile.recording import EpisodeWriter, EpisodeReader, flatten, post_observation
from visuotactile.observations import nominal_keys


def sequence(n=2,length=7):
    rows=[packet(n,t) for t in range(length)]
    obs={k:torch.stack([p[k] for p in rows],1) for k in nominal_keys(2)}
    obs['valid']=torch.ones(n,length,dtype=torch.bool)
    return obs


def test_episode_sequence_chunk_online_and_detach():
    torch.manual_seed(7)
    encoder=NominalDiffusionPolicy(PolicyConfig()).encoder.eval()
    obs=sequence(length=13)
    full,hidden=encoder(obs,return_sequence=True,return_state=True)
    state=None;parts=[]
    for lo,hi in ((0,2),(2,5),(5,13)):
        output,state=encoder({k:v[:,lo:hi] for k,v in obs.items()},state,return_sequence=True,return_state=True)
        parts.append(output);state=detach_state(state)
    torch.testing.assert_close(full,torch.cat(parts,1),atol=2e-6,rtol=1e-5)
    state=None;steps=[]
    for t in range(13):
        state=encoder({k:v[:,t:t+1] for k,v in obs.items()},state)
        steps.append(state)
    torch.testing.assert_close(full,torch.stack(steps,1),atol=2e-6,rtol=1e-5)
    # At t12 the old four-frame/stride3 window began at t3: t0 was gone.
    changed=copy.deepcopy(obs);changed['proprio'][:,0]=4
    altered=encoder(changed)
    assert (altered-hidden).abs().max()>1e-6
    zero_prefix=encoder({k:v[:,5:] for k,v in obs.items()})
    assert not torch.allclose(zero_prefix,hidden,atol=1e-5)
    assert torch.equal(detach_state(hidden),hidden) and not detach_state(hidden).requires_grad
    # Gradients to a remote observation exist, but no claim of lossless memory.
    obs['proprio']=obs['proprio'].requires_grad_()
    encoder(obs).square().sum().backward()
    assert obs['proprio'].grad[:,0].abs().sum()>0


def test_controller_updates_between_replans_and_only_episode_reset():
    torch.manual_seed(6)
    cfg=PolicyConfig(history=2,stride=1,inference_steps=1)
    controller=NominalController(NominalDiffusionPolicy(cfg),2)
    for t in range(9):
        p=packet(t=t)
        p['success_stage_diagnostic']=torch.tensor([t//2,t//2])
        if t>=7:
            p['episode_id'][0]=1;p['step_id'][0]=t-7;p['contact_valid'][0]=t>7
        controller.act(p)
        assert controller.memory_updates.tolist()==([t+1,t+1] if t<7 else [t-6,t+1])
        if t==5:
            assert controller.plan_generation_step.tolist()==[0,0]
        if t==6:
            assert controller.plan_generation_step.tolist()==[6,6]
        if t==7:
            current={k:controller.last_packet[k][0:1,None] for k in nominal_keys(2)}
            current['valid']=torch.ones(1,1,dtype=torch.bool)
            torch.testing.assert_close(controller.memory[:1],controller.model.encoder(current),atol=2e-6,rtol=1e-5)
    assert len(controller.memory_ms)==9 and len(controller.inference_ms)==3
    restarted=NominalController(NominalDiffusionPolicy(cfg),2)
    with pytest.raises(ValueError,match='prefix'):
        restarted.act(packet(t=3))


def test_tactile_chunk_boundaries_preserve_derivatives_and_events():
    torch.manual_seed(4)
    encoder=FingerTactileEncoder()
    p=packet(t=1)
    tactile={name:torch.cat([p[src]]*3,1) for src,name in (
        ('contact_substeps_w_N','force_w_N'),('contact_substeps_local_N','force_local_N'),
        ('finger_position_base_m','position_base_m'),('finger_rotation_base','rotation_base'))}
    tactile['time_s']=torch.arange(6,dtype=torch.float64)[None].expand(2,-1)/120
    tactile['force_w_N'][:,2:4]=0;tactile['force_local_N'][:,2:4]=0
    tactile['position_base_m']+=torch.arange(6)[None,:,None,None]*.001
    valid=torch.ones(2,6,dtype=torch.bool)
    full,full_state=encoder(tactile,valid,return_state=True)
    state=None
    for lo,hi in ((0,2),(2,3),(3,6)):
        out,state=encoder({k:v[:,lo:hi] for k,v in tactile.items()},valid[:,lo:hi],state,return_state=True)
    torch.testing.assert_close(full,out,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(full_state['hidden'],state['hidden'],rtol=1e-5,atol=1e-6)
    reset=reset_state(state,torch.tensor([True,False]))
    assert not reset['hidden'][0].any() and torch.equal(reset['hidden'][1],state['hidden'][1])
    assert not reset['seen'][0] and reset['seen'][1]


def test_shadow_state_matches_direct_step_carry_and_reset():
    cfg=PolicyConfig(history=2,stride=1,inference_steps=1)
    nominal=NominalController(NominalDiffusionPolicy(cfg),2)
    shadow=CorrectionShadow(nominal,2)
    expected_state=None
    with torch.no_grad():
        for t in range(8):
            p=packet(t=t)
            if t>=6:
                p['episode_id'][0]=1;p['step_id'][0]=t-6;p['contact_valid'][0]=t>6
            nominal.act(p)
            log=shadow.observe(p,nominal)
            if t==6: expected_state=reset_state(expected_state,torch.tensor([True,False]))
            inputs=latest_correction_inputs(shadow.last_inputs)
            expected,expected_state=shadow.model(**inputs,state=expected_state,return_state=True)
            torch.testing.assert_close(log['untrained_candidate_diagnostic'],expected,rtol=0,atol=0)
            if t>0 and t!=6:
                torch.testing.assert_close(inputs['motion_history']['previous_command'][:,-1],torch.full((2,29),.25))
            shadow.context.commit(torch.full((2,29),.25))


def test_masked_depth_and_invalid_camera_do_not_influence_or_receive_gradient():
    obs=sequence(n=1,length=1)
    obs['depth_valid'][:,:,:, :,0:8]=False
    obs['camera_valid'][:,:,1]=False
    encoder=NominalDiffusionPolicy().encoder
    ref=encoder(obs)
    altered=copy.deepcopy(obs)
    altered['image'][:,:,0,3,0:8]=987
    altered['image'][:,:,1]=333
    torch.testing.assert_close(ref,encoder(altered),rtol=0,atol=0)
    obs['image'].requires_grad_();encoder(obs).square().sum().backward()
    assert not obs['image'].grad[:,:,1].any()
    assert not obs['image'].grad[:,:,0,3,0:8].any()
    assert obs['image'].grad[:,:,0,:3].abs().sum()>0
    assert obs['image'].grad[:,:,0,3,8:].abs().sum()>0


def test_action_delay_trace_rng_and_formulas():
    path=Path(__file__).parents[1]/'isaacsimenvs/tasks/play/utils/action_utils.py'
    spec=importlib.util.spec_from_file_location('action_trace_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    env=SimpleNamespace(device='cpu',num_envs=2,step_dt=1/60,
        cfg=SimpleNamespace(domain_randomization=SimpleNamespace(use_action_delay=True,action_delay_max=3),
            action=SimpleNamespace(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
        _perm_canon_to_lab=torch.arange(29).flip(0),_prev_targets=torch.zeros(2,29),_cur_targets=torch.zeros(2,29),
        _arm_lower=torch.full((2,7),-2.),_arm_upper=torch.full((2,7),2.),
        _hand_lower=torch.zeros(2,22),_hand_upper=torch.full((2,22),2.),
        _arm_joint_ids=list(range(7)),_hand_joint_ids=list(range(7,29)),
        episode_length_buf=torch.zeros(2,dtype=torch.long),_successes=torch.zeros(2),_action_queue=torch.zeros(2,3,29))
    reference=copy.deepcopy(env);trace=[]
    env._record_action_trace=lambda a,idx,init:trace.append((a.clone(),idx.clone(),init.clone()))
    for t in range(15):
        command=torch.sin(torch.arange(58).reshape(2,29).float()+t)
        old=env._prev_targets.clone();rng=torch.get_rng_state()
        module.apply_action_pipeline(env,command);after=torch.get_rng_state()
        torch.set_rng_state(rng);module.apply_action_pipeline(reference,command)
        assert torch.equal(after,torch.get_rng_state())
        torch.testing.assert_close(env._cur_targets,reference._cur_targets,rtol=0,atol=0)
        delayed,idx,init=trace[-1]
        torch.testing.assert_close(delayed,env._action_queue[torch.arange(2),idx],rtol=0,atol=0)
        expected_arm=.1*(old[:,:7]+1.5/60*delayed[:,:7]).clamp(-2,2)+.9*old[:,:7]
        expected_hand=.1*(delayed[:,7:]+1)+.9*old[:,7:]
        torch.testing.assert_close(env._cur_targets,torch.cat((expected_arm,expected_hand),1))
        env.episode_length_buf+=1;reference.episode_length_buf+=1


def fixture_file(path):
    writer=EpisodeWriter(path,dict(perception_version=2,trace_version=2,camera_ids=['global','local']))
    for t in range(4):
        p=packet(n=1,t=t)
        action=torch.full((1,29),t/10)
        tr=dict(episode_id=p['episode_id'],step_id=p['step_id'],executed_action=action,
                command_clipped_pre_delay=action,
                diagnostics_after=dict(successes=torch.ones(1),max_goals=torch.ones(1),retract_succeeded=torch.ones(1,dtype=torch.bool)),
                observation_after=post_observation(packet(n=1,t=t+1),torch.tensor([t==3])))
        writer.append(p,tr,torch.tensor([t==3]),torch.tensor([False]),'released_rl_checkpoint')
    writer.close('released_rl_checkpoint')


def test_hdf5_step_pickle_and_sequential_training_roundtrip(tmp_path):
    path=tmp_path/'trace.h5';fixture_file(path)
    reader=EpisodeReader(path);ep=reader.episodes[0]
    assert reader.trace_version==2 and reader.attributes(ep)['complete']
    arrays=reader.read(ep)
    for t,row in enumerate(reader.steps(ep)):
        for key,value in flatten(row).items():
            np.testing.assert_array_equal(value,flatten(arrays)[key][t])
    final=reader.read(ep,3)['transition']['observation_after']
    assert not final['available'] and not final['image'].any()
    out=reader.export_pickle(ep,tmp_path/'episode.pkl')
    with out.open('rb') as f: restored=pickle.load(f)  # trusted fixture generated here
    for key,value in flatten(arrays).items():
        other=flatten(restored['arrays'])[key]
        assert value.dtype==other.dtype and value.shape==other.shape
        np.testing.assert_array_equal(value,other)
    dataset=EpisodeSequenceDataset(path,horizon=12)
    chunks=list(dataset.chunks(ep,2))
    assert not chunks[0]['expert_history']['valid'][0]
    torch.testing.assert_close(chunks[1]['expert_history']['previous_issued_command'][0],torch.full((29,),.1))
    encoder=NominalDiffusionPolicy().encoder
    one=torch.cat([c for c,_,_ in encode_episode_chunks(encoder,dataset.chunks(ep,4))],1)
    multiple=torch.cat([c for c,_,_ in encode_episode_chunks(encoder,dataset.chunks(ep,2),detach_between_chunks=True)],1)
    torch.testing.assert_close(one,multiple,atol=2e-6,rtol=1e-5)
    with pytest.raises(ValueError,match='prefix'):
        list(encode_episode_chunks(encoder,chunks[1:]))
    loss=NominalDiffusionPolicy().loss_from_condition(multiple[0],torch.cat([c['action_target'] for c in chunks]),torch.cat([c['action_valid'] for c in chunks]))
    assert torch.isfinite(loss)
    loss.backward()  # numeric/gradient check only; no optimizer


def test_legacy_memory_contract_and_checkpoint_state(tmp_path):
    assert PolicyConfig.from_dict(dict(perception_version=2)).memory_version==1
    model=NominalDiffusionPolicy(PolicyConfig(inference_steps=1))
    obs=sequence(n=1,length=3)
    with pytest.raises(ValueError,match='ordered'):
        model.loss(obs,torch.zeros(1,12,29),torch.ones(1,12,dtype=torch.bool))
    path=tmp_path/'weights.pt'
    torch.save(dict(config=asdict(model.cfg),model=model.state_dict()),path)
    saved=torch.load(path,weights_only=False)
    other=NominalDiffusionPolicy(PolicyConfig.from_dict(saved['config']));other.load_state_dict(saved['model'])
    torch.testing.assert_close(model.encoder(obs),other.encoder(obs),rtol=0,atol=0)
