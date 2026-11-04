import copy
import json
from types import SimpleNamespace as N

import pytest
import torch

from visuotactile.position_control import bounded_targets,apply_position_targets,position_robot_history,contract
from visuotactile.anchor_sampling import labels
from visuotactile.state_history import PersistentState


def test_position_mapping_no_native_transform_and_unused_rng_draw():
    perm=torch.randperm(29,generator=torch.Generator().manual_seed(31));inverse=perm.argsort()
    limits=torch.stack((torch.full((29,),-3.),torch.full((29,),3.)),1)[None]
    env=N(device='cpu',num_envs=1,_perm_canon_to_lab=perm,_perm_lab_to_canon=inverse,
        robot=N(data=N(joint_pos_limits=limits)),_cur_targets=torch.zeros(1,29),_prev_targets=torch.ones(1,29),
        cfg=N(domain_randomization=N(use_action_delay=True,action_delay_max=3)),_action_queue=torch.randn(1,3,29))
    queued=env._action_queue.clone();target=torch.linspace(-4,4,29)[None]
    torch.manual_seed(93);expected_draw=torch.randint(0,3,(1,));expected_rng=torch.get_rng_state()
    torch.manual_seed(93);applied,draw=apply_position_targets(env,target)
    assert torch.equal(draw,expected_draw) and torch.equal(torch.get_rng_state(),expected_rng)
    assert torch.equal(env._cur_targets[:,inverse],target.clamp(-3,3))
    assert torch.equal(env._prev_targets,env._cur_targets)
    assert torch.equal(env._action_queue,queued)
    assert torch.equal(applied,target.clamp(-3,3))
    with pytest.raises(ValueError):bounded_targets(target*float('nan'),limits)


def test_position_history_is_past_radians_not_current_or_native_command():
    target=torch.arange(8*29).reshape(8,29).float()/100
    record=dict(robot=torch.randn(8,178),time=torch.arange(8,dtype=torch.float64)/60)
    result=position_robot_history(record,target)
    assert torch.equal(result[0,116:177],torch.zeros(61))
    assert torch.equal(result[1:,116:145],target[:-1])
    assert torch.equal(result[1:,145:174],target[:-1])
    assert torch.equal(result[:,:116],record['robot'][:,:116])
    changed=target.clone();changed[5:]+=100
    assert torch.equal(position_robot_history(record,changed)[:6],result[:6])


@pytest.mark.parametrize('horizon',[24,32])
def test_horizon_labels_masks_and_model_contract(horizon):
    values=dict(time=torch.arange(41),command=torch.randn(41,29))
    target,mask=labels(values,torch.tensor([0,36]),horizon)
    assert target.shape==(2,horizon,29) and mask.sum()==horizon+5
    assert torch.equal(target[1,:5],values['command'][36:]) and not target[1,5:].any()
    with torch.device('meta'):
        model=PersistentState('dp',contract(horizon))
    assert model.horizon==horizon and model.contract['commit']==12
    with pytest.raises(ValueError):PersistentState('bc',contract(horizon))


def test_position_normalization_roundtrip_and_horizon_loss_rejection():
    from visuotactile.native_data import limits_affine
    target=torch.randn(40,29);scale,offset=limits_affine(target)
    torch.testing.assert_close((target*scale+offset-offset)/scale,target,atol=1e-6,rtol=0)
    with torch.device('meta'):model=PersistentState('dp',contract(32))
    with pytest.raises(ValueError,match='horizon'):
        model.loss(torch.empty(1,512),torch.empty(1,64,29),torch.ones(1,64,dtype=torch.bool))


def test_h32_actual_diffusion_backward_and_native_clock_parity():
    from visuotactile.state_history import HistoryBank
    model=PersistentState('dp',contract()).eval()
    n=25
    actor=torch.randn(n,141);actor[:,-1]=50
    record=dict(actor=actor,robot=torch.randn(n,178),geometry=torch.randn(n,85),
        time=torch.arange(n,dtype=torch.float64)/60,command=torch.randn(n,29))
    record['robot']=position_robot_history(record,record['command'])
    bank=HistoryBank();conditions=[]
    with torch.no_grad():
        batch,anchors,_=model.encode(record)
        for tick in range(n):
            condition=bank.observe(0,9,tick,model,record['actor'][tick],record['robot'][tick],
                record['geometry'][tick],record['time'][tick])
            if condition is not None:conditions.append(condition)
        torch.testing.assert_close(batch,torch.cat(conditions),atol=1e-5,rtol=0)
    model.train()
    encoded,anchors,_=model.encode(record,0,25)
    target,mask=labels(record,anchors,32)
    loss=model.loss(encoded,target,mask,torch.Generator().manual_seed(8));loss.backward()
    assert torch.isfinite(loss) and model.history.weight_hh_l0.grad.norm()>0
    assert next(model.denoiser.parameters()).grad.norm()>0


def test_image_contract_and_privileged_input_rejection():
    from visuotactile.position_control import image_contract
    from visuotactile.state_history import PersistentImage
    for group in ('I-OfficialDP','I-HistoryDP'):
        with torch.device('meta'):model=PersistentImage(group,image_contract(group))
        assert model.horizon==32 and not hasattr(model,'actor_encoder') and not hasattr(model,'state_encoder')
        with pytest.raises(ValueError,match='allowlisted'):
            model.encode(dict(robot=None,actor=None,geometry=None))
        with pytest.raises(ValueError):PersistentImage(group,contract())
