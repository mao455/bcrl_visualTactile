"""Boundary/numerical checks, not synthetic learning or policy evaluation."""
from dataclasses import asdict
import copy
import torch
import pytest

from visuotactile.observations import TaskCommand, empty_plan_context, nominal_keys
from visuotactile.models.nominal import NominalDiffusionPolicy, PolicyConfig
from visuotactile.models.correction import CorrectionPolicy
from visuotactile.correction import CorrectionContext
from visuotactile.rollout import NominalController
from visuotactile.kinematics import rotate_force_to_link


def packet(n=2,t=0):
    # Deterministic sentinels only to test interfaces, never imitation targets.
    return dict(image=torch.linspace(0,1,n*2*4*24*32).reshape(n,2,4,24,32),
                depth_valid=torch.ones(n,2,1,24,32,dtype=torch.bool),
                camera_valid=torch.ones(n,2,dtype=torch.bool),
                camera_time_s=torch.full((n,2),t/60,dtype=torch.float64),
                time_s=torch.full((n,),t/60,dtype=torch.float64),
                proprio=torch.zeros(n,58),episode_id=torch.zeros(n,dtype=torch.long),
                step_id=torch.full((n,),t,dtype=torch.long),
                contact_substeps_w_N=torch.ones(n,2,5,3),contact_substeps_local_N=torch.ones(n,2,5,3),
                finger_position_base_m=torch.zeros(n,2,5,3),finger_rotation_base=torch.eye(3).expand(n,2,5,3,3).clone(),
                contact_time_s=torch.tensor([(t-.5)/60,t/60],dtype=torch.float64).expand(n,2).clone(),
                contact_valid=torch.full((n,),t>0),contact_w_N=torch.ones(n,5,3),
                actuator_target_rad=torch.zeros(n,29),tracking_error_rad=torch.zeros(n,29),
                **TaskCommand("tight_insertion").packet(n,"cpu"),**empty_plan_context(n))


def test_v2_full_interfaces_gradients_and_untrained_roundtrip(tmp_path):
    torch.set_num_threads(2)
    cfg=PolicyConfig(history=2,stride=1,inference_steps=2)
    nominal=NominalController(NominalDiffusionPolicy(cfg),2)
    correction=CorrectionPolicy(cfg)
    assert not ({p.data_ptr() for p in nominal.model.parameters()} & {p.data_ptr() for p in correction.parameters()})
    context=CorrectionContext(2,cfg,force_history_steps=2)
    for t in range(3):
        p=packet(t=t)
        nominal.act(p)
        inputs=context.observe(nominal.last_packet,nominal.plan,nominal.cursor-1)
        context.commit(torch.full((2,29),.25))
    out=correction(**inputs)
    assert out.shape==(2,29) and torch.isfinite(out).all()
    # Jacobian probe, no labels, loss function, optimizer or weight update.
    out.sum().backward()
    for module in (correction.visual.vision.rgb,correction.visual.vision.depth,
                   correction.tactile,correction.intent,correction.motion,correction.policy):
        grads=[p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and sum(g.abs().sum() for g in grads)>0
    state=dict(config=asdict(cfg),step=0,model=nominal.model.state_dict(),correction=correction.state_dict())
    path=tmp_path/'untrained.pt';torch.save(state,path)
    saved=torch.load(path,weights_only=False)
    restored=CorrectionPolicy(PolicyConfig.from_dict(saved['config']));restored.load_state_dict(saved['correction'])
    torch.testing.assert_close(out,restored(**inputs),rtol=0,atol=0)
    with pytest.raises(ValueError,match='training'):
        NominalController.load(path,2)


def test_v2_nominal_private_plan_and_no_tactile_leak():
    cfg=PolicyConfig(history=2,stride=1,inference_steps=2)
    model=NominalDiffusionPolicy(cfg)
    a,b=NominalController(model,2),NominalController(model,2)
    for t in range(4):
        p=packet(t=t)
        if t>=2:
            p['episode_id'][0]=1;p['step_id'][0]=t-2
        altered=copy.deepcopy(p)
        for key in ('contact_substeps_w_N','contact_substeps_local_N','actuator_target_rad','tracking_error_rad',
                    'previous_nominal_action','prior_plan'):
            altered[key].fill_(999)
        altered['expert_action']=torch.ones(2,29)
        torch.testing.assert_close(a.act(p),b.act(altered),rtol=0,atol=0)
        if t==2:
            assert not a.last_packet['previous_nominal_valid'][0]
            assert not a.last_packet['prior_plan_valid'][0].any()
            assert a.last_packet['previous_nominal_valid'][1]
        with pytest.raises(ValueError):
            model.encoder({**a.history.batch(),'contact_w_N':p['contact_w_N']})


def test_goal_sources_masks_and_camera_time():
    with pytest.raises(ValueError):
        TaskCommand('tight_insertion',dict(source='simulator_goal',frame='robot_base',position_m=[0,0,0],rotation_columns=[1,0,0,0,1,0]))
    cmd=TaskCommand('tight_insertion',dict(source='operator_command',frame='robot_base',position_m=[.1,0,.5],rotation_columns=[1,0,0,0,1,0]))
    assert cmd.packet(2,'cpu')['goal_valid'].all()
    cfg=PolicyConfig(history=2,stride=1,inference_steps=2)
    c=NominalController(NominalDiffusionPolicy(cfg),2);c.act(packet())
    obs=c.history.batch();ref=c.model.encoder(obs)
    obs['goal_command'].fill_(1e6)
    obs['prior_plan'].fill_(1e6);obs['previous_nominal_action'].fill_(1e6)
    torch.testing.assert_close(ref,c.model.encoder(obs),rtol=0,atol=0)
    obs['camera_time_s']+=1
    with pytest.raises(ValueError,match='future'):
        c.model.encoder(obs)


def test_force_frame_rotation_and_reset_mask():
    from visuotactile.models.tactile import FingerTactileEncoder
    rotation=torch.tensor([[0.,-1,0],[1,0,0],[0,0,1]])
    force=torch.tensor([1.,0,0])
    torch.testing.assert_close(rotate_force_to_link(force,rotation),torch.tensor([0.,-1,0]))
    p=packet(t=1)
    tactile=dict(force_w_N=p['contact_substeps_w_N'],force_local_N=p['contact_substeps_local_N'],
                 position_base_m=p['finger_position_base_m'],rotation_base=p['finger_rotation_base'],time_s=p['contact_time_s'])
    encoder=FingerTactileEncoder()
    out=encoder(tactile,torch.zeros(2,2,dtype=torch.bool));assert torch.equal(out,torch.zeros_like(out))
    tactile['time_s'][:]=0
    with pytest.raises(ValueError,match='timestamps'):
        encoder(tactile,torch.ones(2,2,dtype=torch.bool))


def test_v2_modalities_and_view_identity_reach_encoder():
    cfg=PolicyConfig(history=1,stride=1,inference_steps=2)
    c=NominalController(NominalDiffusionPolicy(cfg),2);c.act(packet())
    obs=c.history.batch()
    image=obs['image'].clone().requires_grad_();obs['image']=image
    c.model.encoder(obs).sum().backward()
    for view in range(2):
        assert image.grad[:,:,view,:3].abs().sum()>0
        assert image.grad[:,:,view,3:].abs().sum()>0
    with torch.no_grad():
        baseline=c.model.encoder(obs)
        swapped={**obs,'image':obs['image'].flip(2)}
        assert not torch.equal(baseline,c.model.encoder(swapped))


def test_legacy_config_explicit_interpretation():
    assert PolicyConfig.from_dict(dict(views=1)).perception_version==1
    assert PolicyConfig().perception_version==2
    assert nominal_keys(1)==('image','proprio')


def test_v2_correction_async_reset_preserves_other_environment():
    cfg=PolicyConfig(history=2,stride=1)
    context=CorrectionContext(2,cfg,force_history_steps=3)
    plan=torch.zeros(2,12,29)
    for t in range(3):
        p=packet(t=t)
        if t==2:
            p['episode_id'][0]=1;p['step_id'][0]=0;p['contact_valid'][0]=False
        inputs=context.observe(p,plan,torch.zeros(2,dtype=torch.long))
        if t==2:
            assert inputs['visual_history']['valid'][0].sum()==1
            assert inputs['visual_history']['valid'][1].sum()==2
            assert not inputs['tactile_valid'][0].any()
            assert inputs['tactile_valid'][1].sum()==4
            assert not inputs['motion_history']['command_valid'][0].any()
            assert inputs['motion_history']['command_valid'][1].sum()==2
        context.commit(torch.full((2,29),float(t)))
