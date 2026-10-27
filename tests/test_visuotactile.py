"""Contract tests. Synthetic sentinels test boundaries; they are not training data."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from visuotactile.history import OnlineHistory, history_indices
from visuotactile.models.nominal import NominalDiffusionPolicy, PolicyConfig
from visuotactile.models.correction import CorrectionPolicy, compose_action


def test_async_history_matches_offline_indices():
    hist = OnlineHistory(2,4,stride=3,keys=("proprio",))
    episodes, local = [0,0], [0,0]
    streams = [[],[]]
    for tick in range(22):
        if tick in (5,13):
            episodes[0] += 1
            local[0] = 0
            streams[0] = []
        values = torch.tensor([[episodes[0]*100+local[0]],[1000+local[1]]],dtype=torch.float)
        hist.append({"episode_id":torch.tensor(episodes),"step_id":torch.tensor(local),"proprio":values})
        for i in range(2):
            streams[i].append(values[i].clone())
            idx,mask = history_indices(local[i],4,3)
            batch = hist.batch()
            torch.testing.assert_close(batch["proprio"][i],torch.stack([streams[i][j] for j in idx]))
            assert np.array_equal(batch["valid"][i].numpy(),mask)
            local[i] += 1


def test_duplicate_observation_is_rejected():
    h=OnlineHistory(1,2,keys=("proprio",))
    p={"episode_id":torch.tensor([0]),"step_id":torch.tensor([0]),"proprio":torch.ones(1,2)}
    h.append(p)
    with pytest.raises(ValueError):
        h.append(p)


def test_native_arm_hand_semantics_and_canonical_mapping():
    # Load only the authoritative action module, without simulator package imports.
    path=Path(__file__).parents[1]/"isaacsimenvs/tasks/play/utils/action_utils.py"
    spec=importlib.util.spec_from_file_location("native_action_contract",path)
    native=importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    permutation=torch.arange(29).flip(0)
    env=SimpleNamespace(device="cpu",num_envs=1,step_dt=1/60,
        cfg=SimpleNamespace(domain_randomization=SimpleNamespace(use_action_delay=False),
                            action=SimpleNamespace(dof_speed_scale=1.5,arm_moving_average=.1,hand_moving_average=.1)),
        _perm_canon_to_lab=permutation,_prev_targets=torch.zeros(1,29),_cur_targets=torch.zeros(1,29),
        _arm_lower=torch.full((1,7),-2.),_arm_upper=torch.full((1,7),2.),
        _hand_lower=torch.zeros(1,22),_hand_upper=torch.full((1,22),2.),
        _arm_joint_ids=list(range(7)),_hand_joint_ids=list(range(7,29)))
    action=torch.linspace(-1,1,29).reshape(1,29)
    native.apply_action_pipeline(env,action)
    mapped=action[:,permutation]
    torch.testing.assert_close(env._cur_targets[:,:7],.0025*mapped[:,:7])
    torch.testing.assert_close(env._cur_targets[:,7:],.1*(mapped[:,7:]+1))


def test_nominal_rejects_tactile_and_privileged_keys():
    torch.set_num_threads(2)
    model=NominalDiffusionPolicy(PolicyConfig(perception_version=1,views=1,history=2))
    obs={"image":torch.zeros(1,2,1,4,24,32),"proprio":torch.zeros(1,2,58),"valid":torch.ones(1,2,dtype=torch.bool)}
    for name in ("contact_w_N","expert_action","object_pose","goal_pose","stage","corrected_action"):
        with pytest.raises(ValueError):
            model.encoder({**obs,name:torch.ones(1)})
    assert model.encoder(obs).shape == (1,128)


def test_correction_is_independent_and_cannot_execute_untrained():
    cfg=PolicyConfig(perception_version=1,views=1,history=2)
    nominal=NominalDiffusionPolicy(cfg)
    correction=CorrectionPolicy(cfg)
    assert not ({p.data_ptr() for p in nominal.parameters()} & {p.data_ptr() for p in correction.parameters()})
    a=torch.linspace(-1,1,58).reshape(2,29)
    torch.testing.assert_close(compose_action(a),a,rtol=0,atol=0)
    with pytest.raises(RuntimeError):
        compose_action(a,-a,torch.tensor([True,False]))
    out=compose_action(a,-a,torch.tensor([True,False]),trained=True)
    torch.testing.assert_close(out[0],-a[0])
    torch.testing.assert_close(out[1],a[1],rtol=0,atol=0)
    obs={"image":torch.zeros(2,2,1,4,24,32),"proprio":torch.zeros(2,2,58),"valid":torch.ones(2,2,dtype=torch.bool)}
    motion={"proprio":obs["proprio"],"previous_command":torch.zeros(2,2,29),
            "command_valid":obs["valid"],"valid":obs["valid"]}
    candidate=correction(obs,torch.zeros(2,7,5,3),torch.ones(2,7,dtype=torch.bool),
                         a[:,None].expand(-1,12,-1),torch.ones(2,12,dtype=torch.bool),motion)
    assert candidate.shape == (2,29) and torch.isfinite(candidate).all()


def test_checkpoint_load_preserves_simulator_rng(tmp_path):
    from dataclasses import asdict
    from visuotactile.rollout import NominalController
    model=NominalDiffusionPolicy(PolicyConfig(perception_version=1,views=1))
    path=tmp_path/"fixture.pt"
    torch.save(dict(step=1,config=asdict(model.cfg),model=model.state_dict(),metadata={},smoke_only=True),path)
    before=torch.random.get_rng_state().clone()
    NominalController.load(path,1)
    assert torch.equal(before,torch.random.get_rng_state())


def test_correction_real_time_alignment_and_async_reset():
    from visuotactile.correction import CorrectionContext
    context=CorrectionContext(2,PolicyConfig(perception_version=1,views=1,history=2,stride=1),force_history_steps=3)
    plan=torch.arange(2*12*29).reshape(2,12,29).float()
    packet=dict(image=torch.zeros(2,1,4,24,32),proprio=torch.zeros(2,58),
                contact_substeps_w_N=torch.ones(2,2,5,3),contact_valid=torch.tensor([False,False]),
                episode_id=torch.tensor([0,0]),step_id=torch.tensor([0,0]))
    first=context.observe(packet,plan,torch.tensor([0,5]))
    assert not first["tactile_valid"].any()
    assert not first["motion_history"]["command_valid"].any()
    assert first["nominal_valid"][1].sum()==7
    torch.testing.assert_close(first["nominal_chunk"][1,:7],plan[1,5:])
    with pytest.raises(RuntimeError):
        context.observe(packet,plan,torch.tensor([0,5]))
    context.commit(torch.ones(2,29))
    packet.update(episode_id=torch.tensor([1,0]),step_id=torch.tensor([0,1]),contact_valid=torch.tensor([False,True]))
    second=context.observe(packet,plan,torch.tensor([0,6]))
    assert not second["tactile_valid"][0].any()
    assert second["tactile_valid"][1].sum()==2
    assert not second["motion_history"]["command_valid"][0,-1]
    assert second["motion_history"]["command_valid"][1,-1]
    torch.testing.assert_close(second["motion_history"]["previous_command"][1,-1],torch.ones(29))


def test_nominal_inference_is_invariant_to_tactile_and_diagnostics():
    from visuotactile.rollout import NominalController
    cfg=PolicyConfig(perception_version=1,views=1,history=2,stride=1,inference_steps=2)
    model=NominalDiffusionPolicy(cfg)
    a,b=NominalController(model,1),NominalController(model,1)
    for t in range(8):
        # Reset only the second test episode; plans and histories must reset too.
        ep,step=(0,t) if t<4 else (1,t-4)
        p=dict(image=torch.zeros(1,1,4,24,32),proprio=torch.full((1,58),float(t)),
               episode_id=torch.tensor([ep]),step_id=torch.tensor([step]),contact_w_N=torch.zeros(1,5,3))
        alt={**p,"contact_w_N":torch.full((1,5,3),1e4),"expert_action":torch.ones(1,29),
             "object_pose_w":torch.ones(1,7),"reward":torch.tensor([999.])}
        torch.testing.assert_close(a.act(p),b.act(alt),rtol=0,atol=0)
        if step==0:
            assert a.cursor.item()==1
            assert a.history.batch()["valid"].sum()==1
