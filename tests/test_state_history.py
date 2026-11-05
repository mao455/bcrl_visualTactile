import random
import numpy as np
import pytest
import torch

from visuotactile.environment_rng import RandomStream, IsolatedEnvironment
from visuotactile.state_history import HistoryBank, PersistentState, STATE_CONTRACT


@pytest.fixture(scope='module')
def model():
    torch.manual_seed(8)
    return PersistentState('bc').eval()


def record(length=133):
    generator = torch.Generator().manual_seed(27)
    actor = torch.randn(length, 141, generator=generator); actor[:, -1] = 50
    robot = torch.randn(length, 178, generator=generator); robot[:, 176] = 1
    return dict(actor=actor, robot=robot, geometry=torch.randn(length, 85, generator=generator),
        time=torch.arange(length, dtype=torch.float64)/60)


def test_full_stream_chunk_and_prefix_parity(model):
    values = record(); bank = HistoryBank(); rows = []
    with torch.no_grad():
        full, _, _ = model.encode(values)
        for tick in range(133):
            condition = bank.observe(0, 100, tick, model, values['actor'][tick], values['robot'][tick],
                values['geometry'][tick], values['time'][tick])
            if condition is not None:
                rows.append(condition)
        # Batched Linear/GRU and single-row BLAS accumulate in different orders.
        torch.testing.assert_close(full, torch.cat(rows), rtol=0, atol=1e-5)
        condition, anchors, _ = model.encode(values, 96, 133)
        torch.testing.assert_close(full[anchors//12], condition, rtol=0, atol=1e-5)


def test_history_bank_independent_lanes_true_reset_and_tick_guard(model):
    values = record(20); bank = HistoryBank()
    with torch.no_grad():
        for tick in range(20):
            for lane in (0, 1):
                bank.observe(lane, 10, tick, model, values['actor'][tick], values['robot'][tick],
                    values['geometry'][tick], values['time'][tick])
        other = bank.lanes[1]['hidden'].clone()
        bank.observe(0, 11, 0, model, values['actor'][0], values['robot'][0], values['geometry'][0], values['time'][0])
        assert torch.equal(other, bank.lanes[1]['hidden'])
        with pytest.raises(ValueError, match='Skipped'):
            bank.observe(1, 10, 0, model, values['actor'][0], values['robot'][0], values['geometry'][0], values['time'][0])
        with pytest.raises(ValueError, match='full prefix'):
            bank.observe(2, 11, 12, model, values['actor'][0], values['robot'][0], values['geometry'][0], values['time'][0])


def test_causal_future_and_middle_segment_gradients(model):
    values = record(); changed = {key: value.clone() for key, value in values.items()}
    changed['actor'][97:, :140] += 90; changed['robot'][97:] += 3; changed['geometry'][97:] -= 5
    first, anchors, _ = model.encode(values, 96, 133)
    second, _, _ = model.encode(changed, 96, 133)
    assert torch.equal(first[0], second[0])
    model.zero_grad(); first.square().mean().backward()
    assert model.history.weight_hh_l0.grad.abs().sum() > 0
    assert model.actor_encoder[0].weight.grad.abs().sum() > 0


def test_environment_stream_independent_of_policy_draws():
    class Dummy:
        device = 'cpu'
        def reset(self, seed):
            torch.manual_seed(seed)
            return self.step(None)
        def step(self, command):
            return torch.rand(3), np.random.rand(), random.random()
    a, b = IsolatedEnvironment(Dummy(), 42), IsolatedEnvironment(Dummy(), 42)
    a.reset(42); b.reset(42)
    for tick in range(20):
        first = a.step(None)
        torch.randn(100+tick); random.random(); np.random.rand(17)
        second = b.step(None)
        assert torch.equal(first[0], second[0]) and first[1:] == second[1:]


def test_rng_scope_restores_after_exception_and_rejects_nested():
    torch.manual_seed(17); initial = torch.get_rng_state().clone(); stream = RandomStream(91)
    assert torch.equal(initial, torch.get_rng_state())
    with pytest.raises(RuntimeError):
        with stream.scope():
            torch.randn(20)
            with stream.scope():
                pass
    assert torch.equal(initial, torch.get_rng_state())


def test_unknown_actor_group_and_old_contract_rejected(model):
    actor = record(1)['actor']; actor[:, -1] = 49
    with pytest.raises(ValueError, match='Unknown'):
        model.actor_input(actor)
    from visuotactile.native_data import CONTRACT
    assert STATE_CONTRACT != CONTRACT


def test_live_controller_initialization_reads_real_asset_dimensions(model):
    import json
    from pathlib import Path
    import h5py
    from visuotactile.native_data import object_dimensions
    from visuotactile.state_history import HistoryController
    root=Path('outputs/visuotactile/native_dp_3way/native_20261101_02')
    data=json.loads((root/'data.json').read_text());identity=data['splits']['train'][0]
    with h5py.File(data['entries'][identity]['path'],'r') as stream:
        metadata=json.loads(stream.attrs['metadata_json'])
    cached=torch.load(root/'numeric'/(identity+'.pt'),map_location='cpu',weights_only=False)
    controller=HistoryController(dict(kind='bc',updates=1,contract=STATE_CONTRACT,ema=model.state_dict()),metadata,device='cpu')
    torch.testing.assert_close(controller.dimensions,cached['geometry'][0,-3:],rtol=0,atol=0)
    assert torch.equal(object_dimensions(metadata),controller.dimensions)
    assert controller.episode is None and controller.plan is None


def test_diffusion_path_keeps_horizon_and_trains_history():
    from visuotactile.anchor_sampling import labels
    model = PersistentState('dp').train()
    values = record(13); values['command'] = torch.zeros(13, 29)
    condition, anchors, _ = model.encode(values, 0, 12)
    command, valid = labels(values, anchors)
    assert command.shape == (1, 64, 29) and valid.sum() == 13
    assert model.denoiser.final_conv[-1].out_channels == 29
    assert model.noise.config.prediction_type == 'sample'
    loss = model.loss(condition, command, valid, torch.Generator().manual_seed(18))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.history.weight_ih_l0.grad.abs().sum() > 0
    assert next(model.denoiser.parameters()).grad.abs().sum() > 0


def test_empty_disabled_or_nonworking_scene_queries_cannot_admit(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace as NS
    from visuotactile.state_benchmark import collision_evidence
    omni = ModuleType('omni'); usd = ModuleType('omni.usd'); physx = ModuleType('omni.physx')
    root = '/World/Object/body'
    prim = NS(GetPath=lambda: root+'/collisions/box', HasAPI=lambda api: True)
    usd.get_context = lambda: NS(get_stage=lambda: NS(Traverse=lambda: [prim]))
    omni.usd = usd
    physx.get_physx_scene_query_interface = lambda: NS(overlap_shape=lambda *args: 0, overlap_box=lambda *args: 0)
    pxr = ModuleType('pxr'); pxr.PhysicsSchemaTools = NS(encodeSdfPath=lambda path: (1, 2))
    pxr.Sdf = NS(Path=lambda path: path)
    pxr.UsdPhysics = NS(CollisionAPI=lambda prim: NS(GetCollisionEnabledAttr=lambda: NS(Get=lambda: True)))
    carb = ModuleType('carb'); carb.Float3 = lambda *v: v; carb.Float4 = lambda *v: v
    for name, module in [('omni', omni), ('omni.usd', usd), ('omni.physx', physx), ('pxr', pxr), ('carb', carb)]:
        monkeypatch.setitem(sys.modules, name, module)
    env = NS(cfg=NS(sim=NS(enable_scene_query_support=False)),
        object=NS(root_physx_view=NS(prim_paths=[root]), data=NS(root_pos_w=torch.zeros(1, 3))))
    with pytest.raises(RuntimeError, match='disabled'):
        collision_evidence(env)
    env.cfg.sim.enable_scene_query_support = True
    with pytest.raises(RuntimeError, match='positive control'):
        collision_evidence(env)
