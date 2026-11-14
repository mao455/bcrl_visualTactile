from types import SimpleNamespace
import random

import numpy as np
import pytest
import torch

from tools.tight_state import kit_startup_scheduling
from visuotactile.execution_backend import (execution_backend, backend_receipt, startup_arguments,
    DESCRIPTOR_VERSION, renderer_resource_readback, require_renderer_resources)
from visuotactile.environment_rng import RandomStream, DeviceRandomStream, LaneRandomStreams, IsolatedEnvironment
from visuotactile.camera_calibration import ordered_camera_paths, renderer_poses


def test_default_and_registered_thread_settings_are_distinct():
    config = execution_backend()
    assert startup_arguments(config) == kit_startup_scheduling({})[0]
    candidate = execution_backend(dict(kit_tasking_threads=16, physics_threads=8), available_cpus=32)
    args, receipt = kit_startup_scheduling({}, execution_backend=candidate)
    assert 'threadCount=16' in args and 'numThreads=8' in args
    assert receipt['physics_threads'] == 8
    with pytest.raises(ValueError, match='conflicts'):
        kit_startup_scheduling({'TIGHT_KIT_TASKING_THREADS': '4'}, execution_backend=candidate)


@pytest.mark.parametrize('change', [dict(physics_device='cuda'), dict(physics_device='cuda:-1'),
    dict(camera_backend='raycaster'), dict(physics_threads=True), dict(physics_threads=0),
    dict(kit_tasking_threads=129), dict(physics_dt=.02), dict(version='unknown')])
def test_invalid_execution_changes_fail_before_simulator_import(change):
    with pytest.raises(ValueError):
        execution_backend(change, available_cpus=128)


def test_backend_identity_and_environment_count_have_no_four_lane_cap():
    a = backend_receipt(execution_backend(), num_envs=256)
    b = backend_receipt(execution_backend(dict(physics_device='cuda:0', camera_backend='tiled')), num_envs=200)
    assert a['configuration_sha256'] != b['configuration_sha256']
    assert b['num_envs'] == 200 and b['live_accepted'] is False
    assert '1x1' in b['camera_calibration']
    assert startup_arguments(b['configuration'], num_envs=200).endswith('--/rtx/viewTile/limit=200')
    with pytest.raises(ValueError):
        backend_receipt(execution_backend(), num_envs=True)


def test_descriptor_capacity_is_explicit_v2_with_unchanged_v1_arguments():
    ordinary=execution_backend(dict(camera_backend='tiled'))
    candidate=execution_backend(dict(ordinary,version=DESCRIPTOR_VERSION))
    assert candidate['renderer_descriptor_sets']==360000 and candidate['renderer_reserved_descriptors']==900000
    expected=startup_arguments(ordinary,num_envs=8)+' --/rtx/descriptorSets=360000 --/rtx/reservedDescriptors=900000'
    assert startup_arguments(candidate,num_envs=8)==expected
    args,receipt=kit_startup_scheduling({},execution_backend=candidate,num_envs=8)
    assert args==expected and receipt['execution_backend']==candidate
    custom=execution_backend(dict(candidate,renderer_descriptor_sets=60000,renderer_reserved_descriptors=500000))
    assert startup_arguments(custom,num_envs=8).endswith('--/rtx/descriptorSets=60000 --/rtx/reservedDescriptors=500000')
    assert backend_receipt(candidate,num_envs=8)['configuration_sha256']!=backend_receipt(custom,num_envs=8)['configuration_sha256']
    with pytest.raises(ValueError,match='Unknown execution_backend fields'):
        execution_backend(dict(renderer_descriptor_sets=360000))
    assert renderer_resource_readback(ordinary,lambda path:pytest.fail('v1 must not read extra settings')) is None
    require_renderer_resources(None)


@pytest.mark.parametrize('value',[0,-1,True,3.5,'360000',2147483648])
def test_v2_descriptor_counts_reject_invalid_or_injected_values(value):
    for field in ('renderer_descriptor_sets','renderer_reserved_descriptors'):
        with pytest.raises(ValueError,match='signed 32-bit integer'):
            execution_backend(dict(version=DESCRIPTOR_VERSION,**{field:value}))


def test_capacity_readback_rejects_missing_coerced_or_overridden_values():
    config=execution_backend(dict(version=DESCRIPTOR_VERSION))
    settings={'/rtx/descriptorSets':360000,'/rtx/reservedDescriptors':900000,'/rtx/viewTile/limit':8}
    receipt=renderer_resource_readback(config,settings.get)
    require_renderer_resources(receipt)
    assert receipt['matched'] and receipt['view_tile_limit_observed']==8
    for value in (None,True,360000.,60000):
        settings['/rtx/descriptorSets']=value
        receipt=renderer_resource_readback(config,settings.get)
        assert not receipt['matched']
        with pytest.raises(RuntimeError,match='Renderer resource settings'):
            require_renderer_resources(receipt)


def test_lane_stream_replacement_preserves_other_lane_and_external_rng():
    external = RandomStream.snapshot()
    one, two = LaneRandomStreams([11, 12]), LaneRandomStreams([11, 12])
    with one.scope(1):
        first = torch.rand(7)
    with two.scope(1):
        assert torch.equal(first, torch.rand(7))
    one.replace(0, 100)
    with one.scope(0):
        torch.rand(913); random.random(); np.random.rand(9)
    with one.scope(1):
        expected = torch.rand(7)
    with two.scope(1):
        assert torch.equal(expected, torch.rand(7))
    assert torch.equal(external[2], torch.get_rng_state())
    assert external[0] == random.getstate()
    assert np.array_equal(external[1][1], np.random.get_state()[1])


def test_registered_reset_never_calls_global_seed_and_indexed_reset_keeps_other_lane():
    class Env:
        device = 'cpu'
        num_envs = 3
        execution_backend = execution_backend()
        def __init__(self):
            self.positions = torch.tensor([1., 2., 3.])
        def reset(self, seed):
            assert seed is None
            return torch.rand(1)
        def _reset_idx(self, ids):
            self.positions[ids] = torch.rand(len(ids))
        def step(self, command):
            return torch.rand(1)
    env = Env(); isolated = IsolatedEnvironment(env, 17)
    external = torch.get_rng_state().clone()
    isolated.reset(17)
    isolated.reset_lanes([1])
    assert env.positions[[0, 2]].tolist() == [1., 3.]
    assert torch.equal(external, torch.get_rng_state())
    with pytest.raises(ValueError):
        isolated.reset_lanes([1, 1])


def test_gpu_rng_scopes_only_selected_device_using_mock_cuda(monkeypatch):
    # No CUDA initialization or GPU test job: mock state storage with CPU tensors.
    original_generator = torch.Generator
    states = {'cuda:0': original_generator().manual_seed(8).get_state(),
              'cuda:1': original_generator().manual_seed(9).get_state()}
    def get(device):
        return states[str(device)].clone()
    def put(state, device):
        states[str(device)] = state.clone()
    monkeypatch.setattr(torch.cuda, 'get_rng_state', get)
    monkeypatch.setattr(torch.cuda, 'set_rng_state', put)
    monkeypatch.setattr(torch, 'Generator', lambda device=None: original_generator())
    first, other, cpu = states['cuda:0'].clone(), states['cuda:1'].clone(), torch.get_rng_state().clone()
    stream = DeviceRandomStream(27, 'cuda:0')
    with pytest.raises(RuntimeError, match='intentional'):
        with stream.scope():
            assert not torch.equal(states['cuda:0'], first)
            torch.rand(10)
            raise RuntimeError('intentional')
    assert torch.equal(states['cuda:0'], first) and torch.equal(states['cuda:1'], other)
    assert torch.equal(torch.get_rng_state(), cpu)


def test_tiled_calibration_requires_exact_sensor_lane_order_and_live_matrices():
    def prim(path):
        return SimpleNamespace(GetPath=lambda: path)
    sensor = SimpleNamespace(_sensor_prims=[prim('/World/envs/env_0/Wrist/Cam'), prim('/World/envs/env_1/Wrist/Cam')])
    assert len(ordered_camera_paths(sensor, 2)) == 2
    sensor._sensor_prims.reverse()
    with pytest.raises(RuntimeError, match='order'):
        ordered_camera_paths(sensor, 2)
    transforms = [np.eye(4), np.eye(4)]
    transforms[1][3, 0] = -8
    annotators = [SimpleNamespace(get_data=lambda value=value: {'cameraViewTransform': value}) for value in transforms]
    poses = renderer_poses(annotators, 2)
    assert poses.shape == (2, 7) and poses[1, 0] - poses[0, 0] == 8
    with pytest.raises(RuntimeError, match='one independent'):
        renderer_poses(annotators[:1], 2)
    annotators[0] = SimpleNamespace(get_data=lambda: {})
    with pytest.raises(RuntimeError, match='Missing or invalid'):
        renderer_poses(annotators, 2)
