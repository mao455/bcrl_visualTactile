from types import SimpleNamespace

import pytest
import torch

from visuotactile.continuous_vision import encode_dense
from visuotactile.models.native_dp import NativePolicy
from visuotactile.position_control import continuous_image_contract
from visuotactile.state_image_loss import GeometryObjective
from visuotactile.dp_objectives import masked_objectives
from visuotactile.trajectory_batch import encode_trajectories, trajectory_objective


class SmallDenoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(512, 29)
        self.scale = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, sample, timestep, global_cond):
        return self.scale * sample + self.linear(global_cond)[:, None] + timestep[:, None, None] / 100


@pytest.fixture
def model():
    torch.set_num_threads(2)
    torch.manual_seed(71)
    value = NativePolicy('I-HistoryDP', 24, modality='G00')
    value.denoiser = SmallDenoiser()
    value.contract = continuous_image_contract('u_expert')
    return value.train()


def episode(length, seed):
    generator = torch.Generator().manual_seed(seed)
    time = torch.arange(length, dtype=torch.float64) / 60
    robot = torch.randn(length, 178, generator=generator) * .01
    robot[:, 176] = 1
    record = dict(time=time, robot=robot)
    ep = SimpleNamespace(visual_stride=1,
        record=dict(camera_time=time[:, None].expand(-1, 3).clone(),
                    camera_valid=torch.ones(length, 3, dtype=torch.bool)),
        perception=dict(patches=torch.randn(length, 3, 384, 18, 32, generator=generator) * .01,
                        depth=torch.rand(length, 3, 2, 18, 32, generator=generator)))
    return record, ep


def test_real_visual_and_gru_forward_gradient_match_complete_serial_histories(model):
    pairs = [episode(13, 1), episode(25, 2)]
    records, episodes = zip(*pairs)
    serial = [encode_dense(model, record, episode=ep) for record, ep in pairs]
    serial_loss = sum(condition.square().mean() for condition, _ in serial) / 2
    serial_loss.backward()
    expected = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    shapes = []
    hook = model.native_gru.register_forward_pre_hook(lambda _, args: shapes.append(len(args[0])))
    batched = encode_trajectories(model, records, episodes)
    hook.remove()
    assert shapes == [2] * 13 + [1] * 12
    for (a, ai), (b, bi) in zip(serial, batched):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        assert torch.equal(ai, bi)
    sum(condition.square().mean() for condition, _ in batched).div(2).backward()
    for name, parameter in model.named_parameters():
        if name in expected:
            torch.testing.assert_close(parameter.grad, expected[name], atol=2e-5, rtol=2e-5)


def test_other_trajectory_and_future_frames_never_change_past_condition(model):
    pairs = [episode(25, 4), episode(13, 5)]
    records, episodes = zip(*pairs)
    with torch.no_grad():
        before = encode_trajectories(model, records, episodes)
        episodes[0].perception['patches'][13:].add_(.4)
        episodes[1].perception['patches'].add_(.8)
        after = encode_trajectories(model, records, episodes)
    torch.testing.assert_close(before[0][0][:2], after[0][0][:2], atol=1e-7, rtol=0)
    assert not torch.allclose(before[0][0][-1], after[0][0][-1], atol=1e-7, rtol=0)
    assert not torch.allclose(before[1][0], after[1][0], atol=1e-7, rtol=0)


class Objective:
    forward = GeometryObjective.forward

    def __init__(self):
        self.weights = {'geometry': .3}
        self.gain = 1.
        self.context_calls = []
        self.last_geometry_metrics = None

    def components(self, prediction, truth, valid, scale, offset, context):
        self.context_calls.append(context)
        self.last_geometry_metrics = dict(selected=context['selected'])
        return dict(base=masked_objectives(prediction, truth * scale + offset, valid)['B'],
                    geometry=prediction[:context['selected'], 0, 0].square().mean())


def test_loss_keeps_per_episode_weight_noise_order_geometry_and_gradients(model):
    batches = []
    for n in [2, 7]:
        valid = torch.ones(n, 24, dtype=torch.bool)
        valid[-1, 3:] = False
        batches.append(dict(condition=torch.randn(n, 512), truth=torch.randn(n, 24, 29), valid=valid))
    contexts = [{'selected': 1}, {'selected': 2}]
    serial_objective = Objective()
    generator = torch.Generator().manual_seed(999)
    serial = [serial_objective.forward(model, b['condition'], b['truth'], b['valid'], generator,
        geometry_context=context) for b, context in zip(batches, contexts)]
    expected_loss = torch.stack([row[0] for row in serial]).mean()
    expected_loss.backward()
    expected_grad = {name: p.grad.clone() for name, p in model.denoiser.named_parameters()}
    model.zero_grad(set_to_none=True)
    generator2 = torch.Generator().manual_seed(999)
    objective = Objective()
    loss, reports, prediction = trajectory_objective(model, objective, batches, generator2, contexts)
    assert torch.equal(generator.get_state(), generator2.get_state())
    assert objective.context_calls == contexts
    assert [r['geometry']['selected'] for r in reports] == [1, 2]
    torch.testing.assert_close(prediction, torch.cat([row[2] for row in serial]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(loss, expected_loss, atol=2e-5, rtol=2e-5)
    loss.backward()
    for name, parameter in model.denoiser.named_parameters():
        torch.testing.assert_close(parameter.grad, expected_grad[name], atol=2e-5, rtol=2e-5)
    # Unequal trajectory lengths must not become a pooled valid-tick loss.
    pooled = masked_objectives(prediction,
        torch.cat([b['truth'] for b in batches]), torch.cat([b['valid'] for b in batches]))['B']
    equal = sum(row[1]['base'] for row in serial) / 2
    assert not torch.isclose(pooled, equal, atol=1e-6, rtol=0)


@pytest.mark.parametrize('problem', ['future', 'nan_time', 'missing', 'sparse', 'empty'])
def test_invalid_history_rejected_before_training(model, problem):
    record, ep = episode(13, 8)
    if problem == 'future': ep.record['camera_time'][5, 0] += 1
    if problem == 'nan_time': ep.record['camera_time'][5, 0] = float('nan')
    if problem == 'missing': ep.record['camera_valid'][7, 2] = False
    if problem == 'sparse': ep.visual_stride = 12
    if problem == 'empty':
        with pytest.raises(ValueError): encode_trajectories(model, [], [])
        return
    with pytest.raises(ValueError): encode_trajectories(model, [record], [ep])
