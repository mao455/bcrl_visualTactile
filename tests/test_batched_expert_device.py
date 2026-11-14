"""Expert input crosses the physics/player device boundary before concatenation."""
from types import SimpleNamespace

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from visuotactile.batched_rollout import expert_actor_input


def test_cpu_expert_input_preserves_exact_observation_and_embedding():
    policy = torch.arange(14, dtype=torch.float32).reshape(2, 7)
    embedding = torch.tensor([[50.], [50.]])
    actor = expert_actor_input({'policy': policy}, SimpleNamespace(intr_reward_coef_embd=embedding))
    assert actor.device.type == 'cpu'
    assert torch.equal(actor[:, :-1], policy)
    assert torch.equal(actor[:, -1:], embedding)


def test_cuda_physics_observation_moves_to_cpu_expert_without_cuda_runtime():
    # Fake tensors exercise the real PyTorch device-dispatch checks, including
    # the formerly invalid CUDA/CPU cat, without allocating any GPU storage.
    with FakeTensorMode():
        policy = torch.empty((8, 153), device='cuda:0')
        embedding = torch.empty((8, 1), device='cpu')
        actor = expert_actor_input({'policy': policy}, SimpleNamespace(intr_reward_coef_embd=embedding))
        assert actor.device.type == 'cpu'
        assert actor.shape == (8, 154)
        assert policy.device == torch.device('cuda:0')

