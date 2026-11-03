import numpy as np
import pytest
import torch

from tools.state_fk_audit import (quat_matrix, rotation_error, local_xyz, phase_ids,
    native_targets, target_errors, fk_pose)
from visuotactile.kinematics import EncoderKinematics


def arm_fk():
    return EncoderKinematics('assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf',
        [f'iiwa14_joint_{joint}' for joint in range(1, 8)], links=('iiwa14_link_7',))


def test_fk_world_transform_and_geodesic_use_wxyz_and_base_rotation():
    quat = np.array([2**-.5, 0, 0, 2**-.5])
    base = np.array([4., -8., .2, *quat]); point = np.array([.3, -.1, .6])
    world = quat_matrix(quat)@point+base[:3]
    assert np.allclose(local_xyz(world, base), point)
    assert rotation_error(np.eye(3), quat_matrix(quat)) == pytest.approx(90.)
    assert rotation_error(np.eye(3), quat_matrix([0., 1., 0., 0.])) == pytest.approx(180.)
    assert np.allclose(quat_matrix(quat), quat_matrix(-quat))


def test_target_error_never_uses_actual_q_and_handles_no_direction():
    fk = arm_fk(); truth = np.zeros((3, 29)); pred = truth.copy(); pred[:, 1] = .1
    errors = target_errors(pred, truth, truth, fk)
    assert np.all(errors['distance'] > 1) and not errors['directional_valid'].any()
    perfect = target_errors(truth, truth, pred, fk)
    assert np.max(perfect['distance']) == 0 and np.max(perfect['rotation']) == 0
    assert perfect['directional_valid'].all()


def test_fk_inverse_affine_gradient_and_padding():
    fk = arm_fk(); scale = torch.linspace(.2, 2., 29, dtype=torch.double)
    offset = torch.linspace(-1., 1., 29, dtype=torch.double)
    target = torch.zeros(3, 29, dtype=torch.double)
    normalized = (target*scale+offset).clone(); normalized[:, 1] += .05
    normalized.requires_grad_(); valid = torch.tensor([True, True, False])
    def objective(values):
        position, _ = fk((values-offset)/scale); reference, _ = fk(target)
        return (position[valid]-reference[valid]).square().sum()/valid.sum()
    assert torch.autograd.gradcheck(objective, (normalized,), eps=1e-6, atol=1e-6)
    grad, = torch.autograd.grad(objective(normalized), normalized)
    assert grad[:2, :7].norm() > 0
    assert grad[2].abs().sum() == 0 and grad[:, 7:].abs().sum() == 0
    position, _ = fk_pose(fk, np.zeros((1, 29)))
    assert np.isfinite(position).all()


def test_phase_proxies_respect_validity_native_lift_and_subgoals():
    force = np.zeros((16, 2, 5, 3)); force[2:, :, :2, 0] = .2
    valid = np.ones(16, bool); valid[:3] = False
    lifted = np.zeros(16, bool); lifted[10:12] = True
    goals = np.zeros(16); goals[12:] = 1
    phase = phase_ids(force, valid, lifted, goals)
    assert (phase[:3] == 0).all() and (phase[3:8] == 1).all()
    assert (phase[8:10] == 3).all()
    assert (phase[10:12] == 4).all() and (phase[12:] == 5).all()


def test_native_queue_startup_age_and_target_processing_order():
    commands = np.arange(4, dtype=np.float32)[:, None]*np.ones((4, 29), np.float32)/4
    index = np.array([2, 2, 1, 2]); tick = np.arange(4)
    selected = np.maximum(0, tick-index)
    assert np.array_equal(tick-selected, [0, 1, 1, 2])
    before = np.full((4, 29), .99, np.float32); limits = np.tile([-1., 1.], (29, 1))
    config = dict(dof_speed_scale=1.5, arm_moving_average=.1, hand_moving_average=.1)
    targets = native_targets(before, commands[selected], limits, config, 1/60)
    assert targets[0, 0] == pytest.approx(.99)
    assert targets[2, 0] == pytest.approx(.990625)
    assert targets[2, 7] == pytest.approx(.916)
    saturated = native_targets(before, np.ones_like(before), limits, config, 1.)
    assert saturated[0, 0] == pytest.approx(.991)
