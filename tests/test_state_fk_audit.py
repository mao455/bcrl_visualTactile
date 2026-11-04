import numpy as np
import torch
from tools.trajectory_metrics import fk_pose
from visuotactile.kinematics import EncoderKinematics

def arm_fk():
    return EncoderKinematics('assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf',
        [f'iiwa14_joint_{joint}' for joint in range(1, 8)], links=('iiwa14_link_7',))

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
