import copy

import pytest
import torch

from visuotactile.execution_learning import SPAN_PROJECTION_FY_LOSS, projection_fy_elements, projection_loss_scale
from visuotactile.models.native_dp import TargetResidual, load_residual, validate_residual_loss_contract
from visuotactile.position_control import REACHABLE_POSITION_EXECUTION as ADAPTER
from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
from test_target_residual import statistics


@pytest.mark.parametrize('raw_value,target_fraction', [(-3., 1.), (3., 0.), (3., .3), (3., 1.), (.2, .8)])
def test_span_metric_equalizes_true_raw_gradient(raw_value, target_fraction):
    span = torch.tensor([[.0025, .05, .01]], dtype=torch.float64)
    bounds = torch.stack((torch.zeros_like(span), span), -1)
    raw = torch.full_like(span, raw_value, requires_grad=True)
    target = span*target_fraction
    scale = projection_loss_scale(torch.tensor([8., 1., 30.]), bounds, SPAN_PROJECTION_FY_LOSS)
    def objective(value):
        return (projection_fy_elements(value*span, target, bounds, torch.ones(1))[0]*scale.square()).sum()
    gradient = torch.autograd.grad(objective(raw), raw)[0]
    torch.testing.assert_close(gradient, torch.full_like(span, 2*(min(max(raw_value, 0.), 1.)-target_fraction)))
    torch.autograd.gradcheck(objective, (raw,), eps=1e-6, atol=1e-6)


def test_span_loss_masks_roundoff_and_constant_coordinates():
    bounds = torch.tensor([[[0., .0025], [.2, .2]], [[0., .0025], [.2, .2]]])
    proposal = torch.tensor([[1., 1.], [-1., -1.]], requires_grad=True)
    target = torch.tensor([[.0025005, .2], [float('nan'), float('nan')]])
    mask = torch.tensor([1., 0.])
    elements, _ = projection_fy_elements(proposal, target, bounds, mask)
    loss = (elements*projection_loss_scale(torch.ones(2), bounds, SPAN_PROJECTION_FY_LOSS).square()).sum()
    assert loss == 0
    assert torch.equal(torch.autograd.grad(loss, proposal)[0], torch.zeros_like(proposal))
    with pytest.raises(ValueError, match='Unreachable'):
        projection_fy_elements(proposal, torch.ones_like(proposal), bounds, mask)


def test_explicit_span_contract_checkpoint_and_unchanged_executor():
    model = TargetResidual(statistics(), [.2]*7+[.9]*22, execution_adapter=ADAPTER,
        loss_contract=SPAN_PROJECTION_FY_LOSS, output_contract=REACHABLE_CENTERED_OUTPUT)
    previous = torch.zeros(1, 29)
    limits = torch.tensor([[-1., 1.]]).expand(29, 2)
    raw = torch.full((1, 29), 3., requires_grad=True)
    chain = model.compose_chain(previous, raw, None, limits, previous=previous)
    target = chain['execution_bounds'][..., 0]
    values = model.objective_components(chain['u_exec'], target, chain['delta'], torch.ones(1),
        u_pre=chain['u_pre'], execution_bounds=chain['execution_bounds'])
    assert (torch.autograd.grad(values['total'], raw, retain_graph=True)[0] > 0).all()
    assert torch.equal(torch.autograd.grad(values['exec_fit'], raw)[0], torch.zeros_like(raw))
    other = copy.deepcopy(model)
    other.loss_contract = None
    assert torch.equal(chain['u_exec'], other.compose_chain(previous, raw, None, limits, previous=previous)['u_exec'])
    payload = dict(model=model.state_dict(), updates=2, residual_version=model.version,
        residual_loss_version=SPAN_PROJECTION_FY_LOSS['version'],
        residual_output_version=REACHABLE_CENTERED_OUTPUT['version'],
        configuration=dict(loss_contract=SPAN_PROJECTION_FY_LOSS, output_contract=REACHABLE_CENTERED_OUTPUT,
            execution_adapter=ADAPTER))
    assert load_residual(payload).loss_contract == SPAN_PROJECTION_FY_LOSS
    for key, value in [('minimum_span_rad', 0.), ('regularizer_weight', 1e-4), ('extra', True)]:
        with pytest.raises(ValueError):
            validate_residual_loss_contract(dict(SPAN_PROJECTION_FY_LOSS, **{key: value}))
