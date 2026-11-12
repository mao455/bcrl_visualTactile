"""Conditional supervision for the unchanged postprocessed position interface.

The v1 executor is NOT a monotone clip within its 1e-6 rad allowance. This
module explicitly uses the nested ideal box, reports real execution separately,
and admits label roundoff only within that recorded allowance. No straight
through autograd, new delay, speed processing, or counterfactual labels.
"""
import torch

PROJECTION_FY_LOSS = dict(version='ideal_box_fy_roundoff_v1',
    idealization_tolerance_rad=1e-6, multiplier=2., regularizer_weight=1e-4)


def is_projection_fy_loss(value):
    return isinstance(value,dict) and value.get('version')==PROJECTION_FY_LOSS['version']


def ideal_execution_bounds(previous, limits, adapter):
    from .position_control import reachable_position_bounds
    tolerance = adapter['roundoff_tolerance_rad']
    lo, hi = limits[..., 0], limits[..., 1]
    if (not torch.isfinite(limits).all() or (lo > hi).any()
            or (previous < lo).any() or (previous > hi).any()):
        raise ValueError('Ideal nested box requires legal finite previous targets and joint limits')
    bounds = reachable_position_bounds(previous, limits, adapter)
    if (not torch.isfinite(bounds).all() or (bounds[..., 0] > bounds[..., 1]).any()
            or (bounds[..., 0] < lo-tolerance).any() or (bounds[..., 1] > hi+tolerance).any()):
        raise ValueError('Reachable box is not nested in joint limits')
    # Only endpoint arithmetic roundoff is intersected. Reject a truly empty box.
    lower, upper = torch.maximum(bounds[..., 0], lo), torch.minimum(bounds[..., 1], hi)
    if (lower > upper).any():
        raise ValueError('Empty effective execution box')
    return torch.stack((lower, upper), -1).detach()


def projection_fy_elements(x, target, bounds, valid, *, tolerance=1e-6):
    """2*FY = (p-y)^2 + 2*(x-p)*(p-y), derivative 2*(p-y).

    Same interior scaling as squared error. Endpoint labels retain half-line
    equivalence; near-boundary interior labels have a small continuous gradient,
    unlike squared distance to an exact inverse image. Large errors in masked
    student prefixes are excluded, not relabeled as expert supervision.
    """
    lower, upper = bounds[..., 0], bounds[..., 1]
    active = valid[..., None] > 0
    if (not torch.isfinite(x).all() or not torch.isfinite(bounds).all()
            or (lower > upper).any() or not torch.isfinite(target[active.expand_as(target)]).all()):
        raise ValueError('Nonfinite command/label or invalid execution interval')
    y = torch.where(active, target.detach(), lower)
    canonical = torch.maximum(torch.minimum(y, upper), lower)
    if ((y-canonical).abs()[active.expand_as(y)] > tolerance).any():
        raise ValueError('Unreachable expert label exceeds declared rad roundoff tolerance')
    p = torch.maximum(torch.minimum(x, upper), lower)
    error = p-canonical
    value = error.square()+2*(x-p)*error
    return torch.where(active, value, 0.), (y-canonical).abs()


def execution_expressibility(base, scales, action_scale, target, bounds, *, tolerance=1e-6, stable_tanh=.95):
    """Classify P(U_model), with open finite-logit and closed limit domains.

    Exact categories are separate from tolerance-feasible and non-saturated
    categories. The unchanged v1 executor differs from this ideal P by <=tol.
    """
    lower, upper = bounds[..., 0], bounds[..., 1]
    radius = scales/action_scale
    if (radius <= 0).any() or not torch.isfinite(radius).all():
        raise ValueError('Positive finite output radius required')
    def projected(value):return torch.maximum(torch.minimum(value, upper), lower)
    left, right = base-radius, base+radius
    image_lo, image_hi = projected(left), projected(right)
    feasible_label = (target >= lower-tolerance)&(target <= upper+tolerance)
    y = projected(target)
    closure = (y >= image_lo)&(y <= image_hi)&feasible_label
    finite = closure & torch.where(lower == upper, torch.ones_like(closure),
        torch.where(y == lower, left < lower,
        torch.where(y == upper, right > upper, (y > left)&(y < right))))
    stable_lo, stable_hi = projected(base-stable_tanh*radius), projected(base+stable_tanh*radius)
    stable = (y >= stable_lo)&(y <= stable_hi)&feasible_label
    within_tolerance = (y >= image_lo-tolerance)&(y <= image_hi+tolerance)&feasible_label
    return dict(unreachable=~closure, limit_only=closure&~finite,
        finite_saturated=finite&~stable, stable=stable,
        feasible_label=feasible_label, within_tolerance=within_tolerance,
        legacy_pre_range_exceeded=((target-base)*action_scale).abs()>scales+1e-7)


def actual_execution_expressibility(base, scales, action_scale, target, bounds, limits, *, tolerance=1e-6, stable_tanh=.95):
    """Exact real-arithmetic inverse sets of joint clamp followed by v1 P_tol.

    Within the tolerance band the map is identity. At a reachable endpoint its
    inverse additionally has a STRICT exterior ray (the threshold belongs to
    the identity branch). Joint-limit inverse rays are included separately.
    'limit_only' permits a closed tanh endpoint, not an invented closure of a
    discontinuous projection's inverse. Floating-point rounding is reported by
    the actual control numerical audit, not treated as a mathematical infinity.
    """
    lo, hi = bounds[..., 0], bounds[..., 1]
    jl, ju = limits[..., 0], limits[..., 1]
    radius = scales/action_scale
    if (radius<=0).any() or not torch.isfinite(radius).all():raise ValueError('Positive finite output radius required')
    identity_allowed=(target>=lo-tolerance)&(target<=hi+tolerance)&(target>=jl)&(target<=ju)
    def intersects(left,right,closed):
        ge=(target>=left) if closed else (target>left)
        le=(target<=right) if closed else (target<right)
        singleton=torch.where(target==jl,(left<=jl) if closed else (left<jl),
            torch.where(target==ju,(right>=ju) if closed else (right>ju),ge&le))
        # Degenerate joint limit projects every nonempty output domain to jl.
        singleton=torch.where(jl==ju,target==jl,singleton)
        identity=identity_allowed&singleton
        lower_ray=(target==lo)&(jl<lo-tolerance)&(left<lo-tolerance)
        upper_ray=(target==hi)&(ju>hi+tolerance)&(right>hi+tolerance)
        return identity|lower_ray|upper_ray
    finite=intersects(base-radius,base+radius,False)
    closed=intersects(base-radius,base+radius,True)
    stable=intersects(base-stable_tanh*radius,base+stable_tanh*radius,True)
    return dict(unreachable=~closed,limit_only=closed&~finite,finite_saturated=finite&~stable,stable=stable)
