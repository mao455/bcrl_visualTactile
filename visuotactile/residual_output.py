"""Versioned proposal maps; the final common position executor is unchanged."""
import math
import torch

REACHABLE_TAIL_OUTPUT=dict(version='reachable_tanh_linear_tail_v1',tanh_threshold=.95,tail_span_gain=1.)
REACHABLE_CENTERED_OUTPUT=dict(version='reachable_nominal_centered_linear_v1',span_gain=1.)

def validate_output_contract(value):
    if value is None:return None
    if value not in (REACHABLE_TAIL_OUTPUT,REACHABLE_CENTERED_OUTPUT):raise ValueError('Unknown residual output contract')
    return dict(value)


def proposal_delta(raw,scales,action_scale,bounds,contract):
    if contract is None:return scales*raw.tanh()
    validate_output_contract(contract)
    if contract==REACHABLE_CENTERED_OUTPUT:raise ValueError('Centered output requires nominal command')
    if bounds is None:raise ValueError('Reachable tail output requires real conditional bounds')
    threshold=contract['tanh_threshold'];limit=math.atanh(threshold)
    center=raw.clamp(-limit,limit)
    radius=scales/action_scale
    span=bounds[...,1]-bounds[...,0]
    # Identical to historical tanh in |tanh(z)| <= .95. Beyond it, a true
    # linear slope in radians covers every finite legal target, even when
    # tiny train-affine variance made the old physical residual radius tiny.
    slope=torch.maximum(radius*(1-threshold**2),span*contract['tail_span_gain'])
    return scales*center.tanh()+(slope*action_scale)*(raw-center)


def centered_proposal(base,raw,bounds,contract):
    if contract!=REACHABLE_CENTERED_OUTPUT:raise ValueError('Expected centered output contract')
    if bounds is None:raise ValueError('Centered output requires real conditional bounds')
    lower,upper=bounds[...,0],bounds[...,1]
    anchor=torch.maximum(torch.minimum(base,upper),lower)
    return anchor+(upper-lower)*raw
