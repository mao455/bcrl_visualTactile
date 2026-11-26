"""Versioned proposal maps; the final common position executor is unchanged."""
import math
import torch

REACHABLE_TAIL_OUTPUT=dict(version='reachable_tanh_linear_tail_v1',tanh_threshold=.95,tail_span_gain=1.)
REACHABLE_CENTERED_OUTPUT=dict(version='reachable_nominal_centered_linear_v1',span_gain=1.)
REACHABLE_PREVIOUS_TARGET_OUTPUT=dict(version='reachable_previous_actual_target_linear_v1',span_gain=1.)
MATCHED_FINAL_HEAD_RESET=dict(version='matched_final_linear_zero_v1',module='policy.2',
    requires_training_before_deployment=True)

def validate_output_contract(value):
    if value is None:return None
    if value not in (REACHABLE_TAIL_OUTPUT,REACHABLE_CENTERED_OUTPUT,REACHABLE_PREVIOUS_TARGET_OUTPUT):raise ValueError('Unknown residual output contract')
    return dict(value)


def validate_head_reset(value):
    if value is None:return None
    if value!=MATCHED_FINAL_HEAD_RESET:raise ValueError('Unknown residual continuation head-reset contract')
    return dict(value)


def proposal_delta(raw,scales,action_scale,bounds,contract):
    if contract is None:return scales*raw.tanh()
    validate_output_contract(contract)
    if contract in (REACHABLE_CENTERED_OUTPUT,REACHABLE_PREVIOUS_TARGET_OUTPUT):raise ValueError('Centered output requires its explicit command anchor')
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


def centered_proposal(base,raw,bounds,contract,*,previous=None):
    if contract not in (REACHABLE_CENTERED_OUTPUT,REACHABLE_PREVIOUS_TARGET_OUTPUT):raise ValueError('Expected centered output contract')
    if bounds is None:raise ValueError('Centered output requires real conditional bounds')
    lower,upper=bounds[...,0],bounds[...,1]
    if contract==REACHABLE_PREVIOUS_TARGET_OUTPUT:
        if (previous is None or previous.shape!=raw.shape or not torch.isfinite(previous).all()
                or ((previous<lower)|(previous>upper)).any()):
            raise ValueError('Previous-target output requires the finite actual pre-action target inside its bounds')
        anchor=previous
    else:
        anchor=torch.maximum(torch.minimum(base,upper),lower)
    return anchor+(upper-lower)*raw
