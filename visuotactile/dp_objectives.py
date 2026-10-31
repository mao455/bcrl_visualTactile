"""Explicit historical A and default formal B diffusion objectives."""
import torch


DEFAULT_DP_OBJECTIVE = 'B'


def masked_objectives(prediction, target, valid):
    if prediction.shape != target.shape or prediction.shape[-1] != 29 or valid.shape != prediction.shape[:-1]:
        raise ValueError('Expected matching canonical29 predictions/targets and position validity')
    error = (prediction.float()-target.float()).square()
    error = torch.where(valid[...,None],error,0.)
    denominator = valid.sum().clamp_min(1)
    arm = error[...,:7].mean(-1).sum()/denominator
    hand = error[...,7:].mean(-1).sum()/denominator
    return dict(A=error.mean(-1).sum()/denominator,B=.5*(arm+hand),arm=arm,hand=hand)


def recorded_objective(configuration):
    objective = configuration.get('loss', 'A')
    if objective not in ('A', 'B'):
        raise ValueError('Unknown stored DP loss identity')
    return objective
