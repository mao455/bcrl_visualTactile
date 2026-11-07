"""Opt-in expert target progression and physical replay time parametrization.

These functions never change physics dt, recurrent state or native success.
Inputs to progression are AFTER the one native delay/EMA/limit pipeline.
"""
import math
import numpy as np
import torch


def progress_target(previous, native, arm=1., hand=1.):
    if not (0 < arm <= 1 and 0 < hand <= 1):
        raise ValueError('Progress factors must be in (0,1]')
    if previous.shape != native.shape or native.shape[-1] != 29:
        raise ValueError('Expected matching canonical 29D targets')
    if not torch.isfinite(native).all() or not torch.isfinite(previous).all():
        raise ValueError('Nonfinite target')
    if arm == hand == 1:
        return native  # Identity must not introduce even one rounding operation.
    result = native.clone()
    for selection, factor in ((slice(0,7),arm),(slice(7,29),hand)):
        if factor != 1:
            result[...,selection] = previous[...,selection] + factor * (native[...,selection]-previous[...,selection])
    return result


def cap_hand_target(previous, target, maximum_rad_s=None, dt=1/60):
    """Once per native tick, bound each Hand target increment from last issued.

    None is the exact old path. This is an actuator-request slew limit, not a
    guarantee about physical joint or task speed. Convex clipping preserves
    native position bounds when both incoming endpoints are within them.
    """
    if maximum_rad_s is None:
        return target
    if previous.shape != target.shape or target.shape[-1] != 29:
        raise ValueError('Expected matching canonical29 targets')
    if not math.isfinite(dt) or dt<=0:
        raise ValueError('Positive finite native control dt required')
    limits=torch.as_tensor(maximum_rad_s,device=target.device,dtype=target.dtype)
    if limits.shape not in (torch.Size([]),torch.Size([22])) or not torch.isfinite(limits).all() or not (limits>0).all():
        raise ValueError('Hand cap must be positive finite scalar or22 canonical rad/s values')
    if not torch.isfinite(previous).all() or not torch.isfinite(target).all():
        raise ValueError('Nonfinite target')
    delta=target[...,7:]-previous[...,7:];bound=limits*dt
    result=target.clone()
    # Preserve every uncapped element bytewise, including subtraction/addition
    # rounding. Arm commands do not pass through the limiter arithmetic.
    result[...,7:]=torch.where(delta.abs()>bound,
        previous[...,7:]+torch.maximum(torch.minimum(delta,bound),-bound),target[...,7:])
    return result


def retime_targets(before, after, stretch):
    """q(0)=initial applied target; q(k)=original postprocessed target[k-1].

    Each new control transition k requests q(k/stretch), not q((k-1)/stretch).
    The final sample holds q(T) if ceil(T*stretch) is fractional. No state labels
    or images are synthesized; these commands must be executed in physics.
    """
    after=np.asarray(after);before=np.asarray(before)
    if not math.isfinite(stretch) or stretch < 1:
        raise ValueError('Only finite slowdown >=1 is supported')
    if after.ndim!=2 or after.shape[1]!=29 or before.shape!=(29,) or not len(after):
        raise ValueError('Expected initial29 and nonempty T x29 sequence')
    if not np.isfinite(after).all() or not np.isfinite(before).all():raise ValueError('Nonfinite replay')
    if stretch==1:return after.copy()
    points=np.concatenate([before[None],after],axis=0)
    times=np.minimum(np.arange(1,math.ceil(len(after)*stretch)+1,dtype=np.float64)/stretch,len(after))
    lower=np.floor(times).astype(np.int64);upper=np.minimum(lower+1,len(after));alpha=(times-lower)[:,None]
    return (points[lower]*(1-alpha)+points[upper]*alpha).astype(after.dtype)


def reliability(rows, baseline, minimum=10, success_floor=.8):
    """Preregistered expansion gate, with failures retained in the denominator."""
    valid=[r for r in rows if r.get('legal') and r.get('software_complete')]
    refs={r['identity']:r for r in baseline if r.get('success')}
    successes=[r for r in valid if r['success']]
    paired=[r['duration_sim_s']/refs[r['identity']]['duration_sim_s'] for r in successes if r['identity'] in refs]
    median=float(np.median([r['duration_sim_s'] for r in successes])) if successes else None
    ratio=float(np.median(paired)) if paired else None
    rate=len(successes)/len(valid) if valid else None
    return dict(valid=len(valid),success=len(successes),rate=rate,median_success_s=median,
        paired_success_count=len(paired),median_paired_duration_ratio=ratio,
        passed=bool(len(valid)>=minimum and rate>=success_floor and median is not None and 5<=median<=7
                    and ratio is not None and 1.4<=ratio<=2.2),
        rule=dict(minimum=minimum,success_floor=success_floor,median_success_s=[5,7],paired_duration_ratio=[1.4,2.2]),
        caveat='Duration ratio conditions on both successes; all valid failures remain in reported success rate')
