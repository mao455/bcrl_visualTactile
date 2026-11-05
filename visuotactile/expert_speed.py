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
