"""Read-only H24 diagnosis identities and preregistered event definitions."""
import hashlib
import json
from pathlib import Path

import numpy as np

OLD = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/native_dp_v2/h24_parameterization_20261103_01')
ROOT = OLD.parent / 'h24_diagnosis_20261104_01'
TRAIN = 'r931005_1793209376_e02_p000059'
THRESHOLDS = dict(ee_divergence_mm=5., arm_divergence_rad=.01, hand_divergence_rad=.05,
    object_divergence_mm=5., object_rotation_deg=5., consecutive_ticks=3,
    object_contact_N=.1, prefix_joint_tolerance_rad=1e-6, prefix_object_tolerance=1e-6,
    clipping_tolerance_rad=1e-6, movement_arm_rms_rad=.01)
INFERENCE = dict(inference_steps=16, clip_sample=False, overlap_steps=0, sampler_seed=7091)

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def first_sustained(values, threshold, count=3):
    hits = np.asarray(values) > threshold
    if len(hits) < count:
        return None
    indices = np.flatnonzero(np.convolve(hits.astype(int), np.ones(count, int), 'valid') == count)
    return int(indices[0]) if len(indices) else None

def first_true(values):
    indices = np.flatnonzero(np.asarray(values).reshape(-1))
    return int(indices[0]) if len(indices) else None

def contact_force(trace):
    force = np.linalg.norm(trace['object_contact'], axis=-1)
    return force.reshape(len(force), -1).max(-1)

def events(trace, result):
    return dict(contact_tick=first_sustained(contact_force(trace), THRESHOLDS['object_contact_N']),
        lift_tick=first_true(trace['lifted']), subgoal_tick=first_true(trace['subgoals'] > 0),
        alignment_tick=first_true(trace['alignment']),
        fall_tick=len(trace['joint'])-1 if result.get('reasons', {}).get('fall') else None,
        success_tick=len(trace['joint'])-1 if result.get('success') else None)

def prefix_feedback(controller, transition):
    """Postprocessed expert target is the teacher-forced position history label."""
    target = transition['target_after_rad'].detach().cpu().clone()
    controller.command = target
    controller.observe_execution(dict(transition, position_applied_rad=target))

def prefix_comparison(rows, reference, stop):
    errors = {}
    for key in ('command', 'target', 'joint', 'velocity', 'object', 'contact', 'object_contact'):
        actual = np.stack([r[key] for r in rows[:stop]])
        expected = reference[key][:stop]
        errors[key] = float(np.max(np.abs(actual-expected))) if len(expected) == stop else None
    passed = all(value is not None and value <= 1e-6 for value in errors.values())
    return dict(passed=passed, ticks=stop, maximum_absolute_errors=errors,
        missing='PhysX solver/contact warm-start is not serialized; observable prefix parity only')
