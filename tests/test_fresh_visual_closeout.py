import json

import h5py
import numpy as np
import pytest

from tools.tight_fresh_visual_closeout import contact_motion
from visuotactile.compact import sha256


def recorded(tmp_path):
    pose = np.zeros((80, 7)); pose[:, 6] = 1.; pose[48:, 2] = .03
    # Large in-hand rotation must not remove contact/height eligibility.
    angle = np.linspace(0, 2, 80)
    pose[:, 3] = np.sin(angle / 2); pose[:, 6] = np.cos(angle / 2)
    force = np.zeros((80, 2, 5, 3)); force[:, :, :2, 2] = 1.
    goal = pose.copy(); goal[:, 0] = .1
    subgoals = np.zeros(80, dtype=int); subgoals[70:] = 1
    distance = np.ones(80)*.2; distance[64] = .04; distance[75] = .01
    with h5py.File(tmp_path/'numeric.h5', 'w') as stream:
        for name, value in {
            'diagnostics_after/object_pose_w':pose, 'diagnostics_before/goal_pose_w':goal,
            'object_contact_force_w':force, 'native_keypoints_max_dist':distance,
            'native_success_diagnostic/insertion_tolerance_m':np.ones(80)*.015,
            'diagnostics_before/successes':subgoals}.items():
            stream.create_dataset('episode/transition/'+name, data=value)
    (tmp_path/'result.json').write_text(json.dumps({'success':False}))
    return dict(numeric_sha256=sha256(tmp_path/'numeric.h5'),
        result_sha256=sha256(tmp_path/'result.json'), steps=80)


def test_rotation_allowed_and_best_snapshot_uses_same_tick_original_goal(tmp_path):
    result = contact_motion(tmp_path, recorded(tmp_path))
    assert result['first_contact_height_window_end_tick'] == 60
    assert result['longest_contact_and_height_run_ticks'] == 32
    best = result['first_goal_best_keypoint_during_contact_height_window']
    assert best['completed_tick'] == 65  # The closer later target is excluded.
    assert best['root_angle_rad'] == pytest.approx(0., abs=1e-7)
    assert best['root_distance_m'] == pytest.approx(.1)
    assert best['native_keypoint_distance_m'] == pytest.approx(.04)


def test_changed_numeric_evidence_rejected(tmp_path):
    expected = recorded(tmp_path)
    expected['numeric_sha256'] = 'wrong'
    with pytest.raises(ValueError, match='trajectory changed'):
        contact_motion(tmp_path, expected)
