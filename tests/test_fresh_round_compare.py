"""Fail-closed provenance and same-observation held-angle diagnostics, CPU only."""
import hashlib
import json
import subprocess

import h5py
import numpy as np
import pytest

from tools import tight_fresh_round_compare as review


def test_registered_source_drift_is_rejected(tmp_path):
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    source = tmp_path / 'source.py'
    source.write_text('value = 1\n')
    subprocess.run(['git', 'add', '.'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                    'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=tmp_path, text=True).strip()
    plan = dict(source_commit=commit, source_sha256={'source.py': review.sha256(source)})
    assert review.check_sources(plan, tmp_path)['verified_source_files'] == 1
    source.write_text('value = 2\n')
    with pytest.raises(ValueError, match='Frozen file changed'):
        review.check_sources(plan, tmp_path)
    plan['source_commit'] = '0' * 40
    with pytest.raises(ValueError, match='Frozen commit differs'):
        review.check_sources(plan, tmp_path)


def test_angles_use_same_tick_and_active_first_goal(tmp_path, monkeypatch):
    # A later goal has a smaller angle, but must not enter first-goal comparison.
    pose = np.array([[0, 0, 0, 2, 0, 0, 0],
                     [1, 0, 0, 1, 0, 0, 1],
                     [2, 0, 0, 1, 0, 0, 0]], dtype=float)
    goal = np.array([[0, 0, 0, 1, 0, 0, 0]] * 3, dtype=float)
    with h5py.File(tmp_path / 'numeric.h5', 'w') as stream:
        stream['episode/transition/diagnostics_after/object_pose_w'] = pose
        stream['episode/transition/diagnostics_before/goal_pose_w'] = goal
        stream['episode/transition/diagnostics_before/successes'] = [0, 0, 1]
        stream['episode/transition/object_contact_force_w'] = np.zeros((3, 2, 5, 3))
        stream['episode/transition/native_keypoints_max_dist'] = [.3, .1, 0]
        stream['episode/transition/native_success_diagnostic/insertion_tolerance_m'] = [.015] * 3
    monkeypatch.setattr(review, 'causal_contact_height', lambda p, f: (np.ones(3, dtype=bool), 0.), raising=False)
    value = review.held_orientation(tmp_path)
    assert value['eligible_contact_height_windows'] == 2
    assert value['smallest_root_angle']['completed_tick'] == 1
    assert value['smallest_root_angle']['root_angle_rad'] == 0  # Quaternion scale is normalized.
    assert value['smallest_root_angle']['native_keypoint_distance_m'] == .3  # Not another tick's minimum.
    monkeypatch.setattr(review, 'causal_contact_height', lambda p, f: (np.zeros(3, dtype=bool), 0.))
    assert review.held_orientation(tmp_path) == {'eligible_contact_height_windows': 0}


def test_existing_analysis_is_never_overwritten(tmp_path):
    (tmp_path / 'autonomy.json').write_text('{}')
    with pytest.raises(FileExistsError, match='Preserve previous'):
        review.run(tmp_path, tmp_path, tmp_path, tmp_path, tmp_path)


def test_explicit_paths_and_json_digest(tmp_path):
    arguments = review.parser().parse_args([
        '--source-root', 'source', '--candidate-root', 'candidate',
        '--source-tree', 'old', '--candidate-tree', 'new', '--output', 'report'])
    assert str(arguments.candidate_tree) == 'new'
    destination = tmp_path / 'report.json'
    review.atomic_json(destination, {'value': .015})
    assert json.loads(destination.read_text()) == {'value': .015}
    assert review.sha256(destination) == hashlib.sha256(destination.read_bytes()).hexdigest()
