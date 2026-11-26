"""CPU-only recorded grasp loss and actual correction stage coverage; no policy forward."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import h5py
from visuotactile.takeover_trigger import stable_windows
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_phases import ROOT_POSE_SEMANTICS

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def rms(value):
    return float(np.sqrt(np.mean(value * value)))

def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0] + 1) if len(indices) else None

def build(root):
    root = Path(root).resolve()
    plan = json.loads((root / 'development/plan.json').read_text())
    planned = {c['name'] for c in plan['cases']}
    actual = {p.name for p in (root / 'development').glob('*_development_*') if p.is_dir()}
    if planned != actual:
        raise ValueError('Recorded opportunity directories differ from registered plan')
    rows = []
    for directory in sorted((root / 'development').glob('*_development_*')):
        result = json.loads((directory / 'result.json').read_text())
        if not result.get('legal') or not result.get('software_complete'):
            raise ValueError('Slip report requires complete legal trials; retain other outcomes separately')
        with h5py.File(directory / 'numeric.h5') as f:
            t = f['episode/transition']
            d = t['diagnostics_after']
            obj = d['object_pose_w'][:]
            palm = d['palm_body_pose_w'][:]
            goal = d['goal_pose_w'][:]
            q = d['joint_position_rad'][:]
            target = t['target_after_rad'][:]
            force = t['object_contact_force_w'][:]
            prev = t['target_before_rad'][:]
            obsforce = f['episode/obs/contact_w_N'][:]
        metadata = json.loads((directory / 'metadata.json').read_text())
        names = metadata['joint_order']
        n = len(obj)
        if n != result['steps'] or n < 48:
            raise ValueError('Complete recorded tick count and settled reference required')
        height = float(np.median(obj[24:48, 2]))
        stable, contact = stable_windows(obj, palm, force, height)
        fn = np.linalg.norm(force, axis=-1)
        two = (contact.sum(-1) >= 2).all(-1)
        inverse = palm[:, 3:].copy()
        inverse[:, 1:] *= -1
        qv = inverse[:, 1:]
        v = obj[:, :3] - palm[:, :3]
        cross = 2 * np.cross(qv, v)
        rel = v + inverse[:, :1] * cross + np.cross(qv, cross)
        rq = np.concatenate((inverse[:, :1] * obj[:, 3:4] - (qv * obj[:, 4:]).sum(-1, keepdims=True), inverse[:, :1] * obj[:, 4:] + obj[:, 3:4] * qv + np.cross(qv, obj[:, 4:])), axis=-1)
        dist = np.linalg.norm(obj[:, :3] - goal[:, :3], axis=-1)
        angle = 2 * np.arccos(np.clip(np.abs((obj[:, 3:] * goal[:, 3:]).sum(-1)), 0, 1))

        def snapshot(tick):
            i = tick - 1
            return dict(tick=tick, force_max_per_finger_N=fn[i].max(0).tolist(), observed_force_N=np.linalg.norm(obsforce[i], axis=-1).tolist(), target=target[i].tolist(), actual=q[i].tolist(), arm_track=rms(target[i, :7] - q[i, :7]), hand_track=rms(target[i, 7:] - q[i, 7:]), object_height_over_settled_m=float(obj[i, 2] - height), goal_dist_m=float(dist[i]), goal_angle_rad=float(angle[i]), stable=bool(stable[i]), two=bool(two[i]))

        def window(a, b):
            lo = a - 1
            hi = b
            sl = slice(lo, hi)
            delta = target[hi - 1] - target[lo]
            rank = np.argsort(abs(delta[7:]))[-6:][::-1] + 7
            return dict(start_tick=a, end_tick=b, arm_target_change_rms=rms(delta[:7]), hand_target_change_rms=rms(delta[7:]), largest_hand_target_changes=[dict(joint=names[j], delta_rad=float(delta[j]), actual_delta_rad=float(q[hi - 1, j] - q[lo, j])) for j in rank], arm_tracking_rmse=rms(target[sl, :7] - q[sl, :7]), hand_tracking_rmse=rms(target[sl, 7:] - q[sl, 7:]), mean_object_force_per_finger_N=fn[sl].mean((0, 1)).tolist(), contact_finger_fraction=contact[sl].mean((0, 1)).tolist(), both_two_ticks=int(two[sl].sum()), palm_translation_m=float(np.linalg.norm(palm[hi - 1, :3] - palm[lo, :3])), object_relative_palm_drift_m=float(np.linalg.norm(rel[hi - 1] - rel[lo])), relative_angle_change_rad=float(2 * np.arccos(np.clip(abs(rq[hi - 1] @ rq[lo]), 0, 1))), goal_distance_start_end=[float(dist[lo]), float(dist[hi - 1])])
        contact_tick = first(contact.any((1, 2)))
        r = dict(case=directory.name, pose_reference_semantics=ROOT_POSE_SEMANTICS, numeric_sha256=sha(directory / 'numeric.h5'), result_sha256=sha(directory / 'result.json'), metadata_sha256=sha(directory / 'metadata.json'), contact=contact_tick, stable_window_ticks=(np.flatnonzero(stable) + 1).tolist(), first20_after_contact=window(contact_tick, min(contact_tick + 19, n)) if contact_tick else None, post_contact_tracking_arm=rms((target - q)[contact_tick - 1:, :7]) if contact_tick else None)
        if stable.any():
            ends = np.flatnonzero(stable)
            last = int(ends[-1] + 1)
            r['snapshots'] = [snapshot(x) for x in sorted(set([int(ends[0] + 1), last, min(last + 1, n), min(last + 6, n), min(last + 12, n), min(last + 24, n), min(last + 48, n), n]))]
            r['windows'] = [window(max(1, last - 11), last), window(last, min(last + 12, n)), window(last, min(last + 24, n)), window(last, min(last + 48, n))]
            r['lost_window_conditions'] = []
            for stop in range(last + 1, min(last + 25, n + 1)):
                sl = slice(stop - 12, stop)
                drift = float(np.linalg.norm(rel[sl] - rel[stop - 12], axis=-1).max())
                rot = float((2 * np.arccos(np.clip(abs((rq[sl] * rq[stop - 12]).sum(-1)), 0, 1))).max())
                r['lost_window_conditions'].append(dict(tick=stop, two=bool(two[sl].all()), height=bool((obj[sl, 2] >= height + 0.02).all()), drift_m=drift, rotation_rad=rot))
        rows.append(r)
    data = json.loads((root / 'A/state_data/data.json').read_text())
    other = json.loads((root / 'B/state_data/data.json').read_text())
    if data['splits'] != other['splits']:
        raise ValueError('A/B population differs')
    coverage = []
    for eid in data['splits']['train']:
        e = data['entries'][eid]
        if e.get('trajectory_kind') != 'expert_takeover_v1':
            continue
        if sha(Path(e['path'])) != e['source_sha256']:
            raise ValueError('Actual accepted correction bytes changed: ' + eid)
        if any((e[key] != other['entries'][eid][key] for key in ('path', 'source_sha256', 'family'))):
            raise ValueError('A/B accepted source differs')
        with h5py.File(e['path']) as f:
            t = f['episodes/' + eid + '/transition']
            d = t['diagnostics_after']
            obj = d['object_pose_w'][:]
            palm = d['palm_body_pose_w'][:]
            goal = d['goal_pose_w'][:]
            force = t['object_contact_force_w'][:]
            src = t['dagger/control_source'][:]
            mask = t['dagger/expert_label_valid'][:].astype(bool)
            sub = d['successes'][:]
        height = float(np.median(obj[24:48, 2]))
        stable, _ = stable_windows(obj, palm, force, height)
        dist = np.linalg.norm(obj[:, :3] - goal[:, :3], axis=-1)
        ang = 2 * np.arccos(np.clip(abs((obj[:, 3:] * goal[:, 3:]).sum(-1)), 0, 1))
        student = src == 0
        sm = stable & student
        ix = np.flatnonzero(sm)
        before = mask & (sub == 0)
        near = dist < 0.03
        coverage.append(dict(id=eid, round=e['round_index'], role=e['source_group'], takeover_tick=e['takeover_tick'], legal_expert_ticks=int(mask.sum()), student_stable_ticks=int(sm.sum()), first_student_stable=int(ix[0] + 1) if len(ix) else None, student_stable_at_takeover=bool(stable[e['takeover_tick'] - 1]), expert_stable_ticks=int((stable & mask).sum()), expert_first_goal_near3cm_ticks=int((before & near).sum()), expert_first_goal_near3cm_angle_lt0p5_ticks=int((before & near & (ang < 0.5)).sum()), expert_after_first_subgoal_ticks=int((mask & (sub > 0)).sum()), source_sha256=e['source_sha256']))
    return dict(version='recorded_slip_review_v1', source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), source_sha256={str(p): sha(p) for p in (Path(__file__), Path('visuotactile/takeover_trigger.py'))}, input_sha256={str(root / p): sha(root / p) for p in ('development/plan.json', 'development/results.json', 'A/state_data/data.json', 'B/state_data/data.json')}, tick_index='1-based completed control transitions; stable entries are ends of past12tick windows', height_reference='median object root z in completed ticks25-48; diagnostic only', joint_target_scope='actual same-transition target_after_rad, not raw prediction or next reached state', contact_finger_order=['index', 'middle', 'ring', 'thumb', 'pinky'], cases=rows, correction_phase_ticks=coverage, coverage_summary={key: sum((r[key] for r in coverage)) for key in ('legal_expert_ticks', 'student_stable_ticks', 'expert_stable_ticks', 'expert_first_goal_near3cm_ticks', 'expert_first_goal_near3cm_angle_lt0p5_ticks', 'expert_after_first_subgoal_ticks')}, coverage_episodes=dict(accepted=len(coverage), student_previously_stable=sum((r['student_stable_ticks'] > 0 for r in coverage)), student_stable_at_switch=sum((r['student_stable_at_takeover'] for r in coverage))), diagnostic_thresholds=dict(near_goal_distance_m=0.03, near_goal_angle_rad=0.5), interpretation='Descriptive only: drift, contact and commands do not identify causal failure. Near-goal thresholds do not alter native success. Expert suffixes are not autonomous performance. Different seeds or recorded states do not establish exact hidden-state replay.', model_forward_calls=0, optimizer_updates=0, simulator_starts=0, test_used=False)

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Completed round root with A/state_data and development')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, build(args.root))
if __name__ == '__main__':
    main()
