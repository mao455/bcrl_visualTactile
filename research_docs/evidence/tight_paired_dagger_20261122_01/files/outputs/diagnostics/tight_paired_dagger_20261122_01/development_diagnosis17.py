"""Read saved development data only; no model import, policy call, or simulation."""
from pathlib import Path
import csv
import hashlib
import json
import os
import time

import h5py
import numpy as np

ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
DEV = ROOT / 'outputs/visuotactile/run/tight_paired_dagger_20261122_01/evaluation/development'
OUT = ROOT / 'outputs/diagnostics/tight_paired_dagger_20261122_01'
ROLES = ('A_new', 'B_new', 'A_parent', 'B_parent')
START = time.time()
INPUTS = {}


def record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    INPUTS[str(path.relative_to(ROOT))] = digest.hexdigest()
    return path


def read_json(path):
    return json.loads(record(path).read_text())


def rms(value):
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64)**2)))


def first(mask):
    found = np.flatnonzero(mask)
    return int(found[0]) if len(found) else None


def quaternion_angle_degrees(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return float(np.degrees(2*np.arccos(np.clip(abs(a@b), 0, 1))))


def relative_position(obj, palm):
    quat = palm[3:].astype(np.float64)
    quat /= np.linalg.norm(quat)
    quat[1:] *= -1
    value = obj[:3] - palm[:3]
    cross = 2*np.cross(quat[1:], value)
    return value + quat[0]*cross + np.cross(quat[1:], cross)


summary = read_json(DEV / 'summary/primary_summary.json')
with record(DEV / 'summary/rollouts.csv').open() as stream:
    registered = list(csv.DictReader(stream))
assert summary['cohort'] == 'development' and summary['n'] == 8
assert len(registered) == 32
episodes = []
arrays = {}
for row in registered:
    directory = Path(row['result_path']).parent
    assert directory.is_relative_to(DEV)
    result_path = directory / 'result.json'
    entry = dict(seed=int(row['seed']), role=row['role'], family=row['family'],
                 summary_result_path=str(result_path.relative_to(ROOT)))
    if not result_path.exists():
        assert row['missing_result'] == 'True'
        entry.update(software_complete=False, geometry_observed=False,
                     stage='software_unfinished_or_missing', primary_success=False)
        episodes.append(entry)
        continue
    result = read_json(result_path)
    numeric = record(directory / 'numeric.h5')
    with h5py.File(numeric, 'r') as handle:
        tr = handle['episode/transition']
        names = ('diagnostics_after/object_pose_w', 'diagnostics_after/palm_body_pose_w',
                 'diagnostics_before/goal_pose_w', 'native_keypoints_max_dist',
                 'object_contact_force_w', 'diagnostics_before/successes',
                 'diagnostics_after/joint_position_rad', 'target_after_rad',
                 'submitted_action', 'position_requested_rad', 'position_applied_rad',
                 'control_trace/applied_position_target_rad', 'actuator_substep_target_rad')
        data = {name: tr[name][:] for name in names}
    obj = data['diagnostics_after/object_pose_w']
    goal = data['diagnostics_before/goal_pose_w']
    keypoints = data['native_keypoints_max_dist']
    count = (np.linalg.norm(data['object_contact_force_w'], axis=-1) > .1).sum(-1).min(-1)
    settled = float(np.median(obj[24:48, 2]))
    good = (count >= 2) & (obj[:, 2] >= settled + .02)
    held = np.zeros(len(obj), dtype=bool)
    run = 0
    contact_run = longest_contact_run = 0
    for i, value in enumerate(good):
        run = run + 1 if value else 0
        held[i] = run >= 12 and i >= 47
        contact_run = contact_run + 1 if count[i] >= 2 else 0
        longest_contact_run = max(longest_contact_run, contact_run)
    gate = (keypoints <= .015) & (data['diagnostics_before/successes'] == 0)
    near = held & (np.linalg.norm(obj[:, :3] - goal[:, :3], axis=-1) <= .06)
    assert len(obj) == result['steps']
    assert bool(held.any()) == (row['stable_hold'] == 'True')
    assert bool(gate.any()) == (row['first_15mm_gate'] == 'True')
    complete = bool(result['software_complete'] and result['legal'])
    native = bool(result['native_subgoals'] > 0)
    success = bool(result['success'] and complete)
    if not complete:
        stage = 'software_unfinished_or_missing'
    elif success:
        stage = 'strict_success'
    elif native:
        stage = 'native_first_subgoal_without_insertion'
    elif gate.any():
        stage = 'entered_gate_without_native_confirmation'
    elif held.any():
        stage = 'stable_held_never_gate'
    else:
        stage = 'never_stable_held_never_gate'
    entry.update(software_complete=complete, geometry_observed=True, primary_success=success,
                 stage=stage, steps=result['steps'], termination=result['reasons'],
                 stable_held12=bool(held.any()), stable_held12_ticks=int(held.sum()),
                 longest_both_substeps_two_finger_run_ticks=int(longest_contact_run),
                 first_held12_tick=first(held), first_15mm_gate=bool(gate.any()),
                 native_first_subgoal=native, stable_held_root_near60mm=bool(near.any()),
                 stable_held_root_near60mm_ticks=int(near.sum()),
                 minimum_native_keypoint_mm=float(keypoints.min()*1000),
                 numeric_path=str(numeric.relative_to(ROOT)))
    episodes.append(entry)
    arrays[(entry['role'], entry['seed'])] = (data, held, count, settled)

stages = ('software_unfinished_or_missing', 'never_stable_held_never_gate',
          'stable_held_never_gate', 'entered_gate_without_native_confirmation',
          'native_first_subgoal_without_insertion', 'strict_success')
counts = []
for role in ROLES:
    entries = [row for row in episodes if row['role'] == role]
    counts.append(dict(role=role, n=8, **{stage: sum(row['stage'] == stage for row in entries)
                                      for stage in stages},
                       stable_held_root_near60mm=sum(row.get('stable_held_root_near60mm', False)
                                                    for row in entries)))
    assert sum(counts[-1][stage] for stage in stages) == 8

cases = []
for role in ('A_new', 'B_new'):
    candidates = [row for row in episodes if row['role'] == role and row['software_complete']
                  and not row['primary_success']]
    stage_count = {stage: sum(row['stage'] == stage for row in candidates) for stage in stages[1:-1]}
    dominant = max(stage_count, key=lambda stage: stage_count[stage])
    chosen = min((row for row in candidates if row['stage'] == dominant), key=lambda row: row['seed'])
    data, held, count, settled = arrays[(role, chosen['seed'])]
    obj = data['diagnostics_after/object_pose_w']
    palm = data['diagnostics_after/palm_body_pose_w']
    goal = data['diagnostics_before/goal_pose_w']
    command = data['target_after_rad']
    joint = data['diagnostics_after/joint_position_rad']
    keypoints = data['native_keypoints_max_dist']
    names = read_json(ROOT / Path(chosen['numeric_path']).parent / 'metadata.json')['joint_order']
    if held.any():
        start = int(np.argmin(keypoints))
        window_rule = 'minimum recorded native keypoint tick through next23 ticks; retrospective only'
    else:
        start = 48
        window_rule = 'first24 ticks after settled-reference interval [24,48)'
    stop = min(start + 23, len(obj) - 1)
    sl = slice(start, stop+1)
    du = command[stop] - command[start]
    dq = joint[stop] - joint[start]
    snapshots = []
    ticks = sorted(set([start, min(start+1, stop), min(start+2, stop), stop, len(obj)-1]
                       + ([first(held)] if held.any() else [])))
    for tick in ticks:
        snapshots.append(dict(tick=tick, held12=bool(held[tick]),
            minimum_contact_fingers_both_substeps=int(count[tick]),
            object_height_over_settled_mm=float((obj[tick, 2]-settled)*1000),
            native_keypoint_max_mm=float(keypoints[tick]*1000),
            root_goal_distance_mm=float(np.linalg.norm(obj[tick, :3]-goal[tick, :3])*1000),
            root_goal_quaternion_angle_degrees=quaternion_angle_degrees(obj[tick, 3:], goal[tick, 3:]),
            same_transition_command_rad=command[tick].tolist(),
            same_transition_actual_joint_rad=joint[tick].tolist(),
            object_pose_w=obj[tick].tolist(), palm_pose_w=palm[tick].tolist()))
    chain = {name: float(np.abs(data[name]-command).max()) for name in
             ('submitted_action', 'position_requested_rad', 'position_applied_rad',
              'control_trace/applied_position_target_rad')}
    chain['actuator_substep_target_rad'] = float(np.abs(data['actuator_substep_target_rad']-command[:, None]).max())
    cases.append(dict(role=role, seed=chosen['seed'], stage=chosen['stage'],
        window_rule=window_rule, window_start_tick=start, window_end_tick=stop,
        actual_command_chain_maximum_difference_rad=chain,
        arm_command_change_rms_rad=rms(du[:7]), hand_command_change_rms_rad=rms(du[7:]),
        arm_actual_joint_change_rms_rad=rms(dq[:7]), hand_actual_joint_change_rms_rad=rms(dq[7:]),
        arm_command_to_q_after_rmse_rad=rms(command[sl, :7]-joint[sl, :7]),
        hand_command_to_q_after_rmse_rad=rms(command[sl, 7:]-joint[sl, 7:]),
        both_substeps_two_finger_ticks=int((count[sl] >= 2).sum()),
        object_displacement_xyz_mm=((obj[stop, :3]-obj[start, :3])*1000).tolist(),
        palm_displacement_xyz_mm=((palm[stop, :3]-palm[start, :3])*1000).tolist(),
        object_relative_palm_displacement_mm=float(np.linalg.norm(
            relative_position(obj[stop], palm[stop])-relative_position(obj[start], palm[start]))*1000),
        largest_hand_command_changes=[dict(joint=names[j], command_delta_rad=float(du[j]),
            actual_joint_delta_rad=float(dq[j])) for j in np.argsort(abs(du[7:]))[-4:][::-1]+7],
        snapshots=snapshots))

for path in ('tools/tight_paired_evaluate.py', 'visuotactile/sim/environment.py',
             'outputs/diagnostics/tight_paired_dagger_20261122_01/REPORT_FACTS10.json',
             str(Path(__file__).relative_to(ROOT))):
    record(ROOT / path)
output = dict(version='development_only_failure_diagnosis17_v1',
    input_scope='fixed8-family development aggregate, result JSON, numeric HDF5, selected metadata and source schemas; prior non-outcome REPORT_FACTS10 collection-coverage context',
    main64_test_outcomes_read=False, weight_or_cache_payloads_read=False,
    selection_rule='for each new role: most numerous completed failure stage, then smallest seed; no test input',
    tick_index='zero-based completed control transitions; after-state; native first-subgoal confirmation is cumulative10 ticks',
    stable_held12_rule=summary['stable_hold_rule'],
    near60mm_rule='held12 and object-root to current first-goal-root distance <=60mm; auxiliary diagnostic, not15mm keypoint gate',
    native_gate_rule='recorded native_keypoints_max_dist<=15mm with pretransition native first-goal count0',
    limitations=['Stages are observed diagnostic categories, not identified causes.',
        'held12 and native gate need not be nested; assignment prioritizes native milestone before held12.',
        'No recorded expert action counterfactual is available; zero expert shadow is a placeholder.',
        'same-transition target-to-q_after difference includes normal dynamics and contact; it is not a model target error.',
        'Eight development families and one software-missing family per role do not establish population ranking.',
        'Main64 results are required for final efficacy conclusions; no checkpoint or runtime changed.'],
    stage_counts=counts, episodes=episodes, selected_cases=cases,
    recommendation_conditional_on_main='If main64 also remains weak, validate and admit genuinely stable-held near-hole never-entry expert corrections before another training round; current development contains this stage but prior collection audit found zero takeovers meeting held12.',
    input_sha256=dict(sorted(INPUTS.items())),
    resources=dict(cpu_threads_limit=2, environment_threads={key: os.environ.get(key) for key in
        ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        analysis_wall_seconds=time.time()-START, optimizer_updates=0,
        policy_forward_calls=0, gpu_calls=0, simulator_starts=0, new_trajectory_data_bytes=0))
target = OUT / 'DEVELOPMENT_DIAGNOSIS17.json'
with target.open('w') as stream:
    json.dump(output, stream, ensure_ascii=False, indent=2)
print(json.dumps(dict(output=str(target), counts=counts,
                     cases=[dict(role=row['role'], seed=row['seed'], window=[row['window_start_tick'], row['window_end_tick']])
                            for row in cases], inputs=len(INPUTS),
                     resources=output['resources']), ensure_ascii=False, indent=2))
