"""Read-only, development-only physical comparison, gated on primary delivery.

Run only after the frozen 96-rollout primary table is complete.  No inference,
simulation, expert queries, training, or source mutation occurs here.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import shutil

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
DEV = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01/development_20261122_01'
PUBLIC = PROJECT/'research_docs/evidence/tight_entry_final24_20261122_01/development_cases'
OUTPUT = PROJECT/'outputs/diagnostics/tight_entry_final24_development_cases_20261122_01'
MARKER = PUBLIC.parent/'primary_summary.json'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def qconj(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1
    return q


def qmul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate((aw*bw-(av*bv).sum(-1, keepdims=True),
                           aw*bv+bw*av+np.cross(av, bv)), axis=-1)


def rotate(q, v):
    vq = np.concatenate((np.zeros_like(v[..., :1]), v), axis=-1)
    return qmul(qmul(q, vq), qconj(q))[..., 1:]


def relative(parent, child):
    inv = qconj(parent[..., 3:])
    return np.concatenate((rotate(inv, child[..., :3]-parent[..., :3]),
                           qmul(inv, child[..., 3:])), axis=-1)


def angle(q, p):
    q = q/np.linalg.norm(q, axis=-1, keepdims=True)
    p = p/np.linalg.norm(p, axis=-1, keepdims=True)
    return np.rad2deg(2*np.arccos(np.clip(np.abs((q*p).sum(-1)), 0, 1)))


def first_run(mask, count=3):
    mask = np.asarray(mask, dtype=bool)
    if len(mask) < count:
        return None
    hits = np.flatnonzero(np.convolve(mask.astype(int), np.ones(count, int), 'valid') == count)
    return int(hits[0]) if len(hits) else None


def first(mask):
    hits = np.flatnonzero(mask)
    return int(hits[0]) if len(hits) else None


def load(directory):
    from visuotactile.deviation_takeover import DeviationTrigger, profile
    from evaluation.problems import PROBLEM_REGISTRY
    from visuotactile.success_audit import keypoint_distance
    with h5py.File(directory/'numeric.h5', 'r') as f:
        tr = f['episode/transition']
        after = tr['diagnostics_after']
        base = after['base_pose_w'][:].astype(np.float64)
        obj = after['object_pose_w'][:].astype(np.float64)
        palm = after['palm_body_pose_w'][:].astype(np.float64)
        # "before" is the goal evaluated on this transition.  "after" may have
        # already switched to the next native goal at the first subgoal.
        goal = tr['diagnostics_before/goal_pose_w'][:].astype(np.float64)
        metadata = json.loads(f.attrs['metadata_json'])
        cfg = metadata['config']
        problem = PROBLEM_REGISTRY[cfg['precise_assembly']['problem']]
        final_rel = np.asarray(problem.final_insert_pose_rel_receptive)
        final_rel = np.r_[final_rel[:3], final_rel[6], final_rel[3:6]]
        hole = after['hole_pose_w'][:].astype(np.float64)
        final_pose = np.concatenate((hole[:, :3]+rotate(hole[:, 3:], np.broadcast_to(final_rel[:3], (len(hole), 3))),
            qmul(hole[:, 3:], np.broadcast_to(final_rel[3:], (len(hole), 4)))), axis=-1)
        final_key = keypoint_distance(obj.astype(np.float32), final_pose.astype(np.float32), cfg['reward'])
        force = tr['object_contact_force_w'][:]
        fingers = after['finger_position_w'][:]
        count = (np.linalg.norm(force, axis=-1) > .1).sum(-1)
        initial_z = float(tr['diagnostics_before/object_pose_w'][0, 2])
        before_goals = tr['diagnostics_before/successes'][:]
        after_goals = after['successes'][:]
        goals_after = after['goal_pose_w'][:]
        result = dict(
            object_base=relative(base, obj), palm_base=relative(base, palm),
            object_palm=relative(palm, obj),
            height_mm=(obj[:, 2]-initial_z)*1000,
            settled_lift_mm=(obj[:, 2]-np.median(obj[24:48, 2]))*1000,
            keypoint_mm=tr['native_keypoints_max_dist'][:]*1000,
            object_goal_distance_mm=np.linalg.norm(obj[:, :3]-goal[:, :3], axis=-1)*1000,
            object_goal_angle_deg=angle(obj[:, 3:], goal[:, 3:]),
            object_palm_distance_mm=np.linalg.norm(obj[:, :3]-palm[:, :3], axis=-1)*1000,
            object_contact_min_fingers=count.min(axis=-1),
            object_contact_max_fingers=count.max(axis=-1),
            object_contact_peak_N=np.linalg.norm(force, axis=-1).max(axis=(1, 2)),
            fingertip_min_mm=np.linalg.norm(fingers-obj[:, None, :3], axis=-1).min(axis=-1)*1000,
            fingertip_mean_mm=np.linalg.norm(fingers-obj[:, None, :3], axis=-1).mean(axis=-1)*1000,
            final_keypoint_mm=final_key*1000,
            final_root_distance_mm=np.linalg.norm(obj[:, :3]-final_pose[:, :3], axis=-1)*1000,
            final_angle_deg=angle(obj[:, 3:], final_pose[:, 3:]),
            subgoals=after['successes'][:],
            native_complete=tr['native_success_diagnostic/complete'][:],
            native_near_counter=tr['native_success_diagnostic/near_goal_steps_after_goal_reset'][:],
            native_goal_before=before_goals,
            lifted=after['lifted_object'][:],
            native_alignment=tr['native_alignment_condition'][:],
            observation_step_id=f['episode/obs/step_id'][:],
            transition_step_id=tr['step_id'][:],
            final_geometry_config=dict(final_pose_hole_wxyz=final_rel.tolist(),
                insertion_keypoint_tolerance_mm=1000*cfg['precise_assembly']['insertion_success_tolerance']*cfg['reward']['keypoint_scale'],
                retract_keypoint_tolerance_mm=1000*cfg['precise_assembly']['retract_success_tolerance']*cfg['reward']['keypoint_scale'],
                retract_finger_distance_threshold_mm=1000*cfg['precise_assembly']['retract_distance_threshold']),
            native_success_config=dict(success_steps=cfg['termination']['success_steps'],
                force_consecutive_near_goal_steps=cfg['termination']['force_consecutive_near_goal_steps']),
        )
        for name in ('object_base', 'palm_base', 'object_palm'):
            for axis, coordinate in enumerate('xyz'):
                result[f'{name}_{coordinate}_mm'] = result[name][:, axis]*1000
        # Replay an existing read-only physical regression screen.  It names a
        # declared observable condition; it does not prove causality or whether
        # an expert could recover.  No policy, teacher, or simulator is called.
        screen = DeviationTrigger(profile('any'), 10**9)
        for tick in range(len(obj)):
            triggered = screen.observe(tick, obj[tick], palm[tick], force[tick],
                keypoint_max_distance_m=float(tr['native_keypoints_max_dist'][tick]),
                goal_index_before=before_goals[tick], goal_index_after=after_goals[tick],
                goal_pose_before=goal[tick], goal_pose_after=goals_after[tick])
            if triggered is not None:
                break
        result['first_declared_regression_screen'] = screen.receipt()
    assert np.array_equal(result['transition_step_id'], np.arange(len(obj)))
    result['outcome'] = read(directory/'result.json')
    return result


def pick():
    candidates = []
    frozen = read(PUBLIC.parent/'registration.json')['pairs']
    for wave in sorted(DEV.glob('wave[0-9][0-9]')):
        assert read(wave/'paired_initial_check.json')['passed'] is True
        for family in ('A', 'B'):
            old_dirs = list(wave.glob(f'{family}_130_50_*'))
            new_dirs = list(wave.glob(f'{family}_entry_*'))
            assert len(old_dirs) == len(new_dirs) == 1
            old, new = old_dirs[0], new_dirs[0]
            old_r, new_r = read(old/'result.json'), read(new/'result.json')
            assert old_r['complete_student_policy'] == frozen[f'{family}_130_50']
            assert new_r['complete_student_policy'] == frozen[f'{family}_entry']
            assert old_r['legal'] and new_r['legal']
            assert old_r['software_complete'] and new_r['software_complete']
            if bool(old_r['success']) == bool(new_r['success']):
                continue
            candidates.append(dict(family=family, seed=int(old_r['seed']), wave=wave.name,
                direction='original_only' if old_r['success'] else 'entry_only',
                original=str(old.relative_to(PROJECT)), entry=str(new.relative_to(PROJECT)),
                both_videos=(old/'autonomous.mp4').exists() and (new/'autonomous.mp4').exists()))
    chosen = []
    # Reproducible outcome/video-only selection, before loading physical traces:
    # prefer the earliest wave with videos, then lexicographic (wave, A/B).
    for direction in ('original_only', 'entry_only'):
        pool = [c for c in candidates if c['direction'] == direction]
        assert pool, direction
        chosen.append(sorted(pool, key=lambda c: (not c['both_videos'], c['wave'], c['family']))[0])
    return candidates, chosen


def metrics(value):
    outcome = value['outcome']
    fixed_final = ((value['final_keypoint_mm'] <= value['final_geometry_config']['retract_keypoint_tolerance_mm']) &
                  (value['fingertip_mean_mm'] > value['final_geometry_config']['retract_finger_distance_threshold_mm']))
    stage_outcome = ('native_complete' if outcome['success'] else
        ('first_subgoal_unconfirmed_despite_fixed_final_geometry'
         if value['subgoals'].max()==0 and fixed_final.any() else
         'native_failure_without_complete_ordered_goal_sequence'))
    return dict(success=bool(outcome['success']), terminal_reasons=outcome['reasons'],
        observed_native_stage_outcome=stage_outcome,
        ticks=len(value['height_mm']), first_lift_tick=first(value['lifted']),
        first_20mm_lift_tick=first_run(value['height_mm'] >= 20),
        first_20mm_lift_from_settled_tick=first_run((value['settled_lift_mm'] >= 20)&(np.arange(len(value['settled_lift_mm']))>=48)),
        first_2finger_object_contact_tick=first_run(value['object_contact_min_fingers'] >= 2),
        first_subgoal_tick=first(value['subgoals'] >= 1),
        first_native_complete_tick=first(value['native_complete']),
        max_height_mm=float(value['height_mm'].max()),
        max_lift_from_settled_mm=float(value['settled_lift_mm'].max()),
        max_subgoals=int(value['subgoals'].max()),
        min_keypoint_mm=float(value['keypoint_mm'].min()),
        min_keypoint_tick=int(value['keypoint_mm'].argmin()),
        final_height_mm=float(value['height_mm'][-1]),
        final_object_palm_distance_mm=float(value['object_palm_distance_mm'][-1]),
        final_geometry_config=value['final_geometry_config'],
        native_success_config=value['native_success_config'],
        first_goal_near_ticks=int(np.count_nonzero((value['native_goal_before']==0)&(value['keypoint_mm']<=value['final_geometry_config']['insertion_keypoint_tolerance_mm']))),
        terminal_native_near_counter=int(value['native_near_counter'][-1]),
        min_final_keypoint_mm=float(value['final_keypoint_mm'].min()),
        first_final_15mm_tick=first(value['final_keypoint_mm'] <= value['final_geometry_config']['insertion_keypoint_tolerance_mm']),
        first_final_retract_geometry_tick=first((value['final_keypoint_mm'] <= value['final_geometry_config']['retract_keypoint_tolerance_mm']) &
            (value['fingertip_mean_mm'] > value['final_geometry_config']['retract_finger_distance_threshold_mm'])),
        first_3tick_no_object_contact_after200_start=first_run((value['object_contact_max_fingers']==0)&
            (np.arange(len(value['height_mm']))>=200)),
        terminal_final_keypoint_mm=float(value['final_keypoint_mm'][-1]),
        terminal_final_root_distance_mm=float(value['final_root_distance_mm'][-1]),
        terminal_final_angle_deg=float(value['final_angle_deg'][-1]),
        first_declared_regression_screen=value['first_declared_regression_screen'])


def video_panel(case, tag):
    if not case['both_videos']:
        return None
    import cv2
    from PIL import Image, ImageDraw
    # Paired same observation frame; a video frame f is BEFORE control action f
    # and therefore depicts the recorded state after transition f-1.
    frames = (61, 151, 196, 241, 301)
    canvas = Image.new('RGB', (2688, len(frames)*346), 'white')
    draw = ImageDraw.Draw(canvas)
    for column, role in enumerate(('original', 'entry')):
        capture = cv2.VideoCapture(str(PROJECT/case[role]/'autonomous.mp4'))
        assert capture.isOpened()
        for row, frame_id in enumerate(frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = capture.read()
            assert ok
            draw.text((column*1344+10, row*346+8),
                f'{role}: observation frame {frame_id} = after transition {frame_id-1}', fill='black')
            canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (column*1344, row*346+30))
        capture.release()
    path = OUTPUT/f'{tag}_paired_video_frames.jpg'
    canvas.save(path, quality=91)
    return path.name


def analyze(case):
    old_dir, new_dir = PROJECT/case['original'], PROJECT/case['entry']
    old, new = load(old_dir), load(new_dir)
    n = min(len(old['height_mm']), len(new['height_mm']))
    divergence = dict(
        object_translation_mm=np.linalg.norm(old['object_base'][:n, :3]-new['object_base'][:n, :3], axis=-1)*1000,
        palm_translation_mm=np.linalg.norm(old['palm_base'][:n, :3]-new['palm_base'][:n, :3], axis=-1)*1000,
        object_orientation_deg=angle(old['object_base'][:n, 3:], new['object_base'][:n, 3:]),
        palm_orientation_deg=angle(old['palm_base'][:n, 3:], new['palm_base'][:n, 3:]))
    threshold = dict(object_translation_mm=1., palm_translation_mm=1.,
                     object_orientation_deg=1., palm_orientation_deg=1.)
    crossings = {key:first_run(value > threshold[key]) for key,value in divergence.items()}
    tag = f"{case['direction']}_{case['family']}_{case['seed']}"
    with (OUTPUT/f'{tag}_paired_pose_divergence.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['tick0', *divergence])
        writer.writeheader()
        for tick in range(n):
            writer.writerow(dict(tick0=tick, **{key:float(value[tick]) for key,value in divergence.items()}))
    snapshots = sorted(set([0, *[x for x in crossings.values() if x is not None],
                            *range(0, max(len(old['height_mm']), len(new['height_mm'])), 30)]))
    columns = ['tick0', 'policy', 'height_mm', 'settled_lift_mm', 'keypoint_mm', 'object_goal_distance_mm',
        'object_goal_angle_deg', 'object_palm_distance_mm', 'object_contact_min_fingers',
        'object_contact_max_fingers', 'object_contact_peak_N', 'fingertip_min_mm', 'subgoals',
        'final_keypoint_mm', 'final_root_distance_mm', 'final_angle_deg', 'fingertip_mean_mm',
        'native_goal_before', 'native_near_counter',
        *[f'{frame}_{coordinate}_mm' for frame in ('object_base', 'palm_base', 'object_palm')
          for coordinate in 'xyz']]
    for suffix, selected in [('all', None), ('snapshots', snapshots)]:
        with (OUTPUT/f'{tag}_{suffix}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for policy, value in [('130_50', old), ('entry', new)]:
                for tick in (range(len(value['height_mm'])) if selected is None else selected):
                    if tick >= len(value['height_mm']):
                        continue
                    writer.writerow(dict(tick0=tick, policy=policy,
                        **{key:value[key][tick].item() for key in columns[2:]}))
    fig, axes = plt.subplots(5, 1, figsize=(11, 12), sharex=True)
    colors = dict(original='#315caa', entry='#cc6325')
    for label, value in [('original', old), ('entry', new)]:
        x = np.arange(len(value['height_mm']))
        axes[0].plot(x, value['settled_lift_mm'], label=label, color=colors[label])
        axes[1].plot(x, value['keypoint_mm'], label=label, color=colors[label])
        axes[1].plot(x, value['final_keypoint_mm'], ls='--', alpha=.65, color=colors[label])
        axes[2].plot(x, value['object_palm_distance_mm'], label=label, color=colors[label])
        axes[3].plot(x, value['object_contact_min_fingers'], label=label, color=colors[label])
        for t in [first(value['subgoals'] >= 1)]:
            if t is not None:
                for ax in axes[:4]:
                    ax.axvline(t, color=colors[label], ls=':', alpha=.7)
    axes[4].plot(np.arange(n), divergence['object_translation_mm'], label='object position gap')
    axes[4].plot(np.arange(n), divergence['palm_translation_mm'], label='palm position gap')
    axes[4].axhline(1, color='grey', ls=':', label='1 mm')
    for key, label, height in [('palm_translation_mm', 'palm', .94), ('object_translation_mm', 'object', .80)]:
        tick = crossings[key]
        if tick is not None:
            axes[4].axvline(tick, color='black', ls='--', alpha=.6)
            axes[4].text(tick+4, height, f'{label}>{threshold[key]:g} mm starts {tick}\nconfirmed {tick+2}',
                transform=axes[4].get_xaxis_transform(), fontsize=8, va='top')
    for label, value in [('original', old), ('entry', new)]:
        if not value['outcome']['success']:
            fixed = metrics(value)['first_final_retract_geometry_tick']
            if fixed is not None:
                axes[1].annotate('Final geometry satisfied;\nordered native task still fails',
                    xy=(fixed, value['final_keypoint_mm'][fixed]), xytext=(fixed+40, 120),
                    arrowprops=dict(arrowstyle='->', color=colors[label]), color=colors[label], fontsize=8)
    axes[1].axhline(15, color='grey', ls=':', alpha=.7)
    for i, ylabel in enumerate(('Lift above settled\nheight [mm]', 'Keypoint max [mm]\nsolid=active; dash=final',
                               'Object/palm root\ndistance [mm]', 'Object-contact fingers\n(min of 2 substeps)',
                               'Base-frame trajectory\ngap [mm]')):
        axes[i].set_ylabel(ylabel)
        axes[i].grid(alpha=.25)
    axes[0].legend()
    axes[4].legend(loc='upper right')
    axes[4].set_xlabel('0-based control tick (after 120 Hz substeps; control 60 Hz)')
    fig.suptitle(f"{case['family']} seed {case['seed']}: {case['direction']}\n"
                 'Development records only; dotted vertical lines = first native subgoal')
    fig.tight_layout()
    fig.savefig(OUTPUT/f'{tag}.png', dpi=150)
    plt.close(fig)
    panel = video_panel(case, tag)
    return dict(**case, original_metrics=metrics(old), entry_metrics=metrics(new),
        first_sustained_divergence_tick0=crossings, threshold=threshold,
        first_sustained_divergence_confirmation_tick0={key:None if tick is None else tick+2
                                                      for key,tick in crossings.items()},
        sustain_ticks=3, common_prefix_ticks=n,
        base_frame_scope='Rigid base-relative pose; translation and wxyz quaternion are transformed separately.',
        contact_scope='object-filtered fingertip force >0.1 N; minimum finger count over the two 120 Hz substeps',
        height_scope='height_mm is relative to pre-action initial airborne pose; settled_lift_mm uses median object world Z at transitions24..47. The plot uses settled_lift_mm.',
        goal_scope='Original native active-goal keypoint metric; goal changes at native subgoal, do not compare across goal resets as one fixed target.',
        final_geometry_scope='Independent fixed final-pose keypoint/finger-clear geometry only. It never replaces ordered native subgoal and complete-success requirements.',
        interpretation_limits=['Loss of fingertip contact alone does not establish an accidental drop; it may occur during intended release or insertion.',
            'A fixed final-pose geometric match without ordered native subgoal completion remains a native failure.',
            'No causal attribution to sampling, perception, planning, or residual feedback follows from these selected development trajectories.'],
        plot=f'{tag}.png', all_trace_csv=f'{tag}_all.csv', snapshots_csv=f'{tag}_snapshots.csv',
        paired_pose_divergence_csv=f'{tag}_paired_pose_divergence.csv',
        paired_video_frames=panel,
        source_hashes={str(p.relative_to(PROJECT)):sha(p) for p in
            (old_dir/'numeric.h5', old_dir/'trace.npz', old_dir/'result.json',
             new_dir/'numeric.h5', new_dir/'trace.npz', new_dir/'result.json',
             old_dir.parent/'paired_initial_check.json', old_dir.parent/'plan.json',
             PROJECT/'evaluation/problems/peg/problems.py', PROJECT/'evaluation/problems/__init__.py')})


def main():
    assert MARKER.exists(), 'Wait for completed primary result before development diagnosis.'
    marker = read(MARKER)
    assert marker.get('complete') is True, 'Primary summary is not complete.'
    OUTPUT.mkdir(parents=True, exist_ok=True)
    candidates, chosen = pick()
    write(OUTPUT/'selection.json', dict(
        gate_primary_summary_sha256=sha(MARKER),
        selection_rule='For each discordant direction, prefer both existing videos; then earliest wave and A before B. Selected by outcome and video availability before physical traces.',
        all_discordant_development_pairs=candidates, chosen=chosen))
    cases = [analyze(case) for case in chosen]
    write(OUTPUT/'cases.json', dict(complete=True, cases=cases,
        zero_training_updates=True, zero_new_dagger=True, no_simulation=True,
        tick_index='0-based control transition index; each row is state AFTER that control transition.',
        limits=['Threshold crossings identify first confirmed recorded pose difference under a declared threshold, not the first causal policy error.',
                'Contact-force thresholds are diagnostics, not grasp or insertion success definitions.',
                'Development cases are selected illustrations, not heldout success-rate evidence.']))
    shutil.copy2(__file__, OUTPUT/'development_cases.py')
    PUBLIC.mkdir(parents=True, exist_ok=True)
    for path in OUTPUT.iterdir():
        if path.is_file():
            shutil.copy2(path, PUBLIC/path.name)
    print(json.dumps(dict(output=str(OUTPUT), public=str(PUBLIC), cases=cases), ensure_ascii=False))


if __name__ == '__main__':
    main()
