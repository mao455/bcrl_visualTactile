"""CPU-only companion to state_history_report; no optimizer, simulator or run writes.

All model errors compare target FK to target FK. Actual-state FK is used only for
asset validation and the separately named closed-loop achieved-motion analysis.
"""
import csv
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.state_history_report import digest, write
from visuotactile.kinematics import EncoderKinematics

PHASES = ('approach_proxy', 'first_contact_proxy', 'contact_other',
          'sustained_multifinger_proxy', 'native_lift', 'after_subgoal')


def stats(a):
    a = np.asarray(a, dtype=float).ravel()
    if not a.size:
        return dict(n=0, mean=None, rms=None, p05=None, p50=None, p95=None, max=None)
    assert np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), rms=float(np.sqrt(np.mean(a*a))),
                **{k: float(np.quantile(a, p)) for k, p in
                   [('p05', .05), ('p50', .5), ('p95', .95), ('max', 1)]})


def quat_matrix(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                     2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                     2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(q.shape[:-1]+(3, 3))


def rotation_error(a, b):
    r = np.swapaxes(a, -1, -2) @ b
    sine = np.linalg.norm(np.stack((r[..., 2, 1]-r[..., 1, 2],
        r[..., 0, 2]-r[..., 2, 0], r[..., 1, 0]-r[..., 0, 1]), -1), axis=-1)/2
    return np.rad2deg(np.arctan2(sine, (np.trace(r, axis1=-2, axis2=-1)-1)/2))


def local_xyz(p, base):
    return np.einsum('...ji,...j->...i', quat_matrix(base[..., 3:]), p-base[..., :3])


def fk_pose(fk, q):
    shape = q.shape[:-1]
    with torch.no_grad():
        p, r = fk(torch.as_tensor(np.array(q, copy=True)).double().reshape(-1, 29))
    return p[:, 0].numpy().reshape(shape+(3,)), r[:, 0].numpy().reshape(shape+(3, 3))


def phase_ids(force, valid, lifted, subgoals):
    """Disclosed aggregate-force proxies, never object-contact/grasp ground truth."""
    active = (np.linalg.norm(force, axis=-1).max(axis=1) > .1) & valid[:, None]
    contact = active.any(-1)
    seen = np.maximum.accumulate(contact)
    phase = np.where(seen, 2, 0)
    hits = np.flatnonzero(contact)
    if len(hits):
        phase[hits[0]:hits[0]+6] = 1
    sustained = np.convolve((active.sum(-1) >= 2).astype(int), np.ones(6, int), 'full')[:len(force)] == 6
    phase[sustained] = 3
    phase[np.asarray(lifted).astype(bool)] = 4
    phase[np.asarray(subgoals) > 0] = 5
    return phase


def native_targets(before, delayed, limits, config, dt):
    """Same float32 operation order as apply_action_pipeline, canonical joint order."""
    b, a, lim = (torch.as_tensor(x, dtype=torch.float32) for x in (before, delayed, limits))
    arm = torch.clamp(b[:, :7]+config['dof_speed_scale']*dt*a[:, :7], lim[:7, 0], lim[:7, 1])
    arm = torch.clamp(config['arm_moving_average']*arm+(1-config['arm_moving_average'])*b[:, :7], lim[:7, 0], lim[:7, 1])
    hand = lim[7:, 0]+.5*(a[:, 7:]+1)*(lim[7:, 1]-lim[7:, 0])
    hand = torch.clamp(config['hand_moving_average']*hand+(1-config['hand_moving_average'])*b[:, 7:], lim[7:, 0], lim[7:, 1])
    return torch.cat((arm, hand), -1).numpy()


def read_episode(entry):
    with h5py.File(entry['path'], 'r') as f:
        meta = json.loads(f.attrs['metadata_json']); g = f['episodes'][entry['episode']]
        before = {k: v[:] for k, v in g['transition/diagnostics_before'].items()}
        after = {k: v[:] for k, v in g['transition/diagnostics_after'].items()}
        row = dict(before=before, after=after, meta=meta,
            target=g['transition/target_after_rad'][:], previous=g['transition/target_before_rad'][:],
            force=g['obs/contact_substeps_w_N'][:], contact_valid=g['obs/contact_valid'][:])
        row['phase'] = phase_ids(row['force'], row['contact_valid'], before['lifted_object'], before['successes'])
        row['control'] = {k: v[:] for k, v in g['transition/control_trace'].items()}
        for name in ('raw_action', 'submitted_action', 'command_clipped_pre_delay', 'executed_action', 'time_before_s', 'time_after_s'):
            row[name] = g['transition/'+name][:]
        row['obs_target'] = g['obs/actuator_target_rad'][:]
        row['time'] = g['obs/time_s'][:]
    return row


def csv_rows(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader(); writer.writerows(rows)


def data_audit(data, output):
    validation = []; delay = []; rows = []; all_indices = []; all_lags = []
    distribution = {s: {} for s in ('train', 'validation')}
    sample_windows = {s: {h: [] for h in (12, 32, 96)} for s in distribution}
    actual_windows = {s: {h: [] for h in (12, 32, 96)} for s in distribution}
    phase_counts = {s: Counter() for s in distribution}; anchor_counts = {s: Counter() for s in distribution}
    label_counts = {s: Counter() for s in distribution}
    trajectory = {s: [] for s in distribution}; identities = {s: [] for s in distribution}
    split_of = {i: s for s, ids in data['splits'].items() for i in ids}
    fk = None
    for i, (identity, entry) in enumerate(sorted(data['entries'].items())):
        a = read_episode(entry); m = a['meta']; n = len(a['target']); b = a['before']; split = split_of[identity]
        if fk is None:
            fk = EncoderKinematics(m['config']['assets']['robot_urdf'], m['joint_order'], links=('iiwa14_link_7',))
            reference_order = m['joint_order']
        assert m['joint_order'] == reference_order and m['fk_urdf_sha256'] == fk.sha256
        assert m['joint_order'][:7] == [f'iiwa14_joint_{j}' for j in range(1, 8)]
        assert fk.root == m['fk_root']
        for timing in ('before', 'after'):
            diag = a[timing]; p, r = fk_pose(fk, diag['joint_position_rad'])
            base = diag['base_pose_w']; world = np.einsum('nij,nj->ni', quat_matrix(base[:, 3:]), p)+base[:, :3]
            error = np.linalg.norm(world-diag['palm_body_pose_w'][:, :3], axis=-1)*1000
            angle = rotation_error(quat_matrix(base[:, 3:])@r, quat_matrix(diag['palm_body_pose_w'][:, 3:]))
            validation.append(np.stack((error, angle), -1))
        c = a['control']; tick = np.arange(n); index = c['delay_index']; selected = np.maximum(0, tick-index)
        cmd = a['command_clipped_pre_delay']; expected = cmd[selected]
        actual = native_targets(a['previous'], c['delayed_command'], np.asarray(m['joint_limits_rad'])[0], m['config']['action'], m['control_dt'])
        errors = dict(raw_submitted=float(np.max(np.abs(a['raw_action']-a['submitted_action']))),
            preclip=float(np.max(np.abs(np.clip(a['submitted_action'], -1, 1)-cmd))),
            legacy_executed_preclip=float(np.max(np.abs(a['executed_action']-cmd))),
            queue=float(np.max(np.abs(expected-c['delayed_command']))),
            targets=float(np.max(np.abs(actual-a['target']))),
            obs_target=float(np.max(np.abs(a['previous']-a['obs_target']))),
            continuity=float(np.max(np.abs(a['target'][:-1]-a['previous'][1:]))),
            selected_steps=int(np.max(np.abs(selected-c['selected_command_step']))),
            effective_lag=int(np.max(np.abs(tick-selected-c['effective_lag_steps']))),
            selected_time=float(np.max(np.abs(a['time_before_s']-(tick-selected)*m['control_dt']-c['selected_command_time_s']))),
            observation_time=float(np.max(np.abs(a['time']-a['time_before_s']))),
            execution_dt=float(np.max(np.abs(a['time_after_s']-a['time_before_s']-m['control_dt']))))
        assert c['valid'].all() and c['delay_enabled'].all() and (c['queue_length'] == 3).all()
        delay.append(dict(identity=identity, ticks=n, exact_target_values=int((actual == a['target']).sum()),
            maximum_arm_target_step_rad=float(np.max(np.abs(a['target'][:, :7]-a['previous'][:, :7]))),
            submitted_outside_unit=int((np.abs(a['submitted_action']) > 1).sum()),
            delayed_differs_current_ticks=int(np.any(c['delayed_command'] != cmd, axis=-1).sum()), **errors))
        all_indices.extend(index.tolist()); all_lags.extend(c['effective_lag_steps'].tolist())
        if split not in distribution:
            continue
        phase = a['phase']; phase_counts[split].update(phase.tolist()); anchor_counts[split].update(phase[::12].tolist())
        q = b['joint_position_rad']; base = b['base_pose_w']
        features = dict(initial_arm_rad=q[0, :7], initial_object_m=local_xyz(b['object_pose_w'][:1, :3], base[:1])[0],
            initial_goal_m=local_xyz(b['goal_pose_w'][:1, :3], base[:1])[0],
            joint_range_rad=np.ptp(q, axis=0), target_range_rad=np.ptp(a['target'], axis=0), length_ticks=np.array([n]))
        for name, values in features.items():
            distribution[split].setdefault(name, []).append(values)
        rows.append(dict(identity=identity, split=split, ticks=n,
            arm_actual_range_rms_rad=float(np.sqrt(np.mean(np.ptp(q[:, :7], axis=0)**2))),
            **{name+'_ticks': int((phase == k).sum()) for k, name in enumerate(PHASES)}))
        for horizon in (12, 32, 96):
            stride = 96 if horizon == 96 else 12
            for start in range(0, n, stride):
                end = min(start+horizon, n)
                if horizon == 32:
                    label_counts[split].update(phase[start:end].tolist())
                if end-start != horizon:
                    continue
                values = np.concatenate((a['previous'][start:start+1], a['target'][start:end]), 0)
                sample_windows[split][horizon].append([float(np.sqrt(np.mean(np.ptp(values[:, sl], axis=0)**2))) for sl in (slice(0, 7), slice(7, 29))])
                actual_values = np.concatenate((q[start:start+1], a['after']['joint_position_rad'][start:end]), 0)
                actual_windows[split][horizon].append([float(np.sqrt(np.mean(np.ptp(actual_values[:, sl], axis=0)**2))) for sl in (slice(0, 7), slice(7, 29))])
        trajectory[split].append(np.stack([np.interp(np.linspace(0, n-1, 64), np.arange(n), q[:, j]) for j in range(7)], -1))
        identities[split].append(identity)
        if (i+1) % 100 == 0:
            print('data audited', i+1, flush=True)
    checks = np.concatenate(validation)
    fk_result = dict(poses=len(checks), reference='iiwa14_link_7 actor/link frame, not COM; base<-link and full root transform',
        urdf_sha256=fk.sha256, root=fk.root, joint_order=reference_order,
        position_mm=stats(checks[:, 0]), rotation_degree=stats(checks[:, 1]),
        admission_mm=.1, admission_degree=.05,
        passed=bool(checks[:, 0].max() < .1 and checks[:, 1].max() < .05))
    write(output/'fk_validation.json', fk_result)
    dr = dict(episodes=len(delay), ticks=sum(r['ticks'] for r in delay),
        maximum_errors={k: max(r[k] for r in delay) for k in errors},
        exact_target_values=sum(r['exact_target_values'] for r in delay),
        delay_index_counts=dict(Counter(all_indices)), effective_lag_counts=dict(Counter(all_lags)),
        effective_lag_ms=stats(np.array(all_lags)*1000/60),
        index_vs_lag_different_ticks=int(np.sum(np.array(all_indices) != all_lags)),
        submitted_outside_unit=sum(r['submitted_outside_unit'] for r in delay),
        delayed_differs_current_ticks=sum(r['delayed_differs_current_ticks'] for r in delay),
        maximum_arm_target_step_rad=max(r['maximum_arm_target_step_rad'] for r in delay),
        queue_length=3, numeric_reconstruction_only=True)
    write(output/'delay_audit.json', dr); csv_rows(output/'delay_episodes.csv', delay)
    summary = {}
    for split, values in distribution.items():
        summary[split] = dict(episodes=len(identities[split]), ticks=sum(phase_counts[split].values()),
            distributions={k: [stats(column) for column in np.array(v).T] for k, v in values.items()},
            phases={name: dict(ticks=phase_counts[split][k], anchors=anchor_counts[split][k],
                horizon32_valid_labels=label_counts[split][k]) for k, name in enumerate(PHASES)}, windows={})
        for h, values in sample_windows[split].items():
            v = np.array(values); summary[split]['windows'][h] = dict(full=len(v),
                arm_range_rms_rad=stats(v[:, 0]), hand_range_rms_rad=stats(v[:, 1]),
                static_fraction={str(th): dict(arm=float((v[:, 0] < th).mean()),
                    both=float((v.max(-1) < th).mean())) for th in (.001, .005, .01)})
            actual = np.array(actual_windows[split][h])
            summary[split]['windows'][h]['actual_static_fraction'] = {
                str(th): dict(arm=float((actual[:, 0] < th).mean()), both=float((actual.max(-1) < th).mean()))
                for th in (.001, .005, .01)}
    from scipy.stats import ks_2samp, wasserstein_distance
    shift = {}
    for key in distribution['train']:
        train, val = (np.array(distribution[s][key]) for s in distribution)
        shift[key] = [dict(ks=float(ks_2samp(t, v).statistic), wasserstein=float(wasserstein_distance(t, v)),
            standardized_mean_difference=float((v.mean()-t.mean())/max(t.std(), 1e-9)),
            val_outside_train_range=int(((v < t.min()) | (v > t.max())).sum())) for t, v in zip(train.T, val.T)]
    summary['train_validation_shift'] = shift
    train = np.array(trajectory['train']); neighbours = []
    for split in trajectory:
        for j, q in enumerate(trajectory[split]):
            dist = np.sqrt(np.mean((train-q)**2, axis=(1, 2)))
            if split == 'train':
                dist[j] = np.inf
            k = int(dist.argmin()); neighbours.append(dict(split=split, identity=identities[split][j],
                nearest_train=identities['train'][k], arm64_rms_rad=float(dist[k])))
        summary[split]['nearest_train_trajectory_rms_rad'] = stats([r['arm64_rms_rad'] for r in neighbours if r['split'] == split])
    write(output/'data_coverage.json', summary); csv_rows(output/'data_episodes.csv', rows)
    csv_rows(output/'trajectory_neighbours.csv', neighbours)
    plot_data(output, distribution, sample_windows, summary)
    if not fk_result['passed']:
        raise ValueError('FK asset/frame validation failed; no spatial conclusions allowed')
    return fk


def target_errors(pred, truth, hold, fk):
    p, r = fk_pose(fk, pred); y, yr = fk_pose(fk, truth); base, _ = fk_pose(fk, hold)
    displacement = y-base; norm = np.linalg.norm(displacement, axis=-1)
    along = ((p-y)*displacement).sum(-1)/np.maximum(norm, 1e-12)*1000
    return dict(joint=pred[..., :7]-truth[..., :7], xyz=(p-y)*1000,
        distance=np.linalg.norm(p-y, axis=-1)*1000, rotation=rotation_error(r, yr),
        along=along, directional_valid=norm > .001, xyz_position=p)


def error_summary(e, mask):
    along = mask & e['directional_valid']
    return dict(ticks=int(mask.sum()), arm7_rmse_rad=np.sqrt(np.mean(e['joint'][mask]**2, axis=0)).tolist(),
        xyz_rmse_mm=np.sqrt(np.mean(e['xyz'][mask]**2, axis=0)).tolist(), ee_mm=stats(e['distance'][mask]),
        orientation_degree=stats(e['rotation'][mask]), along_expert_mm=stats(e['along'][along]),
        direction_scope='expert target displacement from anchor observed target >1mm; signed student-expert projection')


def offline_audit(data, session, output, fk):
    from visuotactile.state_history import PersistentState, load_records
    from visuotactile.state_history_training import labels
    from visuotactile.position_control import bounded_targets
    checkpoint = Path(session['motion_checkpoint'])
    assert digest(checkpoint) == session['motion_checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = PersistentState('dp', saved['contract']); model.load_state_dict(saved['ema']); model.eval(); del saved
    rows = []; aggregate = {name: [] for name in ('Student', 'Hold', 'Expert')}; phases = []; plot_rows = []
    for identity in sorted(data['splits']['validation']):
        a = read_episode(data['entries'][identity]); record = load_records(session['source_root'], [identity], model.contract)[identity]
        with torch.no_grad():
            condition, anchors, _ = model.encode(record); truth, valid = labels(record, anchors, model.horizon)
            pred = model.predict(condition, torch.Generator().manual_seed(7091), inference_steps=16, clip_sample=False)
            pred = bounded_targets(pred.flatten(0, 1), record['joint_limits']).reshape_as(pred)
        anchors = anchors.numpy(); mask = valid[:, :12].numpy()
        target = truth[:, :12].numpy(); hold = np.broadcast_to(record['robot'][anchors, 58:87].numpy()[:, None], target.shape)
        index = (anchors[:, None]+np.arange(12))[mask]; phase = a['phase'][index]; phases.append(phase)
        obj = local_xyz(a['before']['object_pose_w'][index, :3], a['before']['base_pose_w'][index])
        goal = local_xyz(a['before']['goal_pose_w'][index, :3], a['before']['base_pose_w'][index])
        for name, values in [('Student', pred[:, :12].numpy()), ('Hold', hold), ('Expert', target)]:
            e = target_errors(values[mask], target[mask], hold[mask], fk); aggregate[name].append(e)
            for label, phase_mask in [('all', np.ones(len(index), bool)), ('approach_proxy', phase == 0), ('other', phase != 0)]:
                if phase_mask.any():
                    rows.append(dict(identity=identity, model=name, phase=label, **error_summary(e, phase_mask)))
            for tick, distance, orient, object_dist, goal_dist in zip(index, e['distance'], e['rotation'],
                    np.linalg.norm(e['xyz_position']-obj, axis=-1)*1000, np.linalg.norm(e['xyz_position']-goal, axis=-1)*1000):
                plot_rows.append(dict(identity=identity, model=name, time_s=float(tick/60),
                    ee_error_mm=float(distance), orientation_degree=float(orient),
                    target_fk_to_observed_object_mm=float(object_dist), target_fk_to_observed_goal_mm=float(goal_dist)))
        print('offline FK', len(phases), '/40', flush=True)
    phase = np.concatenate(phases); summary = {}
    for name, values in aggregate.items():
        e = {k: np.concatenate([v[k] for v in values]) for k in values[0]}
        summary[name] = {label: error_summary(e, mask) for label, mask in
            [('all', np.ones(len(phase), bool)), ('approach_proxy', phase == 0), ('other', phase != 0)] if mask.any()}
    paired = []
    for identity in sorted(data['splits']['validation']):
        values = {row['model']: row['ee_mm']['rms'] for row in rows
                  if row['identity'] == identity and row['phase'] == 'approach_proxy'}
        if 'Student' in values and 'Hold' in values:
            paired.append(values['Student']-values['Hold'])
    bootstrap = np.random.default_rng(6193).choice(paired, (5000, len(paired)), replace=True).mean(-1)
    result = dict(episodes=40, checkpoint_sha256=digest(checkpoint), inference='CPU float32, seed7091 per episode, DDIM16 clipFalse; CPU and CUDA RNG are not bitwise paired',
        scope='validation40 only; teacher-forced observations/history; first12 valid labels; no optimizer or test inference',
        paired_approach_student_minus_hold_episode_rmse_mm=dict(mean=float(np.mean(paired)),
            bootstrap95=np.quantile(bootstrap, [.025, .975]).tolist(), episodes=len(paired),
            unit='episode bootstrap, fixed one policy seed; not success confidence'),
        summary=summary, episodes_by_phase=rows)
    write(output/'offline_fk.json', result); csv_rows(output/'offline_fk_curves.csv', plot_rows)
    plot_errors(output, plot_rows, 'offline_fk', 'Target FK, teacher-forced validation40 (CPU RNG)')


def closed_loop_audit(root, session, output, fk):
    rows = []; curves = []; rate_rows = []
    for path in sorted((root/'C0').glob('*/trace.npz')):
        identity = path.parent.name; expert_path = Path(session['previous_root'])/'expert_learning'/identity/'trace.npz'
        if not expert_path.exists():
            continue
        student = np.load(path); expert = np.load(expert_path)
        initial = json.loads((path.parent/'initial_observable.json').read_text())
        reference = json.loads((expert_path.parent/'initial_observable.json').read_text())
        assert initial == reference
        base = np.array(initial['base_pose_w']).reshape(-1, 7)[0]
        n = min(len(student['joint']), len(expert['joint']))
        obj_contact = np.linalg.norm(expert['object_contact'], axis=-1).reshape(len(expert['joint']), -1).max(-1) > .1
        before_contact = np.cumsum(obj_contact[:n]) == 0
        truth = expert['joint'][:n]; hold = np.broadcast_to(np.array(initial['joint_position_rad']).reshape(-1, 29)[0], truth.shape)
        result = json.loads((path.parent/'result.json').read_text())
        with h5py.File(path.parent/'numeric.h5', 'r') as stream:
            group = stream['episode']
            target_before = group['transition/target_before_rad'][:]
        jump = student['target'][:, :7]-target_before[:, :7]
        rate_rows.append(dict(identity=identity, ticks=len(jump), arm_values=jump.size,
            beyond_native_delta_values=int((np.abs(jump) > .002501).sum()),
            beyond_native_delta_ticks=int(np.any(np.abs(jump) > .002501, axis=-1).sum()),
            max_delta_rad=float(np.max(np.abs(jump))), rms_delta_rad=float(np.sqrt(np.mean(jump**2)))))
        for name, values, obj in [('Student', student['joint'][:n], student['object'][:n, :3]),
                ('Expert', truth, expert['object'][:n, :3]), ('Hold', hold, expert['object'][:n, :3])]:
            e = target_errors(values, truth, hold, fk)
            row = dict(identity=identity, model=name, observable_initial_equal=True,
                approach_proxy=result.get('approach_proxy') if name == 'Student' else None,
                object_contact=result.get('contact', {}).get('object_contact') if name == 'Student' else None,
                success=result.get('success') if name == 'Student' else (True if name == 'Expert' else None),
                approach=error_summary(e, before_contact) if before_contact.any() else None,
                other=error_summary(e, ~before_contact) if (~before_contact).any() else None)
            rows.append(row)
            world = e['xyz_position']@quat_matrix(base[3:]).T+base[:3]
            goal = np.array(initial['goal_pose_w']).reshape(-1, 7)[0, :3]
            for tick in range(n):
                curves.append(dict(identity=identity, model=name, time_s=(tick+1)/60,
                    ee_error_mm=float(e['distance'][tick]), orientation_degree=float(e['rotation'][tick]),
                    actual_fk_to_object_mm=float(np.linalg.norm(world[tick]-obj[tick])*1000),
                    actual_fk_to_initial_goal_mm=float(np.linalg.norm(world[tick]-goal)*1000)))
        with h5py.File(expert_path.parent/'numeric.h5', 'r') as stream:
            expert_before = stream['episode/transition/target_before_rad'][:n]
        target_hold = expert_before[np.arange(n)//12*12]
        e = target_errors(student['target'][:n], expert['target'][:n], target_hold, fk)
        rows.append(dict(identity=identity, model='Student_target_vs_expert_target',
            approach=error_summary(e, before_contact) if before_contact.any() else None,
            other=error_summary(e, ~before_contact) if (~before_contact).any() else None))
    write(output/'closed_loop_fk.json', dict(rows=rows, cohort='existing C0 only', target_rate=rate_rows,
        hold='counterfactual fixed initial q in expert scene; no Hold rollout/success rate',
        scope='Achieved q vs achieved expert q at equal elapsed ticks, object-filtered expert contact phases. Observable reset equality is not full PhysX identity. Initial goal distance excludes later goal resampling.'))
    csv_rows(output/'closed_loop_fk_curves.csv', curves)
    plot_errors(output, curves, 'closed_loop_fk', 'Actual joint FK; matched observable starts, no exact internal-state claim')


def plot_errors(output, rows, stem, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys = [k for k in rows[0] if k not in ('identity', 'model', 'time_s')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    aggregate = []
    for model in ('Expert', 'Student', 'Hold'):
        values = [r for r in rows if r['model'] == model]
        bins = sorted({int(round(r['time_s']*10)) for r in values})
        for ax, key in zip(axes.flat, keys):
            groups = [[r[key] for r in values if int(round(r['time_s']*10)) == b] for b in bins]
            ax.plot(np.array(bins)/10, [np.mean(v) for v in groups], label=model)
            ax.set(xlabel='elapsed expert clock (s)', ylabel=key); ax.grid(alpha=.2); ax.legend(fontsize=8)
        for bucket in bins:
            points = [r for r in values if int(round(r['time_s']*10)) == bucket]
            aggregate.append(dict(model=model, time_bin_s=bucket/10, samples=len(points),
                episodes=len({r['identity'] for r in points}), **{key: float(np.mean([r[key] for r in points])) for key in keys}))
    fig.suptitle(title); fig.tight_layout(); fig.savefig(output/(stem+'.png'), dpi=150); plt.close(fig)
    csv_rows(output/(stem+'_aggregate_curves.csv'), aggregate)


def plot_data(output, distribution, windows, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for split, values in distribution.items():
        for ax, key in zip(axes.flat[:3], ('initial_arm_rad', 'initial_object_m', 'initial_goal_m')):
            a = np.array(values[key]); mean = a.mean(0); low, high = np.quantile(a, [.05, .95], axis=0)
            ax.errorbar(np.arange(a.shape[1])+(0 if split == 'train' else .12), mean, yerr=np.maximum(0,np.stack((mean-low, high-mean))), fmt='o', label=split)
            ax.set_title(key+' mean / 5–95%')
        axes[1, 0].hist(np.array(values['length_ticks']).ravel(), bins=25, density=True, alpha=.4, label=split)
        a = np.array(values['joint_range_rad']); axes[1, 1].plot(np.arange(29), a.mean(0), 'o-', label=split)
        v = np.sort(np.array(windows[split][32])[:, 0]); axes[1, 2].plot(v, np.arange(1,len(v)+1)/len(v), label=split)
    for ax, title in zip(axes[1], ('length (60Hz ticks)', 'mean actual joint range (rad), 7 arm +22 hand', 'H32 Arm7 range RMS CDF (rad), full windows')):
        ax.set_title(title)
    for ax in axes.flat:
        ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output/'data_distributions.png', dpi=150); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for j, split in enumerate(('train', 'validation')):
        for offset, field in [(-.2, 'ticks'), (0, 'anchors'), (.2, 'horizon32_valid_labels')]:
            v = np.array([summary[split]['phases'][p][field] for p in PHASES]); axes[j].bar(np.arange(len(v))+offset, v/v.sum(), width=.2, label=field)
        axes[j].set_xticks(np.arange(len(PHASES)), PHASES, rotation=30, ha='right', fontsize=7)
        axes[j].set(title=split, ylabel='fraction'); axes[j].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(output/'phase_coverage.png', dpi=150); plt.close(fig)


def gradient_audit(data, session, output, fk):
    """Measure candidate auxiliary gradients on train-only x0 without updating weights."""
    from visuotactile.state_history import PersistentState, load_records
    from visuotactile.state_history_training import labels
    from visuotactile.dp_objectives import masked_objectives
    saved = torch.load(session['motion_checkpoint'], map_location='cpu', weights_only=False)
    model = PersistentState('dp', saved['contract']); model.load_state_dict(saved['ema']); model.eval(); del saved
    parameters = list(model.denoiser.parameters())
    rows = []
    for identity in sorted(data['splits']['train'])[:3]:
        record = load_records(session['source_root'], [identity], model.contract)[identity]
        with torch.no_grad():
            condition, anchors, _ = model.encode(record, stop=min(96, len(record['time'])))
            command, valid = labels(record, anchors, model.horizon)
            target = torch.where(valid[..., None], command*model.action_scale+model.action_offset, 0.)
        generator = torch.Generator().manual_seed(6193)
        noise = torch.randn(target.shape, generator=generator)
        timestep = torch.randint(100, (len(target),), generator=generator)
        prediction = model.denoiser(model.noise.add_noise(target, noise, timestep), timestep, global_cond=condition)
        base = masked_objectives(prediction, target, valid)['B']
        prediction_rad = (prediction-model.action_offset)/model.action_scale
        pos, _ = fk(prediction_rad.reshape(-1, 29))
        reference, _ = fk(command.reshape(-1, 29))
        auxiliary = ((pos-reference).square().sum((-1, -2)).reshape(valid.shape)*valid).sum()/valid.sum()
        output_base, = torch.autograd.grad(base, prediction, retain_graph=True)
        output_fk, = torch.autograd.grad(auxiliary, prediction, retain_graph=True)
        base_grads = torch.autograd.grad(base, parameters, retain_graph=True)
        base_norm = float(torch.stack([g.detach().double().square().sum() for g in base_grads]).sum().sqrt())
        del base_grads
        fk_grads = torch.autograd.grad(auxiliary, parameters)
        fk_norm = float(torch.stack([g.detach().double().square().sum() for g in fk_grads]).sum().sqrt())
        del fk_grads
        rows.append(dict(identity=identity, anchors=anchors.tolist(), timesteps=timestep.tolist(), valid_labels=int(valid.sum()),
            loss_B=float(base.detach()), loss_EE_m2=float(auxiliary.detach()),
            denoiser_grad_B=base_norm, denoiser_grad_FK=fk_norm,
            output_arm_grad_B=float(output_base[..., :7].norm()), output_arm_grad_FK=float(output_fk[..., :7].norm()),
            output_hand_grad_FK=float(output_fk[..., 7:].norm()),
            padded_output_grad_FK=float(output_fk[~valid].norm()), finite=bool(torch.isfinite(output_fk).all())))
        print('FK gradient diagnostic', len(rows), '/3', flush=True)
    parameter_ratio = np.array([r['denoiser_grad_FK']/r['denoiser_grad_B'] for r in rows])
    arm_ratio = np.array([r['output_arm_grad_FK']/r['output_arm_grad_B'] for r in rows])
    coefficient = float(min(.1/parameter_ratio.max(), .2/arm_ratio.max()))
    assert all(r['finite'] and r['output_hand_grad_FK'] == 0 and r['padded_output_grad_FK'] == 0 for r in rows)
    write(output/'fk_gradient_diagnostic.json', dict(rows=rows, optimizer_updates=0,
        scope='original EMA, first96 ticks of sorted first3 train IDs, CPU float32 denoiser, no optimizer; preliminary candidate, not trained',
        formula='q=(predicted_clean_x0-action_offset)/action_scale; mean_valid ||FK(q)-FK(expert_target)||^2 in m^2',
        gradient_scope='all denoiser parameters; conditioning detached; also normalized-x0 Arm7 output',
        candidate_lambda_per_m2=coefficient,
        selection_rule='min(0.10/max_train_denoiser_FK_to_B_ratio, 0.20/max_train_arm_output_FK_to_B_ratio)',
        weighted_denoiser_ratio=(coefficient*parameter_ratio).tolist(),
        weighted_arm_output_ratio=(coefficient*arm_ratio).tolist(),
        deployment_unchanged=True, recalibration_required_at_independent_training_initialization=True))


def run(root, output, mode='all'):
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU analysis must not initialize CUDA')
    torch.set_num_threads(2)
    analysis_sha256 = digest(Path(__file__))
    session = json.loads((root/'session.json').read_text())
    frozen = session['implementation_sha256']
    assert all(digest(Path(p)) == h for p, h in frozen.items())
    output.mkdir(parents=True, exist_ok=True)
    data_path = Path(session['source_root'])/'data.json'
    assert digest(data_path) == session['data_sha256']
    data = json.loads(data_path.read_text())
    if mode in ('all', 'data'):
        fk = data_audit(data, output)
    else:
        gate = json.loads((output/'fk_validation.json').read_text()); assert gate['passed']
        m = read_episode(data['entries'][sorted(data['splits']['train'])[0]])['meta']
        fk = EncoderKinematics(m['config']['assets']['robot_urdf'], m['joint_order'], links=('iiwa14_link_7',))
        assert fk.sha256 == gate['urdf_sha256']
    if mode in ('all', 'offline'):
        offline_audit(data, session, output, fk)
    if mode in ('all', 'closed'):
        closed_loop_audit(root, session, output, fk)
    if mode == 'gradient':
        gradient_audit(data, session, output, fk)
    assert all(digest(Path(p)) == h for p, h in frozen.items())
    assert digest(Path(session['motion_checkpoint'])) == session['motion_checkpoint_sha256']
    assert digest(Path(__file__)) == analysis_sha256
    write(output/('preservation_'+mode+'.json'), dict(frozen_sources=len(frozen), all_unchanged=True,
        checkpoint_sha256=session['motion_checkpoint_sha256'], data_sha256=session['data_sha256'],
        analysis_sha256=analysis_sha256, cuda_initialized=torch.cuda.is_initialized(), optimizer_updates=0))
