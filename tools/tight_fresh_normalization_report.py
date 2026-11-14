"""CPU-only audit of robot178 normalization against actual executed histories."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from visuotactile.compact import sha256
from visuotactile.models.native_dp import NativePolicy
from visuotactile.native_data import robot_rows
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def actual_robot(group):
    """Same canonical q/qd and past issued/applied fields as NativeController."""
    get = lambda path: torch.from_numpy(group[path][:])
    stamp = get('obs/time_s')
    before = get('obs/actuator_target_rad')
    target = get('transition/target_after_rad')
    if not torch.equal(target[:-1], before[1:]):
        raise ValueError('Recorded actuator target continuity differs')
    issued = get('transition/position_requested_rad') if 'transition/position_requested_rad' in group else target
    applied = get('transition/position_applied_rad') if 'transition/position_applied_rad' in group else target
    if not torch.equal(issued, target) or not torch.equal(applied, target):
        raise ValueError('Expected actual direct-target execution acknowledgement')
    past = torch.zeros(len(stamp), 61)
    past[1:, :29] = issued[:-1]
    past[1:, 29:58] = applied[:-1]
    past[1:, 58:60] = (stamp[1:] - stamp[:-1]).float()[:, None]
    past[1:, 60] = 1.
    return robot_rows(get('transition/diagnostics_before/joint_position_rad'),
        get('transition/diagnostics_before/joint_velocity_rad_s'), before, past, stamp-stamp[0])


def first_contact(group):
    forces = group['transition/object_contact_force_w'][:]
    detected = np.linalg.norm(forces, axis=-1) > .1
    indices = np.flatnonzero(detected.any(axis=(1, 2)))
    return int(indices[0]) if len(indices) else None


def field_names(joints):
    return [kind+'.'+joint for kind in ('q', 'qd', 'target', 'target_error',
        'previous_issued', 'previous_applied') for joint in joints] + [
        'issued_age', 'applied_age', 'history_valid', 'elapsed']


def summarize(rows, names, normalizer):
    raw = torch.cat(rows)
    z = NativePolicy.normalized_robot(normalizer, raw)
    magnitude = z.abs()
    maxima, ticks = magnitude.max(0)
    top = torch.argsort(maxima, descending=True)[:12]
    return dict(ticks=len(raw), values=raw.numel(),
        max_abs_z=float(magnitude.max()),
        p99_abs_z=float(torch.quantile(magnitude.flatten(), .99)),
        p999_abs_z=float(torch.quantile(magnitude.flatten(), .999)),
        fraction_abs_z_above={str(t):float((magnitude>t).float().mean()) for t in (5,10,20)},
        top_fields=[dict(index=int(i), name=names[i], max_abs_z=float(maxima[i]),
            p99_abs_z=float(torch.quantile(magnitude[:,i], .99)),
            fraction_above10=float((magnitude[:,i]>10).float().mean()),
            raw_value_at_max=float(raw[ticks[i],i]), mean=float(normalizer.robot_mean[i]),
            std=float(normalizer.robot_std[i])) for i in top])


def phase_rows(robot, onset):
    # Contact is recorded after action k; observation k+1 first includes it.
    if onset is None:
        return dict(pre_detected_contact=robot)
    return dict(pre_detected_contact=robot[:onset+1],
        first24_observations_after_detected_contact=robot[onset+1:onset+25],
        later_observations=robot[onset+25:])


def crossings(robot, names, normalizer):
    z = NativePolicy.normalized_robot(normalizer, robot).abs()
    output = {}
    for threshold in (5,10,20):
        rows = torch.nonzero((z>threshold).any(-1)).flatten()
        if not len(rows):
            output[str(threshold)] = None
            continue
        tick = int(rows[0]); field = int(z[tick].argmax())
        output[str(threshold)] = dict(observation_step_id=tick,
            field=names[field], abs_z=float(z[tick,field]), raw_value=float(robot[tick,field]))
    return output


def report(round_root):
    torch.set_num_threads(2)
    data_path = round_root/'A/state_data'
    data = read(data_path/'data.json')
    cache = torch.load(data_path/'numeric.pt', weights_only=False, map_location='cpu')
    if cache['data_sha256'] != sha256(data_path/'data.json'):
        raise ValueError('Numeric cache manifest identity differs')
    records = cache['records']
    stats = torch.load(data_path/'statistics.pt', weights_only=False, map_location='cpu')
    other_stats = torch.load(round_root/'B/state_data/statistics.pt', weights_only=False, map_location='cpu')
    if not all(torch.equal(stats[k], other_stats[k]) for k in ('robot_mean','robot_std')):
        raise ValueError('A/B robot normalization differs')
    original_ids = data['normalization_train_ids']
    if len(original_ids)!=30 or len(data['splits']['train'])!=71 or len(data['splits']['validation'])!=6 or data['splits']['test']:
        raise ValueError('This audit expects the completed round03 original30 plus41 corrections and validation6 cohort')
    original = torch.cat([records[i]['robot'] for i in original_ids])
    original_std = original.std(0)
    if not torch.equal(original.mean(0), stats['robot_mean']) or not torch.equal(original_std.clamp_min(1e-4), stats['robot_std']):
        raise ValueError('Normalizer differs from original successful30 all-tick statistics')
    normalizer = SimpleNamespace(robot_mean=stats['robot_mean'], robot_std=stats['robot_std'])
    buckets = {}; dataset_rows = []; names = None
    def add(name, value):
        if len(value): buckets.setdefault(name, []).append(value)
    for identity, entry in data['entries'].items():
        cached = records[identity]['robot']
        with h5py.File(entry['path']) as f:
            group = f['episodes'][entry['episode']]
            joints = json.loads(f.attrs['metadata_json'])['joint_order']
            if names is None: names = field_names(joints)
            if names != field_names(joints): raise ValueError('Joint order differs')
            actual = actual_robot(group)
            if not torch.equal(cached, actual):
                raise ValueError('Training cache and actual canonical/history construction differ: '+identity)
            onset = first_contact(group)
            if identity in original_ids:
                cohort = 'original30'
            elif identity in data['splits']['validation']:
                cohort = 'validation6'
            else:
                cohort = 'corrections41'
                source = torch.from_numpy(group['transition/dagger/control_source'][:])
                mask = records[identity]['supervision_mask']
                add('corrections41/student_prefix', actual[source==0])
                add('corrections41/expert_queue_masked', actual[(source==1)&~mask])
                add('corrections41/legal_expert_suffix', actual[mask])
                add('corrections_round'+str(entry['round_index'])+'/legal_expert_suffix', actual[mask])
            add(cohort+'/all', actual)
            for phase, value in phase_rows(actual, onset).items(): add(cohort+'/'+phase, value)
            dataset_rows.append(dict(id=identity, cohort=cohort, ticks=len(actual),
                cached_robot_exact=True, first_detected_contact_completed_tick=None if onset is None else onset+1,
                recorded_source_sha256=entry['source_sha256'], source_hash_recomputed=False))
    plan = read(round_root/'development/plan.json')
    checkpoint_audit = []
    for role, pair in plan['pairs'].items():
        for kind in ('frontend', 'backend'):
            if sha256(Path(pair[kind])) != pair[kind+'_sha256']:
                raise ValueError('Registered checkpoint hash differs')
            saved = torch.load(pair[kind], weights_only=False, map_location='cpu')
            states = ['model', 'ema'] if kind=='frontend' else ['model']
            for state in states:
                for name in ('robot_mean', 'robot_std'):
                    if not torch.equal(saved[state][name], stats[name]):
                        raise ValueError('Actual checkpoint robot normalization differs')
            checkpoint_audit.append(dict(role=role,kind=kind,sha256=pair[kind+'_sha256'],
                states_checked=states,robot_mean_std_exact=True))
    autonomy = []
    for case in plan['cases']:
        directory = round_root/'development'/case['name']; result = read(directory/'result.json')
        pair = plan['pairs'][case['role']]
        if not result.get('software_complete') or result['checkpoint_sha256']!=pair['backend_sha256']:
            raise ValueError('Autonomy completeness/model identity differs')
        with h5py.File(directory/'numeric.h5') as f:
            group = f['episode']; actual = actual_robot(group); onset = first_contact(group)
        add('autonomy_'+case['role']+'/all', actual)
        phase_summaries = {}
        for phase, value in phase_rows(actual, onset).items():
            add('autonomy_'+case['role']+'/'+phase, value)
            if len(value): phase_summaries[phase] = summarize([value], names, normalizer)
        autonomy.append(dict(case=case['name'], role=case['role'], ticks=len(actual),
            success=result['success'], first_detected_contact_completed_tick=None if onset is None else onset+1,
            numeric_sha256=sha256(directory/'numeric.h5'),
            first_abs_z_crossings=crossings(actual,names,normalizer), phases=phase_summaries))
    smallest = torch.argsort(stats['robot_std'])[:24]
    sources = [Path(__file__), Path('visuotactile/native_controller.py'),
        Path('visuotactile/native_data.py'), Path('visuotactile/models/native_dp.py'),
        Path('visuotactile/position_control.py'), Path('visuotactile/narrow_data.py')]
    return dict(version='fresh_robot178_normalization_v1',
        sources={str(p):sha256(p) for p in sources},
        data_sha256=sha256(data_path/'data.json'), statistics_sha256=sha256(data_path/'statistics.pt'),
        plan_sha256=sha256(round_root/'development/plan.json'), model_pairs=plan['pairs'],
        original_statistics_exact=True, original_statistics_ticks=len(original),
        dimensions_below_std_floor=int((original_std<1e-4).sum()),
        ab_robot_statistics_exact=True, numeric_cache_all77_actual_robot_exact=True,
        checkpoint_normalizers=checkpoint_audit,
        smallest_std_fields=[dict(index=int(i),name=names[i],mean=float(stats['robot_mean'][i]),
            std=float(stats['robot_std'][i])) for i in smallest],
        groups={name:summarize(rows,names,normalizer) for name,rows in buckets.items()},
        dataset_episodes=dataset_rows, autonomy=autonomy,
        semantics=dict(normalizer='Actual NativePolicy.normalized_robot, including zero invalid past history and raw history-valid bit.',
            robot='Canonical pre-action observed q/qd, actual current actuator target and strictly previous actual issued/applied target; ages and elapsed on actual native timestamps.',
            phases='First contact is a completed transition with any fingertip/object force >0.1N in either physics sample. Observation at that transition still precedes contact; following observation begins first24. Unmeasured other hand-body contacts can precede detection.',
            source_identity='Cached source SHA256 is reported; large original HDF5 images are not rehashed. Numeric fields are reread from all77 HDF5 records; autonomous numeric files are hashed.',
            interpretation='Large z values show scale/distribution excursions, not clipping or causal normalization failure. No policy input, statistics, weights or train sampling changed.'),
        model_forward_calls=0, optimizer_updates=0, simulator_starts=0,
        cuda_initialized=torch.cuda.is_initialized(), test_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, report(args.round_root))


if __name__ == '__main__':
    main()
