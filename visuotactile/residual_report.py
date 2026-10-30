"""Small sanitized aggregate, recomputed from local residual artifacts."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from .compact import sha256
from .residual_data import BASE_HASH
from .training_resources import atomic_json


def metric_summary(result):
    fields = ('rows', 'contact_rows', 'arm_rmse', 'hand_rmse', 'contact_arm_rmse', 'contact_hand_rmse',
              'base_arm_rmse', 'base_hand_rmse', 'base_contact_arm_rmse', 'base_contact_hand_rmse',
              'selection_mse', 'saturation_fraction', 'bypass_exact', 'updates')
    return {name:result[name] for name in fields}


def resource_summary(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    area, measured, missing = 0., 0., 0.
    for previous, current in zip(rows, rows[1:]):
        duration = current['elapsed_s']-previous['elapsed_s']
        if previous['gpu_util_percent'] is None or duration>15:
            missing += duration
        else:
            area += duration*previous['gpu_util_percent']
            measured += duration
    complete = [row['rolling_60s_percent'] for row in rows if row['rolling_60s_percent'] is not None]
    free = [row['gpu_free_mib'] for row in rows if row['gpu_free_mib'] is not None]
    return dict(samples=len(rows), time_weighted_mean_percent=area/measured if measured else None,
        max_complete_60s_mean_percent=max(complete) if complete else None,
        complete_windows=len(complete), measured_seconds=measured, missing_seconds=missing,
        missing_samples=sum(row['gpu_util_percent'] is None for row in rows),
        max_sample_gap_s=max((current['elapsed_s']-previous['elapsed_s'] for previous,current in zip(rows,rows[1:])), default=None),
        min_gpu_free_mib=min(free) if free else None,
        min_ram_available_gib=min(row['ram_available_bytes'] for row in rows)/1024**3,
        started_epoch=rows[0]['utc_epoch'], finished_epoch=rows[-1]['utc_epoch'],
        scope='whole card including unrelated workloads; sampled means, not a quota or process attribution')


def paired_trace_summary(root, trials):
    lookup = {(row['mode'], row['seed'], row['variant']): row for row in trials}
    comparisons = []
    for (mode, seed, variant), row in lookup.items():
        reference = lookup.get((mode, seed, 'a'))
        if variant == 'a' or reference is None or 'trace_sha256' not in row or 'trace_sha256' not in reference:
            continue
        first_path = root.parent / reference.get('trace_directory', root.name) / (reference['attempt']+'.npz')
        second_path = root.parent / row.get('trace_directory', root.name) / (row['attempt']+'.npz')
        with np.load(first_path) as first, np.load(second_path) as second:
            common = min(len(first['command']), len(second['command']))
            prefix = min(reference['prefix_steps'], row['prefix_steps'])
            switch = row['switch_step']
            comparisons.append(dict(mode=mode, seed=seed, reference='a', variant=variant, common_steps=common,
                first_command_max_abs=float(np.abs(first['command'][0]-second['command'][0]).max()),
                whole_command_max_abs=float(np.abs(first['command'][:common]-second['command'][:common]).max()),
                both_observed_gate_always_off=bool(not first['gate'].any() and not second['gate'].any()),
                expert_prefix_commands_equal=bool(prefix == reference['prefix_steps'] == row['prefix_steps'] and
                    np.array_equal(first['command'][:prefix], second['command'][:prefix])) if mode=='expert_rollin' else None,
                switch_nominal_max_abs=float(np.abs(first['base'][switch]-second['base'][switch]).max())
                    if switch is not None and switch == reference['switch_step'] and switch < common else None))
    return comparisons


def pose_difference(first, second):
    first_quaternion, second_quaternion = first[3:].astype(float), second[3:].astype(float)
    cosine = abs(np.dot(first_quaternion, second_quaternion) /
                 (np.linalg.norm(first_quaternion)*np.linalg.norm(second_quaternion)))
    return dict(position_distance_m=float(np.linalg.norm(first[:3]-second[:3])),
                orientation_distance_rad=float(2*np.arccos(np.clip(cosine, 0, 1))))


def switch_physical_summary(root, trials):
    references = {row['seed']:row for row in trials if row['mode']=='expert_rollin' and row['variant']=='a'}
    result = []
    for row in trials:
        if row['mode']!='expert_rollin' or row['variant']=='a' or row.get('switch_step') is None:
            continue
        reference = references[row['seed']]
        first_path = root / reference.get('trace_directory', 'closed_loop') / (reference['attempt']+'_switch.npz')
        second_path = root / row.get('trace_directory', 'closed_loop') / (row['attempt']+'_switch.npz')
        if not first_path.exists() or not second_path.exists():
            continue
        with np.load(first_path) as first, np.load(second_path) as second:
            velocity = first['object_velocity_w_m_s_rad_s']-second['object_velocity_w_m_s_rad_s']
            result.append(dict(seed=row['seed'], variant=row['variant'],
                poses={name:pose_difference(first[name+'_pose_w'], second[name+'_pose_w'])
                       for name in ('object', 'goal', 'palm_body')},
                joint_position_max_rad=float(np.abs(first['joint_position_rad']-second['joint_position_rad']).max()),
                joint_velocity_max_rad_s=float(np.abs(first['joint_velocity_rad_s']-second['joint_velocity_rad_s']).max()),
                object_linear_velocity_distance_m_s=float(np.linalg.norm(velocity[:3])),
                object_angular_velocity_distance_rad_s=float(np.linalg.norm(velocity[3:])),
                finger_force_max_distance_N=float(np.linalg.norm(first['contact']-second['contact'], axis=-1).max())))
    return result


def execution_summary(root, trials):
    result = {}
    for row in trials:
        if 'trace_sha256' not in row:
            continue
        path = root / row.get('trace_directory', 'closed_loop') / (row['attempt']+'.npz')
        with np.load(path) as trace:
            student = trace['student'].astype(bool)
            active = np.flatnonzero(student & trace['gate'].astype(bool))
            adjacent = student[1:] & student[:-1]
            boundary = adjacent & (trace['cursor'][1:]==0)
            within = adjacent & ~boundary
            handoff = student[1:] & ~student[:-1]
            jumps = np.diff(trace['command'], axis=0)
            result[row['attempt']] = dict(student_steps=int(student.sum()), contact_student_steps=len(active),
                first_contact_step=int(active[0]) if len(active) else None,
                last_contact_step=int(active[-1]) if len(active) else None,
                student_native_lifted=bool(trace['lifted'][student].any()),
                jumps={kind:{name:float(np.sqrt(np.mean(jumps[mask, selected]**2))) if mask.any() else None
                    for name,selected in (('arm', slice(0, 7)), ('hand', slice(7, 29)))}
                    for kind,mask in (('student_replan', boundary), ('student_within_plan', within), ('expert_to_student', handoff))})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    index = json.loads((root / 'cache/index.json').read_text())
    pilot = json.loads((root / 'training_pilot/report.json').read_text())
    full = json.loads((root / 'training_full/report.json').read_text())
    test = json.loads((root / 'training_test/report.json').read_text())
    diagnostic_root = root / 'closed_loop_remaining_01' if (root / 'closed_loop_remaining_01/report.json').exists() else root / 'closed_loop'
    closed = json.loads((diagnostic_root / 'report.json').read_text())
    session = json.loads((root / 'session.json').read_text())
    launch = json.loads((diagnostic_root / 'launch.json').read_text())
    if not closed.get('allocation_complete', closed['complete']) or len(closed['trials']) != session['diagnostics_max']:
        raise ValueError('Unresolved diagnostic allocations cannot become a final report')
    aggregate = dict(report_version=1, base_sha256=BASE_HASH, manifest_sha256=index['manifest_sha256'],
        original_formal_results='unchanged B0 0/50 and B1 0/50; no new formal samples',
        cache_index_sha256=sha256(root / 'cache/index.json'), cached_episodes=len(index['entries']),
        cached_steps=sum(row['steps'] for row in index['entries'].values()),
        split_sizes={name:len(value) for name,value in index['splits'].items()},
        architecture=json.loads((root / 'parameter_counts.json').read_text()),
        pilot=dict(epochs=pilot['completed_epochs'], updates=pilot['effective_updates'],
                   validation={name:metric_summary(value) for name,value in pilot['history'][-1]['validation'].items()}),
        full=dict(epochs=full['completed_epochs'], requested_epochs=full['requested_epochs'],
            selected_epoch=full['selected_epoch'], updates=full['effective_updates'],
            selected_updates=next(row['updates'] for row in full['history'] if row['epoch']==full['selected_epoch']),
            episode_visits=full['episode_visits'], unique_training_episodes=full['unique_training_episodes'],
            base_gradients_absent=full['base_gradients_absent'], frozen_base_unchanged=full['frozen_base_unchanged'],
            zero_validation={name:metric_summary(value) for name,value in full['baseline'].items()},
            epochs_metrics=[dict(epoch=row['epoch'], updates=row['updates'],
                train={name:metric_summary(value) for name,value in row['train'].items()},
                validation={name:metric_summary(value) for name,value in row['validation'].items()}) for row in full['history']],
            source_commit=full['source_and_config']['source_commit'],
            source_sha256={Path(name).name:value for name,value in full['source_and_config']['source_sha256'].items()},
            derived_index_at_training_sha256=full['source_and_config']['cache_index_sha256'],
            started_epoch=full['source_and_config']['started_epoch'], finished_epoch=full['finished_epoch']),
        test={name:metric_summary(value) for name,value in test['test'].items()},
        checkpoints={name:sha256(root / 'training_full' / (name+'_best.pt')) for name in ('b', 'c')},
        diagnostic_trials=[{name:value for name,value in row.items() if name not in ('causal_events', 'geometry')}
                           for row in closed['trials']],
        resources=resource_summary(root / 'resources/resources.jsonl'),
        waits=dict(pilot_s=pilot['wait_s'], full_training_s=full['wait_s'], closed_loop_final_process_s=closed['wait_s']),
        causal_validation=json.loads((root / 'causal_validation.json').read_text()),
        limitations=['teacher-forced expert state/history and one cached policy-noise sequence per demonstration',
            'force-masked control retains true tactile gate; not a strict tactile-free ablation',
            'aggregate normal-force contacts have unknown partners',
            'diagnostics are two paired starts per mode; local roll-in is not autonomous success',
            'no critic, replay buffer, reward training, or RL implementation'])
    amplitude = json.loads((root / 'amplitude.json').read_text())
    aggregate['amplitude'] = dict(scales=amplitude['scales'],
        statistics={name:value for name,value in amplitude['statistics'].items() if name!='episodes'})
    aggregate['wait_records'] = [json.loads(path.read_text()) for path in sorted(root.glob('waits_*.json'))]
    historical_waits = root / 'cache_completed_waits.json'
    if historical_waits.exists():
        aggregate['early_stage_waits'] = json.loads(historical_waits.read_text())
    aggregate['session'] = session
    aggregate['gpu_stage'] = json.loads((root / 'gpu_stage.json').read_text())
    aggregate['training_configuration'] = {name:value for name,value in full['source_and_config'].items()
        if name not in ('pid', 'source_sha256', 'parameter_counts')}
    aggregate['closed_loop_source'] = dict(source_commit=launch['source_commit'], checkpoints=launch['checkpoints'],
        source_sha256={Path(name).name:value for name,value in launch['source_sha256'].items()})
    aggregate['diagnostic_completion'] = dict(all_allocations_resolved=True,
        all_trials_naturally_terminated=closed['complete'], naturally_terminated=closed.get('naturally_terminated', 12),
        mismatch_policy=launch.get('mismatch_policy', 'stop_before_student'),
        software_errors=sum(row.get('outcome')=='switch_validation_error' for row in closed['trials']))
    aggregate['paired_trace_comparisons'] = paired_trace_summary(root / 'closed_loop', closed['trials'])
    aggregate['switch_physical_comparisons'] = switch_physical_summary(root, closed['trials'])
    aggregate['execution_metrics'] = execution_summary(root, closed['trials'])
    if diagnostic_root.name != 'closed_loop':
        original_launch = json.loads((root / 'closed_loop/launch.json').read_text())
        aggregate['original_closed_loop_source'] = dict(source_commit=original_launch['source_commit'],
            checkpoints=original_launch['checkpoints'], source_sha256={Path(name).name:value
                for name,value in original_launch['source_sha256'].items()})
    else:
        original_launch = launch
    if (root / 'training_test/report.json').stat().st_mtime > original_launch['started_epoch']:
        raise ValueError('Offline test was not completed before the closed-loop cohort')
    monitor = json.loads((root / 'monitor_final.json').read_text())
    aggregate['budget_observations'] = dict(
        offline_gpu_stage_completion_upper_bound_s=original_launch['started_epoch']-aggregate['gpu_stage']['started_epoch'],
        offline_gpu_stage_limit_s=session['first_gpu_stage_seconds'],
        session_through_monitor_shutdown_s=monitor['ended_epoch']-session['started_epoch'],
        session_limit_s=session['deadline_epoch']-session['started_epoch'])
    aggregate['limitations'].append('matched physical resets do not guarantee pixel-identical observations or bitwise policy replay; action differences alone are not residual effects')
    aggregate['final_disk_free_bytes'] = shutil.disk_usage(root).free
    aggregate['disk_scope'] = '12 GB reserve checked before work; final free space is not a recorded continuous minimum'
    preservation = root / 'preservation.json'
    if preservation.exists():
        aggregate['original_preservation'] = json.loads(preservation.read_text())
    validation = root / 'validation_final.json'
    if validation.exists():
        aggregate['validation'] = json.loads(validation.read_text())
    aggregate['diagnostic_aggregates'] = {}
    for mode in ('autonomous', 'expert_rollin'):
        aggregate['diagnostic_aggregates'][mode] = {}
        for variant in ('a', 'b', 'c'):
            selected = [row for row in closed['trials'] if row['mode']==mode and row['variant']==variant]
            aggregate['diagnostic_aggregates'][mode][variant] = dict(allocated=len(selected),
                evaluated=sum(row.get('success') is not None for row in selected),
                successes=sum(row.get('success') is True for row in selected),
                strict_pair_admission_passes=sum(row.get('paired_comparison_valid', row.get('switch_max_state_error')==0)
                                                 for row in selected) if mode=='expert_rollin' else None,
                first_subgoal_hits=sum(row.get('success') is not None and row.get('max_subgoals', 0)>0
                                       for row in selected))
    atomic_json(root / 'public_report.json', aggregate)
    print(json.dumps(dict(full_epochs=full['completed_epochs'], diagnostics=aggregate['diagnostic_aggregates'],
                          resources=aggregate['resources']), indent=2))


if __name__ == '__main__':
    main()
