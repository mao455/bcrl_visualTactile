"""Read recorded full-student shadow errors at actual expert recovery onset.

CPU-only numeric analysis: no model inference, training, or simulated replay.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import h5py
import numpy as np

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


BINS = ('first12', 'next12', 'after24')
PARTS = {'arm': slice(0, 7), 'hand': slice(7, 29)}
COHORTS = ('expert_success', 'expert_failure', 'no_expert_takeover',
           'student_success_without_takeover', 'no_legal_expert_labels')


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def finish(sums, ticks):
    result = {'ticks': ticks}
    for name, part in PARTS.items():
        count = ticks * (part.stop - part.start)
        predicted, held = sums[name]
        result[name] = dict(
            shadow_rmse_rad=float(np.sqrt(predicted / count)) if count else None,
            previous_target_rmse_rad=float(np.sqrt(held / count)) if count else None,
            shadow_to_previous_target_rmse_ratio=float(np.sqrt(predicted / held)) if held else None,
            shadow_squared_error_rad2=predicted,
            previous_target_squared_error_rad2=held)
    return result


def fit_eligible(row):
    """Successful expert control alone is insufficient without executed labels."""
    return bool(row['expert_takeover_success'] and row['legal_expert_ticks'] > 0)


def population(cases):
    """Keep the registered opportunity denominator separate from fitting rows."""
    return dict(completed_opportunities=len(cases),
        takeover_cases=sum(row['masked_expert_transition_ticks'] + row['legal_expert_ticks'] > 0 for row in cases),
        legal_label_cases=sum(row['legal_expert_ticks'] > 0 for row in cases),
        zero_label_cases=sum(row['legal_expert_ticks'] == 0 for row in cases),
        successful_legal_suffix_cases=sum(fit_eligible(row) for row in cases),
        legal_expert_ticks=sum(row['legal_expert_ticks'] for row in cases),
        cohorts={cohort: sum(row['cohort'] == cohort for row in cases) for cohort in COHORTS})


def read_case(collection, case, pair):
    directory = collection / case['name']
    result = json.loads((directory / 'result.json').read_text())
    if not result.get('legal') or not result.get('software_complete'):
        raise ValueError('Expected completed legal recovery: ' + case['name'])
    if result['complete_student_policy'] != pair or result['checkpoint_sha256'] != pair['backend_sha256']:
        raise ValueError('Recorded full student differs from collection plan')
    if result['identity'] != case['name'] or result['split'] != 'train':
        raise ValueError('Expected registered training-family recovery')
    numeric = directory / 'numeric.h5'
    with h5py.File(numeric) as f:
        transition = f['episode/transition']
        dagger = transition['dagger']
        mask_raw = dagger['expert_label_valid'][:]
        if not np.isin(mask_raw, (0, 1)).all():
            raise ValueError('Invalid legal supervision mask')
        mask = mask_raw.astype(bool)
        source = dagger['control_source'][:]
        if mask.ndim != 1 or not len(mask) or source.shape != mask.shape:
            raise ValueError('Expected nonempty complete control-source and mask arrays')
        indices = np.flatnonzero(mask)
        student_ticks = int((source == 0).sum())
        expert_ticks = len(mask) - student_ticks
        if (not np.isin(source, (0, 1)).all()
                or not np.array_equal(source == 0, np.arange(len(source)) < student_ticks)
                or not (source[mask] == 1).all()
                or (len(indices) and not np.array_equal(indices, np.arange(indices[0], len(mask))))):
            raise ValueError('Expected one student prefix and continuous legal expert suffix')
        segment = dagger['segment_id'][:]
        if segment.shape != mask.shape or (len(indices) and len(np.unique(segment[mask])) != 1):
            raise ValueError('Recovery bins must not bridge control segments')
        for name in ('switch_clean', 'expert_source_is_expert_issued'):
            provenance = dagger[name][:]
            if (provenance.shape != mask.shape or not np.isin(provenance, (0, 1)).all()
                    or not provenance[mask].astype(bool).all()):
                raise ValueError('Selected labels include an uncleared expert queue')
        predicted = dagger['student_proposed_target_rad'][:].astype(np.float64)
        target = dagger['u_expert'][:].astype(np.float64)
        previous = f['episode/obs/actuator_target_rad'][:].astype(np.float64)
        actual = transition['target_after_rad'][:].astype(np.float64)
        if any(x.shape != (len(mask), 29) for x in (predicted, target, previous, actual)):
            raise ValueError('Expected complete absolute target arrays')
        if (not all(np.isfinite(x).all() for x in (predicted, previous, actual))
                or not np.isfinite(target[mask]).all()):
            raise ValueError('Nonfinite recovery command')
        if not np.array_equal(actual[:-1], previous[1:]):
            raise ValueError('Previous-target baseline lacks actual execution continuity')
        label_error = float(np.abs(target[mask] - actual[mask]).max()) if len(indices) else None
        if label_error is not None and label_error > 1e-6:
            raise ValueError('Expert label differs from actual executed target')
        if (len(mask) != result['steps'] or result['residual_forward_ticks'] != len(mask)
                or result['dagger']['expert_valid_ticks'] != len(indices)
                or result['dagger']['student_ticks'] != student_ticks):
            raise ValueError('Saved result and numeric exposure differ')
        expert_success = bool(result['success'] and expert_ticks > 0)
        expected = dict(expert_ticks=expert_ticks, takeover_triggered=expert_ticks > 0,
            expert_switch_masked_ticks=expert_ticks - len(indices),
            expert_takeover_success=expert_success,
            student_autonomous_success=bool(result['success'] and expert_ticks == 0))
        if any(key in result['dagger'] and result['dagger'][key] != value for key, value in expected.items()):
            raise ValueError('Saved result and actual takeover outcome differ')
        if not expert_ticks:
            cohort = 'student_success_without_takeover' if result['success'] else 'no_expert_takeover'
        elif not len(indices):
            cohort = 'no_legal_expert_labels'
        else:
            cohort = 'expert_success' if expert_success else 'expert_failure'
        bins = {}
        for name, rows in zip(BINS, (indices[:12], indices[12:24], indices[24:])):
            sums = {part: [float(np.square(predicted[rows, section] - target[rows, section]).sum()),
                           float(np.square(previous[rows, section] - target[rows, section]).sum())]
                    for part, section in PARTS.items()}
            bins[name] = dict(finish(sums, len(rows)),
                first_completed_tick=int(rows[0]) + 1 if len(rows) else None,
                last_completed_tick=int(rows[-1]) + 1 if len(rows) else None)
        return dict(case=case['name'], role=case['role'], family=result['family'],
            expert_takeover_success=expert_success, cohort=cohort,
            student_ticks=student_ticks, masked_expert_transition_ticks=expert_ticks - len(indices),
            legal_expert_ticks=len(indices), bins=bins,
            frontend_sha256=pair['frontend_sha256'], backend_sha256=pair['backend_sha256'],
            numeric_sha256=sha256(numeric), result_sha256=sha256(directory / 'result.json'),
            legal_mask_sha256_uint8=array_sha(mask.astype(np.uint8)),
            control_source_sha256_uint8=array_sha(source.astype(np.uint8)),
            legal_indices_sha256_int64_le=array_sha(indices.astype('<i8')),
            label_vs_actual_max_abs_rad=label_error)


def report(collection):
    plan = json.loads((collection / 'plan.json').read_text())
    completed = json.loads((collection / 'results.json').read_text())
    if plan.get('kind') != 'takeover' or plan.get('test_used'):
        raise ValueError('Expected training-only takeover collection')
    if len(completed) != len(plan['cases']) or {r['case'] for r in completed} != {c['name'] for c in plan['cases']}:
        raise ValueError('Collection is incomplete or case population differs')
    for pair in plan['pairs'].values():
        for name in ('frontend', 'backend'):
            if sha256(Path(pair[name])) != pair[name + '_sha256']:
                raise ValueError('Saved student model differs from collection plan')
    cases = [read_case(collection, case, plan['pairs'][case['role']]) for case in plan['cases']]
    roles = {}
    for role in sorted(plan['pairs']):
        roles[role] = {'population': population([row for row in cases if row['role'] == role])}
        for cohort in COHORTS:
            selected = [row for row in cases if row['role'] == role and row['cohort'] == cohort]
            bins = {}
            for name in BINS:
                ticks = sum(row['bins'][name]['ticks'] for row in selected)
                sums = {part: [sum(row['bins'][name][part][key] for row in selected)
                               for key in ('shadow_squared_error_rad2', 'previous_target_squared_error_rad2')]
                        for part in PARTS}
                bins[name] = finish(sums, ticks)
            roles[role][cohort] = dict(cases=len(selected), bins=bins,
                first12_hand_worse_than_previous_target_cases=sum(
                    row['bins']['first12']['hand']['shadow_rmse_rad'] > row['bins']['first12']['hand']['previous_target_rmse_rad']
                    for row in selected if row['bins']['first12']['ticks']))
    return dict(version='recorded_full_student_recovery_onset_v1',
        collection_source_commit=plan['source_commit'],
        report_source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        report_source_sha256=sha256(Path(__file__)),
        plan_sha256=sha256(collection / 'plan.json'), results_sha256=sha256(collection / 'results.json'),
        completed_cases=len(cases), population=population(cases), roles=roles, cases=cases,
        bin_definition='Ordinal first 12 legal expert ticks, next 12, remaining ticks. Masked queue transitions excluded; one contiguous legal segment required.',
        aggregation='Squared errors summed over actual ticks and joint dimensions, then square root; not an average of per-case RMSE.',
        tick_index='1-based completed control transitions; mask and index hashes explicitly name byte encoding.',
        empty_label_rule='All completed opportunities remain in the denominator. No takeover and no legal expert labels are separate cohorts; empty-bin RMSE and label discrepancy are null, and these cases never enter suffix fitting.',
        scope='Saved old full DP+backend shadow proposals on actual expert recovery observations and actual executed history. After takeover the student proposals do not control the robot. Expert completion and shadow error are not autonomous student outcomes. Different bins contain different physical states, so an error difference does not establish its cause.',
        optimizer_updates=0, model_forward_calls=0, simulator_starts=0, test_used=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    atomic_json(args.output, report(args.collection))


if __name__ == '__main__':
    main()
