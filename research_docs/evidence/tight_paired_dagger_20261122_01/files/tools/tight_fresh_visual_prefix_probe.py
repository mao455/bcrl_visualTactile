"""CPU reproduction of original30 ridge decoding on actual student prefixes.

This fits diagnostic coefficients only. It never trains or loads a student,
opens test trajectories, regenerates caches, or initializes a simulator/CUDA.
"""
import argparse
import json
import os
from pathlib import Path
import resource
import time

import h5py
import torch

from tools.tight_fresh_position_probe import RIDGE, fit, inputs, labels, predict, read, tensor_sha
from visuotactile.compact import sha256
from visuotactile.narrow_data import aggregate_family_partitions


METHODS = ('visual_only', 'robot178_only', 'combined')
VALIDATION_PHASES = ('all', 'before_contact', 'after_contact')
CORRECTION_PHASES = ('student_prefix', 'student_after_contact', 'legal_expert')


def admit_population(reference, other):
    """All partition/family checks precede any source trajectory/cache payload."""
    if (reference['splits'] != other['splits'] or reference['entries'] != other['entries']
            or reference['normalization_train_ids'] != other['normalization_train_ids']):
        raise ValueError('A/B populations or source entries differ')
    families = aggregate_family_partitions(reference)
    original = reference['normalization_train_ids']
    validation = reference['splits']['validation']
    training = reference['splits']['train']
    if (len(original) != 30 or len(set(original)) != 30 or len(validation) != 6
            or not set(original) <= set(training)):
        raise ValueError('Expected original30 and independent validation6')
    original_families = {reference['entries'][i]['family'] for i in original}
    if len(original_families) != 30:
        raise ValueError('Original30 must represent 30 distinct families')
    corrections = [i for i in training if i not in original]
    for identity in original + validation + corrections:
        entry = reference['entries'][identity]
        if (Path(identity).name != identity or identity in ('.', '..')
                or not isinstance(entry['length'], int) or entry['length'] <= 0
                or len(entry.get('source_sha256', '')) != 64):
            raise ValueError('Malformed source episode metadata')
        if identity in corrections:
            if (entry.get('trajectory_kind') != 'expert_takeover_v1'
                    or entry.get('split') != 'train' or entry['family'] not in original_families):
                raise ValueError('Corrections require known original training families')
        elif entry.get('trajectory_kind') == 'expert_takeover_v1':
            raise ValueError('Original/validation population contains takeover data')
        declared = reference.get('families', {}).get(entry['family'])
        if declared is not None and declared != families[entry['family']]:
            raise ValueError('Declared family partition contradicts split entries')
    return dict(original=original, validation=validation, corrections=corrections)


def authority_masks(control, valid, clean, segment, after_contact):
    """An actual expert authority row is not automatically a legal label."""
    n = len(after_contact)
    if (valid.dtype != torch.bool or clean.dtype != torch.bool or after_contact.dtype != torch.bool
            or control.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)
            or segment.dtype != torch.int64
            or any(value.shape != (n,) for value in (control, valid, clean, segment, after_contact))):
        raise ValueError('Malformed authority/eligibility arrays')
    if (not n or control[0] != 0 or not ((control == 0) | (control == 1)).all()
            or (control[1:] < control[:-1]).any() or not valid.any()
            or (valid & ((control != 1) | ~clean)).any()):
        raise ValueError('Accepted correction requires student prefix and actual clean expert labels')
    change = (control[1:] != control[:-1]) | (valid[1:] != valid[:-1])
    if (change & (segment[1:] == segment[:-1])).any():
        raise ValueError('Authority or eligibility change lacks a new segment')
    return dict(student_prefix=control == 0,
                student_after_contact=(control == 0) & after_contact, legal_expert=valid)


def source_labels(entry, split, correction):
    target, phases, stamp, report = labels(entry)
    truth = target['geometric_aabb_center_palm']
    if (truth.shape != (entry['length'], 3) or not torch.isfinite(truth).all()
            or stamp.shape != (entry['length'],) or not torch.isfinite(stamp).all()
            or (stamp[1:] <= stamp[:-1]).any()):
        raise ValueError('Invalid current-position labels or observation clock')
    masks = dict(all=phases['all'], before_contact=phases['before_first_observed_contact'],
                 after_contact=phases['from_first_observed_contact'])
    segment = None
    with h5py.File(entry['path'], 'r') as stream:
        metadata = json.loads(stream.attrs['metadata_json'])
        group = stream['episodes'][entry['episode']]
        if metadata.get('dataset_split') != split or metadata.get('initial_family') != entry['family']:
            raise ValueError('Recorded source family/split differs from manifest')
        terminal = torch.from_numpy(group['terminated'][:]).bool() | torch.from_numpy(group['truncated'][:]).bool()
        if terminal.shape != stamp.shape or terminal[:-1].any():
            raise ValueError('Source crosses an episode reset/termination boundary')
        if correction:
            dagger = group['transition/dagger']
            control, valid, clean, segment = [torch.from_numpy(dagger[k][:]) for k in
                ('control_source', 'expert_label_valid', 'switch_clean', 'segment_id')]
            masks.update(authority_masks(control, valid, clean, segment, masks['after_contact']))
            report['authority_sha256'] = {k:tensor_sha(v) for k,v in
                dict(control_source=control, expert_label_valid=valid, switch_clean=clean, segment_id=segment).items()}
    report.update(source_manifest_sha256=entry['source_sha256'], family=entry['family'], split=split,
                  time_sha256=tensor_sha(stamp), phase_frames={k:int(v.sum()) for k,v in masks.items()})
    return truth, masks, stamp, segment, report


def cache_binding(directory, residual_statistics_sha):
    index_path = directory / 'backend' / directory.name / 'nominal/index.json'
    index = read(index_path)
    binding = dict(base_checkpoint_sha256=sha256(directory/'training/best.pt'),
                   data_sha256=sha256(directory/'state_data/data.json'),
                   statistics_sha256=sha256(directory/'state_data/statistics.pt'),
                   residual_statistics_sha256=residual_statistics_sha)
    for key, value in binding.items():
        index_key = 'base_sha256' if key == 'base_checkpoint_sha256' else key
        if index.get(index_key) != value:
            raise ValueError('Current model/data/statistics cache binding differs: ' + key)
    return index_path, index, binding


def validate_cache(record, identity, stamp, binding, masks, segment):
    if (record.get('identity') != identity or record.get('cache_identity') != binding
            or not torch.equal(record['time'], stamp)):
        raise ValueError('Cache identity or pre-action timestamp differs')
    for key, width in (('visual', 512), ('robot', 178)):
        value = record[key]
        if value.shape != (len(stamp), width) or not torch.isfinite(value).all() or value.device.type != 'cpu':
            raise ValueError('Invalid CPU feature dimensions or values')
    expected = masks.get('legal_expert', masks['all'])
    if (record['supervision_mask'].dtype != torch.bool
            or not torch.equal(record['supervision_mask'], expected)):
        raise ValueError('Cache action supervision differs from actual legal authority mask')
    if segment is not None and not torch.equal(record['control_segment'], segment):
        raise ValueError('Cached authority segment differs')


def summarize(errors, masks, identities, phases):
    result = {}
    for phase in phases:
        rows = [errors[i][masks[i][phase]] for i in identities if masks[i][phase].any()]
        out = dict(frames=sum(len(v) for v in rows), episodes=len(rows),
                   empty_episodes=len(identities)-len(rows),
                   frame_weighted_vector_rmse_mm=None, episode_equal_mean_vector_rmse_mm=None)
        if rows:
            merged = torch.cat(rows)
            out.update(frame_weighted_vector_rmse_mm=float(merged.square().sum(-1).mean().sqrt()*1000),
                       episode_equal_mean_vector_rmse_mm=sum(float(v.square().sum(-1).mean().sqrt()*1000) for v in rows)/len(rows))
        result[phase] = out
    return result


def fit_original(records, truth, original, method):
    """No correction/validation row can affect probe coefficients or scaling."""
    return fit(torch.cat([inputs(records[i], method) for i in original]),
               torch.cat([truth[i] for i in original]))


def write_new(output, result):
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


def run(root, output):
    if output.exists() or output.is_symlink():
        raise FileExistsError('Preserve previous diagnostic output')
    if torch.cuda.is_initialized():
        raise ValueError('Run this CPU-only diagnostic in a fresh CPU process')
    began = time.monotonic()
    project = Path(__file__).resolve().parents[1]
    names = ('tools/tight_fresh_visual_prefix_probe.py', 'tools/tight_fresh_position_probe.py',
             'visuotactile/native_data.py', 'visuotactile/narrow_data.py', 'visuotactile/compact.py')
    sources = {name:sha256(project/name) for name in names}
    reference = read(root/'A/state_data/data.json')
    population = admit_population(reference, read(root/'B/state_data/data.json'))
    identities = population['original'] + population['validation'] + population['corrections']
    residual_sha = sha256(root/'B/state_data/statistics.pt')
    # Both roles' complete index identities are admitted before HDF5 or .pt reads.
    indexes = {role:cache_binding(root/role, residual_sha) for role in ('A', 'B')}
    truth, masks, stamps, segments, label_reports = {}, {}, {}, {}, {}
    for identity in identities:
        truth[identity], masks[identity], stamps[identity], segments[identity], label_reports[identity] = source_labels(
            reference['entries'][identity], 'validation' if identity in population['validation'] else 'train',
            identity in population['corrections'])
    roles, robot_hashes = {}, {}
    for role in ('A', 'B'):
        index_path, index, binding = indexes[role]
        records, cache_hashes = {}, {}
        for identity in identities:
            path = index_path.parent/(identity+'.pt')
            digest = sha256(path)
            if (digest != index['episodes'][identity]['sha256']
                    or index['episodes'][identity]['ticks'] != len(stamps[identity])):
                raise ValueError('Nominal feature cache file changed')
            record = torch.load(path, map_location='cpu', weights_only=False)
            validate_cache(record, identity, stamps[identity], binding, masks[identity], segments[identity])
            robot_sha = tensor_sha(record['robot'])
            if role == 'A':
                robot_hashes[identity] = robot_sha
            elif robot_hashes[identity] != robot_sha:
                raise ValueError('A/B robot178 features differ')
            records[identity] = {k:record[k].clone() for k in ('visual', 'robot')}
            cache_hashes[identity] = digest
        methods = {}
        for method in METHODS:
            parameters, fit_report = fit_original(records, truth, population['original'], method)
            errors = {i:predict(parameters, inputs(records[i], method))-truth[i] for i in identities}
            methods[method] = dict(fit=fit_report, metrics=dict(
                validation=summarize(errors, masks, population['validation'], VALIDATION_PHASES),
                correction=summarize(errors, masks, population['corrections'], CORRECTION_PHASES)))
        roles[role] = dict(cache_identity=binding, cache_index_sha256=sha256(index_path),
                           cache_sha256=cache_hashes, methods=methods)
        print(json.dumps(dict(role=role, complete=True)), flush=True)
    if torch.cuda.is_initialized() or any(sha256(project/name) != digest for name,digest in sources.items()):
        raise ValueError('CUDA initialization or diagnostic source drift')
    result = dict(version='original30_visual_student_prefix_ridge_v1', source_run=root.parent.name,
        source_round=root.name, population=population, roles=roles, labels=label_reports,
        registration=dict(ridge_lambda=RIDGE, fit_population='original30_only',
            fit_frames=sum(len(truth[i]) for i in population['original']),
            input_dimensions=dict(visual_only=512, robot178_only=178, combined=690),
            fit_scope='Fixed float64 ridge and feature standardization use original30 frames only; no parameter selection.',
            objective='mean_over_frames(sum_xyz((prediction-target)^2)) + 0.01*sum(weights^2); unpenalized intercept'),
        representation='Current nominal cache visual512: last three real RGB-D frames through residual_visual; excludes robot fusion and persistent visual GRU.',
        coordinates=dict(target='Current pre-action collision-geometry local-AABB center in palm-body xyz, meters',
            equation='R(palm_quaternion)^T*(object_root_position+R(object_quaternion)*[0,.025,0]-palm_position)',
            quaternion_order='wxyz', label_role='Training target for diagnostic ridge only; never policy input'),
        phases=dict(student_prefix='Actual control_source==0, including noncontact observations.',
            student_after_contact='Student prefix from first valid observed >0.1 N fingertip contact onward, including later contact loss.',
            legal_expert='Recorded expert_label_valid; actual expert authority and clean switch required. Transition/invalid rows excluded.',
            before_contact='Before first valid observed contact; initial invalid contact excluded.',
            after_contact='From first valid observed contact onward; observed forces may contact any body, not only the object.'),
        aggregation='Frame-weighted vector RMSE and arithmetic mean of per-episode vector RMSE; empty phases reported with null errors.',
        provenance=dict(source_sha256=sources, source_hash_scope='Code used by this diagnostic, not a claim of cache-generation code identity.',
            payload_hash_scope='Full nominal shard/model/statistics/manifest hashes verified; raw HDF5 source SHA is manifest provenance, not rehashed here. Read labels, clocks and authority tensors hashed.',
            robot178_sha256=robot_hashes),
        wall_seconds=time.monotonic()-began,
        process_peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        cuda_initialized=False, student_optimizer_updates=0, simulator_starts=0, test_payloads_read=0,
        interpretation='Corrections were already used by these student models; only ridge fitting excludes them. These are actual-state distribution diagnostics, not held-out generalization or autonomous results. Prefix error can reflect inadequate original30 probe fit coverage, not lost geometric information in visual512. No proof of visual neglect, causal failure, or need to modify vision. Validation6 is reused for student selection/development. Different frame and episode means expose duration weighting.')
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    run(args.round_root, args.output)


if __name__ == '__main__':
    main()
