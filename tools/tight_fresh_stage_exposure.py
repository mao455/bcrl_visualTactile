"""Reconstruct stage target-slot exposure and reconcile it with real update rows.

CPU only. Uses the registered train-only projection and frozen target-mask code.
Does not rewrite the earlier stage-fit registration's then-unknown exposure.
"""
import argparse
import json
from pathlib import Path

from tools.tight_fresh_selected_fit import read, sha256, checked_hash, exclusive_json, import_frozen_trainer

COUNTERS = ('updates', 'anchors_seen', 'target_ticks')


def require(value, message):
    if not value:
        raise ValueError(message)


def update_counts(rows, identities, limit):
    counts = {i: dict(visits=0, anchors=0, target_slots=0) for i in identities}
    previous = dict(updates=0, anchors_seen=0, target_ticks=0)
    for row in rows[:limit]:
        require(row['updates'] == previous['updates'] + 1, 'Optimizer row gap or duplicated update')
        da = row['anchors_seen'] - previous['anchors_seen']
        dt = row['target_ticks'] - previous['target_ticks']
        require(da > 0 and dt > 0, 'Nonpositive observed supervision increment')
        if row['episode'] in counts:
            value = counts[row['episode']]
            value['visits'] += 1
            value['anchors'] += da
            value['target_slots'] += dt
        previous = row
    require(previous['updates'] == limit, 'Selected checkpoint exceeds recorded updates')
    return counts, {k: previous[k] for k in COUNTERS}


def reconcile_checkpoints(training, selected, rows, finished):
    """Read actual checkpoint metadata on CPU; never infer last from best.

    The registered best digest remains the policy identity. These additional
    checks reconcile counters and every training episode's visits, including
    episodes outside the stage-mask projection; they do not hash weight bytes.
    """
    import torch
    training = Path(training)
    require(finished.get('complete') is True and rows and
        len(rows) == finished['updates'] and
        finished['checkpoint_sha256'] == selected['sha256'],
        'Completed actual selected training required')
    require(all(rows[-1][key] == finished[key] for key in COUNTERS),
        'Learning terminal counters differ from finished training')
    require(Path(selected['path']).resolve(strict=True) == (training/'best.pt').resolve(strict=True),
        'Registered selected checkpoint is not the actual training best')
    receipts = {}
    for label, filename, epoch, limit in (
            ('selected', 'best.pt', selected['epoch'], selected['updates']),
            ('last', 'last.pt', finished['epochs'], finished['updates'])):
        saved = torch.load(training/filename, map_location='cpu', mmap=True, weights_only=False)
        require(saved.get('kind') == 'dp' and saved.get('group') == 'I-HistoryDP' and
            saved['configuration'] == selected['configuration'] and
            saved['epoch'] == epoch and saved['updates'] == limit,
            label+' checkpoint metadata differs from the registered/completed training')
        identities = saved['train_ids']
        require(identities and len(identities) == len(set(identities)) and
            set(saved.get('sampling_exposure', {})) == set(identities),
            label+' checkpoint lacks exact per-episode sampling exposure')
        require(all(row['episode'] in saved['sampling_exposure'] for row in rows[:limit]),
            label+' learning row is outside checkpoint training identities')
        counts, total = update_counts(rows, identities, limit)
        require(all(saved[key] == total[key] for key in COUNTERS),
            label+' checkpoint counters differ from actual learning rows')
        require(all(saved['sampling_exposure'][identity] == value['visits']
            for identity, value in counts.items()),
            label+' checkpoint per-episode sampling exposure differs from actual learning rows')
        receipts[label] = dict(checkpoint=filename, epoch=epoch, counters=total,
            training_episodes_checked=len(identities), all_episode_visits_match=True,
            loader='torch.load(map_location=cpu, mmap=True)', weight_bytes_rehashed=False)
        del saved
    return receipts


def run(registration, output):
    import numpy as np
    import torch
    path = Path(registration).resolve(strict=True)
    plan = read(path)
    require(plan['version'] == 'train_only_selected_frontend_stage_fit_v3', 'Bound union registration required')
    selected = plan['selected']; frozen = Path(selected['frozen_source_root'])
    for name, digest in selected['configuration']['source_sha256'].items():
        checked_hash(frozen / name, digest)
    trainer = import_frozen_trainer(dict(frozen_source_root=str(frozen)))
    from visuotactile.anchor_sampling import planning_anchors
    from visuotactile.target_windows import with_target_window
    projection = plan['train_projection']; dr = Path(projection['root'])
    checked_hash(dr/'data.json', projection['data_sha256'])
    checked_hash(dr/'numeric.pt', projection['numeric_sha256'])
    checked_hash(plan['masks_path'], plan['masks_sha256'])
    data = read(dr/'data.json'); ids = plan['identities']
    require(data['splits'] == dict(train=ids, validation=[], test=[]), 'Only registered training projection allowed')
    container = torch.load(dr/'numeric.pt', map_location='cpu', mmap=True, weights_only=False)
    require(container['data_sha256'] == projection['data_sha256'], 'Numeric projection binding differs')
    records = with_target_window(container['records'], plan['selected_receipt']['target_window'])
    training = Path(plan['root'])/plan['role']/'training'
    rows = [json.loads(line) for line in (training/'learning.jsonl').read_text().splitlines()]
    finished = read(training/'finished.json')
    checkpoint_checks = reconcile_checkpoints(training, selected, rows, finished)
    populations = {}
    names = [*plan['stages'], 'legal_expert']
    with np.load(plan['masks_path'], allow_pickle=False) as archive:
        for identity in ids:
            record = records[identity]
            anchors = planning_anchors(record)
            _, valid = trainer.labels(record, anchors, 24)
            masks = torch.from_numpy(archive[identity])
            require(masks.dtype == torch.bool and masks.shape == (len(names), len(record['time'])) and
                torch.equal(masks[-1], record['supervision_mask']), 'Stage masks differ from actual legal supervision')
            indices = (anchors[:, None]+torch.arange(24)).clamp_max(len(record['time'])-1)
            stage_counts = {name:int((masks[j][indices] & valid).sum()) for j,name in enumerate(names)}
            require(stage_counts['legal_expert'] == int(valid.sum()), 'Stage slots include invalid expert targets')
            populations[identity] = dict(anchors=len(anchors), target_slots=int(valid.sum()), stages=stage_counts)
    checkpoints = {}
    for label, limit in [('selected', selected['updates']), ('last', finished['updates'])]:
        counts, total = update_counts(rows, ids, limit)
        episodes = {}
        for identity in ids:
            observed = counts[identity]; pop = populations[identity]; visits = observed['visits']
            require(observed['anchors'] == pop['anchors']*visits and
                observed['target_slots'] == pop['target_slots']*visits,
                'Reconstructed masks disagree with actual optimizer supervision increments: '+identity)
            if label == 'selected':
                require(visits == plan['episodes'][identity]['selected_full_history_visits'], 'Selected visit count differs')
            episodes[identity] = dict(**observed, stages={k:v*visits for k,v in pop['stages'].items()})
        strata = {}
        for name, members in plan['strata'].items():
            stages = {stage:sum(episodes[i]['stages'][stage] for i in members) for stage in names}
            strata[name] = dict(trajectories=len(members), visits=sum(episodes[i]['visits'] for i in members),
                target_slots=stages['legal_expert'], target_slot_fraction_of_all_training=stages['legal_expert']/total['target_ticks'],
                stages=stages)
        checkpoints[label] = dict(all_training=total, episodes=episodes, strata=strata)
    result = dict(version='reconstructed_stage_target_exposure_v1', role=plan['role'],
        registration_sha256=sha256(path), masks_sha256=plan['masks_sha256'],
        learning_sha256=sha256(training/'learning.jsonl'), finished_sha256=sha256(training/'finished.json'),
        evaluator_sha256=sha256(__file__), checkpoint_sha256=selected['sha256'], checkpoints=checkpoints,
        checkpoint_metadata_checks=checkpoint_checks,
        optimizer_updates_performed=0, cuda_initialized=torch.cuda.is_initialized(),
        semantics='H24 action-target slots at actual 5Hz/onset anchors, including repeated future slots, reconciled per-episode against real learning rows through selected/last update. Stage masks overlap. This reconstructs action-target exposure, not loss weight, gradient magnitude, random diffusion timestep, geometry subsampling, or cumulative inherited training. Full-history visits and physical unique actions are separate quantities.',
        previous_unknown='Earlier immutable stage-fit registration remains unchanged; this additional audit reconstructs its previously unreported stage target-slot counts.')
    exclusive_json(Path(output), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = run(**vars(args))
    print(json.dumps(dict(role=result['role'], cuda_initialized=result['cuda_initialized'],
        selected=result['checkpoints']['selected']['strata']['all'])))
