"""Reconstruct stage target-slot exposure and reconcile it with real update rows.

CPU only. Uses the registered train-only projection and frozen target-mask code.
Does not rewrite the earlier stage-fit registration's then-unknown exposure.
"""
import argparse
import json
from pathlib import Path

from tools.tight_fresh_selected_fit import read, sha256, checked_hash, exclusive_json, import_frozen_trainer
from tools.tight_fresh_learning_report import batch_checkpoint_exposure, sampled_visit_coverage
from visuotactile.training_receipts import BATCH_ROW_VERSION, iter_visits, learning_counters

COUNTERS = ('updates', 'anchors_seen', 'target_ticks')


def require(value, message):
    if not value:
        raise ValueError(message)


def update_counts(rows, identities, limit):
    counts = {i: dict(visits=0, anchors=0, target_slots=0) for i in identities}
    if any(row.get('version') == BATCH_ROW_VERSION for row in rows[:limit]):
        require(all(row.get('version') == BATCH_ROW_VERSION for row in rows[:limit]),
            'Mixed single-trajectory and B2 optimizer receipts')
        for visit in iter_visits(rows, limit=limit):
            if visit['episode'] in counts:
                value = counts[visit['episode']]
                value['visits'] += 1
                value['anchors'] += visit['anchors']
                value['target_slots'] += visit['target_ticks']
        return counts, learning_counters(rows, limit=limit)
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
    if rows[0].get('version') == BATCH_ROW_VERSION:
        require(finished.get('visits_seen') == learning_counters(rows)['visits_seen'],
            'Learning terminal visit counter differs from finished training')
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
        batch_total = batch_checkpoint_exposure(saved, rows)
        visited = (v['episode'] for v in iter_visits(rows, limit=limit)) if batch_total is not None else (
            row['episode'] for row in rows[:limit])
        require(all(identity in saved['sampling_exposure'] for identity in visited),
            label+' learning row is outside checkpoint training identities')
        counts, total = update_counts(rows, identities, limit)
        require(all(saved[key] == total[key] for key in total),
            label+' checkpoint counters differ from actual learning rows')
        require(all(saved['sampling_exposure'][identity] == value['visits']
            for identity, value in counts.items()),
            label+' checkpoint per-episode sampling exposure differs from actual learning rows')
        coverage=sampled_visit_coverage(rows,saved['configuration'].get('training_sampling'),limit=limit)
        if coverage is not None:
            require(coverage.state_dict()==saved.get('visit_sampling_coverage'),
                label+' checkpoint random sampling coverage differs from actual learning rows')
        receipts[label] = dict(checkpoint=filename, epoch=epoch, counters=total,
            training_episodes_checked=len(identities), all_episode_visits_match=True,
            loader='torch.load(map_location=cpu, mmap=True)', weight_bytes_rehashed=False)
        del saved
    return receipts


def sampled_stage_totals(records, stage_masks, names, rows, limit, profile, label_fn):
    """Reconstruct each actual sampled H24 window with the same legal mask."""
    import torch
    from visuotactile.anchor_sampling import sample_training_anchors
    sampled_visit_coverage(rows,profile,limit=limit)
    result={i:dict(visits=0,anchors=0,target_slots=0,stages={name:0 for name in names}) for i in records}
    for visit in iter_visits(rows,limit=limit):
        identity=visit['episode']
        if identity not in records:continue
        record=records[identity]
        anchors,receipt=sample_training_anchors(record,profile['anchors'],identity=identity,
            epoch=visit['epoch'],visit_index=visit['episode_index'])
        require(receipt==visit['anchor_sampling'],'Actual random starts differ from bound legal population/seed: '+identity)
        _,valid=label_fn(record,anchors,24)
        masks=stage_masks[identity]
        require(masks.dtype==torch.bool and masks.shape==(len(names),len(record['time'])) and
            torch.equal(masks[-1],record['supervision_mask']),'Stage masks differ from actual legal supervision')
        require(len(anchors)==visit['anchors'] and int(valid.sum())==visit['target_ticks'],
            'Actual sampled target masks disagree with learning increments: '+identity)
        indices=(anchors[:,None]+torch.arange(24)).clamp_max(len(record['time'])-1)
        stages={name:int((masks[j][indices]&valid).sum()) for j,name in enumerate(names)}
        require(stages['legal_expert']==int(valid.sum()),'Sampled stage slots include invalid targets')
        value=result[identity];value['visits']+=1;value['anchors']+=len(anchors);value['target_slots']+=int(valid.sum())
        for name in names:value['stages'][name]+=stages[name]
    return result


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
    sampling=selected['configuration'].get('training_sampling')
    populations = {}
    stage_masks={}
    names = [*plan['stages'], 'legal_expert']
    with np.load(plan['masks_path'], allow_pickle=False) as archive:
        for identity in ids:
            record = records[identity]
            masks = torch.from_numpy(archive[identity])
            stage_masks[identity]=masks
            if sampling is not None:continue
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
        episodes = (sampled_stage_totals(records,stage_masks,names,rows,limit,sampling,trainer.labels)
            if sampling is not None else {})
        for identity in ids:
            if sampling is not None:
                require(all(episodes[identity][k]==counts[identity][k] for k in ('visits','anchors','target_slots')),
                    'Actual sampled stage totals differ from observed visit counters')
                if label=='selected':require(episodes[identity]['visits']==plan['episodes'][identity]['selected_full_history_visits'],
                    'Selected visit count differs')
                continue
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
    if selected['configuration'].get('full_trajectory_batch') is not None:
        result.update(full_trajectory_batch=selected['configuration']['full_trajectory_batch'],
            visit_semantics='visits_seen counts full-trajectory visits; updates counts actual optimizer/EMA transactions, including one-visit odd tails. Target slots are exposure, not the per-trajectory normalized gradient weight.')
    if sampling is not None:
        result.update(version='reconstructed_stage_target_exposure_v2',training_sampling=sampling,
            anchor_source='Actual per-visit selected_anchors receipts, checked against deterministic seed and legal action population.',
            semantics='H24 target slots at actual recorded random legal training starts, with original legal-suffix masks. Repeated window slots are exposure, not unique actions, loss weight, gradient magnitude or geometry exposure. Deployment remains5Hz; no grid-population times visits inference.')
    exclusive_json(Path(output), result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = run(**vars(args))
    print(json.dumps(dict(role=result['role'], cuda_initialized=result['cuda_initialized'],
        selected=result['checkpoints']['selected']['strata']['all'])))
