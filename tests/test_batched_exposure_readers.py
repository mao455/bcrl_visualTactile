"""B2 publication reads actual visit receipts without halving training exposure."""
import copy
import json
from pathlib import Path

import pytest
import torch

from tools import tight_fresh_learning_report as learning
from tools import tight_fresh_stage_exposure as stage
from tools import tight_fresh_stage_fit as fit
from visuotactile.training_receipts import BATCH_ROW_VERSION, learning_counters
from visuotactile.trajectory_training import profile


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def batch_artifacts(tmp_path):
    root = tmp_path/'root'; role = root/'B'; training = role/'training'
    training.mkdir(parents=True)
    # Each epoch has three visits, requiring two optimizer steps. Epoch 2
    # deliberately puts the same correction in both lanes of one batch.
    rows = []; anchors = slots = visits_seen = 0
    for update, identities in enumerate([['old', 'new'], ['new'], ['new', 'new'], ['old']], 1):
        visits = []
        for identity in identities:
            da, dt, frames = (2, 30, 10) if identity == 'old' else (3, 54, 20)
            visits.append(dict(episode=identity, anchors=da, target_ticks=dt,
                visual_frames=frames, loss_weight=1/len(identities)))
            anchors += da; slots += dt; visits_seen += 1
        rows.append(dict(version=BATCH_ROW_VERSION, full_trajectory_batch=profile(), updates=update,
            visits=visits, visits_seen=visits_seen, anchors_seen=anchors, target_ticks=slots,
            visual_frames=sum(v['visual_frames'] for v in visits)))
    data = dict(splits=dict(train=['old', 'new'], validation=['val'], test=['test']),
        normalization_train_ids=['old'], aggregation=dict(new_training_ids=['new']),
        entries={i:dict(source_sha256='source-'+i, family='family-'+i) for i in ['old', 'new']})
    write(role/'state_data/data.json', data)
    torch.save(dict(mean=torch.zeros(1)), role/'state_data/statistics.pt')
    config = dict(version='fresh30_continuous_full_trajectory_batch_v1',
        source_sha256={}, full_trajectory_batch=profile(), episodes_per_epoch=3,
        optimizer_updates_per_epoch=2, episode_multiplicities=dict(old=1, new=2),
        data_sha256=learning.sha256(role/'state_data/data.json'))
    write(training/'configuration.json', config)
    saved = {}
    history = []
    for epoch, filename, limit in [(1, 'best.pt', 2), (2, 'last.pt', 4)]:
        item = dict(kind='dp', group='I-HistoryDP', configuration=config, epoch=epoch,
            train_ids=['old', 'new'], sampling_exposure=dict(old=epoch, new=2*epoch),
            ema=dict(real_tensor=torch.arange(8)), **learning_counters(rows, limit=limit))
        saved[filename] = item; torch.save(item, training/filename)
        history.append(dict(epoch=epoch, train=dict(selection=1.), validation=dict(selection=1.),
            correction=dict(selection=.5), **learning_counters(rows, limit=limit)))
    selected = dict(path=str(training/'best.pt'), sha256=learning.sha256(training/'best.pt'),
        configuration=config, epoch=1, updates=2)
    finished = dict(complete=True, epochs=2, checkpoint_sha256=selected['sha256'],
        full_trajectory_batch=profile(), sampling_exposure=dict(old=2, new=4),
        **learning_counters(rows))
    write(training/'finished.json', finished)
    (training/'learning.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (training/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
    return root, training, selected, rows, finished, saved, data


def test_b2_visits_same_episode_and_singleton_keep_actual_slots(batch_artifacts):
    _, _, _, rows, _, _, _ = batch_artifacts
    counts, buckets = learning.exposure_rows(rows, {'old', 'new'}, {'new'})
    assert counts == dict(old=2, new=4)
    assert buckets['original'] == dict(episode_visits=2, visual_frames=20, anchors_seen=4, target_ticks=60)
    assert buckets['correction'] == dict(episode_visits=4, visual_frames=80, anchors_seen=12, target_ticks=216)
    projected, total = stage.update_counts(rows, ['new'], 3)
    assert projected == dict(new=dict(visits=4, anchors=12, target_slots=216))
    assert total == dict(updates=3, anchors_seen=14, target_ticks=246, visits_seen=5)


def test_b2_role_report_separates_optimizer_steps_and_selected_visits(batch_artifacts):
    root, _, _, _, _, _, _ = batch_artifacts
    result = learning.role_report(root, 'B')
    assert 'actual_optimizer_episodes' not in result
    assert result['actual_optimizer_updates'] == 4
    assert result['actual_episode_visits'] == 6
    assert result['selected_episode_visits'] == 3
    assert result['selected_sampling_exposure'] == dict(old=1, new=2)
    assert not result['correction_metrics_match_selected']
    assert not result['test_used'] and not torch.cuda.is_initialized()


def test_b2_selected_last_reconcile_real_cpu_checkpoints(batch_artifacts):
    _, training, selected, rows, finished, _, _ = batch_artifacts
    receipt = stage.reconcile_checkpoints(training, selected, rows, finished)
    assert receipt['selected']['counters'] == dict(updates=2, anchors_seen=8, target_ticks=138, visits_seen=3)
    assert receipt['last']['counters'] == dict(updates=4, anchors_seen=16, target_ticks=276, visits_seen=6)
    assert all(v['all_episode_visits_match'] for v in receipt.values())


@pytest.mark.parametrize('fault', ['missing_visits', 'update_equals_visits', 'wrong_multiplicity',
    'missing_sampling', 'same_total_wrong_episode', 'wrong_optimizer_budget'])
def test_b2_source_and_selected_reject_mislabeled_exposure(batch_artifacts, fault):
    _, _, _, rows, _, checkpoints, _ = batch_artifacts
    saved = copy.deepcopy(checkpoints['best.pt'])
    if fault == 'missing_visits': saved.pop('visits_seen')
    elif fault == 'update_equals_visits': saved['updates'] = saved['visits_seen']
    elif fault == 'wrong_multiplicity': saved['configuration']['episode_multiplicities'] = dict(old=2, new=1)
    elif fault == 'missing_sampling': saved.pop('sampling_exposure')
    elif fault == 'same_total_wrong_episode': saved['sampling_exposure'] = dict(old=2, new=1)
    elif fault == 'wrong_optimizer_budget': saved['configuration']['optimizer_updates_per_epoch'] = 3
    with pytest.raises(ValueError): learning.batch_checkpoint_exposure(saved, rows)
    with pytest.raises(ValueError): fit.validate_batch_checkpoint_visits(saved)


def test_b2_source_exposure_uses_raw_identity_and_visits_not_optimizer_count(batch_artifacts):
    _, training, _, _, _, checkpoints, data = batch_artifacts
    current = copy.deepcopy(data); current['entries']['renamed'] = current['entries'].pop('new')
    rows, receipt = fit.source_training_exposure(dict(path=str(training/'best.pt')),
        checkpoints['best.pt'], current, ['renamed'])
    assert rows == dict(renamed=dict(source_training_id='new', source_seen=True, full_history_visits=2))
    assert 'distinct from optimizer' in receipt['authority']


@pytest.mark.parametrize('fault', ['local_slots', 'local_frames', 'cumulative_visits', 'mixed_mode'])
def test_malformed_b2_row_cannot_enter_source_buckets_or_stage_projection(batch_artifacts, fault):
    _, _, _, original, _, _, _ = batch_artifacts; rows = copy.deepcopy(original)
    if fault == 'local_slots': rows[0]['visits'][0]['target_ticks'] += 1
    elif fault == 'local_frames': rows[0]['visits'][0]['visual_frames'] += 1
    elif fault == 'cumulative_visits': rows[0]['visits_seen'] += 1
    elif fault == 'mixed_mode': rows[0] = dict(episode='old', updates=1, anchors_seen=5, target_ticks=84, visual_frames=30)
    with pytest.raises(ValueError): learning.exposure_rows(rows, {'old', 'new'}, {'new'})
    with pytest.raises(ValueError): stage.update_counts(rows, ['new'], 4)


def test_b2_cannot_be_inferred_from_legacy_checkpoint_or_unseen_episode(batch_artifacts):
    _, _, _, rows, _, checkpoints, _ = batch_artifacts
    saved = copy.deepcopy(checkpoints['best.pt']); saved['configuration'].pop('full_trajectory_batch')
    with pytest.raises(ValueError, match='explicit checkpoint profile'):
        learning.batch_checkpoint_exposure(saved, rows)
    rows = copy.deepcopy(rows); rows[0]['visits'][0]['episode'] = 'test'
    with pytest.raises(ValueError, match='Unexpected episode'):
        learning.exposure_rows(rows, {'old', 'new'}, {'new'})


def test_old_rows_keep_original_public_bucket_schema():
    rows = [dict(episode='old', updates=1, anchors_seen=2, target_ticks=30, visual_frames=10)]
    _, buckets = learning.exposure_rows(rows, {'old'}, set())
    assert buckets == dict(original=dict(optimizer_episodes=1, visual_frames=10, anchors_seen=2, target_ticks=30),
        correction=dict(optimizer_episodes=0, visual_frames=0, anchors_seen=0, target_ticks=0))
