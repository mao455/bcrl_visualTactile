"""Real CPU checkpoint metadata must agree with observed optimizer rows."""
import copy
from pathlib import Path

import pytest
import torch

from tools import tight_fresh_stage_exposure as exposure


@pytest.fixture
def checkpoints(tmp_path):
    training = tmp_path/'training'; training.mkdir()
    rows = []
    anchors = slots = 0
    for update, identity in enumerate(('historical', 'new', 'historical', 'new'), 1):
        da, dt = (2, 30) if identity == 'historical' else (3, 54)
        anchors += da; slots += dt
        rows.append(dict(updates=update, episode=identity, anchors_seen=anchors, target_ticks=slots))
    configuration = dict(source_sha256={}, experiment='actual full-episode visits')
    for filename, limit in [('best.pt', 2), ('last.pt', 4)]:
        saved = dict(kind='dp', group='I-HistoryDP', configuration=configuration,
            epoch=limit//2, train_ids=['historical', 'new'],
            sampling_exposure=dict(historical=limit//2, new=limit//2),
            ema=dict(real_tensor=torch.arange(8)),
            **{key: rows[limit-1][key] for key in exposure.COUNTERS})
        torch.save(saved, training/filename)
    selected = dict(path=str(training/'best.pt'), sha256=exposure.sha256(training/'best.pt'),
        epoch=1, updates=2, configuration=configuration)
    finished = dict(complete=True, epochs=2, checkpoint_sha256=selected['sha256'],
        **{key: rows[-1][key] for key in exposure.COUNTERS})
    return training, selected, rows, finished


def test_selected_and_last_load_separately_on_cpu_with_every_episode_checked(checkpoints, monkeypatch):
    load = torch.load; calls = []
    def checked_load(path, **kwargs):
        assert kwargs == dict(map_location='cpu', mmap=True, weights_only=False)
        calls.append(Path(path).name)
        return load(path, **kwargs)
    monkeypatch.setattr(torch, 'load', checked_load)
    result = exposure.reconcile_checkpoints(*checkpoints)
    assert calls == ['best.pt', 'last.pt']
    assert result['selected']['counters'] == dict(updates=2, anchors_seen=5, target_ticks=84)
    assert result['last']['counters'] == dict(updates=4, anchors_seen=10, target_ticks=168)
    assert all(row['training_episodes_checked'] == 2 and row['all_episode_visits_match']
        for row in result.values())
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('key', ['anchors_seen', 'target_ticks'])
def test_finished_drift_rejected_before_checkpoint_load(checkpoints, monkeypatch, key):
    training, selected, rows, finished = checkpoints
    finished[key] += 1
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('Drift must fail before checkpoint read'))
    with pytest.raises(ValueError, match='terminal counters'):
        exposure.reconcile_checkpoints(training, selected, rows, finished)


@pytest.mark.parametrize('filename,field', [
    ('best.pt', 'anchors_seen'), ('last.pt', 'target_ticks'),
    ('best.pt', 'epoch'), ('last.pt', 'configuration'),
    ('best.pt', 'sampling_exposure'), ('last.pt', 'sampling_exposure')])
def test_actual_checkpoint_drift_is_not_hidden_by_valid_learning_rows(checkpoints, filename, field):
    training, selected, rows, finished = checkpoints
    saved = torch.load(training/filename, map_location='cpu', weights_only=False)
    if field == 'sampling_exposure':
        # Same total visits, wrong episode allocation. Historical data outside
        # the stage-fit union must be checked too, not just the new trajectory.
        saved[field]['historical'] -= 1
        saved[field]['new'] += 1
    elif field == 'configuration':
        saved[field] = dict(saved[field], experiment='different source')
    else:
        saved[field] += 1
    torch.save(saved, training/filename)
    with pytest.raises(ValueError, match='checkpoint'):
        exposure.reconcile_checkpoints(training, selected, rows, finished)


def test_missing_last_cannot_fall_back_to_registered_best(checkpoints):
    training, selected, rows, finished = checkpoints
    (training/'last.pt').unlink()
    with pytest.raises(FileNotFoundError):
        exposure.reconcile_checkpoints(training, selected, rows, finished)


def test_learning_episode_outside_real_checkpoint_cannot_disappear_from_counts(checkpoints):
    training, selected, rows, finished = checkpoints
    rows = copy.deepcopy(rows); rows[0]['episode'] = 'unregistered'
    with pytest.raises(ValueError, match='outside checkpoint training identities'):
        exposure.reconcile_checkpoints(training, selected, rows, finished)
