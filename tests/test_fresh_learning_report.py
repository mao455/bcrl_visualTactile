import pytest

from tools.tight_fresh_learning_report import exposure_rows, learning_sources


def test_second_round_counts_prior_recoveries_as_corrections():
    data = dict(splits=dict(train=['demo', 'old_recovery', 'new_recovery']),
        normalization_train_ids=['demo'], aggregation=dict(new_training_ids=['new_recovery']))
    corrections, current, groups = learning_sources(data)
    assert corrections == {'old_recovery', 'new_recovery'}
    assert current == {'new_recovery'}
    rows = [dict(episode=identity, updates=i+1, visual_frames=10+i,
        anchors_seen=2*(i+1), target_ticks=24*(i+1))
        for i, identity in enumerate(data['splits']['train'])]
    _, total = exposure_rows(rows, set(groups), corrections)
    assert total['original']['optimizer_episodes'] == 1
    assert total['correction']['optimizer_episodes'] == 2
    assert total['correction']['target_ticks'] == 48
    _, detail = exposure_rows(rows, set(groups), corrections, source_groups=groups)
    assert detail['historical_correction']['visual_frames'] == 11
    assert detail['current_correction']['visual_frames'] == 12
    assert sum(r['target_ticks'] for r in detail.values()) == rows[-1]['target_ticks']


def test_original_affine_population_cannot_be_mislabeled_as_current_correction():
    data = dict(splits=dict(train=['demo']), normalization_train_ids=['demo'],
        aggregation=dict(new_training_ids=['demo']))
    with pytest.raises(ValueError, match='overlap'):
        learning_sources(data)
