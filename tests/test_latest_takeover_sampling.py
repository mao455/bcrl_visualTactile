import copy
import io
import json
from collections import Counter

import pytest
import torch

from visuotactile.latest_takeover_sampling import (
    SAMPLING_VERSION, episode_deck, sampling_plan, validate_sampling_exposure,
)


def cohort():
    return dict(splits=dict(train=['demo0', 'demo1', 'old', 'latest0', 'latest1'],
        validation=['validation'], test=['test'], correction_dev=['correction_dev']),
        normalization_train_ids=['demo0', 'demo1'],
        aggregation=dict(version='expert_takeover_aggregate_v1', student_prefix_supervised=False,
            new_training_ids=['latest0', 'latest1']),
        entries={identity: dict(trajectory_kind='expert_takeover_v1')
            for identity in ['old', 'latest0', 'latest1']})


def spec(repeat=4):
    return dict(version=SAMPLING_VERSION, latest_repeat=repeat)


@pytest.mark.parametrize('contract', [None, spec(1)])
def test_default_and_repeat_one_preserve_old_order_and_rng_bytes(contract):
    data = cohort()
    _, multiplicities = sampling_plan(data, contract)
    old = torch.Generator().manual_seed(2026111402)
    new = torch.Generator().manual_seed(2026111402)
    global_before = torch.get_rng_state().clone()
    for _ in range(4):
        indices = torch.randperm(len(data['splits']['train']), generator=old).tolist()
        expected = [data['splits']['train'][index] for index in indices]
        assert episode_deck(data['splits']['train'], multiplicities, new) == expected
        assert torch.equal(old.get_state(), new.get_state())
    assert torch.equal(global_before, torch.get_rng_state())


@pytest.mark.parametrize('repeat', [1, 2, 4])
def test_only_latest_takeovers_repeat_without_mutating_inputs(repeat):
    data = cohort()
    requested = spec(repeat)
    before = copy.deepcopy((data, requested))
    validated, multiplicities = sampling_plan(data, requested)
    assert validated == requested and validated is not requested
    assert multiplicities == dict(demo0=1, demo1=1, old=1, latest0=repeat, latest1=repeat)
    deck = episode_deck(data['splits']['train'], multiplicities, torch.Generator().manual_seed(7))
    assert Counter(deck) == multiplicities
    assert len(deck) == 3 + 2 * repeat
    assert (data, requested) == before


def test_absent_contract_needs_no_takeover_metadata():
    assert sampling_plan(dict(splits=dict(train=['demo'])), None) == (None, {'demo': 1})


def test_checkpointed_schedule_restores_next_deck_and_visit_counts():
    data = cohort()
    contract, multiplicities = sampling_plan(data, spec())
    generator = torch.Generator().manual_seed(813)
    exposure = dict.fromkeys(multiplicities, 0)
    for _ in range(2):
        for identity in episode_deck(data['splits']['train'], multiplicities, generator):
            exposure[identity] += 1
    validate_sampling_exposure(exposure, multiplicities, completed_epochs=2, updates=22)
    buffer = io.BytesIO()
    torch.save(dict(schedule_rng=generator.get_state(), sampling_exposure=exposure,
        sampling_contract=contract, episode_multiplicities=multiplicities), buffer)
    expected = episode_deck(data['splits']['train'], multiplicities, generator)
    expected_after = generator.get_state().clone()
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    continuation_generator = torch.Generator().manual_seed(999)
    continuation_generator.set_state(restored['schedule_rng'])
    assert restored['sampling_contract'] == contract
    validate_sampling_exposure(restored['sampling_exposure'], restored['episode_multiplicities'],
        completed_epochs=2, updates=22)
    assert episode_deck(data['splits']['train'], multiplicities, continuation_generator) == expected
    assert torch.equal(continuation_generator.get_state(), expected_after)


@pytest.mark.parametrize('contract', [spec(0), spec(3), spec(8), spec(True), spec(4.0),
    dict(version='unknown', latest_repeat=4), dict(version=SAMPLING_VERSION),
    dict(spec(), latest_ids=['latest0']), [], 'latest_takeover_repeat_v1'])
def test_invalid_contract_is_rejected(contract):
    with pytest.raises(ValueError, match='Sampling contract'):
        sampling_plan(cohort(), contract)


@pytest.mark.parametrize('mutation', ['duplicate_train', 'duplicate_new', 'empty_new', 'missing_new',
    'unknown_new', 'original_new', 'validation_new', 'heldout_overlap', 'fake_takeover',
    'missing_original', 'unknown_original', 'student_labels', 'unknown_aggregate'])
def test_invalid_population_or_provenance_is_rejected(mutation):
    data = cohort()
    if mutation == 'duplicate_train': data['splits']['train'].append('demo0')
    elif mutation == 'duplicate_new': data['aggregation']['new_training_ids'].append('latest0')
    elif mutation == 'empty_new': data['aggregation']['new_training_ids'] = []
    elif mutation == 'missing_new': del data['aggregation']['new_training_ids']
    elif mutation == 'unknown_new': data['aggregation']['new_training_ids'] = ['missing']
    elif mutation == 'original_new': data['aggregation']['new_training_ids'] = ['demo0']
    elif mutation == 'validation_new': data['aggregation']['new_training_ids'] = ['validation']
    elif mutation == 'heldout_overlap': data['splits']['validation'].append('latest0')
    elif mutation == 'fake_takeover': data['entries']['latest0']['trajectory_kind'] = 'student'
    elif mutation == 'missing_original': del data['normalization_train_ids']
    elif mutation == 'unknown_original': data['normalization_train_ids'] = ['missing']
    elif mutation == 'student_labels': data['aggregation']['student_prefix_supervised'] = True
    elif mutation == 'unknown_aggregate': data['aggregation']['version'] = 'unknown'
    with pytest.raises(ValueError):
        sampling_plan(data, spec())


@pytest.mark.parametrize('multiplicities', [dict(a=1), dict(a=1, b=1, extra=4),
    dict(a=True, b=4), dict(a=1, b=3), dict(a=1, b=4.0)])
def test_deck_rejects_invalid_multiplicities_before_consuming_rng(multiplicities):
    generator = torch.Generator().manual_seed(7)
    before = generator.get_state().clone()
    with pytest.raises(ValueError):
        episode_deck(['a', 'b'], multiplicities, generator)
    assert torch.equal(before, generator.get_state())


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'wrong_count', 'bool_count', 'bad_updates'])
def test_resume_rejects_incomplete_or_inconsistent_visit_ledger(mutation):
    _, multiplicities = sampling_plan(cohort(), spec())
    counts = {identity: 2 * count for identity, count in multiplicities.items()}
    updates = 22
    if mutation == 'missing': del counts['latest0']
    elif mutation == 'extra': counts['unexpected'] = 0
    elif mutation == 'wrong_count': counts['latest0'] -= 1
    elif mutation == 'bool_count': counts['demo0'] = True
    elif mutation == 'bad_updates': updates = 21
    with pytest.raises(ValueError, match='Sampling exposure'):
        validate_sampling_exposure(counts, multiplicities, completed_epochs=2, updates=updates)


@pytest.mark.parametrize('mutation',[None,'started','different_population'])
def test_cycle_binds_both_frontends_before_training_and_leaves_backend_uniform(tmp_path,mutation):
    from tools.tight_fresh_cycle import configure_frontend_sampling
    for role in ('A','B'):
        directory=tmp_path/role;(directory/'state_data').mkdir(parents=True)
        data=cohort()
        if role=='B' and mutation=='different_population':data['aggregation']['new_training_ids']=['latest0']
        (directory/'state_data/data.json').write_text(json.dumps(data))
        (directory/'continuation.json').write_text(json.dumps(dict(path='source',sha256='digest')))
    if mutation=='started':(tmp_path/'B/training').mkdir()
    if mutation:
        with pytest.raises(ValueError):configure_frontend_sampling(tmp_path,2)
        for role in ('A','B'):
            assert 'sampling_contract' not in json.loads((tmp_path/role/'continuation.json').read_text())
    else:
        receipt=configure_frontend_sampling(tmp_path,2)
        assert receipt['backend_latest_repeat']==1
        assert receipt['roles']['A']['episode_multiplicities']==receipt['roles']['B']['episode_multiplicities']
        assert receipt['roles']['A']['episode_multiplicities']['latest0']==2
        with pytest.raises(ValueError):configure_frontend_sampling(tmp_path,2)
