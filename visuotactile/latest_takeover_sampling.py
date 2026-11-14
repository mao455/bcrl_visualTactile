"""Explicit episode reuse for latest legal takeovers; no synthetic samples.

The absent contract and repeat=1 each consume exactly one historical-size
randperm call. A visit is a complete episode traversal, not a cached prefix.
Callers retain the original supervision masks and recompute learned history.
"""

import torch


SAMPLING_VERSION = 'latest_takeover_repeat_v1'
REPEAT_FACTORS = (1, 2, 4)


def _identities(values, label):
    if (not isinstance(values, (list, tuple)) or not values
            or any(not isinstance(identity, str) or not identity for identity in values)
            or len(set(values)) != len(values)):
        raise ValueError(label + ' requires nonempty unique episode IDs')
    return list(values)


def sampling_plan(data, spec):
    """Return a copied validated opt-in spec and per-epoch episode visit counts."""
    identities = _identities(data['splits']['train'], 'Training population')
    multiplicities = dict.fromkeys(identities, 1)
    if spec is None:
        return None, multiplicities
    if (not isinstance(spec, dict) or set(spec) != {'version', 'latest_repeat'}
            or spec['version'] != SAMPLING_VERSION
            or type(spec['latest_repeat']) is not int
            or spec['latest_repeat'] not in REPEAT_FACTORS):
        raise ValueError('Sampling contract requires latest_takeover_repeat_v1 and integer repeat 1, 2 or 4')
    aggregation = data.get('aggregation', {})
    if (aggregation.get('version') != 'expert_takeover_aggregate_v1'
            or aggregation.get('student_prefix_supervised') is not False):
        raise ValueError('Latest sampling requires an actual legal expert takeover aggregate')
    latest = _identities(aggregation.get('new_training_ids'), 'Latest takeovers')
    original = _identities(data.get('normalization_train_ids'), 'Original affine population')
    training = set(identities)
    if not set(original) <= training or not set(latest) <= training:
        raise ValueError('Original and latest IDs must belong to training')
    if set(original) & set(latest):
        raise ValueError('Original and latest takeover IDs overlap')
    if any(training & set(values) for split, values in data['splits'].items() if split != 'train'):
        raise ValueError('Training IDs overlap a held-out split')
    if any(data.get('entries', {}).get(identity, {}).get('trajectory_kind') != 'expert_takeover_v1'
            for identity in latest):
        raise ValueError('Latest IDs require actual expert takeover trajectory provenance')
    for identity in latest:
        multiplicities[identity] = spec['latest_repeat']
    return dict(spec), multiplicities


def episode_deck(ids, multiplicities, generator):
    """Shuffle one base population plus extra copies with one randperm call."""
    identities = _identities(ids, 'Episode deck')
    if (set(multiplicities) != set(identities)
            or any(type(value) is not int or value not in REPEAT_FACTORS
                for value in multiplicities.values())):
        raise ValueError('Episode multiplicities require exact IDs and integer repeat 1, 2 or 4')
    deck = list(identities)
    for identity in identities:
        deck.extend([identity] * (multiplicities[identity] - 1))
    return [deck[index] for index in torch.randperm(len(deck), generator=generator).tolist()]


def validate_sampling_exposure(exposure, multiplicities, *, completed_epochs, updates):
    """Validate the completed-epoch visit ledger before saving or restoring it."""
    if (type(completed_epochs) is not int or completed_epochs < 0
            or type(updates) is not int or updates != completed_epochs * sum(multiplicities.values())
            or not isinstance(exposure, dict) or set(exposure) != set(multiplicities)
            or any(type(exposure[identity]) is not int
                or exposure[identity] != completed_epochs * count
                for identity, count in multiplicities.items())):
        raise ValueError('Sampling exposure differs from the complete epoch deck')
