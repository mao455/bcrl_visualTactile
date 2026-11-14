"""Opt-in loss weights for actual fresh-cohort expert recovery onsets.

This does not change sampling, validation, histories, supervision eligibility,
or the fixed physical-tick optimizer denominator.
"""
import math

import torch

from .anchor_sampling import recovery_prefix_rows, supervision_rows
from .fresh_contracts import FRESH_AGGREGATE_CONTRACT, residual_normalization
from .residual_tick_window import TICK_WINDOW_CONTRACT


FRESH_RECOVERY_WEIGHTING_VERSION = 'fresh_actual_takeover_onset_weighting_v1'


def fresh_recovery_weighting_config(value):
    if value is None:
        return None
    if (not isinstance(value, dict)
            or set(value) != {'version', 'ticks', 'backend_weight'}
            or value['version'] != FRESH_RECOVERY_WEIGHTING_VERSION):
        raise ValueError('Unknown fresh recovery weighting contract')
    ticks, weight = value['ticks'], value['backend_weight']
    if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
        raise ValueError('Fresh recovery ticks must be a positive integer')
    if (isinstance(weight, bool) or not isinstance(weight, (int, float))
            or not math.isfinite(weight) or weight < 1):
        raise ValueError('Fresh recovery backend weight must be finite and at least one')
    return dict(value)


def fresh_recovery_exposure(records, masks, counts, contract):
    """Count real legal observations separately from their loss weight mass."""
    contract = fresh_recovery_weighting_config(contract)
    if contract is None:
        raise ValueError('Fresh recovery exposure requires an explicit contract')
    if any(i not in masks or isinstance(n, bool) or not isinstance(n, int) or n < 0
            for i, n in counts.items()):
        raise ValueError('Fresh recovery exposure requires audited masks and episode counts')
    legal = sum(int(supervision_rows(records[i])[0].sum()) * n for i, n in counts.items())
    initial = sum(int(masks[i].sum()) * n for i, n in counts.items())
    return dict(contract=contract, supervised_ticks=legal,
        initial_recovery_supervised_ticks=initial, remaining_supervised_ticks=legal-initial,
        loss_weight_mass=legal+(contract['backend_weight']-1)*initial,
        loss_weight_mass_is_exposure=False, episode_draws=sum(counts.values()))


def fresh_recovery_plan(data, configuration, records):
    """Validate the actual aggregate and build train-only, legal onset masks.

    Missing opt-in leaves all historical paths untouched. Original episodes are
    identified only by the frozen normalizer population; every extra episode
    must declare an actual expert takeover with explicit control provenance.
    """
    contract = fresh_recovery_weighting_config(configuration.get('fresh_recovery_weighting'))
    if contract is None:
        return None, {}, None
    if configuration.get('fresh_data_contract') != FRESH_AGGREGATE_CONTRACT:
        raise ValueError('Fresh recovery weighting requires the fresh actual-takeover aggregate')
    if (data.get('sampling_strata') is not None or configuration.get('sampling_strata') is not None
            or configuration.get('recovery_prefix') is not None):
        raise ValueError('Fresh recovery weighting cannot change sampling or combine legacy recovery_prefix')
    if configuration.get('tick_window_contract') != TICK_WINDOW_CONTRACT:
        raise ValueError('Fresh recovery weighting requires the unchanged fixed128 tick window')
    residual_normalization(data, configuration)
    originals = set(data['normalization_train_ids'])
    masks = {}
    for identity in data['splits']['train']:
        if identity not in records:
            raise ValueError('Fresh recovery training record is missing')
        record = records[identity]
        if identity in originals:
            masks[identity] = torch.zeros(len(record['robot']), dtype=torch.bool)
            continue
        entry = data['entries'][identity]
        if (entry.get('trajectory_kind') != 'expert_takeover_v1'
                or not {'supervision_mask', 'control_segment', 'time'} <= set(record)):
            raise ValueError('Fresh recovery corrections require actual expert control provenance')
        legal, _ = supervision_rows(record)
        if len(legal) != len(record['robot']) or not legal.any():
            raise ValueError('Fresh recovery correction has no aligned legal expert ticks')
        masks[identity] = recovery_prefix_rows(record, contract['ticks']).cpu()
    population = fresh_recovery_exposure(records, masks, {i: 1 for i in masks}, contract)
    return contract, masks, population
