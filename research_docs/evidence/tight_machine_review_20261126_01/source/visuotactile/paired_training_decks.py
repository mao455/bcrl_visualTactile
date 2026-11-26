"""Strict own-pool replay and explicit complete-group rear epoch contracts.

All schedules contain actual whole episodes. No padding or synthetic targets
contribute to their update preflight. The opt-in callers retain complete causal
histories,32tick chunks and the original128tick gradient denominator.
"""
from collections import Counter
import copy
import hashlib
import json

import torch

POOL_VERSION = 'paired_own_training_pool_v1'
DECK_VERSION = 'paired_complete_group_decks_v1'
SAMPLING_VERSION = 'paired_dagger_replay_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _ids(values, label, *, empty=False):
    if (not isinstance(values, list) or (not values and not empty)
            or any(not isinstance(i, str) or not i for i in values) or len(set(values)) != len(values)):
        raise ValueError(label+' requires unique actual episode IDs')
    return list(values)


def validate_pool(data, *, role=None):
    """Validate all own-training partitions; no inferred or cross-role labels."""
    pool = data.get('paired_pool')
    keys = {'version', 'role', 'original_ids', 'shared_full_ids', 'own_correction_ids'}
    if not isinstance(pool, dict) or set(pool) != keys or pool['version'] != POOL_VERSION or pool['role'] not in ('A', 'B'):
        raise ValueError('Explicit paired own-pool identity required')
    if role is not None and pool['role'] != role:
        raise ValueError('Paired own-pool role differs')
    train = _ids(data['splits']['train'], 'Own train')
    parts = {k:_ids(pool[k], k, empty=k=='own_correction_ids') for k in ('original_ids', 'shared_full_ids', 'own_correction_ids')}
    flat = sum(parts.values(), [])
    if len(flat) != len(set(flat)) or set(flat) != set(train):
        raise ValueError('Paired partitions must disjointly exhaust own training IDs')
    original = _ids(data.get('normalization_train_ids'), 'Original normalization population')
    if len(original) != 30 or not set(original) <= set(parts['original_ids']):
        raise ValueError('Protected original30 normalization population required')
    if any(set(train) & set(v) for k,v in data['splits'].items() if k!='train'):
        raise ValueError('Own train overlaps held-out split')
    aggregate = data.get('aggregation', {})
    if aggregate.get('version') != 'expert_takeover_aggregate_v1' or aggregate.get('student_prefix_supervised') is not False:
        raise ValueError('Legal takeover aggregate with unsupervised student prefixes required')
    entries = data['entries']
    for i in train:
        if i not in entries:
            raise ValueError('Missing actual trajectory entry')
    for i in original + parts['shared_full_ids']:
        if entries[i].get('trajectory_kind') == 'expert_takeover_v1':
            raise ValueError('Full demonstration partition contains a takeover')
    for i in set(parts['original_ids'])-set(original):
        if entries[i].get('trajectory_kind') != 'expert_takeover_v1':
            raise ValueError('Old correction partition requires actual takeovers')
    for i in parts['own_correction_ids']:
        if entries[i].get('trajectory_kind') != 'expert_takeover_v1' or entries[i].get('source_group') != pool['role']:
            raise ValueError('Own corrections require this student actual takeover provenance')
    return copy.deepcopy(pool)


def validate_pool_pair(data, command_data, *, role, enabled=False):
    """The ordinary equality restriction changes only under this explicit gate."""
    if not enabled:
        if data['splits'] != command_data['splits']:
            raise ValueError('A/B cohort differs')
        return
    own, other = validate_pool(data, role=role), validate_pool(command_data, role='B')
    if (len(data['splits']['validation']) != 6
            or {k:v for k,v in data['splits'].items() if k!='train'} != {k:v for k,v in command_data['splits'].items() if k!='train'}
            or own['original_ids'] != other['original_ids'] or own['shared_full_ids'] != other['shared_full_ids']
            or data['normalization_train_ids'] != command_data['normalization_train_ids']):
        raise ValueError('Paired pools must retain common original/shared/validation populations')
    for i in own['original_ids'] + own['shared_full_ids']:
        a, b = data['entries'][i], command_data['entries'][i]
        if any(a.get(k) != b.get(k) for k in ('path', 'episode', 'source_sha256', 'trajectory_kind')):
            raise ValueError('Shared actual trajectory identities differ')


def mask_bindings(records, ids):
    result = {}
    for i in ids:
        mask = records[i]['supervision_mask'].detach().cpu()
        if mask.dtype != torch.bool or mask.ndim != 1 or len(mask) != len(records[i]['robot']) or not mask.any():
            raise ValueError('Every replay requires a nonempty actual legal tick population')
        result[i] = dict(length=len(mask), legal_ticks=int(mask.sum()), mask_sha256=hashlib.sha256(mask.numpy().tobytes()).hexdigest())
    return result


def group_windows(group, masks):
    """Exact unsplit32tick / >=128tick / whole-group-tail update intervals."""
    if not isinstance(group, list) or not 1 <= len(group) <= 4 or any(i not in masks for i in group):
        raise ValueError('A complete rear group requires1..4 own actual episodes')
    pending, windows, ticks = {}, [], 0
    length = max(len(masks[i]) for i in group)
    for start in range(0, length, 32):
        count = 0
        for i in group:
            indices = masks[i][start:start+32].nonzero().flatten().tolist()
            count += len(indices)
            pending.setdefault(i, set()).update(start+j for j in indices)
        ticks += count
        if ticks and (ticks >= 128 or start+32 >= length):
            windows.append(dict(supervised_ticks=ticks, contributions={i:sorted(js) for i,js in pending.items() if js}))
            pending, ticks = {}, 0
    return windows


def group_updates(group, masks):
    # Keep the preflight inexpensive: count chunks without materializing tick sets.
    pending = updates = 0
    length = max(len(masks[i]) for i in group)
    for start in range(0, length, 32):
        pending += sum(int(masks[i][start:start+32].sum()) for i in group)
        if pending and (pending >= 128 or start+32 >= length):
            updates += 1; pending = 0
    return updates


def _binding(data, records, data_sha256):
    ids = data['splits']['train']
    return dict(data_sha256=data_sha256, train_ids=list(ids), masks=mask_bindings(records, ids),
        sources={i:data['entries'][i]['source_sha256'] for i in ids})


def _counts(groups, ids):
    counts = Counter(i for group in groups for i in group)
    return {i:counts[i] for i in ids}


def build_pair(data_by_role, records_by_role, data_sha256_by_role, *, sampling_by_role, epochs=60, seed=2026111404):
    """Build60 fixed complete decks then add real singleton replays to match updates."""
    from .latest_takeover_sampling import sampling_plan, episode_deck
    if type(epochs) is not int or epochs != 60 or type(seed) is not int:
        raise ValueError('Register60 complete paired epochs and integer schedule seed')
    validate_pool_pair(data_by_role['A'], data_by_role['B'], role='A', enabled=True)
    decks, masks = {}, {}
    for role in ('A', 'B'):
        data, records = data_by_role[role], records_by_role[role]
        validate_pool(data, role=role)
        sampling, counts = sampling_plan(data, sampling_by_role[role])
        if sampling['version'] != SAMPLING_VERSION:
            raise ValueError('Paired group decks require the paired real replay contract')
        ids = data['splits']['train']; masks[role] = {i:records[i]['supervision_mask'].cpu().bool() for i in ids}
        generator = torch.Generator().manual_seed(seed)
        groups = []
        for _ in range(epochs):
            order = episode_deck(ids, counts, generator)
            groups.append([order[j:j+4] for j in range(0, len(order), 4)])
        decks[role] = dict(version=DECK_VERSION, role=role, epochs=epochs, seed=seed,
            binding=_binding(data, records, data_sha256_by_role[role]), sampling=sampling, base_multiplicities=counts,
            epoch_groups=groups, base_group_counts=[len(gs) for gs in groups])
    budget = max(sum(group_updates(g, masks[r]) for g in gs) for r,d in decks.items() for gs in d['epoch_groups'])
    for role, deck in decks.items():
        ids = deck['binding']['train_ids']
        candidates = sorted((i for i in ids if 0 < int(masks[role][i].sum()) <= 128), key=lambda i:(int(masks[role][i].sum()), i))
        extra_counts = Counter()
        for epoch, groups in enumerate(deck['epoch_groups']):
            missing = budget-sum(group_updates(g, masks[role]) for g in groups)
            if missing and not candidates:
                raise ValueError('Matching needs a real episode with1..128 legal ticks; never fabricate padding')
            for _ in range(missing):
                # Spread real visits before reusing the same candidate again.
                identity = min(candidates, key=lambda i:(extra_counts[i], i))
                groups.append([identity]); extra_counts[identity] += 1
        deck['updates_per_epoch'] = budget
        deck['epoch_group_updates'] = [[group_updates(g, masks[role]) for g in gs] for gs in deck['epoch_groups']]
        deck['epoch_multiplicities'] = [_counts(gs, ids) for gs in deck['epoch_groups']]
        deck['extra_replay_visits'] = dict(extra_counts)
        deck['sha256'] = digest(deck)
        validate_deck(deck, data_by_role[role], records_by_role[role], data_sha256=data_sha256_by_role[role])
    return decks


def validate_deck(deck, data, records=None, *, data_sha256=None, role=None):
    """Reject altered data, masks, group boundaries or deterministic base order."""
    from .latest_takeover_sampling import sampling_plan, episode_deck
    if (not isinstance(deck, dict) or deck.get('version') != DECK_VERSION
            or deck.get('sha256') != digest({k:v for k,v in deck.items() if k!='sha256'})
            or deck.get('epochs') != 60 or type(deck.get('seed')) is not int
            or deck.get('role') != validate_pool(data, role=role)['role']):
        raise ValueError('Paired complete group deck identity/hash differs')
    binding = deck['binding']; ids = data['splits']['train']
    if (binding['train_ids'] != ids or set(binding['masks']) != set(ids)
            or binding['sources'] != {i:data['entries'][i]['source_sha256'] for i in ids}
            or data_sha256 is not None and binding['data_sha256'] != data_sha256):
        raise ValueError('Paired deck data/source population binding differs')
    if records is not None and binding != _binding(data, records, binding['data_sha256']):
        raise ValueError('Paired deck actual supervision mask binding differs')
    _, counts = sampling_plan(data, deck['sampling'])
    if counts != deck['base_multiplicities'] or deck['sampling']['version'] != SAMPLING_VERSION:
        raise ValueError('Paired group deck base replay differs')
    if (len(deck['epoch_groups']) != 60 or len(deck['base_group_counts']) != 60
            or len(deck['epoch_multiplicities']) != 60 or len(deck.get('epoch_group_updates', [])) != 60
            or type(deck['updates_per_epoch']) is not int or deck['updates_per_epoch'] <= 0):
        raise ValueError('Exactly60 positive fixed-update epoch group decks required')
    generator = torch.Generator().manual_seed(deck['seed']); extras = Counter()
    masks = {i:records[i]['supervision_mask'].cpu().bool() for i in ids} if records is not None else None
    for index, groups in enumerate(deck['epoch_groups']):
        order = episode_deck(ids, counts, generator)
        base = [order[j:j+4] for j in range(0, len(order), 4)]
        if deck['base_group_counts'][index] != len(base) or groups[:len(base)] != base:
            raise ValueError('Paired deterministic base group/order changed')
        for group in groups:
            if not isinstance(group, list) or not 1 <= len(group) <= 4 or any(i not in ids for i in group):
                raise ValueError('Paired group contains held-out/unknown IDs or invalid group width')
        for group in groups[len(base):]:
            if len(group) != 1 or not 0 < binding['masks'][group[0]]['legal_ticks'] <= 128:
                raise ValueError('Matching replay must be a real1..128legal-tick singleton')
            extras[group[0]] += 1
        if _counts(groups, ids) != deck['epoch_multiplicities'][index]:
            raise ValueError('Paired epoch actual multiplicities differ')
        expected_updates = deck['epoch_group_updates'][index]
        if (len(expected_updates) != len(groups) or any(type(n) is not int or n <= 0 for n in expected_updates)
                or sum(expected_updates) != deck['updates_per_epoch']
                or masks is not None and expected_updates != [group_updates(g, masks) for g in groups]):
            raise ValueError('Paired exact32chunk/128window update preflight differs')
    if dict(extras) != deck['extra_replay_visits']:
        raise ValueError('Paired explicit replay ledger differs')
    return copy.deepcopy(deck)


def completed_counts(deck, epoch, next_group):
    if (type(epoch) is not int or not 1 <= epoch <= deck['epochs'] or type(next_group) is not int
            or not 0 <= next_group <= len(deck['epoch_groups'][epoch-1])):
        raise ValueError('Paired restore requires an exact complete-group boundary')
    groups = sum(deck['epoch_groups'][:epoch-1], []) + deck['epoch_groups'][epoch-1][:next_group]
    return _counts(groups, deck['binding']['train_ids'])


def validate_progress(deck, saved):
    epoch, cursor = saved['epoch'], saved['next_episode_group']
    counts = completed_counts(deck, epoch, cursor)
    order = sum(deck['epoch_groups'][epoch-1], [])
    expected_updates = (epoch-1)*deck['updates_per_epoch'] + sum(deck['epoch_group_updates'][epoch-1][:cursor])
    if saved['episode_order'] != order or saved.get('episode_counts') != counts or saved['updates'] != expected_updates:
        raise ValueError('Paired restored group/order/actual visit ledger differs')
    for key, field in (('supervised_ticks', 'legal_ticks'), ('control_ticks', 'length')):
        if key in saved and saved[key] != sum(counts[i]*deck['binding']['masks'][i][field] for i in counts):
            raise ValueError('Paired restored actual tick ledger differs')
    return counts
