"""Explicit train-only, per-visit geometry sampling over real legal ticks."""
import math

import torch

VERSION = 'legal_tick_half_phase_v1'
STAGES = ('grasp_without_contact', 'contact_not_held', 'held_far', 'held_rotating',
    'held_near_goal60', 'after_first_native_subgoal', 'contact_lost')


def profile(max_samples=None):
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError('Geometry cap must be None or a positive integer')
    return dict(version=VERSION, fraction=.5, max_samples=max_samples, commit=12,
        population='unique legal identity/tick in this visit selected anchors commit12',
        allocation='proportional stage population with one per nonempty stage when feasible',
        rounding='ceil population/2; stage floor then largest remaining proportional deficit',
        reduction='mean selected unique configurations; no additional stage weights',
        rng='explicit independent CPU generator per visit; never model/diffusion RNG',
        labels='recorded diagnostic phases for sampling only; never network inputs')


def validate_profile(value):
    if value is None: return None
    if not isinstance(value, dict) or value != profile(value.get('max_samples')):
        raise ValueError('Exact legal_tick_half_phase_v1 profile required')
    return dict(value)


def diagnostic_phases(pose, palm, force, before_goal, after_goal, before_success, after_success, keypoint):
    """Exclusive after-transition labels; holding is distinct from native lift.

    The reference is original completed transitions 25--48. No contact loss is
    labeled a physical drop; goal completion only labels the following stage.
    """
    import numpy as np
    from .takeover_trigger import contact_height_windows
    values = [np.asarray(x) for x in (pose, palm, force, before_goal, after_goal,
        before_success, after_success, keypoint)]
    pose, palm, force, before_goal, after_goal, before_success, after_success, keypoint = values
    n = len(pose)
    if ([x.shape for x in values] != [(n,7),(n,7),(n,2,5,3),(n,7),(n,7),(n,),(n,),(n,)]
            or not all(np.isfinite(x).all() for x in values)):
        raise ValueError('Aligned finite real transition diagnostics required for geometry phases')
    contact = np.linalg.norm(force[:, -1], axis=-1) > .1
    current = contact.any(-1); ever = np.maximum.accumulate(current)
    phases = np.zeros(n, dtype=np.int64)
    phases[current] = 1; phases[ever & ~current] = 6
    held = np.zeros(n, dtype=bool); rotating = np.zeros(n, dtype=bool)
    if n >= 48:
        held = contact_height_windows(pose, force, float(np.median(pose[24:48,2])))
        held[:47] = False
        inverse = palm[:,3:].copy(); inverse[:,1:] *= -1
        q = pose[:,3:]
        relative = np.concatenate((inverse[:,:1]*q[:,:1]-(inverse[:,1:]*q[:,1:]).sum(-1,keepdims=True),
            inverse[:,:1]*q[:,1:]+q[:,:1]*inverse[:,1:]+np.cross(inverse[:,1:],q[:,1:])),axis=-1)
        for tick in np.flatnonzero(held):
            window = relative[tick-11:tick+1]
            cosine = np.clip(np.abs((window*window[0]).sum(-1)),0,1)
            rotating[tick] = (2*np.arccos(cosine)).max() > .2
    phases[held] = 2; phases[held & rotating] = 3
    same = (before_goal == after_goal).all(-1) & (before_success == after_success)
    phases[held & same & (keypoint <= .06)] = 4
    phases[before_success > 0] = 5
    return torch.from_numpy(phases)


def _quotas(populations, count, generator):
    # Floors preserve proportional allocation. Minimum one makes rare recorded
    # phases visible; actual per-stage rates are reported, not called unbiased.
    keys = list(populations); total = sum(populations.values())
    if count < len(keys): raise ValueError('Geometry budget cannot cover all nonempty phases')
    ideal = {key:count*size/total for key,size in populations.items()}
    result = {key:max(1, math.floor(ideal[key])) for key in keys}
    tie_order = torch.randperm(len(keys), generator=generator).tolist()
    rank = {keys[index]:position for position,index in enumerate(tie_order)}
    while sum(result.values()) > count:
        pool = [key for key in keys if result[key] > 1]
        key = max(pool, key=lambda k:(result[k]-ideal[k],-rank[k])); result[key] -= 1
    while sum(result.values()) < count:
        pool = [key for key in keys if result[key] < populations[key]]
        key = max(pool, key=lambda k:(ideal[k]-result[k],-rank[k])); result[key] += 1
    return result


def select(objective, selected, lengths, *, configuration, generator):
    """Return commit mask and exact sampled identity/tick receipt, without replacement."""
    configuration = validate_profile(configuration)
    if configuration is None or not isinstance(generator, torch.Generator) or generator.device.type != 'cpu':
        raise ValueError('Explicit sampling profile and independent CPU generator required')
    records = getattr(objective, 'supervision_records', None)
    if records is None: raise ValueError('New geometry sampling requires actual legal supervision records')
    from .anchor_sampling import labels
    by_identity = {}
    for row,(identity,tick) in enumerate(selected):
        if type(tick) is not int or tick < 0 or tick >= lengths[identity]:
            raise ValueError('Real selected planning anchors required')
        by_identity.setdefault(identity, []).append((row,tick))
    candidates = {}; raw_slots = 0; episode_population = {}
    for identity, rows in by_identity.items():
        record = records[identity]; context = objective.loss_context[identity]
        phase = context.get('sampling_phase')
        n = lengths[identity]
        if (phase is None or phase.dtype != torch.long or phase.shape != (n,)
                or ((phase < 0) | (phase >= len(STAGES))).any() or len(record['time']) != n):
            raise ValueError('Exact recorded phase labels aligned with the real episode required')
        _, valid = labels(record, torch.tensor([tick for _,tick in rows],device=record['command'].device),24)
        valid = valid.detach().cpu()
        legal = record.get('supervision_mask', torch.ones(n,dtype=torch.bool)).detach().cpu().bool()
        episode_population[identity] = int(legal.sum())
        for local,(row,tick) in enumerate(rows):
            for offset in range(min(12,n-tick)):
                if not bool(valid[local,offset]): continue
                raw_slots += 1; real_tick = tick+offset
                candidates.setdefault((identity,real_tick), dict(identity=identity,tick=real_tick,
                    stage=STAGES[int(phase[real_tick])],row=row,offset=offset))
    if not candidates: raise ValueError('No legal committed geometry configurations')
    pools = {}
    for item in candidates.values(): pools.setdefault(item['stage'],[]).append(item)
    count = max(math.ceil(len(candidates)*configuration['fraction']), len(pools))
    if configuration['max_samples'] is not None: count = min(count,configuration['max_samples'])
    quotas = _quotas({key:len(pool) for key,pool in pools.items()},count,generator)
    chosen = []
    for key,pool in pools.items():
        chosen.extend(pool[index] for index in torch.randperm(len(pool),generator=generator)[:quotas[key]].tolist())
    chosen.sort(key=lambda x:(x['row'],x['offset']))
    mask = torch.zeros((len(selected),24),dtype=torch.bool)
    for item in chosen: mask[item['row'],item['offset']] = True
    report = dict(version=VERSION, selected_count=len(chosen),eligible_commit_count=raw_slots,
        eligible_unique_commit_count=len(candidates),episode_legal_tick_count=sum(episode_population.values()),
        episode_legal_tick_count_by_identity=episode_population,
        duplicate_commit_slots_removed=raw_slots-len(candidates),
        selected_ticks=[{k:item[k] for k in ('identity','tick','stage')} for item in chosen],
        selected_locations=[[item['row'],item['offset']] for item in chosen],
        phase_counts={key:dict(eligible=len(pool),selected=quotas[key],fraction=quotas[key]/len(pool)) for key,pool in pools.items()},
        actual_fraction_of_candidate_ticks=len(chosen)/len(candidates),
        sampling=configuration, normalization='equal selected unique configuration mean; no extra stage weights',
        population_unbiased=False, rate_note='Small-stage minimum-one and integer rounding can change stage selection rates; see phase_counts')
    return mask,report


def validate_mask(objective, selected, mask):
    """Explicit locations obey the same real-tick legality/uniqueness rule."""
    from .anchor_sampling import labels
    if mask.dtype != torch.bool or mask.shape != (len(selected),24) or mask[:,12:].any():
        raise ValueError('Geometry mask must select committed H24 row/offset locations')
    seen = set(); masks = {}; rows = []
    for row,offset in mask.nonzero().tolist():
        identity,anchor = selected[row]; tick = anchor+offset
        record = objective.supervision_records[identity]
        if row not in masks:
            _,valid=labels(record,torch.tensor([anchor],device=record['command'].device),24)
            masks[row]=valid[0].detach().cpu()
        if not bool(masks[row][offset]) or (identity,tick) in seen:
            raise ValueError('Explicit geometry contains illegal or duplicated real ticks')
        seen.add((identity,tick))
        phase=objective.loss_context[identity]['sampling_phase']
        rows.append(dict(identity=identity,tick=tick,stage=STAGES[int(phase[tick])]))
    return rows
