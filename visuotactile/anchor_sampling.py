"""Episode-diverse causal anchors. No changes to observation or action contracts."""
import math
import torch


class AnchorSampler:
    """One anchor per distinct episode; shuffled episode and per-episode anchor decks.

    Each cohort-sized deck visits every training episode once, including a short
    final batch without dropping or copying episodes. Each episode's native
    12-tick anchors are exhausted before its anchor deck is reshuffled. Thus this
    is episode-balanced, not uniform over all anchors when episodes differ in length.
    An anchor-equivalent epoch is an exposure count, not independent new data.
    """
    def __init__(self, lengths, generator, batch_size=64, *, records=None, strata=None, weights=None, recovery_prefix=None):
        self.ids=list(lengths);self.lengths=dict(lengths);self.generator=generator
        if not self.ids or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Positive batch and nonempty episode cohort required')
        if any(n < 1 for n in lengths.values()):
            raise ValueError('Empty episodes are not training examples')
        self.batch_size=batch_size
        self.order=[];self.cursor=0;self.decks={};self.positions={}
        self.anchors={i:list(range(0,n,12)) for i,n in lengths.items()}
        self.supervised_ticks=sum(lengths.values())
        target_masks={}
        if records is not None:
            self.supervised_ticks=0
            for identity,n in lengths.items():
                mask,_=supervision_rows(records[identity])
                mask=mask.cpu()
                if len(mask)!=n:raise ValueError('Sampler record length mismatch')
                self.anchors[identity]=planning_anchors(records[identity]).cpu().tolist()
                self.supervised_ticks+=int(mask.sum())
                from .target_windows import window_mask
                _,segment=supervision_rows(records[identity])
                target_masks[identity]=window_mask(mask,segment.cpu(),torch.tensor(self.anchors[identity],dtype=torch.long),24,
                    records[identity].get('target_window_contract'))
            if any(not a for a in self.anchors.values()):
                raise ValueError('Every training episode needs a supervised native planning anchor')
        self.counts={i:[0]*len(a) for i,a in self.anchors.items()}
        self.draws=0;self.prefix_ticks=0;self.gradient_ticks=0
        self.strata,self.weights=sampling_strata(self.ids,strata,weights)
        self.stratum_orders={};self.stratum_positions={};self.stratum_draws={k:0 for k in self.strata}
        self.recovery_prefix=recovery_prefix_config(recovery_prefix)
        self.recovery_pools={};self.recovery_decks={};self.recovery_positions={};self.recovery_draws={}
        self.recovery_target_counts={}
        if self.recovery_prefix:
            if records is None or not self.strata:raise ValueError('Recovery sampling requires explicit records and correction strata')
            for name,identities in self.strata.items():
                if name=='core':continue
                for identity in identities:
                    prefix=recovery_prefix_rows(records[identity],self.recovery_prefix['ticks']).cpu()
                    offsets=torch.tensor(self.anchors[identity],dtype=torch.long)[:,None]+torch.arange(24)
                    early_targets=target_masks[identity] & prefix[offsets.clamp_max(len(prefix)-1)]
                    initial=early_targets[:,:12].any(-1).tolist()
                    pools={key:[j for j,t in enumerate(self.anchors[identity]) if initial[j]==early]
                        for key,early in (('initial',True),('remaining',False))}
                    self.recovery_pools[identity]={k:v for k,v in pools.items() if v}
                    self.recovery_draws[identity]={k:0 for k in self.recovery_pools[identity]}
                    self.recovery_decks[identity]={};self.recovery_positions[identity]={}
                    self.recovery_target_counts[identity]=early_targets.sum(-1).tolist()
        self.target_counts={i:(target_masks[i].sum(-1).tolist() if records is not None else
            [min(24,lengths[i]-t) for t in self.anchors[i]]) for i in self.ids}

    def draw(self):
        if self.strata:
            chosen=[]
            for _ in range(self.batch_size):
                # Cumulative quota deficit avoids rounding away a small stratum.
                total=sum(self.stratum_draws.values())+1
                name=max(self.strata,key=lambda k:self.weights[k]*total-self.stratum_draws[k])
                if self.stratum_positions.get(name,0)==len(self.stratum_orders.get(name,[])):
                    self.stratum_orders[name]=torch.randperm(len(self.strata[name]),generator=self.generator).tolist()
                    self.stratum_positions[name]=0
                j=self.stratum_orders[name][self.stratum_positions[name]];self.stratum_positions[name]+=1
                chosen.append(self.ids.index(self.strata[name][j]));self.stratum_draws[name]+=1
        else:
            if self.cursor==len(self.order):
                self.order=torch.randperm(len(self.ids),generator=self.generator).tolist();self.cursor=0
            chosen=self.order[self.cursor:self.cursor+self.batch_size];self.cursor+=len(chosen)
        selected=[]
        for index in chosen:
            identity=self.ids[index];n=len(self.counts[identity])
            if identity in self.recovery_pools:
                pools=self.recovery_pools[identity];draws=self.recovery_draws[identity]
                fraction=self.recovery_prefix['frontend_fraction']
                weights=dict(initial=fraction,remaining=1-fraction)
                total=sum(draws.values())+1
                bucket=max(pools,key=lambda k:weights[k]*total-draws[k])
                decks=self.recovery_decks[identity];positions=self.recovery_positions[identity]
                if positions.get(bucket,0)==len(decks.get(bucket,[])):
                    decks[bucket]=torch.randperm(len(pools[bucket]),generator=self.generator).tolist();positions[bucket]=0
                anchor=pools[bucket][decks[bucket][positions[bucket]]];positions[bucket]+=1;draws[bucket]+=1
            else:
                if self.positions.get(identity,0)==len(self.decks.get(identity,[])):
                    self.decks[identity]=torch.randperm(n,generator=self.generator).tolist();self.positions[identity]=0
                anchor=self.decks[identity][self.positions[identity]];self.positions[identity]+=1
            tick=self.anchors[identity][anchor];self.counts[identity][anchor]+=1
            self.prefix_ticks+=tick+1;self.gradient_ticks+=tick%96+1
            selected.append((identity,tick))
        assert 0 < len(selected) <= self.batch_size
        if not self.strata:assert len({i for i,t in selected})==len(selected)
        self.draws+=len(selected)
        return selected

    def state_dict(self):
        keys=('order','cursor','decks','positions','counts','draws','prefix_ticks','gradient_ticks','anchors',
            'strata','weights','stratum_orders','stratum_positions','stratum_draws')
        if self.recovery_prefix:keys+=('recovery_prefix','recovery_pools','recovery_decks','recovery_positions','recovery_draws')
        return {k:getattr(self,k) for k in keys}

    def load_state_dict(self,state):
        if set(state['counts'])!=set(self.ids):raise ValueError('Sampler episode identity mismatch')
        if state.get('anchors',self.anchors)!=self.anchors:raise ValueError('Sampler supervision population changed')
        if state.get('strata',{})!=self.strata or state.get('weights',{})!=self.weights:
            raise ValueError('Sampler stratum snapshot or weights changed')
        if state.get('recovery_prefix')!=self.recovery_prefix or state.get('recovery_pools',{})!=self.recovery_pools:
            raise ValueError('Sampler recovery-prefix configuration or population changed')
        if self.recovery_prefix and any(k not in state for k in ('recovery_decks','recovery_positions','recovery_draws')):
            raise ValueError('Sampler recovery-prefix resume state is incomplete')
        if any(len(state['counts'][i])!=len(self.anchors[i]) for i in self.ids):raise ValueError('Sampler anchor count differs')
        for k,v in state.items():setattr(self,k,v)
        assert sum(sum(v) for v in self.counts.values())==self.draws
        if self.recovery_prefix:
            for identity,pools in self.recovery_pools.items():
                if any(sum(self.counts[identity][j] for j in anchors)!=self.recovery_draws[identity][bucket]
                        for bucket,anchors in pools.items()):
                    raise ValueError('Sampler recovery-prefix exposure counters differ')

    def coverage(self):
        counts=torch.tensor([v for values in self.counts.values() for v in values])
        episode=[sum(self.counts[i]) for i in self.ids]
        segments=[sum(count for tick,count in zip(self.anchors[i],self.counts[i]) if tick//96==segment)
            for i,n in self.lengths.items() for segment in range(math.ceil(n/96))]
        result=dict(sampled_anchors=self.draws,anchor_population=len(counts),
            supervised_anchor_population=len(counts),supervised_tick_population=self.supervised_ticks,
            anchor_equivalent_epochs=self.draws/len(counts),unique_anchors=int((counts>0).sum()),
            per_episode_selection_min=min(episode),per_episode_selection_max=max(episode),
            episode_selection_passes=self.draws/len(self.ids),
            visited_original_segments=sum(v>0 for v in segments),original_segment_population=len(segments),
            anchor_based_segment_equivalents=self.draws/(len(counts)/len(segments)),
            full_segment_traversals=0,prefix_ticks_recomputed=self.prefix_ticks,gradient_ticks=self.gradient_ticks)
        result['repeated_samples']=self.draws-int((counts>0).sum())
        if self.strata:
            result['strata']={}
            for name,ids in self.strata.items():
                values=[v for i in ids for v in self.counts[i]];draws=sum(values)
                result['strata'][name]=dict(sampled_anchors=draws,requested_fraction=self.weights[name],
                    actual_fraction=draws/max(1,self.draws),anchor_population=len(values),
                    unique_anchors=sum(v>0 for v in values),repeated_samples=sum(max(0,v-1) for v in values),
                    unique_episodes=sum(any(self.counts[i]) for i in ids),episode_population=len(ids),
                    supervised_target_ticks=sum(c*n for i in ids for c,n in zip(self.counts[i],self.target_counts[i])),
                    initial_24_supervised_anchor_draws=sum(sum(self.counts[i][:2]) for i in ids),
                    effective_anchor_exposures=draws/len(values))
                if self.recovery_prefix and name!='core':
                    early=[self.counts[i][j] for i in ids for j in self.recovery_pools[i].get('initial',[])]
                    rest=[self.counts[i][j] for i in ids for j in self.recovery_pools[i].get('remaining',[])]
                    result['strata'][name]['recovery_prefix']=dict(ticks=self.recovery_prefix['ticks'],
                        requested_anchor_fraction=self.recovery_prefix['frontend_fraction'],
                        initial_anchor_population=len(early),remaining_anchor_population=len(rest),
                        initial_anchor_draws=sum(early),remaining_anchor_draws=sum(rest),
                        actual_initial_anchor_fraction=sum(early)/max(1,draws),
                        unique_initial_anchors=sum(v>0 for v in early),unique_remaining_anchors=sum(v>0 for v in rest),
                        initial_supervised_target_ticks=sum(c*n for i in ids for c,n in zip(self.counts[i],self.recovery_target_counts[i])),
                        episodes_without_initial_anchor=sum('initial' not in self.recovery_pools[i] for i in ids),
                        episodes_without_remaining_anchor=sum('remaining' not in self.recovery_pools[i] for i in ids))
        return result


def recovery_prefix_config(spec):
    """Opt-in sampling/loss weighting; missing configuration preserves old paths."""
    if spec is None:return None
    if not isinstance(spec,dict) or set(spec)!={'ticks','frontend_fraction','backend_weight'}:
        raise ValueError('Recovery prefix requires ticks, frontend_fraction and backend_weight')
    ticks=spec['ticks'];fraction=spec['frontend_fraction'];weight=spec['backend_weight']
    if isinstance(ticks,bool) or not isinstance(ticks,int) or ticks<1:
        raise ValueError('Positive integer recovery-prefix length required')
    if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in (fraction,weight)):
        raise ValueError('Finite recovery-prefix fraction/weight required')
    if not 0<fraction<1 or weight<1:raise ValueError('Recovery fraction must be between zero and one; loss weight at least one')
    return dict(spec)


def recovery_prefix_rows(record,ticks=48):
    """First continuous legal expert segment only; gaps/switches never get bridged."""
    mask,segment=supervision_rows(record)
    prefix=torch.zeros_like(mask);valid=mask.nonzero().flatten()
    if len(valid):
        start=int(valid[0]);end=min(start+ticks,len(mask))
        contiguous=(mask[start:end]&(segment[start:end]==segment[start])).long().cumprod(0).bool()
        prefix[start:end]=contiguous
    return prefix


def residual_recovery_weights(prefix_masks,start,stop,supervised,weight):
    """Pass float weights to the existing final-command objective, never to history.

    The objective normalizes by this actual masked weight sum. Physical tick
    exposure and optimizer counters continue to use the original Boolean mask.
    """
    early=torch.zeros_like(supervised)
    if len(prefix_masks)!=len(supervised):raise ValueError('Recovery masks and residual batch differ')
    for row,prefix in enumerate(prefix_masks):
        count=max(0,min(stop,len(prefix))-start)
        if count:early[row,:count]=prefix[start:start+count].to(supervised.device)
    early &= supervised
    return supervised.to(torch.float32)*(1+(weight-1)*early),early


def sampling_strata(ids,strata=None,weights=None):
    """A frozen, disjoint train-only partition; no group-name target special cases."""
    if not strata:return {},{}
    names=('core','historical_correction','current_correction')
    if set(strata)!=set(names):raise ValueError('Exactly Core/history/current sampling strata required')
    flat=[i for name in names for i in strata[name]]
    if len(flat)!=len(set(flat)) or set(flat)!=set(ids):raise ValueError('Strata must partition the training snapshot exactly')
    requested=dict(weights or dict(core=.5,historical_correction=.2,current_correction=.3))
    if set(requested)!=set(names) or any(not math.isfinite(x) or x<0 for x in requested.values()):
        raise ValueError('Finite nonnegative stratum weights required')
    if abs(sum(requested.values())-1)>1e-8 or requested['core']>.5:
        raise ValueError('Corrections require at least half the requested sampling mass')
    active={k:list(strata[k]) for k in names if strata[k]}
    corrections=[k for k in active if k!='core']
    if not corrections:raise ValueError('Stratified incremental training needs qualified corrections')
    missing=sum(requested[k] for k in names if k not in active)
    result={k:requested[k] for k in active}
    mass=sum(result[k] for k in corrections)
    for k in corrections:result[k]+=missing*(result[k]/mass if mass else 1/len(corrections))
    if any(v<=0 for v in result.values()):raise ValueError('Every populated stratum must receive positive exposure')
    return active,result


def stratified_episode_order(ids,records,strata,weights,generator,target_ticks=None):
    """Whole causal episodes, allocated by supervised ticks, with explicit repeats.

    This is an exposure deck, not a collection of new episodes. Whole-episode
    rounding can exceed the requested tick count by less than two episodes.
    """
    pools,weights=sampling_strata(ids,strata,weights)
    if not pools:return [ids[i] for i in torch.randperm(len(ids),generator=generator).tolist()]
    lengths={i:int(supervision_rows(records[i])[0].sum()) for i in ids}
    if min(lengths.values())<1:raise ValueError('Every training episode needs valid supervision')
    target_ticks=sum(lengths.values()) if target_ticks is None else target_ticks
    if target_ticks<1:raise ValueError('Positive episode-deck tick budget required')
    exposure={k:0 for k in pools};decks={};positions={};order=[]
    while sum(exposure.values())<target_ticks or exposure.get('core',0)>sum(v for k,v in exposure.items() if k!='core'):
        total=sum(exposure.values())+min(lengths.values())
        name=max(pools,key=lambda k:weights[k]*total-exposure[k])
        if positions.get(name,0)==len(decks.get(name,[])):
            decks[name]=torch.randperm(len(pools[name]),generator=generator).tolist();positions[name]=0
        identity=pools[name][decks[name][positions[name]]];positions[name]+=1
        order.append(identity);exposure[name]+=lengths[identity]
    return order


def episode_exposure(records,counts,strata,weights=None,*,recovery_prefix=None):
    pools,weights=sampling_strata(list(counts),strata,weights)
    recovery=recovery_prefix_config(recovery_prefix)
    total=sum(int(supervision_rows(records[i])[0].sum())*count for i,count in counts.items())
    result=dict(supervised_ticks=total,unique_episodes=sum(v>0 for v in counts.values()),
        repeated_samples=sum(max(0,v-1) for v in counts.values()),episode_draws=sum(counts.values()),strata={})
    for name,ids in pools.items():
        population=sum(int(supervision_rows(records[i])[0].sum()) for i in ids)
        draws=sum(int(supervision_rows(records[i])[0].sum())*counts[i] for i in ids)
        unique=sum(int(supervision_rows(records[i])[0].sum()) for i in ids if counts[i])
        result['strata'][name]=dict(supervised_tick_population=population,supervised_ticks=draws,
            unique_supervised_ticks=unique,repeated_supervised_ticks=draws-unique,
            actual_fraction=draws/max(total,1),requested_fraction=weights[name],effective_tick_exposures=draws/population,
            unique_episodes=sum(counts[i]>0 for i in ids),episode_population=len(ids),
            episode_draws=sum(counts[i] for i in ids),repeated_samples=sum(max(0,counts[i]-1) for i in ids),
            initial_24_supervised_tick_draws=sum(min(24,int(supervision_rows(records[i])[0].sum()))*counts[i] for i in ids))
        if recovery:
            early={i:int(recovery_prefix_rows(records[i],recovery['ticks']).sum()) if name!='core' else 0 for i in ids}
            initial=sum(early[i]*counts[i] for i in ids)
            result['strata'][name]['recovery_prefix']=dict(ticks=recovery['ticks'],
                initial_supervised_tick_population=sum(early.values()),initial_supervised_tick_draws=initial,
                remaining_supervised_tick_draws=draws-initial,backend_weight=recovery['backend_weight'],
                loss_weight_mass=draws+(recovery['backend_weight']-1)*initial,
                loss_weight_mass_is_exposure=False)
    return result


def relative_selection(scores,baseline=None):
    """Frozen update-zero component scales: old validation, correction, early correction."""
    if set(scores)!={'old_validation','correction_dev','correction_initial24'} or any(
            not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in scores.values()):
        raise ValueError('Three finite nonnegative selection components required')
    baseline=dict(scores) if baseline is None else baseline
    if set(baseline)!=set(scores):raise ValueError('Selection baseline differs')
    ratios={k:scores[k]/max(float(baseline[k]),1e-12) for k in scores}
    return .5*ratios['old_validation']+.25*ratios['correction_dev']+.25*ratios['correction_initial24'],baseline,ratios


def anchor_batch(model,records,selected):
    from .anchor_sampling import labels
    conditions=[];commands=[];masks=[]
    for identity,tick in selected:
        record=records[identity]
        if tick%12 or not 0<=tick<len(record['time']):raise ValueError('Invalid native plan anchor')
        # Same TBPTT boundaries as the old trainer. The complete earlier prefix is
        # recomputed by the unchanged encoder at current weights, without gradients.
        c,anchors,_=model.encode(record,tick//96*96,tick+1)
        assert int(anchors[-1])==tick
        anchor=anchors[-1:].to(record['command'].device)
        y,m=labels(record,anchor,24)
        conditions.append(c[-1:]);commands.append(y);masks.append(m)
    return torch.cat(conditions),torch.cat(commands),torch.cat(masks)


def mask_totals(lengths,selected,geometry_commit=None,*,records=None):
    if records is not None:
        masks=[labels(records[i],torch.tensor([t],device=records[i]['command'].device),24)[1] for i,t in selected]
        totals=(sum(int(m.sum()) for m in masks),sum(int((m[:,1:]&m[:,:-1]).sum()) for m in masks))
        return totals if geometry_commit is None else totals+(sum(int(m[:,:geometry_commit].sum()) for m in masks),)
    n=[min(24,lengths[i]-t) for i,t in selected]
    totals=(sum(n),sum(max(0,x-1) for x in n))
    return totals if geometry_commit is None else totals+(sum(min(geometry_commit,x) for x in n),)


def weighted_parts(parts,valid,totals,weights,geometry_commit=12,global_hand_geometry=False):
    """Exactly the old global valid-target/pair means across microbatches.

    Weighting every microbatch equally is wrong at padded episode tails.
    Geometry/base terms count valid targets; temporal counts valid adjacent pairs.
    """
    rows=valid.sum()/max(1,totals[0]);pairs=(valid[:,1:]&valid[:,:-1]).sum()/max(1,totals[1])
    geometry=valid[:,:geometry_commit].sum()/max(1,totals[2]) if len(totals)==3 else rows
    contribution={k:v*(1. if k=='hand_geometry' and global_hand_geometry else
        pairs if k=='temporal' else geometry if k in ('position','rotation') else rows) for k,v in parts.items()}
    return contribution['base']+sum(weights[k]*contribution[k] for k in weights),contribution


def continuation_lr(step,total,warmup=200,peak=3e-5,floor=3e-6):
    if not 1<=step<=total or total<=warmup:raise ValueError('Invalid continuation scheduler position')
    if step<=warmup:return floor+(peak-floor)*(step-1)/max(1,warmup-1)
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def supervision_rows(record):
    """Loss eligibility is separate from history existence and never resets memory."""
    if 'time' in record:
        timeline=record['time']
    elif 'robot' in record and not {'supervision_mask','control_segment'}&set(record):
        # Historical residual-only caches had robot rows but no explicit clock.
        # Their original all-tick supervision remains the only legal fallback.
        timeline=record['robot']
    else:raise ValueError('Explicit supervised segments require the complete episode clock')
    n=len(timeline);device=timeline.device
    mask=record.get('supervision_mask',torch.ones(n,dtype=torch.bool,device=device))
    segment=record.get('control_segment',torch.zeros(n,dtype=torch.long,device=device))
    if mask.shape!=(n,) or mask.dtype!=torch.bool or segment.shape!=(n,) or segment.dtype!=torch.long:
        raise ValueError('Boolean supervision and int64 control segments must align with the complete episode')
    if (segment<0).any() or (segment[1:]<segment[:-1]).any():
        raise ValueError('Control segments must be monotonic within one actual episode')
    return mask,segment


def supervision_population(records,horizon=24):
    """Count real eligible anchors/ticks for a bounded training exposure budget."""
    counts=dict(episodes=len(records),history_ticks=0,supervised_ticks=0,
        supervised_anchors=0,supervised_window_targets=0)
    for record in records.values():
        mask,_=supervision_rows(record)
        anchors=planning_anchors(record)
        counts['history_ticks']+=len(mask);counts['supervised_ticks']+=int(mask.sum())
        counts['supervised_anchors']+=len(anchors)
        if len(anchors):counts['supervised_window_targets']+=int(labels(record,anchors,horizon)[1].sum())
    return counts


def labels(record, anchors, horizon=64):
    if horizon<1 or anchors.ndim!=1 or (anchors<0).any() or (anchors>=len(record['time'])).any():
        raise ValueError('Invalid native label anchor')
    indices = anchors[:, None]+torch.arange(horizon, device=anchors.device)
    safe=indices.clamp_max(len(record['time'])-1)
    supervised,segment=supervision_rows(record)
    from .target_windows import window_mask
    valid=window_mask(supervised,segment,anchors,horizon,record.get('target_window_contract'))
    commands = record.get('clean_sample',record['command'])[safe]
    return torch.where(valid[..., None], commands, 0.), valid


def planning_anchors(record):
    """Native anchors with eligible targets in their actual committed prefix."""
    mask,_=supervision_rows(record)
    anchors=torch.arange(0,len(mask),12,device=mask.device)
    if record.get('target_window_contract') is None:
        return anchors[mask[anchors]]
    return anchors[labels(record,anchors,12)[1].any(-1)]
