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
    def __init__(self, lengths, generator, batch_size=64, *, records=None):
        self.ids=list(lengths);self.lengths=dict(lengths);self.generator=generator
        if not self.ids or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Positive batch and nonempty episode cohort required')
        if any(n < 1 for n in lengths.values()):
            raise ValueError('Empty episodes are not training examples')
        self.batch_size=batch_size
        self.order=[];self.cursor=0;self.decks={};self.positions={}
        self.anchors={i:list(range(0,n,12)) for i,n in lengths.items()}
        self.supervised_ticks=sum(lengths.values())
        if records is not None:
            self.supervised_ticks=0
            for identity,n in lengths.items():
                mask,_=supervision_rows(records[identity])
                if len(mask)!=n:raise ValueError('Sampler record length mismatch')
                self.anchors[identity]=[t for t in self.anchors[identity] if bool(mask[t])]
                self.supervised_ticks+=int(mask.sum())
            if any(not a for a in self.anchors.values()):
                raise ValueError('Every training episode needs a supervised native planning anchor')
        self.counts={i:[0]*len(a) for i,a in self.anchors.items()}
        self.draws=0;self.prefix_ticks=0;self.gradient_ticks=0

    def draw(self):
        if self.cursor==len(self.order):
            self.order=torch.randperm(len(self.ids),generator=self.generator).tolist();self.cursor=0
        chosen=self.order[self.cursor:self.cursor+self.batch_size];self.cursor+=len(chosen)
        selected=[]
        for index in chosen:
            identity=self.ids[index];n=len(self.counts[identity])
            if self.positions.get(identity,0)==len(self.decks.get(identity,[])):
                self.decks[identity]=torch.randperm(n,generator=self.generator).tolist();self.positions[identity]=0
            anchor=self.decks[identity][self.positions[identity]];self.positions[identity]+=1
            tick=self.anchors[identity][anchor];self.counts[identity][anchor]+=1
            self.prefix_ticks+=tick+1;self.gradient_ticks+=tick%96+1
            selected.append((identity,tick))
        assert 0 < len(selected) <= self.batch_size and len({i for i,t in selected})==len(selected)
        self.draws+=len(selected)
        return selected

    def state_dict(self):
        return {k:getattr(self,k) for k in ('order','cursor','decks','positions','counts','draws','prefix_ticks','gradient_ticks','anchors')}

    def load_state_dict(self,state):
        if set(state['counts'])!=set(self.ids):raise ValueError('Sampler episode identity mismatch')
        if state.get('anchors',self.anchors)!=self.anchors:raise ValueError('Sampler supervision population changed')
        if any(len(state['counts'][i])!=len(self.anchors[i]) for i in self.ids):raise ValueError('Sampler anchor count differs')
        for k,v in state.items():setattr(self,k,v)
        assert sum(sum(v) for v in self.counts.values())==self.draws

    def coverage(self):
        counts=torch.tensor([v for values in self.counts.values() for v in values])
        episode=[sum(self.counts[i]) for i in self.ids]
        segments=[sum(count for tick,count in zip(self.anchors[i],self.counts[i]) if tick//96==segment)
            for i,n in self.lengths.items() for segment in range(math.ceil(n/96))]
        return dict(sampled_anchors=self.draws,anchor_population=len(counts),
            supervised_anchor_population=len(counts),supervised_tick_population=self.supervised_ticks,
            anchor_equivalent_epochs=self.draws/len(counts),unique_anchors=int((counts>0).sum()),
            per_episode_selection_min=min(episode),per_episode_selection_max=max(episode),
            episode_selection_passes=self.draws/len(self.ids),
            visited_original_segments=sum(v>0 for v in segments),original_segment_population=len(segments),
            anchor_based_segment_equivalents=self.draws/(len(counts)/len(segments)),
            full_segment_traversals=0,prefix_ticks_recomputed=self.prefix_ticks,gradient_ticks=self.gradient_ticks)


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
    n=([int(labels(records[i],torch.tensor([t],device=records[i]['command'].device),24)[1].sum())
        for i,t in selected] if records is not None else [min(24,lengths[i]-t) for i,t in selected])
    totals = (sum(n),sum(max(0,x-1) for x in n))
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
        anchors=torch.arange(0,len(mask),12,device=record['command'].device)
        anchors=anchors[mask[anchors]]
        counts['history_ticks']+=len(mask);counts['supervised_ticks']+=int(mask.sum())
        counts['supervised_anchors']+=len(anchors)
        if len(anchors):counts['supervised_window_targets']+=int(labels(record,anchors,horizon)[1].sum())
    return counts


def labels(record, anchors, horizon=64):
    if horizon<1 or anchors.ndim!=1 or (anchors<0).any() or (anchors>=len(record['time'])).any():
        raise ValueError('Invalid native label anchor')
    indices = anchors[:, None]+torch.arange(horizon, device=anchors.device)
    valid = indices < len(record['time'])
    safe=indices.clamp_max(len(record['time'])-1)
    supervised,segment=supervision_rows(record)
    valid &= supervised[safe] & (segment[safe]==segment[anchors,None])
    # An invalid row ends this target window, even if valid labels resume later.
    valid=valid.long().cumprod(-1).bool()
    commands = record.get('clean_sample',record['command'])[safe]
    return torch.where(valid[..., None], commands, 0.), valid
