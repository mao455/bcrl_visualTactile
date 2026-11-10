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
    def __init__(self, lengths, generator, batch_size=64):
        self.ids=list(lengths);self.lengths=dict(lengths);self.generator=generator
        if not self.ids or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Positive batch and nonempty episode cohort required')
        if any(n < 1 for n in lengths.values()):
            raise ValueError('Empty episodes are not training examples')
        self.batch_size=batch_size
        self.order=[];self.cursor=0;self.decks={};self.positions={}
        self.counts={i:[0]*math.ceil(n/12) for i,n in lengths.items()}
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
            tick=anchor*12;self.counts[identity][anchor]+=1
            self.prefix_ticks+=tick+1;self.gradient_ticks+=tick%96+1
            selected.append((identity,tick))
        assert 0 < len(selected) <= self.batch_size and len({i for i,t in selected})==len(selected)
        self.draws+=len(selected)
        return selected

    def state_dict(self):
        return {k:getattr(self,k) for k in ('order','cursor','decks','positions','counts','draws','prefix_ticks','gradient_ticks')}

    def load_state_dict(self,state):
        if set(state['counts'])!=set(self.ids):raise ValueError('Sampler episode identity mismatch')
        for k,v in state.items():setattr(self,k,v)
        assert sum(sum(v) for v in self.counts.values())==self.draws

    def coverage(self):
        counts=torch.tensor([v for values in self.counts.values() for v in values])
        episode=[sum(self.counts[i]) for i in self.ids]
        segments=[sum(v[s:s+8]) for v in self.counts.values() for s in range(0,len(v),8)]
        return dict(sampled_anchors=self.draws,anchor_population=len(counts),
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


def mask_totals(lengths,selected,geometry_commit=None):
    n=[min(24,lengths[i]-t) for i,t in selected]
    totals = (sum(n),sum(max(0,x-1) for x in n))
    return totals if geometry_commit is None else totals+(sum(min(geometry_commit,x) for x in n),)


def weighted_parts(parts,valid,totals,weights,geometry_commit=12):
    """Exactly the old global valid-target/pair means across microbatches.

    Weighting every microbatch equally is wrong at padded episode tails.
    Geometry/base terms count valid targets; temporal counts valid adjacent pairs.
    """
    rows=valid.sum()/max(1,totals[0]);pairs=(valid[:,1:]&valid[:,:-1]).sum()/max(1,totals[1])
    geometry=valid[:,:geometry_commit].sum()/max(1,totals[2]) if len(totals)==3 else rows
    contribution={k:v*(pairs if k=='temporal' else geometry if k in ('position','rotation') else rows) for k,v in parts.items()}
    return contribution['base']+sum(weights[k]*contribution[k] for k in weights),contribution


def continuation_lr(step,total,warmup=200,peak=3e-5,floor=3e-6):
    if not 1<=step<=total or total<=warmup:raise ValueError('Invalid continuation scheduler position')
    if step<=warmup:return floor+(peak-floor)*(step-1)/max(1,warmup-1)
    return floor+(peak-floor)*.5*(1+math.cos(math.pi*(step-warmup)/(total-warmup)))


def labels(record, anchors, horizon=64):
    indices = anchors[:, None]+torch.arange(horizon, device=anchors.device)
    valid = indices < len(record['time'])
    commands = record.get('clean_sample',record['command'])[indices.clamp_max(len(record['time'])-1)]
    return torch.where(valid[..., None], commands, 0.), valid
