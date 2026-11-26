"""Actual positive-weight label participation, deduplicated per optimizer step.

This is supervision exposure, not proof of a nonzero residual or parameter gradient.
Counters start at zero for each new continuation; restore only this run's ledger.
"""
import json
from pathlib import Path
import torch

VERSION = 'new_optimizer_step_legal_tick_exposure_v1'


def profile():
    return dict(version=VERSION, minimum_visits=50, minimum_tick_steps=50,
        ancestors_counted=False, duplicate_tick_within_step='count_once',
        eligibility='actual_positive_weight_loss_mask', selection='validation_minimum_among_eligible_checkpoints')


def validate(value):
    if value is not None and value != profile():
        raise ValueError('Exact effective exposure contract required')
    return value


def load(root):
    p = Path(root)/'effective_exposure.json'
    return validate(json.loads(p.read_text())) if p.exists() else None


def dp_ticks(record, anchors):
    from .anchor_sampling import labels
    anchors=torch.as_tensor(anchors,dtype=torch.long,device=record['time'].device)
    _, mask=labels(record,anchors,24)
    indices=anchors[:,None]+torch.arange(24,device=anchors.device)
    return indices[mask].unique().cpu().tolist()


def cover_anchors(record, anchors):
    from .anchor_sampling import supervision_rows
    mask,_=supervision_rows(record)
    legal=set(mask.nonzero().flatten().cpu().tolist())
    chosen=set(anchors.cpu().tolist())
    uncovered=legal-set(dp_ticks(record,sorted(chosen)))
    while uncovered:
        tick=min(uncovered)
        chosen.add(tick)
        covered=set(dp_ticks(record,[tick]))
        if tick not in covered:raise ValueError('Legal label is not supervisable at its own causal input')
        uncovered-=covered
    return torch.tensor(sorted(chosen),dtype=torch.long,device=anchors.device)


class Ledger:
    def __init__(self, masks, state=None):
        self.legal={i:torch.as_tensor(m).bool().cpu().clone() for i,m in masks.items()}
        if not self.legal or any(not m.any() for m in self.legal.values()):raise ValueError('Empty supervision population')
        self.counts={i:torch.zeros(len(m),dtype=torch.int64) for i,m in self.legal.items()}
        self.updates=0
        if state is not None:
            if state['version']!=VERSION or set(state['counts'])!=set(self.legal):raise ValueError('Exposure identity mismatch')
            for i,m in self.legal.items():
                if not torch.equal(m,state['legal'][i].bool().cpu()):raise ValueError('Supervision population changed')
                c=state['counts'][i].long().cpu().clone()
                if c.shape!=m.shape or (c<0).any() or c[~m].any() or (c>state['updates']).any():raise ValueError('Invalid exposure counters')
                self.counts[i]=c
            self.updates=state['updates']

    def step(self, update, contributions):
        if update!=self.updates+1:raise ValueError('Exposure requires consecutive successful optimizer steps')
        merged={}
        for identity,ticks in contributions:
            merged.setdefault(identity,set()).update(int(t) for t in ticks)
        if not any(merged.values()):raise ValueError('No positive-weight supervised label in update')
        receipt={}
        for i,ticks in merged.items():
            indices=torch.tensor(sorted(ticks),dtype=torch.long)
            if i not in self.legal or (indices<0).any() or (indices>=len(self.legal[i])).any() or not self.legal[i][indices].all():
                raise ValueError('Masked, padded or unknown tick counted')
            receipt[i]=indices.tolist()
        for i,ticks in receipt.items():
            self.counts[i][torch.tensor(ticks,dtype=torch.long)]+=1
        self.updates=update
        return receipt

    def state_dict(self):
        return dict(version=VERSION,updates=self.updates,legal=self.legal,counts=self.counts)

    def summary(self, visits):
        if set(visits)!=set(self.legal):raise ValueError('Visit population differs')
        x=torch.cat([self.counts[i][m] for i,m in self.legal.items()])
        minimum=int(x.min());visit_min=min(visits.values())
        return dict(version=VERSION,updates=self.updates,trajectories=len(visits),legal_ticks=len(x),
            minimum_visits=visit_min,minimum_tick_steps=minimum,maximum_tick_steps=int(x.max()),
            below50_ticks=int((x<50).sum()),below50_trajectories=sum(v<50 for v in visits.values()),
            quantiles={str(q):float(torch.quantile(x.double(),q)) for q in (0.,.05,.25,.5,.75,.95,1.)},
            eligible=minimum>=50 and visit_min>=50,ancestors_counted=False)
