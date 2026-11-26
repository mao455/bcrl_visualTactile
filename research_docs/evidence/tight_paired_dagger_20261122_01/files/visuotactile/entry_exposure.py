"""Native recovery entry participation, counted after successful optimizer steps."""
import torch
from .anchor_sampling import native_entry_anchor, labels

VERSION='native_entry_commit_optimizer_exposure_v1'

def population(records):
    entries={}
    for identity,r in records.items():
        anchor=native_entry_anchor(r)
        if anchor is None:continue
        _,mask=labels(r,torch.tensor([anchor],device=r['time'].device),24)
        slots=mask[0,:12].nonzero().flatten().cpu().tolist()
        entries[identity]=dict(anchor=anchor,slots=slots,ticks=[anchor+s for s in slots])
    return entries

class EntryLedger:
    def __init__(self,records,state=None):
        self.entries=population(records)
        if not self.entries:raise ValueError('Entry ledger requires actual correction entries')
        self.counts={i:{t:0 for t in e['ticks']} for i,e in self.entries.items()};self.updates=0
        if state is not None:
            if state['version']!=VERSION or state['entries']!=self.entries:raise ValueError('Entry exposure population changed')
            self.updates=state['updates']
            if set(state['counts'])!=set(self.counts):raise ValueError('Entry exposure identity changed')
            for i,values in self.counts.items():
                old={int(t):n for t,n in state['counts'][i].items()}
                if set(old)!=set(values) or any(type(n) is not int or not 0<=n<=self.updates for n in old.values()):
                    raise ValueError('Invalid entry exposure counts')
                self.counts[i]=dict(old)

    def contributions(self,records,selections):
        result=[]
        for identity,anchors in selections:
            if identity not in self.entries:continue
            entry=self.entries[identity];a=torch.as_tensor(anchors,dtype=torch.long,device=records[identity]['time'].device)
            a=a[a==entry['anchor']]
            if not len(a):continue
            _,mask=labels(records[identity],a,24)
            ticks=(a[:,None]+torch.arange(12,device=a.device))[mask[:,:12]].unique().cpu().tolist()
            result.append((identity,ticks))
        return result

    def step(self,update,contributions):
        if update!=self.updates+1:raise ValueError('Entry exposure requires consecutive successful optimizer steps')
        merged={}
        for identity,ticks in contributions:merged.setdefault(identity,set()).update(map(int,ticks))
        for i,ticks in merged.items():
            if i not in self.counts or not ticks<=self.counts[i].keys():raise ValueError('Unknown or masked entry slot')
        receipt={i:sorted(ticks) for i,ticks in merged.items()}
        for i,ticks in receipt.items():
            for t in ticks:self.counts[i][t]+=1
        self.updates=update
        return receipt

    def state_dict(self):
        return dict(version=VERSION,updates=self.updates,entries=self.entries,counts={i:dict(v) for i,v in self.counts.items()})

    def rows(self):
        return [dict(identity=i,native_anchor=e['anchor'],slot=s,absolute_tick=t,optimizer_steps=self.counts[i][t])
                for i,e in self.entries.items() for s,t in zip(e['slots'],e['ticks'])]

    def summary(self):
        values=[n for v in self.counts.values() for n in v.values()]
        return dict(version=VERSION,updates=self.updates,correction_entries=len(self.entries),committed_slots=len(values),
            minimum_entry_steps=min(values),maximum_entry_steps=max(values),below50_slots=sum(n<50 for n in values),
            eligible=min(values)>=50,ancestors_counted=False,duplicate_within_step='count_once')
