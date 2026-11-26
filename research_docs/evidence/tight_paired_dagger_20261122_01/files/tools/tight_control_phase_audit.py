"""Read saved train-only phase coverage without updating parameters or workers."""
import argparse
import collections
import json
from pathlib import Path
import time

import torch

from visuotactile.anchor_sampling import labels, planning_anchors
from visuotactile.target_windows import with_target_window


def read(p):return json.loads(Path(p).read_text())
def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
def phase_sum(values):
    return [int(values[p::12].sum()) for p in range(12)]


def hits(record,anchors,counts=None,horizon=24):
    result=torch.zeros(len(record['time']),dtype=torch.long)
    mask=labels(record,anchors,horizon)[1]
    for row,a in enumerate(anchors.tolist()):
        result[a+mask[row].nonzero().flatten()]+=1 if counts is None else counts[row]
    return result


def dp(root):
    data=read(root/'state_data/data.json');saved=load(root/'training/last.pt')
    raw=load(root/'state_data/numeric.pt')['records'];sampler=saved['anchor_sampler']
    records=with_target_window({i:raw[i] for i in data['splits']['train']},saved['configuration'].get('target_window_contract'))
    fields=('legal_population','eligible_unique_targets','actual_unique_targets','actual_target_draws',
        'eligible_unique_committed_targets','actual_unique_committed_targets','actual_committed_target_draws',
        'observations_in_sampled_prefixes','condition_draws','early48_legal_population','early48_actual_unique_targets')
    stats={name:{f:[0]*12 for f in fields} for name in ('all',*data['sampling_strata'])}
    assert set(sampler['anchors'])==set(records)==set(sampler['counts'])
    assert sum(map(sum,sampler['counts'].values()))==sampler['draws']
    assert sampler['draws']==saved['updates']*saved['configuration']['effective_batch']
    missing=[];onsets=collections.Counter();switches=collections.Counter();old_repairs=[]
    for identity,r in records.items():
        stratum=next(k for k,v in data['sampling_strata'].items() if identity in v)
        mask=r['supervision_mask'];anchors=torch.tensor(sampler['anchors'][identity],dtype=torch.long);counts=sampler['counts'][identity]
        eligible=hits(r,planning_anchors(r));seen=hits(r,anchors,counts)
        committed=hits(r,planning_anchors(r),horizon=12);seen_commit=hits(r,anchors,counts,horizon=12)
        condition=torch.zeros(len(mask),dtype=torch.long);history=torch.zeros_like(condition)
        for a,c in zip(anchors.tolist(),counts):condition[a]+=c;history[:a+1]+=c
        from visuotactile.anchor_sampling import recovery_prefix_rows
        early=recovery_prefix_rows(r) if stratum!='core' else torch.zeros_like(mask)
        values=dict(legal_population=mask,eligible_unique_targets=eligible>0,actual_unique_targets=seen>0,
            actual_target_draws=seen,eligible_unique_committed_targets=committed>0,
            actual_unique_committed_targets=seen_commit>0,actual_committed_target_draws=seen_commit,
            observations_in_sampled_prefixes=history,condition_draws=condition,
            early48_legal_population=early,early48_actual_unique_targets=early&(seen>0))
        for name in ('all',stratum):
            for key,v in values.items():stats[name][key]=[a+b for a,b in zip(stats[name][key],phase_sum(v))]
        if stratum!='core':
            first=int(mask.nonzero()[0]);entry=data['entries'][identity]
            onsets[first%12]+=1;switches[entry['supervision_boundary']['takeover_tick']%12]+=1
            if first%12:
                old_repairs.append(dict(identity=identity,first_legal=first,anchor=first//12*12,
                    anchor_draws=counts[anchors.tolist().index(first//12*12)] if first//12*12 in anchors else 0,
                    early10_actual_unique=int((seen[first:first+10]>0).sum())))
        if (mask&(eligible==0)).any():missing.append(identity)
    assert sum(stats['all']['actual_target_draws'])==saved['supervised_target_ticks']
    assert sum(stats['all']['condition_draws'])==sampler['draws']
    return dict(root=str(root.resolve()),recorded_utc=time.time(),checkpoint_updates=saved['updates'],
        counter_validation=dict(sampled_anchors=sampler['draws'],
            supervised_target_slots=saved['supervised_target_ticks'],
            checkpoint_completed_updates_and_sampler_counts_agree=True),
        configured_target_window=saved['configuration'].get('target_window_contract'),
        train_episodes=len(records),stratum_episodes={k:len(v) for k,v in data['sampling_strata'].items()},
        phase_order=list(range(12)),statistics=stats,takeover_phase=dict(switches),first_legal_phase=dict(onsets),
        missing_eligible_episode_ids=missing,legacy_repaired_samples=old_repairs,
        all_legacy_early10_seen=all(x['early10_actual_unique']==10 for x in old_repairs),
        caveat='Snapshot of completed optimizer updates in a running trainer, reconstructed from saved per-anchor counts, not expected sampling probabilities. Repeated overlapping H24 targets are intentional; committed12 coverage counted separately. History counts include detached prefix replay and are not gradient exposure counts.')


def rear(root):
    data=read(root/'state_data/data.json');saved=load(root/'training/last.pt');counts=saved['episode_counts']
    finished=read(root/'training/finished.json')
    assert saved['updates']==finished['updates']
    assert set(counts)==set(data['splits']['train'])
    raw=load(root/'state_data/numeric.pt')['records'];stats={}
    for stratum,ids in data['sampling_strata'].items():
        population=torch.zeros(12,dtype=torch.long);unique=population.clone();draws=population.clone()
        for identity in ids:
            row=phase_sum(raw[identity]['supervision_mask']);p=torch.tensor(row)
            population+=p;unique+=p*int(counts[identity]>0);draws+=p*counts[identity]
        stats[stratum]=dict(legal_population=population.tolist(),actual_unique=unique.tolist(),actual_draws=draws.tolist(),
            episodes=len(ids),episodes_seen=sum(counts[i]>0 for i in ids))
    total=sum(sum(s['actual_draws']) for s in stats.values())
    assert total==finished['supervised_ticks']
    return dict(root=str(root.resolve()),checkpoint_updates=saved['updates'],statistics=stats,phase_order=list(range(12)),
        counter_validation=dict(supervised_ticks=total,completed_receipt_and_episode_counts_agree=True),
        caveat='Completed prior backend; current round backend has not run. Entire selected episodes retain all legal ticks.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--outer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=dict(current_dp=dp(a.outer/'adaptive_contact_recovery_01/round_01/QG01/dp'),
        completed_previous_rear={g:rear(a.outer/'adaptive_tick_window_01/formal'/g/'backend'/g) for g in ('QG01','G01')},
        test22=False,parameter_updates=0,new_simulations=0)
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
