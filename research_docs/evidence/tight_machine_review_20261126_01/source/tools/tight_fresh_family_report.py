"""Audit actual learning exposure by original reset family, not episode count."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from tools.tight_fresh_rollouts import read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def report(root, output):
    import torch
    roles={}
    for role in ['A','B']:
        directory=root/role;data=read(directory/'state_data/data.json')
        records=torch.load(directory/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
        counts=Counter(data['entries'][i]['family'] for i in data['splits']['train'])
        validation={data['entries'][i]['family'] for i in data['splits']['validation']}
        if set(counts)&validation:raise ValueError('Training/validation family leakage')
        source_hashes=[data['entries'][i]['source_sha256'] for i in data['splits']['train']]
        by_family=defaultdict(Counter);previous=dict(anchors_seen=0,target_ticks=0)
        learning=directory/'training/learning.jsonl'
        for update,line in enumerate(learning.read_text().splitlines(),1):
            row=json.loads(line)
            if row['episode'] not in data['splits']['train'] or row['updates']!=update:
                raise ValueError('Unexpected optimizer receipt')
            bucket=by_family[data['entries'][row['episode']]['family']]
            bucket['frontend_episode_updates']+=1;bucket['frontend_history_frames']+=row['visual_frames']
            for key in previous:
                delta=row[key]-previous[key]
                if delta<=0:raise ValueError('Invalid exposure increment')
                bucket['frontend_'+key]+=delta;previous[key]=row[key]
        frontend_finished=read(directory/'training/finished.json')
        if not frontend_finished.get('complete') or any(
                sum(x['frontend_'+key] for x in by_family.values())!=frontend_finished[key]
                for key in previous):
            raise ValueError('Frontend supervision exposure differs')
        if sum(x['frontend_episode_updates'] for x in by_family.values())!=frontend_finished['updates']:
            raise ValueError('Frontend optimizer count differs')
        rear=directory/'backend'/role/'training'
        checkpoint=torch.load(rear/'last.pt',map_location='cpu',weights_only=False)
        if set(checkpoint['episode_counts'])!=set(data['splits']['train']):
            raise ValueError('Backend episode population differs')
        for identity,count in checkpoint['episode_counts'].items():
            rec=records[identity];n=len(rec['time'])
            valid=int(rec['supervision_mask'].sum()) if 'supervision_mask' in rec else n
            bucket=by_family[data['entries'][identity]['family']]
            bucket['backend_episode_draws']+=count;bucket['backend_history_frames']+=count*n
            bucket['backend_legal_supervised_ticks']+=count*valid
        finished=read(rear/'finished.json')
        if sum(x['backend_legal_supervised_ticks'] for x in by_family.values())!=finished['supervised_ticks']:
            raise ValueError('Backend supervision exposure differs')
        roles[role]=dict(episodes=len(data['splits']['train']),families=len(counts),
            episodes_per_family=dict(counts),duplicate_source_hash_count=len(source_hashes)-len(set(source_hashes)),
            validation_families=sorted(validation),actual_family_exposure=dict(by_family),
            data_sha256=sha256(directory/'state_data/data.json'),learning_sha256=sha256(learning),
            backend_last_sha256=sha256(rear/'last.pt'),backend_finished_sha256=sha256(rear/'finished.json'))
    if roles['A']['episodes_per_family']!=roles['B']['episodes_per_family']:
        raise ValueError('A/B family population differs')
    atomic_json(output,dict(roles=roles,test_used=False,
        interpretation='Actual optimizer exposure. DAgger episodes share original reset families but contain new student deviations and expert suffixes. Equal episode weighting is not equal family weighting; legal supervised ticks exclude student prefixes and transition masks. Unique source hashes do not prove physical state diversity.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();report(a.root,a.output)
