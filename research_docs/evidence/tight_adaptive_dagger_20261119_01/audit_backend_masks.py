"""Rebuild backend optimizer-step participation from raw masks and seeded decks."""
import argparse
import collections
import hashlib
from pathlib import Path

import torch

from tools.tight_frozen_frontend import read, write
from tools.tight_representation_probe import source
from visuotactile.effective_exposure import Ledger
from visuotactile.latest_takeover_sampling import episode_deck, sampling_plan


def audit(root, role, output):
    torch.set_num_threads(2)
    folder=root/role/f'backend/{role}/training'
    with (folder/'best.pt').open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
        selected=torch.load(stream,map_location='cpu',weights_only=False)
    config=read(folder/'configuration.json')
    data=read(root/role/'state_data/data.json')
    ids=data['splits']['train']
    assert config['microbatch']==4 and config['tick_window_contract']['target_supervised_ticks']==128
    assert not config.get('recovery_prefix') and not config.get('fresh_recovery_weighting')
    masks={identity:torch.from_numpy(source(data['entries'][identity])['legal']).bool() for identity in ids}
    ledger=Ledger(masks)
    expected=Ledger(masks,selected['effective_exposure_state'])
    _,multiplicities=sampling_plan(data,config['episode_sampling'])
    generator=torch.Generator().manual_seed(config['initialization_seed'])
    counts=collections.Counter();update=0
    import json
    with (folder/'loss.jsonl').open() as stream:
        for epoch in range(1,selected['epoch']+1):
            order=episode_deck(ids,multiplicities,generator)
            if epoch==selected['epoch']:
                assert order==selected['episode_order']
            for offset in range(0,len(order),4):
                group=order[offset:offset+4]
                length=max(len(masks[i]) for i in group)
                pending=collections.defaultdict(set);mass=chunks=0
                for start in range(0,length,32):
                    n=0
                    for identity in group:
                        ticks=masks[identity][start:start+32].nonzero().flatten()+start
                        n+=len(ticks);pending[identity].update(ticks.tolist())
                    if n:
                        mass+=n;chunks+=1
                    if mass and (mass>=128 or start+32>=length):
                        update+=1
                        row=json.loads(next(stream))
                        actual={i:sorted(t) for i,t in pending.items()}
                        assert row['update']==update and row['epoch']==epoch
                        assert row['supervised_ticks']==mass and row['window_chunks']==chunks
                        assert row['loss_weight_mass']==mass and row['fixed_denominator']==128
                        assert row['effective_tick_contributions']==actual, (epoch,update,
                            {i:len(t) for i,t in row['effective_tick_contributions'].items()},
                            {i:len(t) for i,t in actual.items()})
                        ledger.step(update,list(actual.items()))
                        mass=chunks=0;pending=collections.defaultdict(set)
                counts.update(group)
                if epoch==selected['epoch'] and offset+len(group)==selected['next_episode_group']:
                    break
            if epoch==selected['epoch']:
                break
    assert update==selected['updates']==expected.updates
    assert dict(counts)==selected['episode_counts']
    for identity in masks:
        assert torch.equal(ledger.counts[identity],expected.counts[identity])
    receipt=ledger.summary(dict(counts))
    assert receipt['eligible']
    result=dict(role=role,component='Transformer',selected_epoch=selected['epoch'],selected_updates=update,
        selected_sha256=digest,effective_exposure=receipt,raw_masks_recomputed=True,
        seeded_decks_recomputed=True,chunk_windows_recomputed=True,step_receipts_exact=True,
        no_ancestor_credit=True,same_step_duplicates_counted_once=True,cuda_initialized=torch.cuda.is_initialized())
    write(output,result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--role',choices=['A','B'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.root,a.role,a.output)
