"""Independently reconstruct a durable DP ledger without changing training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import time
import torch
from tools.tight_frozen_frontend import read,write
from visuotactile.effective_exposure import Ledger,dp_ticks
from visuotactile.native_data import visual_records


def audit(root, role, output, checkpoint_name='last.pt'):
    torch.set_num_threads(2)
    assert checkpoint_name in ('last.pt','best.pt')
    directory=root/role/'training';path=directory/checkpoint_name
    # Open one durable inode; an atomic future replacement cannot change this read.
    with path.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
        checkpoint=torch.load(stream,map_location='cpu',weights_only=False)
    state=checkpoint['effective_exposure_state'];ledger=Ledger(state['legal'],state)
    replay=Ledger(state['legal']);visits=collections.Counter()
    contract=read(root/role/'continuation.json').get('target_window_contract')
    records=visual_records(root/role/'state_data',list(state['legal']),'G00',target_window_contract=contract)
    for line in (directory/'learning.jsonl').open():
        row=json.loads(line)
        if row['updates']>checkpoint['updates']:break
        actual=collections.defaultdict(set)
        for visit in row['visits']:
            identity=visit['episode'];visits[identity]+=1
            actual[identity].update(dp_ticks(records[identity],visit['anchor_sampling']['selected_anchors']))
        assert {i:sorted(t) for i,t in actual.items()}==row['effective_tick_contributions']
        replay.step(row['updates'],list(actual.items()))
    assert replay.updates==checkpoint['updates']==state['updates']
    for identity in ledger.legal:
        assert torch.equal(replay.counts[identity],ledger.counts[identity])
    assert dict(visits)==checkpoint['sampling_exposure']
    receipt=ledger.summary(dict(visits))
    assert checkpoint['optimizer']['state']
    result=dict(role=role,checkpoint_sha256=digest,epoch=checkpoint['epoch'],updates=checkpoint['updates'],
        independently_recomputed_mask_and_anchors=True,replay_exact=True,optimizer_present=True,
        eligible_for_selection=receipt['eligible'],effective_exposure=receipt,
        scope='Actual durable checkpoint; no ancestor exposure; no extra training; no CUDA',
        checkpoint_name=checkpoint_name,selected_checkpoint=checkpoint_name=='best.pt',
        cuda_initialized=torch.cuda.is_initialized(),utc=time.time())
    write(output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--role',choices=['A','B'],required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',choices=['last.pt','best.pt'],default='last.pt')
    args=parser.parse_args();audit(args.root,args.role,args.output,args.checkpoint)
