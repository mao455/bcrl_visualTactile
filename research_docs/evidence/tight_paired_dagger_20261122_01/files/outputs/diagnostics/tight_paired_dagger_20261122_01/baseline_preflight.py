import json,time
import torch
from pathlib import Path
from tools.tight_paired_learning import PARENTS,read
from visuotactile.native_data import visual_records
from visuotactile.anchor_sampling import supervision_rows,sample_training_anchors
from visuotactile.latest_takeover_sampling import sampling_plan,episode_deck
from visuotactile.effective_exposure import Ledger,dp_ticks
from visuotactile.entry_exposure import EntryLedger
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
rows={}
for role in 'AB':
 child=PARENTS/role;d=read(child/'state_data/data.json');cont=read(child/'continuation.json')
 records=visual_records(child/'state_data',d['splits']['train'],'G00',target_window_contract=cont['target_window_contract'])
 _,counts=sampling_plan(d,cont['sampling_contract'])
 ledger=Ledger({i:supervision_rows(r)[0] for i,r in records.items()});entries=EntryLedger(records)
 g=torch.Generator().manual_seed(2026111402);profile=read(PARENTS/'training_sampling.json')['anchors'];update=0
 for epoch in range(1,61):
  order=episode_deck(d['splits']['train'],counts,g)
  for offset in range(0,len(order),2):
   update+=1;contributions=[];selections=[]
   for pos in range(offset,min(offset+2,len(order))):
    i=order[pos];r=records[i];anchors,_=sample_training_anchors(r,profile,identity=i,epoch=epoch,visit_index=pos+1)
    contributions.append((i,dp_ticks(r,anchors)));selections.append((i,anchors))
   ledger.step(update,contributions);entries.step(update,entries.contributions(records,selections))
 visits={i:60*counts[i] for i in records}
 summary=ledger.summary(visits);entry=entries.summary()
 assert summary['eligible'] and entry['eligible'] and update==7200
 rows[role]=dict(absolute=summary,entry=entry,data_sha256=sha256(child/'state_data/data.json'),initialization_sha256=cont['sha256'])
 print(json.dumps(dict(role=role,**rows[role])),flush=True)
atomic_json(PARENTS/'preflight.json',dict(passed=True,roles=rows,utc=time.time()))
