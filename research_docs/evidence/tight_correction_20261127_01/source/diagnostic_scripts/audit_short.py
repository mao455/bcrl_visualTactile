from pathlib import Path
import json,csv
import torch
from visuotactile.native_data import visual_records
from visuotactile.effective_exposure import Ledger,dp_ticks
from visuotactile.entry_exposure import EntryLedger
from visuotactile.paired_training_decks import validate_deck,group_windows
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01');read=lambda p:json.loads(p.read_text());summary={};rows=[]
for condition in ['old_only','with_corrections']:
 root=R/('short_'+condition);reg=read(root/'registration.json');summary[condition]={}
 for role in 'AB':
  child=root/role;data=read(child/'state_data/data.json');records=visual_records(child/'state_data',data['splits']['train'],'G00',target_window_contract='recovery_onset_commit12_v1');masks={i:r['supervision_mask'] for i,r in records.items()};dp=Ledger(masks);entry=EntryLedger(records)
  logs=[json.loads(s) for s in (child/'training/learning.jsonl').read_text().splitlines()];assert [r['updates'] for r in logs]==list(range(1,reg['frontend_updates']+1))
  for row in logs:
   contributions=[];selections=[]
   for visit in row['visits']:
    i=visit['episode'];anchors=visit['anchor_sampling']['selected_anchors'];contributions.append((i,dp_ticks(records[i],anchors)));selections.append((i,anchors))
   dp.step(row['updates'],contributions);entry.step(row['updates'],entry.contributions(records,selections))
  deck=validate_deck(read(root/f'paired_group_deck_{role}.json'),data,records,data_sha256=sha256(child/'state_data/data.json'));rear=Ledger(masks);actual=[json.loads(s) for s in (child/'backend'/role/'training/loss.jsonl').read_text().splitlines()];expected=[dict(epoch=e+1,**w) for e,groups in enumerate(deck['epoch_groups']) for g in groups for w in group_windows(g,masks)];assert len(actual)==len(expected)==reg['backend_updates']
  for step,(a,b) in enumerate(zip(actual,expected),1):
   assert a['update']==step and a['epoch']==b['epoch'] and a['supervised_ticks']==b['supervised_ticks'];rear.step(step,list(b['contributions'].items()))
  saved=torch.load(child/'backend'/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True)
  assert saved['updates']==reg['backend_updates'] and all(torch.equal(saved['effective_exposure_state']['counts'][i],rear.counts[i]) for i in masks);del saved
  saved=torch.load(child/'training/best.pt',map_location='cpu',weights_only=False,mmap=True);assert saved['updates']==saved['ema_optimization_step']==reg['frontend_updates'];visits=saved['sampling_exposure'];del saved
  novel=set(data['targeted_correction']['new_own_ids'])
  for i,m in masks.items():
   rows.append(dict(condition=condition,role=role,identity=i,family=data['entries'][i]['family'],new_current_own=i in novel,legal_ticks=int(m.sum()),visits=visits[i],dp_min=int(dp.counts[i][m].min()),dp_max=int(dp.counts[i][m].max()),rear_min=int(rear.counts[i][m].min()),rear_max=int(rear.counts[i][m].max()),entry_min=min(entry.counts[i].values()) if i in entry.counts else None))
  summary[condition][role]=dict(frontend_updates=reg['frontend_updates'],backend_updates=reg['backend_updates'],frontend=dp.summary(visits),entry=entry.summary(),new_current_own=[r for r in rows if r['condition']==condition and r['role']==role and r['new_current_own']],selected='fixed_final1epoch',ancestors_counted=False,padding_counted=False)
with (D/'short_exposure_by_episode.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
atomic_json(D/'short_optimization_audit.json',summary);print(json.dumps(summary))
