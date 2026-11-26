from pathlib import Path
import json
import torch
from visuotactile.native_data import visual_records
from visuotactile.anchor_sampling import labels
from visuotactile.latest_takeover_sampling import sampling_plan,episode_deck
from visuotactile.paired_training_decks import group_windows
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01/b72_fit');D=Path('outputs/diagnostics/tight_correction_20261127_01')
read=lambda p:json.loads(p.read_text())
ids=['B72_28817004_expert','B72_28817008_expert'];out={}
for role in 'AB':
 data=read(R/role/'state_data/data.json');records=visual_records(R/role/'state_data',data['splits']['train'],'G00',target_window_contract='recovery_onset_commit12_v1')
 masks={i:r['supervision_mask'] for i,r in records.items()};exposure={i:{int(t):set() for t in masks[i].nonzero().flatten()} for i in ids}
 rows=[json.loads(s) for s in (R/role/'training/learning.jsonl').read_text().splitlines()]
 assert [r['updates'] for r in rows]==list(range(1,63))
 for row in rows:
  for visit in row['visits']:
   i=visit['episode']
   if i not in exposure:continue
   a=torch.tensor(visit['anchor_sampling']['selected_anchors']);_,mask=labels(records[i],a,24)
   ticks=(a[:,None]+torch.arange(24))[mask].unique().tolist()
   for t in ticks:exposure[i][t].add(row['updates'])
 front={i:dict(legal_ticks=len(ts),minimum_distinct_successful_steps=min(map(len,ts.values())),maximum_distinct_successful_steps=max(map(len,ts.values())),first_legal_steps=len(ts[min(ts)])) for i,ts in exposure.items()}
 saved=torch.load(R/role/'training/best.pt',weights_only=False,map_location='cpu',mmap=True)
 assert saved['updates']==saved['ema_optimization_step']==62
 assert all(saved['sampling_exposure'][i]==32 for i in ids)
 del saved
 rear=R/role/'backend'/role/'training';cfg=read(rear/'configuration.json');v=cfg.get('residual_pair',cfg)
 # Saved rear config is the resolved production configuration; use the original session's schedule spec.
 v=read(R/role/'backend/session.json')['residual_pair']
 _,counts=sampling_plan(data,v['episode_sampling']);generator=torch.Generator().manual_seed(v['initialization_seed'])
 actual=[json.loads(s) for s in (rear/'loss.jsonl').read_text().splitlines()];reconstructed=[]
 for epoch in [1,2]:
  order=episode_deck(data['splits']['train'],counts,generator)
  for start in range(0,len(order),4):
   for window in group_windows(order[start:start+4],masks):reconstructed.append(dict(epoch=epoch,**window))
 assert len(actual)==len(reconstructed)==188
 exposure={i:{int(t):set() for t in masks[i].nonzero().flatten()} for i in ids}
 for row,expected in zip(actual,reconstructed):
  assert row['epoch']==expected['epoch'] and row['supervised_ticks']==expected['supervised_ticks']
  for i,ticks in expected['contributions'].items():
   if i in exposure:
    for t in ticks:exposure[i][t].add(row['update'])
 back={i:dict(legal_ticks=len(ts),minimum_distinct_successful_steps=min(map(len,ts.values())),maximum_distinct_successful_steps=max(map(len,ts.values())),first_legal_steps=len(ts[min(ts)])) for i,ts in exposure.items()}
 saved=torch.load(rear/'best.pt',weights_only=False,map_location='cpu',mmap=True)
 assert saved['updates']==188 and all(saved['episode_counts'][i]==32 for i in ids)
 del saved
 out[role]=dict(frontend_updates=62,backend_updates=188,frontend=front,backend=back,actual_visits_per_suffix=32,old_full_visits_per_episode=2,successful_rear_window_reconstruction_matches_all_logs=True,ancestors_counted=False,padding_counted=False,within_step_duplicates_counted_once=True)
atomic_json(D/'b72_optimization_audit.json',out);print(json.dumps(out))
