"""Verify fixed checkpoint ages, actual label exposure, and complete pair binding."""
from pathlib import Path
import collections,csv,gc,gzip,json,time
import torch
from visuotactile.native_data import visual_records
from visuotactile.effective_exposure import Ledger
from visuotactile.entry_exposure import EntryLedger
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';read=lambda p:json.loads(p.read_text())
reg=read(R/'registration.json');models=[];exposures=[];hashes={};stats={};age=[]
def digest(p):
 p=Path(p);s=p.stat();key=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
 if key not in hashes:hashes[key]=sha256(p)
 return hashes[key]
for role,pair in reg['starting_pairs'].items():
 for component in ['frontend','backend']:
  assert digest(pair[component])==pair[component+'_sha256']
  models.append(dict(condition='frozen_start',role=role,component=component,epoch=0,path=pair[component],sha256=pair[component+'_sha256']))
with gzip.open(D/'actual_selected_tick_exposure_partial11.csv.gz','wt',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=['condition','role','component','epoch','identity','family','tick','optimizer_steps']);writer.writeheader()
 for condition in reg['conditions']:
  root=R/condition
  assert read(root/'frontend_complete.json')
  for role in 'AB':
   child=root/role;data=read(child/'state_data/data.json');cont=read(child/'continuation.json');ids=data['splits']['train']
   records=visual_records(child/'state_data',ids,'G00',target_window_contract=cont['target_window_contract'])
   masks={i:records[i]['supervision_mask'] for i in ids};statsha=digest(child/'state_data/statistics.pt')
   if role in stats:assert stats[role]==statsha
   stats[role]=statsha
   for epoch in [30,60]:
    front=child/'training'/f'milestone_epoch_{epoch}.pt';front_hash=digest(front)
    for component,path in [('DP',front),('Transformer',root/f'rear{epoch}'/role/'backend'/role/'training/best.pt')]:
     if not path.exists() or component=='Transformer' and not (path.parent/'fixed_final_selection.json').exists():continue
     saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
     expected=epoch*(206 if component=='DP' else 812)
     assert saved['epoch']==epoch and saved['updates']==expected
     state=saved['effective_exposure_state'];assert state['updates']==expected
     visits=saved['sampling_exposure'] if component=='DP' else saved['episode_counts']
     ledger=Ledger(masks,state);summary=ledger.summary(visits)
     assert summary['minimum_tick_steps']>=epoch and summary['minimum_visits']>=epoch
     if epoch==60:assert summary['eligible']
     entry=None
     if component=='DP':
      assert saved['configuration']['statistics_sha256']==statsha
      assert saved['configuration']['data_sha256']==digest(child/'state_data/data.json')
      assert saved['ema_optimization_step']==expected
      assert saved['configuration']['recurrent_execution']==read(root/'recurrent_execution.json')
      assert saved['configuration']['continuation']['sha256']==reg['starting_pairs'][role]['frontend_sha256']
      entries=EntryLedger(records,saved['entry_exposure_state']);entry=entries.summary();assert entry['minimum_entry_steps']>=epoch
      if epoch==60:assert entry['eligible']
     else:
      assert (path.parent/'fixed_final_selection.json').exists()
      assert saved['configuration']['base_sha256']==front_hash
      assert saved['continuation_from']['sha256']==reg['starting_pairs'][role]['backend_sha256']
      assert saved['continuation_from']['teacher_supervision']=='u_expert'
      assert saved['ema'] is False
      fixed=read(path.parent/'fixed_final_selection.json');assert fixed['epoch']==epoch and fixed['updates']==expected and fixed['sha256']==digest(path)
     for i,mask in ledger.legal.items():
      for tick in mask.nonzero().flatten().tolist():writer.writerow(dict(condition=condition,role=role,component=component,epoch=epoch,identity=i,family=data['entries'][i]['family'],tick=tick,optimizer_steps=int(ledger.counts[i][tick])))
     exposures.append(dict(condition=condition,role=role,component=component,epoch=epoch,summary=summary,native_entry=entry,ancestors_counted=False,padding_or_duplicate_slots_counted=False))
     models.append(dict(condition=condition,role=role,component=component,epoch=epoch,updates=expected,path=str(path),sha256=digest(path),statistics_sha256=statsha if component=='DP' else saved['configuration']['residual_statistics_sha256'],paired_frontend_sha256=front_hash))
     del saved,ledger;gc.collect()
   fixed=read(child/'training/fixed_final_selection.json');assert fixed['epoch']==60 and fixed['updates']==12360 and fixed['sha256']==digest(child/'training/best.pt')
   del records,masks;gc.collect()
for path,rows in [(D/'model_identities_partial11.csv',models)]:
 fields=list(dict.fromkeys(k for r in rows for k in r))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
atomic_json(D/'selected_training_audit_partial11.json',dict(utc=time.time(),models=models,exposures=exposures,final_minimum_exposure=60,minimum_requirement=50,shared_statistics_by_role=stats,epochs=[0,30,60],fixed_deployment=60,optimizer_updates=dict(DP_each=12360,final_rear_each=48720,extra_midpoint_rear_each=24360),discarded_old_cell_updates=1169,temporary_actual_data_probe_updates=8,pose_auxiliary_enabled=False,architecture_changed=False))
print('Fixed checkpoint identities, paired inputs and real tick exposures verified',flush=True)
