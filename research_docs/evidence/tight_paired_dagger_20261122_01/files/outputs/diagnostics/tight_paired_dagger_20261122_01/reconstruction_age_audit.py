import json,time
from pathlib import Path
import torch
from tools.tight_paired_learning import ROOT,BASE,PARENTS,AUDIT,read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
rows={}
for role in 'AB':
 chain=[];p=BASE/role/'training/best.pt';seen=set()
 while str(p) not in seen:
  seen.add(str(p))
  if not p.exists():
   chain.append(dict(path=str(p),exists=False,missing_source=True));break
  d=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
  chain.append(dict(path=str(p),exists=True,sha256=sha256(p),epoch=d.get('epoch'),updates=d.get('updates'),role=role))
  cont=d.get('configuration',{}).get('continuation') or d.get('continuation_from')
  if not cont:break
  p=Path(cont['path'])
 rear=[];p=BASE/role/'backend'/role/'training/last.pt';seen=set()
 while str(p) not in seen:
  seen.add(str(p))
  if not p.exists():
   rear.append(dict(path=str(p),exists=False,missing_source=True));break
  d=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
  rear.append(dict(path=str(p),exists=True,sha256=sha256(p),epoch=d.get('epoch'),updates=d.get('updates'),base_sha256=d.get('configuration',{}).get('base_sha256')))
  cont=d.get('continuation_from')
  if not cont:break
  p=Path(cont['path'])
 initial=PARENTS/role/'common_seed_initialization.pt';d=torch.load(initial,map_location='cpu',weights_only=False,mmap=True)
 assert d['epoch']==d['updates']==0 and d['ancestors']==[]
 rows[role]=dict(historical_frontend_chain=chain,historical_backend_chain=rear,reconstructed_parent_initial=dict(path=str(initial),sha256=sha256(initial),epoch=0,updates=0,ancestors=[]),planned_complete_parent_frontend_epochs=60,planned_complete_parent_backend_epochs=60)
atomic_json(AUDIT/'reconstruction_age_audit.json',dict(utc=time.time(),roles=rows,decision='No matched verifiable historical ancestry; common180 random matched initialization reconstructs complete parents without historical optimizer age'))
print(json.dumps(rows))
