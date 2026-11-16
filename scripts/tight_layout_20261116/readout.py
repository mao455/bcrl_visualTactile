"""Read closed receipts only; no raw videos/models, no worker restart."""
import json,sys,time
from pathlib import Path
TREE=Path(__file__).resolve().parents[2]
read=lambda p:json.loads(Path(p).read_text())
from visuotactile.training_resources import atomic_json

def report():
 base=TREE/'outputs/visuotactile/run';capture=base/'tight_layout10_20261116_01_recovery02';learning=base/'tight_layout10_20261116_learning01'
 out=dict(utc=time.time(),capture_root=str(capture.resolve()),learning_root=str(learning.resolve()))
 if (capture/'collection_results.json').exists():
  rows=read(capture/'collection_results.json')
  out['collection']={split:dict(attempts=sum(r['split']==split for r in rows),legal=sum(r['split']==split and r['legal'] for r in rows),success=sum(r['split']==split and r['success'] for r in rows)) for split in ['train','validation']}
 for key,root in [('capture',capture),('learning',learning)]:
  out[key]={}
  for name in ['status.json','owner_status.json','failure.json','collection_failure.json','owner_failure.json','prepared.json','complete.json','owner_exit.json','collection_owner_exit.json']:
   if (root/name).exists():
    value=read(root/name)
    if name=='prepared.json':value={k:v for k,v in value.items() if k!='source_sha256'}
    out[key][name]=value
 if learning.exists():
  for role in 'AB':
   out[role]={}
   for label,p in [('dp',learning/role/'training'),('rear',learning/role/'backend'/role/'training')]:
    if (p/'finished.json').exists():out[role][label]=read(p/'finished.json')
    elif (p/'validation.json').exists():
     v=read(p/'validation.json');v=v[-1] if isinstance(v,list) else v
     out[role][label]={k:v[k] for k in ['epoch','updates','loss','selection_score','validation','stopping'] if k in v}
 return out
if __name__=='__main__':
 r=report()
 if '--save' in sys.argv:atomic_json(TREE/'research_docs/evidence/tight_layout_20261116/progress.json',r)
 print(json.dumps(r,ensure_ascii=False,indent=2))
