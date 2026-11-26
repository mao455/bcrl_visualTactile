"""Same real-history and production noise replay, all208 expert suffix ticks."""
from pathlib import Path
from types import SimpleNamespace
import json,time,sys,argparse
import numpy as np
import h5py
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256

def worker(owner):
 from scripts.tight_entry_final24_20261122 import entry_history
 P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';fit=R/'b72_fit'
 read=lambda p:json.loads(p.read_text())
 source=P/'outputs/visuotactile/run/tight_machine_review_20261126_01/fresh_rescue_history02'
 plan=read(source/'entry_same_history_plan.json');pairs={r:{} for r in 'AB'}
 current=read(R/'registration.json')['pairs']
 for role in 'AB':
  pairs[role]['before']=current[f'{role}_new']
  front=fit/role/'training/best.pt';rear=fit/role/'backend'/role/'training/best.pt'
  pairs[role]['after']=dict(frontend=str(front),frontend_sha256=sha256(front),backend=str(rear),backend_sha256=sha256(rear))
 for case in plan['cases']:
  with h5py.File(case['raw_path']) as f:
   g=next(iter(f['episodes'].values()));mask=g['transition/dagger/expert_label_valid'][:].astype(bool)
   case.update(stop=len(mask),ticks=np.flatnonzero(mask).tolist(),new50=True)
 assert [len(c['ticks']) for c in plan['cases']]==[92,116]
 plan.update(pairs=pairs,interpretation='ExistingB72 two-family training fit; all legal expert suffix ticks, same real prefix/feedback/7091noise; no own-student rescue or generalization claim')
 atomic_json(owner/'entry_same_history_plan.json',plan)
 entry_history.NEW=fit;entry_history.AUDIT=owner
 def summarize(rows,p,began):
  out=[]
  for role in 'AB':
   for tag in ['before','after']:
    for first in [True,False,None]:
     selected=[r for r in rows if r['role']==role and r['policy']==tag and (first is None or r['first_legal']==first)]
     if not selected:continue
     item=dict(role=role,policy=tag,population='all_suffix' if first is None else 'first_legal' if first else 'remaining_suffix',ticks=len(selected),families=len({r['identity'] for r in selected}))
     for stage in ['nominal','preprojection','final_command','projection_magnitude']:
      item[stage]={k:float(np.mean([r[stage][k] for r in selected])) for k in ['arm_rmse_rad','hand_rmse_rad']}
     out.append(item)
  return dict(rows=out,wall_s=time.time()-began,physics_steps=0,label='ExistingB72 fit after actual optimization; isolated temporary models')
 entry_history.summarize=summarize
 entry_history.run(SimpleNamespace(output='inference',population='all150',identities='',limit=0,device='cuda:0',roles='AB',tags='before,after'))
 atomic_json(owner/'result.json',dict(software_complete=True,utc=time.time(),physics_steps=0))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['b72_history']=('evaluate','outputs.diagnostics.tight_correction_20261127_01.b72_history',[])
  sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);worker(p.parse_args().root)
