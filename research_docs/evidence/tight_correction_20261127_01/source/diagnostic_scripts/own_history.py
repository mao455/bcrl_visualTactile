"""Current-own captures, actual real history; original inference consumer only."""
from pathlib import Path
from types import SimpleNamespace
import json,time,sys,argparse
import numpy as np,h5py
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127';read=lambda p:json.loads(p.read_text())
def prepare():
 for role in 'AB':
  owner=R/('own_history_'+role);s=session_at(owner,duration=2*3600);s['resources']['min_ram_gib']=6;atomic_json(owner/'session.json',s)
  root=R/'short_with_corrections';data=read(root/role/'state_data/data.json');index=read(root/role/'cache_index.json');cases=[]
  for i in data['targeted_correction']['new_own_ids']:
   e=data['entries'][i];path=Path(e['path'])
   with h5py.File(path) as f:
    tr=f['episodes'][e['episode']]['transition'];mask=tr['dagger/expert_label_valid'][:].astype(bool)
   ticks=np.flatnonzero(mask).tolist();first=ticks[0]
   cases.append(dict(identity=i,new50=False,first_legal=first,anchor=first//12*12,stop=len(mask),ticks=ticks,raw_path=str(path),raw_source_sha256=e['source_sha256'],visual_cache_sha256=index['entries'][i]['sha256']))
  if not cases:
   atomic_json(owner/'not_applicable.json',dict(role=role,reason='No accepted own current correction; no fitting metric invented'));continue
  pairs={role:dict(before=read(R/'registration.json')['pairs'][role+'_new'])}
  for tag,condition in [('new','with_corrections'),('control','old_only')]:
   q=R/('short_'+condition);front=q/role/'training/best.pt';rear=q/role/'backend'/role/'training/best.pt';pairs[role][tag]=dict(frontend=str(front),frontend_sha256=sha256(front),backend=str(rear),backend_sha256=sha256(rear))
  template=read(R/'b72_fit_history/entry_same_history_plan.json');template.update(cases=cases,pairs=pairs,cases_total=len(cases),slots_total=sum(len(c['ticks']) for c in cases),new50_cases=0,new50_slots=0,interpretation='Training fit on this student own current true expert suffixes, same full real histories/actual feedback and7091noise; no independent generalization estimate')
  atomic_json(owner/'entry_same_history_plan.json',template);atomic_json(owner/'plan.json',dict(role=role,source_sha256=sources(),data_root=str(root)))
  atomic_json(C/('own_history_'+role+'.json'),dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=[dict(entry='own_history',args=['--root',str(owner)])]))
  print(owner)
def worker(owner):
 from scripts.tight_entry_final24_20261122 import entry_history
 plan=read(owner/'plan.json');role=plan['role']
 for name,digest in plan['source_sha256'].items():assert sha256(name)==digest,name
 entry_history.NEW=Path(plan['data_root']);entry_history.AUDIT=owner
 def summarize(rows,p,began):
  output=[]
  for tag in ['before','new','control']:
   for first in [True,None]:
    selected=[r for r in rows if r['policy']==tag and (first is None or r['first_legal'])]
    if not selected:continue
    item=dict(role=role,policy=tag,population='first_legal' if first else 'all_legal_suffix',ticks=len(selected),trajectories=len({r['identity'] for r in selected}))
    for stage in ['nominal','preprojection','final_command','projection_magnitude']:
     item[stage]={k:float(np.mean([r[stage][k] for r in selected])) for k in ['arm_rmse_rad','hand_rmse_rad']}
    output.append(item)
  return dict(rows=output,wall_s=time.time()-began,physics_steps=0,interpretation='Own current training fit; all masks and actual feedback retained, model own latent history recomputed')
 entry_history.summarize=summarize
 entry_history.run(SimpleNamespace(output='inference',population='all150',identities='',limit=0,device='cuda:0',roles=role,tags='before,new,control'))
 atomic_json(owner/'result.json',dict(complete=True,utc=time.time(),physics_steps=0))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['own_history']=('evaluate','outputs.diagnostics.tight_correction_20261127_01.own_history',[]);sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('--root',type=Path);a=p.parse_args();worker(a.root) if a.root else prepare()
