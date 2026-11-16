"""Read existing numeric evidence only. Never train, collect or interpolate logs."""
import csv,gzip,json,sys,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.tight_fresh_grasp_evidence import behavior
B=Path('outputs/visuotactile/run')
O=Path('research_docs/evidence/tight_diagnosis_20261116');O.mkdir(exist_ok=True,parents=True)
ROOTS={n:B/p/f'round_{n:02}' for n,p in [(16,'tight_fresh_20261116_01'),(17,'tight_fresh_20261116_02'),(18,'tight_fresh_20261116_02'),(19,'tight_fresh_20261116_04')]}
def read(p):return json.loads(p.read_text())
def lines(p):
 with p.open() as f:
  for line in f:
   if line.strip():yield json.loads(line)
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(n,rows):
 keys=list(dict.fromkeys(k for r in rows for k in r));op=gzip.open if n.endswith('.gz') else open
 with op(O/n,'wt',newline='') as f:
  w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)
def save(fig,n):
 fig.tight_layout();fig.savefig(O/n,dpi=150);plt.close(fig)
def compact(x):
 if isinstance(x,dict):return {k:compact(v) for k,v in x.items() if k not in ('episodes','offline_planning','frozen_visual_io','episode_io','training_epoch_io','latest_episode_visits')}
 if isinstance(x,list):return [compact(v) for v in x]
 return x
DP=[];REAR=[];DU=[];RU=[];V=[];SUMMARY={};FITS=[]
for n,root in ROOTS.items():
 for role in 'AB':
  t=root/role/'training';rear=root/role/'backend'/role/'training';data=read(root/role/'state_data/data.json');new=set(data['aggregation']['new_training_ids']);orig=set(data['normalization_train_ids'])
  for x in lines(t/'learning.jsonl'):
   visits=x['visits'];row=dict(round=n,role=role,epoch=x['epoch'],update=x['updates'],loss=x['loss'],seconds=x['seconds'],gradient_norm=x['gradient_norm'])
   for k in ('base','position','rotation','hand_geometry'):row[k]=sum(v['parts'].get(k,0)*v['loss_weight'] for v in visits)
   row['read_seconds']=sum(v.get('frozen_visual_io',{}).get('tensor_load_s',0) for v in visits);DU.append(row)
   for v in visits:V.append(dict(round=n,role=role,epoch=x['epoch'],update=x['updates'],cohort='latest4' if v['episode'] in new else 'original30' if v['episode'] in orig else 'previous_dagger',loss=v['loss'],base=v['parts']['base']))
  history=list(lines(t/'validation.jsonl'));initial=read(t/'initial_metrics.json');selected=min(history,key=lambda x:x['validation']['selection'])
  for x in [dict(epoch=0,updates=0,**initial)]+history:
   row=dict(round=n,role=role,epoch=x['epoch'],update=x['updates'],optimization_loss=x.get('loss'),best=x['epoch']==selected['epoch'])
   for split in ('train','validation'):
    for k in ('selection','arm_rmse_rad','hand_rmse_rad'):row[split+'_'+k]=x.get(split,{}).get(k)
   DP.append(row)
  rh=read(rear/'validation.json');sel=read(rear/'selected.json')
  for x in lines(rear/'loss.jsonl'):
   RU.append(dict(round=n,role=role,epoch=x['epoch'],update=x['update'],loss=x['loss'],supervised_ticks=x['supervised_ticks'],contact_ticks=x['contact_ticks'],**x.get('loss_components',{})))
  for x in rh:
   row=dict(round=n,role=role,epoch=x['epoch'],update=x['update'],best=x['epoch']==sel['epoch'])
   for split in ('train','validation'):
    for phase,metrics in x[split].items():
     for k,v in metrics.items():row[f'{split}_{phase}_{k}']=v
   REAR.append(row)
  fits={}
  for name,p in [('initial',rear/'latest_fit_initial.json'),('selected',rear.parent/'offline/latest_fit_selected.json')]:
   if p.exists():
    f=read(p);fits[name]=compact(f)
    for phase,m in f['summary'].items():FITS.append(dict(round=n,role=role,checkpoint=name,phase=phase,**m))
  SUMMARY[f'{n}{role}']=dict(dp_initial=compact(initial),dp_best=compact(selected),dp_last=compact(history[-1]),rear_selected=sel,rear_history=compact(rh),latest_fit=fits)
csvout('dp_epochs.csv',DP);csvout('rear_epochs.csv',REAR);csvout('dp_updates.csv.gz',DU);csvout('rear_updates.csv.gz',RU);csvout('dp_cohort_visits.csv.gz',V);csvout('latest_corrections.csv',FITS);dump('training_summary.json',SUMMARY)
# Four measured curves per role and round; stochastic objective != deterministic DDIM metric.
for role in 'AB':
 fig,axs=plt.subplots(4,3,figsize=(14,13))
 for row,n in enumerate(ROOTS):
  d=[x for x in DP if x['round']==n and x['role']==role];u=[x for x in DU if x['round']==n and x['role']==role]
  axs[row,0].scatter([x['update'] for x in u],[x['loss'] for x in u],s=2,alpha=.15,label='actual stochastic batch loss')
  for key,label,style in [('optimization_loss','epoch mean objective','-o'),('train_selection','fixed original train2 DDIM','--o'),('validation_selection','validation6 DDIM','-s')]:
   a=[x for x in d if x.get(key) is not None];axs[row,0].plot([x['update'] for x in a],[x[key] for x in a],style,label=label,ms=3)
  axs[row,0].set_yscale('log')
  for col,part in [(1,'arm'),(2,'hand')]:
   for split in ('train','validation'):
    a=[x for x in d if x.get(f'{split}_{part}_rmse_rad') is not None];axs[row,col].plot([x['update'] for x in a],[x[f'{split}_{part}_rmse_rad'] for x in a],'-o',label=split+('2' if split=='train' else '6'),ms=3)
   axs[row,col].set_ylabel(part+' RMSE (rad)')
  for ax in axs[row]:
   best=next(x for x in d if x['best']);ax.axvline(best['update'],ls=':',color='black',label='selected');ax.set_title(f'{role} round {n}');ax.set_xlabel('optimizer updates in this continuation');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle('A: q[t+1]' if role=='A' else 'B: absolute expert target',y=1.0);save(fig,f'dp_{role}.png')
 fig,axs=plt.subplots(4,3,figsize=(14,13))
 for row,n in enumerate(ROOTS):
  d=[x for x in REAR if x['round']==n and x['role']==role];u=[x for x in RU if x['round']==n and x['role']==role]
  # Epoch means are aggregates of recorded updates, not invented observations.
  for key in ('loss','exec_fit','fy_fit'):
   epochs=sorted(set(x['epoch'] for x in u));axs[row,0].plot(epochs,[np.mean([x[key] for x in u if x['epoch']==e]) for e in epochs],'-o',label=key,ms=3)
  axs[row,0].set_yscale('log');axs[row,0].set_ylabel('recorded objective component mean')
  for col,part in [(1,'arm'),(2,'hand')]:
   for split,phase in [('train','all'),('validation','all'),('validation','contact'),('validation','noncontact')]:
    axs[row,col].plot([x['epoch'] for x in d],[x[f'{split}_{phase}_{part}_rmse_rad'] for x in d],'-o',label=split+' '+phase,ms=3)
   axs[row,col].set_ylabel(part+' final target RMSE (rad)')
  for ax in axs[row]:
   ax.axvline(next(x for x in d if x['best'])['epoch'],ls=':',color='black',label='selected');ax.set_title(f'{role} rear round {n}');ax.set_xlabel('completed control-tick epoch');ax.grid(alpha=.2);ax.legend(fontsize=7)
 save(fig,f'rear_{role}.png')
fig,axs=plt.subplots(2,2,figsize=(12,8))
for j,role in enumerate('AB'):
 for n in ROOTS:
  u=[x for x in DU if x['round']==n and x['role']==role]
  epochs=sorted(set(x['epoch'] for x in u))
  for key in ('base','position','rotation','hand_geometry'):
   vals=[np.mean([x[key] for x in u if x['epoch']==e]) for e in epochs]
   if max(vals)>0:axs[j,0].plot(epochs,vals,'-o',ms=3,label=f'r{n} {key}')
  for cohort in ('latest4','original30','previous_dagger'):
   v=[x for x in V if x['round']==n and x['role']==role and x['cohort']==cohort];axs[j,1].plot(epochs,[np.mean([x['base'] for x in v if x['epoch']==e]) for e in epochs],'-o',ms=3,label=f'r{n} {cohort}')
 for ax in axs[j]:ax.set_yscale('log');ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2);ax.set_xlabel('epoch');ax.set_title(role)
axs[0,0].set_ylabel('raw loss components (different physical scales)');axs[0,1].set_ylabel('sample/x0 noisy training MSE; changing anchors/noise');save(fig,'dp_components_cohorts.png')
# All requested physical stages; protocols remain separate. Full-task trials overlap cohort counts.
E=[];counts=[]
dirs=[('legacy_A11_B09_GPU',B/'tight_fresh_20261116_01/source_backend_baseline')]+[(str(n),root/('development_camera_repaired_01' if n==16 else 'development')) for n,root in ROOTS.items()]
for label,directory in dirs:
 report=behavior(directory);backend=read(directory/'execution_backend_observed.json') if (directory/'execution_backend_observed.json').exists() else {}
 for row in report['rows']:
  row.update(round=label,directory=str(directory),backend=backend);E.append(row)
 cohorts=sorted({r['diagnostic_cohort'] for r in report['rows']})
 for cohort in cohorts+['full_task_only']:
  for role in 'AB':
   allrows=[r for r in report['rows'] if r['role']==role and (r['diagnostic_cohort']==cohort if cohort!='full_task_only' else not r['grasp_stage_censored'])];legal=[r for r in allrows if r['legal']]
   if not allrows:continue
   c=dict(round=label,role=role,cohort=cohort,legal=len(legal),illegal=len(allrows)-len(legal),full_task_legal=sum(not r['grasp_stage_censored'] for r in legal))
   fun={'contact':lambda r:r['first_contact_completed_tick'] is not None,'two_finger':lambda r:r['first_two_finger_completed_tick'] is not None,'native_lift':lambda r:r['native_lift'],'contact_height':lambda r:r['first_contact_height_completed_tick'] is not None,'hold12':lambda r:r['brief_hold12']['sustained'],'hold60':lambda r:r['hold60']['sustained'],'final_hold60':lambda r:r['hold60']['held_at_final_window'],'alignment':lambda r:r['native_alignment'],'subgoal':lambda r:r['native_subgoals']>0,'insertion':lambda r:r['insertion']}
   c.update({key:sum(bool(f(r)) for r in legal) for key,f in fun.items()});counts.append(c)
dump('autonomy_cases.json',E);csvout('autonomy_stages.csv',counts)
fig,axs=plt.subplots(1,2,figsize=(15,6));stages=['contact','two_finger','native_lift','contact_height','hold12','hold60','final_hold60','alignment','subgoal']
for ax,role in zip(axs,'AB'):
 selected=[r for r in counts if r['role']==role and r['cohort']=='fixed_development']
 if not selected:selected=[r for r in counts if r['role']==role and 'development' in r['cohort']]
 matrix=np.array([[r[k]/r['legal'] if r['legal'] else 0 for k in stages] for r in selected]);ax.imshow(matrix,vmin=0,vmax=1,cmap='YlGn',aspect='auto')
 for i,r in enumerate(selected):
  for j,k in enumerate(stages):ax.text(j,i,f"{r[k]}/{r['legal']}",ha='center',va='center',fontsize=8)
 ax.set_xticks(range(len(stages)),stages,rotation=55,ha='right');ax.set_yticks(range(len(selected)),[r['round'] for r in selected]);ax.set_title(role+' fixed development (legal starts)')
fig.suptitle('Separate physical criteria: native lift and contact-height are NOT nested funnel stages');save(fig,'autonomy_stages.png')
# Existing offline phase reports retain exact native phase labels.
phase=[]
for n,root in ROOTS.items():
 for role in 'AB':
  for name in ('stage_errors.csv','base_stage_errors.csv','contact_phase_metrics.csv'):
   p=root/role/'backend'/role/'offline'/name
   if p.exists():
    with p.open() as f:
     phase += [dict(round=n,role=role,source=name,**r) for r in csv.DictReader(f)]
csvout('offline_phase_metrics.csv',phase)
manifest=[]
for n,root in ROOTS.items():
 for p in root.rglob('*'):
  if p.name in ('learning.jsonl','loss.jsonl','validation.json','validation.jsonl','initial_metrics.json','latest_fit_selected.json','latest_fit_initial.json'):
   manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
dump('source_logs.json',manifest)
print(json.dumps(dict(dp_epochs=len(DP),rear_epochs=len(REAR),dp_updates=len(DU),rear_updates=len(RU),autonomy_cases=len(E),counts=counts),indent=2))
