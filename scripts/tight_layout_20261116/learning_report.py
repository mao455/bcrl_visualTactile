"""Small closed-stage summaries; losses are compared only within each label space."""
import argparse,json,time
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

read=lambda p:json.loads(p.read_text())

def summarize(root,output):
 report=dict(root=str(root),utc=time.time(),roles={},complete=(root/'complete.json').exists())
 curves={}
 for role in 'AB':
  report['roles'][role]={}
  for label,folder in [('DP',root/role/'training'),('Transformer',root/role/'backend'/role/'training')]:
   rows=[];path=folder/'validation.jsonl'
   values=[json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
   if not values and (folder/'validation.json').exists():
    value=read(folder/'validation.json');values=value if isinstance(value,list) else [value]
   if values:
    for v in values:
     score=v.get('validation',{}).get('selection',v.get('selection_score'))
     if score is None:continue
     rows.append(dict(epoch=v['epoch'],updates=v.get('updates',v.get('update')),score=score,train_loss=v.get('loss'),stopping=v.get('stopping')))
   item=dict(history=rows,validation_only=True)
   receipt=folder/'finished.json'
   if receipt.exists():
    v=read(receipt);item['finished']={k:x for k,x in v.items() if k!='plateau_state'}
    item['finished_sha256']=sha256(receipt)
   report['roles'][role][label]=item;curves[(role,label)]=rows
 if (root/'development/result.json').exists():
  from tools.tight_fresh_grasp_evidence import behavior
  report['development']=behavior(root/'development')
  report['paired_initial_check']=read(root/'development/paired_initial_check.json')
  report['evaluation_result_sha256']=sha256(root/'development/result.json')
  report['interpretation']='Native full-task insertion is primary. Holding is diagnostic. These development states can guide DAgger; they are not final held-out test performance.'
 output.mkdir(parents=True,exist_ok=True);atomic_json(output/'learning.json',report)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(10,6))
 for i,role in enumerate('AB'):
  for j,label in enumerate(['DP','Transformer']):
   ax=axes[i,j];rows=curves[(role,label)]
   if rows:ax.plot([r['epoch'] for r in rows],[r['score'] for r in rows],label='Independent validation')
   ax.set(title=role+' '+label,xlabel='Epoch',ylabel='Within-role selection score');ax.grid(alpha=.25)
 fig.suptitle('New 10 cm layout: common data, separate target spaces')
 fig.tight_layout();fig.savefig(output/'validation.png',dpi=130);plt.close(fig)
 print(json.dumps(dict(complete=report['complete'],development=report.get('development',{}).get('counts')),ensure_ascii=False))


def budget_audit(root,output):
 """Read actual selected lineage and masked sampled-target exposure, no GPU."""
 import torch
 import numpy as np
 from visuotactile.target_windows import window_mask
 from visuotactile.anchor_sampling import supervision_rows
 report=dict(root=str(root),roles={},semantics=dict(epoch='One registered complete trajectory deck',
  optimizer_update='One Adam step; DP B2, rear fixed128 valid-tick gradient window with microbatch4',
  diffusion_step='Noise index sampled inside a DP update, never an optimizer count',
  control_tick='Native60Hz interaction timestamp',dagger_round='Collection, shared aggregation, paired full-policy update',
  dp_exposure='Valid sampled horizon slots, repeats counted; independent episode/ticks reported separately',
  backend_exposure='Actual valid expert ticks, student prefixes retained only as history'))
 def load(p):return torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 def rates(c):return [{k:v for k,v in g.items() if k!='params'} for g in c['optimizer']['param_groups']]
 def rear_exposure(checkpoint):
  """Count actual completed episode visits, excluding unlabeled prefix ticks."""
  counts=checkpoint.get('episode_counts')
  if counts is None:return dict(available=False,reason='Historical checkpoint lacks per-episode visit counts')
  source=Path(checkpoint['configuration']['source_root'])/'state_data'
  data=read(source/'data.json');records=load(source/'numeric.pt')['records']
  original=set(data['normalization_train_ids']);totals=dict(original_early=0,original_later=0,correction_onset24=0,correction_later=0)
  unique=dict.fromkeys(totals,0);history=0;supervised=0
  for identity,visits in counts.items():
   assert identity in checkpoint['train_ids'] and visits>=0
   mask,_=supervision_rows(records[identity]);tick=mask.nonzero().reshape(-1).numpy()
   history+=len(mask)*visits;supervised+=len(tick)*visits
   if not len(tick):continue
   if identity in original:bins=dict(original_early=tick<150,original_later=tick>=150)
   else:bins=dict(correction_onset24=tick<tick[0]+24,correction_later=tick>=tick[0]+24)
   for name,selected in bins.items():
    n=int(selected.sum());totals[name]+=n*visits;unique[name]+=n if visits else 0
  assert supervised==checkpoint['supervised_ticks'] and history==checkpoint['control_ticks']
  assert sum(totals.values())==supervised
  return dict(available=True,supervised_ticks=supervised,full_history_ticks=history,
   masked_history_ticks=history-supervised,phase_ticks=totals,phase_unique_episode_ticks=unique)
 def exposure(folder,checkpoint):
  role_root=folder.parent;numeric=role_root.parent/'B/state_data/numeric.pt'
  data=read(role_root/'state_data/data.json');records=load(numeric)['records'];original=set(data['normalization_train_ids'])
  totals={};distinct={};windows=0;targets=0;updates=checkpoint['updates']
  contract=checkpoint['configuration'].get('continuation',{}).get('target_window_contract')
  for line in (folder/'learning.jsonl').open():
   row=json.loads(line)
   if row['updates']>updates:break
   for visit in row.get('visits',[row]):
    identity=visit['episode'];rec=records[identity];mask,segment=supervision_rows(rec)
    anchors=torch.tensor(visit['anchor_sampling']['selected_anchors'],dtype=torch.long)
    valid=window_mask(mask,segment,anchors,24,contract)
    tick=(anchors[:,None]+torch.arange(24)[None])[valid].numpy()
    assert len(tick)==visit['target_ticks']
    windows+=len(anchors);targets+=len(tick)
    first=int(mask.nonzero()[0]);is_original=identity in original
    bins=dict(original_early=tick<150 if is_original else np.zeros(len(tick),bool),
      correction_onset24=(tick<first+24) if not is_original else np.zeros(len(tick),bool),
      original_later=tick>=150 if is_original else np.zeros(len(tick),bool),
      correction_later=(tick>=first+24) if not is_original else np.zeros(len(tick),bool))
    for name,selected in bins.items():
     totals[name]=totals.get(name,0)+int(selected.sum())
     distinct.setdefault(name,set()).update((identity,int(t)) for t in tick[selected])
  assert windows==checkpoint['anchors_seen'] and targets==checkpoint['target_ticks']
  return dict(windows=windows,target_slots=targets,phase_slots=totals,
   phase_unique_episode_ticks={k:len(v) for k,v in distinct.items()})
 for role in 'AB':
  report['roles'][role]={}
  for component,start in [('DP',root/role/'training/best.pt'),('Transformer',root/role/'backend'/role/'training/best.pt')]:
   rows=[];path=start;seen=set()
   while path is not None:
    assert str(path) not in seen;seen.add(str(path));c=load(path);folder=path.parent;last=load(folder/'last.pt')
    cfg=c['configuration'];finished=read(folder/'finished.json');prior=cfg.get('continuation') if component=='DP' else c.get('continuation_from')
    row=dict(checkpoint=str(path.resolve()),sha256=sha256(path),selected_epoch=c['epoch'],selected_updates=c['updates'],
      actual_epoch=last['epoch'],actual_updates=last['updates'],train_trajectories=len(c['train_ids']),
      selected_optimizer=rates(c),last_optimizer=rates(last),ema=c.get('ema_optimization_step',c.get('ema')),
      prior=None if not prior else prior['path'],normalizer=cfg.get('statistics_sha256',cfg.get('residual_statistics_sha256')),
      optimizer_initialization='full state with explicit schedule restart' if (folder/'extension_restore.json').exists() else 'fresh optimizer',
      finished={k:finished.get(k) for k in ['epochs','updates','wall_seconds','wall_s','supervised_ticks','control_ticks','microbatch']})
    if component=='DP':row.update(selected_exposure=exposure(folder,c),actual_anchors=last['anchors_seen'],actual_target_slots=last['target_ticks'])
    else:row.update(selected_supervised_ticks=c['supervised_ticks'],selected_control_ticks=c['control_ticks'],
      actual_supervised_ticks=last['supervised_ticks'],frontend=cfg['bases'][role],microbatch=cfg['microbatch'],
      gradient_window=cfg.get('tick_window_contract'),selected_episode_counts=c.get('episode_counts'),
      selected_phase_exposure=rear_exposure(c),actual_phase_exposure=rear_exposure(last))
    rows.append(row);path=Path(prior['path']) if prior else None
   report['roles'][role][component]=rows
 output.parent.mkdir(parents=True,exist_ok=True);atomic_json(output,report)
 print(json.dumps({r:{k:[(v['train_trajectories'],v['selected_updates'],v['actual_updates']) for v in rows] for k,rows in comps.items()} for r,comps in report['roles'].items()}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--budget',action='store_true');a=p.parse_args();(budget_audit if a.budget else summarize)(a.root.resolve(),a.output.resolve())
