"""Independent saved-candidate validation and committed-step ledger rebuild; zero updates."""
from pathlib import Path
import json,sys,gzip,io,contextlib,gc
import torch,numpy as np,zstandard
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from tools.tight_fresh_train import inputs,metrics,fresh_objective
from visuotactile.models.native_dp import load_policy,load_residual
from visuotactile.native_data import Episode
from visuotactile.anchor_sampling import labels
from visuotactile.anchor_continue import residual_metrics,projection_batch
from visuotactile.trajectory_batch import supervised_trajectories,trajectory_objective
from visuotactile.precision import state_fp32
from visuotactile.fresh_visit_sampling import visit as sample_visit
class Lease:
 def block(self):return contextlib.nullcontext()
 def check(self):pass
def read(p):return json.loads(Path(p).read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def rows(p):
 if p.exists():
  with p.open() as f:yield from (json.loads(l) for l in f)
 else:
  with p.with_suffix(p.suffix+'.zst').open('rb') as src,zstandard.ZstdDecompressor().stream_reader(src) as stream,io.TextIOWrapper(stream) as f:yield from (json.loads(l) for l in f)
def exposure(saved,training,records=None):
 state=saved['effective_exposure_state'];counts={i:torch.zeros_like(c) for i,c in state['counts'].items()};visits={i:0 for i in counts};last=0;checks=0
 if records is not None:
  for row in rows(training/'learning.jsonl'):
   if row['updates']>saved['updates']:break
   assert row['updates']==last+1;last=row['updates'];merged={}
   for visit in row['visits']:
    i=visit['episode'];visits[i]+=1;anchors=visit['anchor_sampling']['selected_anchors']
    a=torch.tensor(anchors,device=records[i]['time'].device);_,mask=labels(records[i],a,24);ticks=(a[:,None]+torch.arange(24,device=a.device))[mask].unique().cpu().tolist()
    merged.setdefault(i,set()).update(ticks)
   for i,ticks in merged.items():
    assert sorted(ticks)==row['effective_tick_contributions'][i];counts[i][list(ticks)]+=1
   checks+=1
 else:
  for row in rows(training/'loss.jsonl'):
   update=row.get('updates',row.get('update'))
   if update is None:continue
   if update>saved['updates']:break
   contribution=row.get('effective_tick_contributions')
   if contribution is None:continue
   assert update==last+1;last=update
   for i,ticks in contribution.items():counts[i][list(set(ticks))]+=1
   checks+=1
 differences={i:int((counts[i]!=state['counts'][i]).sum()) for i in counts};return dict(updates_rebuilt=last,rows=checks,count_mismatches=sum(differences.values()),minimum=int(torch.cat([state['counts'][i][state['legal'][i]] for i in counts]).min()),minimum_visits=min(saved.get('sampling_exposure',saved.get('episode_counts',{})).values()),legal_ticks=sum(int(x.sum()) for x in state['legal'].values()))
def gradient(root,role,data,records,saved,identity=None):
 cfg=saved['configuration'];model=load_policy(saved,'cuda').train();model.load_state_dict(saved['model']);i=identity or next(i for i in reversed(data['splits']['train']) if data['entries'][i].get('round_index')==2);rec=records[i]
 objective,_,_=fresh_objective(root,role,data,records,geometry_sampling_profile=cfg['training_sampling']['geometry'])
 a,receipt,gen=sample_visit(rec,cfg['training_sampling'],identity=i,epoch=saved['epoch'],visit_index=1)
 ep=Episode(root/role/'state_data',i,'I-HistoryDP',frozen_visual_io=cfg['frozen_visual_io']);values=supervised_trajectories(model,[rec],[ep],selected_anchors=[a]);context=projection_batch(objective,[(i,int(t)) for t in a],'cuda',**(dict(geometry_generator=gen,sampling_profile=cfg['training_sampling']['geometry']) if role=='A' else {}))
 loss,reports,pred=trajectory_objective(model,objective,values,torch.Generator(device='cuda').manual_seed(8091),[context]);pred.retain_grad();parts=reports[0]['parts'];weighted={k:(v if k=='base' else v*objective.gain*objective.weights[k]) for k,v in parts.items()};component_grad={}
 for k,v in weighted.items():
  g=torch.autograd.grad(v,pred,retain_graph=True)[0];component_grad[k]=dict(arm_l2=float(g[...,:7].norm()),hand_l2=float(g[...,7:].norm()))
 loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.));g=pred.grad;valid=values[0]['valid'];result=dict(identity=i,epoch=saved['epoch'],anchors=a.tolist(),first_legal=int(rec['supervision_mask'].nonzero()[0]),loss=float(loss),parts={k:float(v) for k,v in parts.items()},weights=objective.weights,component_gradient=component_grad,gradient_norm_before_clip=norm,clip_multiplier=min(1.,1./norm),per_joint_output_gradient_rms=g[valid].square().mean(0).sqrt().tolist(),invalid_gradient_max=float(g[~valid].abs().max()) if (~valid).any() else 0.,action_scale=model.action_scale.tolist(),scope='Fixed selected raw state diagnostic; historical per-update gradients are not retained; no optimizer or EMA step')
 del model,objective,ep,loss,pred,values;gc.collect();torch.cuda.empty_cache();return result

def main():
 torch.set_num_threads(2);state_fp32();results=[]
 for round in (1,2):
  root=RUN/f'round{round:02d}/learning'
  for role in ('A','B'):
   data_root,data,records,stats=inputs(root,role);p=root/role/'training';saved=torch.load(p/'best.pt',weights_only=False,map_location='cpu',mmap=True);cfg=saved['configuration'];model=load_policy(saved,'cuda').eval()
   with torch.no_grad():val=metrics(model,records,data['splits']['validation'],offline_planning=cfg['offline_planning'],frozen_visual_io=cfg['frozen_visual_io'])
   hist=[json.loads(l) for l in (p/'validation.jsonl').read_text().splitlines()];chosen=next(x for x in hist if x['epoch']==saved['epoch']);eligible=[x for x in hist if x.get('effective_exposure',{}).get('eligible')]
   item=dict(round=round,role=role,component='DP',epoch=saved['epoch'],updates=saved['updates'],recomputed_validation=val['selection'],recorded_validation=chosen['validation']['selection'],validation_abs_error=abs(val['selection']-chosen['validation']['selection']),eligible_min_epoch=min(eligible,key=lambda x:x['validation']['selection'])['epoch'],exposure=exposure(saved,p,records),lr=[dict(epoch=x['epoch'],lr=x['learning_rates'],next=x['stopping']['next_learning_rates']) for x in hist if x['epoch'] in (1,47,48,49,60)])
   del model;gc.collect();torch.cuda.empty_cache()
   if round==2:item['gradient']=gradient(root,role,data,records,saved)
   results.append(item);write('training_verification.json',results);print(json.dumps({k:v for k,v in item.items() if k not in ('gradient','lr')}),flush=True)
   del saved,records;gc.collect();torch.cuda.empty_cache()
   p=root/role/'backend'/role/'training';saved=torch.load(p/'best.pt',weights_only=False,map_location='cpu',mmap=True);model=load_residual(saved,'cuda').eval();backrecords={i:torch.load(p.parent/'nominal'/(i+'.pt'),weights_only=False,map_location='cpu',mmap=True) for i in data['splits']['validation']}
   with torch.no_grad():val,_=residual_metrics(model,backrecords,data['splits']['validation'],Lease())
   hist=read(p/'validation.json');chosen=next(x for x in hist if x['epoch']==saved['epoch']);eligible=[x for x in hist if x.get('effective_exposure',{}).get('eligible')]
   item=dict(round=round,role=role,component='Transformer',epoch=saved['epoch'],updates=saved['updates'],recomputed_validation=val['all']['normalized_equal_mse'],recorded_validation=chosen['selection_score'],validation_abs_error=abs(val['all']['normalized_equal_mse']-chosen['selection_score']),eligible_min_epoch=min(eligible,key=lambda x:x['selection_score'])['epoch'],exposure=exposure(saved,p))
   results.append(item);write('training_verification.json',results);print(json.dumps(item),flush=True)
   del saved,model,backrecords;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
