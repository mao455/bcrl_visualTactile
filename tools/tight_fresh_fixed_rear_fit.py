"""Bounded train2 capacity diagnosis with the actual DP frozen.

The complete student still contains DP+Transformer. Only the latter is optimized
on two original successful demonstrations. No held-out labels or new simulation
are read by this command; resulting checkpoints are explicitly diagnostic.
"""
import argparse,copy,json,time,subprocess
from pathlib import Path
import torch
from visuotactile.anchor_continue import residual_aux,residual_prefix_state,residual_prediction
from visuotactile.native_data import residual_batch,residual_supervision_mask,residual_previous_targets
from visuotactile.models.native_dp import load_residual
from visuotactile.residual_tick_window import ResidualTickWindow,TICK_WINDOW_CONTRACT
from visuotactile.precision import state_fp32
from visuotactile.compact import sha256
from visuotactile.checkpointing import save,cpu_snapshot
from visuotactile.state_resources import lease_for
from visuotactile.training_resources import atomic_json

read=lambda p:json.loads(Path(p).read_text())

def detach(value):
 return {k:detach(v) for k,v in value.items()} if isinstance(value,dict) else value.detach()

@torch.no_grad()
def metrics(model,records,ids,lease):
 rows={};selected=[]
 for identity in ids:
  record=records[identity];prediction=residual_prediction(model,record,lease)
  mask=record['supervision_mask'].cpu();error=prediction-record['target'].cpu()
  row={}
  for name,keep in [('all_legal',mask),('before150',mask & (torch.arange(len(mask))<150))]:
   e=error[keep];normalized=e*model.action_scale.cpu()
   row[name]=dict(ticks=len(e),arm_rmse_rad=float(e[:,:7].square().mean().sqrt()),
    hand_rmse_rad=float(e[:,7:].square().mean().sqrt()),
    normalized_equal_mse=float(.5*(normalized[:,:7].square().mean()+normalized[:,7:].square().mean())))
  rows[identity]=row;selected.append(row['before150']['normalized_equal_mse'])
 return dict(episodes=rows,train_early_selection_score=sum(selected)/len(selected),
  selection='Train2 early command fit only; not validation selection or evidence of generalization')

def train(source,role,resource_root,output,*,maximum_passes=80,wall_seconds=900):
 if output.exists():raise FileExistsError('Preserve earlier diagnostic; use a new identity')
 if not 1<=maximum_passes<=100 or not 60<=wall_seconds<=1800:raise ValueError('Bounded diagnostic budget required')
 state_fp32();torch.set_num_threads(2);torch.manual_seed(2026111602);began=time.time()
 session=read(resource_root/'session.json');deadline=min(session['deadline']-120,began+wall_seconds)
 lease=lease_for(resource_root,deadline=deadline)
 base=source/role;rear=base/'backend'/role;data=read(base/'state_data/data.json')
 ids=data['normalization_train_ids'][:2]
 assert len(set(ids))==2 and set(ids)<=set(data['splits']['train'])
 assert not set(ids)&set(data['splits']['validation']+data['splits'].get('test',[]))
 path=rear/'training/best.pt';saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 cfg=saved['configuration'];index=read(rear/'nominal/index.json')
 assert sha256(cfg['base_checkpoint'])==cfg['base_sha256']==index['base_sha256']
 assert cfg['tick_window_contract']==TICK_WINDOW_CONTRACT
 records={};bindings={}
 for identity in ids:
  p=rear/'nominal'/(identity+'.pt');digest=sha256(p)
  assert digest==index['episodes'][identity]['sha256']
  records[identity]=torch.load(p,map_location='cpu',weights_only=False)
  assert records[identity]['supervision_mask'].all(), 'Original full successful expert demonstration required'
  bindings[identity]=dict(path=str(p),sha256=digest,source_sha256=data['entries'][identity]['source_sha256'])
 output.mkdir(parents=True)
 registration=dict(version='fixed_original_train2_frozen_dp_rear_fit_v1',source_root=str(source),role=role,
  checkpoint=str(path),checkpoint_sha256=sha256(path),frozen_dp=cfg['base_checkpoint'],frozen_dp_sha256=cfg['base_sha256'],
  identities=ids,bindings=bindings,maximum_passes=maximum_passes,wall_seconds=wall_seconds,deadline=deadline,
  learning_rate=1e-4,weight_decay=1e-4,early_weight=3.,early_ticks=150,tick_window=TICK_WINDOW_CONTRACT,
  history='Actual saved complete observations and past issued actions; recompute current-weight prefix after every update; no cached learned state',
  task='Capacity diagnosis of final complete-policy command; frozen DP means its correctly bound existing inputs are reused',
  test_read=False,formal_candidate=False,started=began)
 atomic_json(output/'registration.json',registration)
 with lease.block():model=load_residual(saved,'cuda:0')
 template={k:v for k,v in saved.items() if k not in ('model','optimizer','torch_rng','cuda_rng','schedule_rng','history')}
 del saved
 configuration=copy.deepcopy(cfg);configuration['fixed_train_diagnostic']=registration
 configuration['selection']='Original train2 early fit; diagnostic only, no validation model selection'
 configuration['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
 configuration['source_sha256']={str(p):sha256(p) for folder in ('tools','visuotactile') for p in sorted(Path(folder).rglob('*.py'))}
 configuration['continuation_from']=dict(path=str(path),sha256=registration['checkpoint_sha256'],teacher_supervision='u_expert')
 configuration['continuation_audit']=dict(source=str(path),source_sha256=registration['checkpoint_sha256'],
  source_updates=template['updates'],source_checkpoint_updates=template['updates'],optimizer_state_restored=False)
 configuration['fresh_recovery_weighting']=dict(version='fresh_early_grasp_control_weighting_v1',ticks=150,backend_weight=3.)
 configuration['recovery_weighting_scope']='Diagnostic original train2 legal controls before150 weight3; later legal controls weight1.'
 opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
 window=ResidualTickWindow(TICK_WINDOW_CONTRACT);updates=draws=0;history=[];best=float('inf');passes=0
 def snapshot():
  return cpu_snapshot(dict(template,configuration=configuration,model=model.state_dict(),optimizer=opt.state_dict(),
   updates=updates,new_optimizer_updates=updates,formal_updates=0,epoch=passes,diagnostic_only=True,
   train_ids=ids,validation_ids=[],history=history,supervised_ticks=draws,
   episode_counts={i:passes for i in ids},torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()))
 def review():
  nonlocal best
  row=dict(passes=passes,updates=updates,supervised_ticks=draws,elapsed_seconds=time.time()-began,
   **metrics(model,records,ids,lease));history.append(row)
  atomic_json(output/'metrics.json',history)
  if row['train_early_selection_score']<best:
   best=row['train_early_selection_score'];save(output/'best.pt',snapshot())
   atomic_json(output/'selected.json',dict(passes=passes,updates=updates,score=best,checkpoint_sha256=sha256(output/'best.pt')))
  return row
 review();selected=[records[i] for i in ids];length=max(len(r['robot']) for r in selected)
 stop='maximum_passes'
 for epoch in range(1,maximum_passes+1):
  if time.time()>deadline-90:stop='bounded_wall';break
  state=None;prefix=[];dirty=False
  for start in range(0,length,32):
   end=min(start+32,length)
   with lease.block():
    model.train();inputs=residual_batch(selected,start,end,'cuda:0')
    legal=residual_supervision_mask(selected,start,end,'cuda:0')&inputs['valid']
    if dirty:state=residual_prefix_state(model,prefix);dirty=False
    raw,state=model(inputs,state);gate=residual_aux(selected,start,end,'gate','cuda:0')&inputs['valid']
    limits=torch.stack([r['limits'] for r in selected]).cuda()[:,None]
    chain=model.compose_chain(residual_aux(selected,start,end,'base','cuda:0'),raw,gate,limits,
      previous=residual_previous_targets(inputs,limits))
    weights=legal.float()*(1+2*(torch.arange(start,end,device='cuda')[None]<150))
    parts=model.objective_components(chain['u_exec'],residual_aux(selected,start,end,'target','cuda:0'),
     chain['delta'],weights,u_pre=chain['u_pre'],execution_bounds=chain['execution_bounds'])
    if not torch.isfinite(parts['total']):raise FloatingPointError('Nonfinite diagnostic loss')
    if legal.any():
     window.backward(parts['total'],opt,weight_mass=float(weights.sum()),supervised_ticks=int(legal.sum()),
      contact_ticks=int((gate&legal).sum()),history_ticks=int(inputs['valid'].sum()),weight_report={})
     draws+=int(legal.sum())
    if window.pending and (window.ready or end==length):
     report=window.step(model,opt);updates+=1;dirty=True
     with (output/'loss.jsonl').open('a') as stream:
      stream.write(json.dumps(dict(pass_index=epoch,updates=updates,**report))+'\n')
    state=detach(state);prefix.append(inputs)
  passes=epoch
  atomic_json(output/'progress.json',dict(passes=passes,updates=updates,supervised_ticks=draws,utc=time.time()))
  print(json.dumps(dict(passes=passes,updates=updates,utc=time.time())),flush=True)
  if epoch==1 or epoch%5==0:
   review();save(output/'last.pt',snapshot())
   if epoch>=20 and len(history)>=5:
    old=history[-5]['train_early_selection_score']
    if (old-best)/max(old,1e-12)<.002:stop='train_fit_plateau';break
 if history[-1]['updates']!=updates:review()
 save(output/'last.pt',snapshot())
 atomic_json(output/'finished.json',dict(complete=True,stop_reason=stop,passes=passes,updates=updates,
  supervised_ticks=draws,seconds=time.time()-began,best=read(output/'selected.json'),
  scope='Real rear optimizer updates on original train2; frozen paired DP unchanged. This fit is not autonomous success or a formal candidate.'))

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__)
 for name in ('source','resource-root','output'):p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--role',choices=['A','B'],required=True);p.add_argument('--maximum-passes',type=int,default=80)
 p.add_argument('--wall-seconds',type=int,default=900);a=p.parse_args()
 train(a.source.resolve(),a.role,a.resource_root.resolve(),a.output.resolve(),maximum_passes=a.maximum_passes,wall_seconds=a.wall_seconds)
