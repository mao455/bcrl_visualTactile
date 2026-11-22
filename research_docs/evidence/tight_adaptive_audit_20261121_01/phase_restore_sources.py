from pathlib import Path
import sys,json,hashlib,copy,io
import torch,numpy as np,zstandard
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.models.native_dp import load_policy
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.checkpointing import restore_training
from visuotactile.latest_takeover_sampling import episode_deck
from visuotactile.native_data import visual_records
from visuotactile.target_windows import with_target_window
from visuotactile.anchor_sampling import labels,planning_anchors,validate_training_anchors
from visuotactile.compact import sha256

def read(p):return json.loads(Path(p).read_text())
def write(n,x):(OUT/n).write_text(json.dumps(x,indent=2)+'\n')
def same(a,b):
 if torch.is_tensor(a):return torch.equal(a.cpu(),b.cpu())
 if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
 return a==b

def main():
 torch.set_num_threads(2);facts=read(OUT/'data_chain.json')['episodes'];phase=[]
 for round in (1,2):
  root=RUN/f'round{round:02d}/learning';selected=[x for x in facts if x['round']==round];ids=[x['identity'] for x in selected];dr=root/'B/state_data';records=with_target_window(visual_records(dr,ids,'G00'),'recovery_onset_commit12_v1')
  for role in ('A','B'):
   saved=torch.load(root/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True);coverage=saved['visit_sampling_coverage']
   for fact in selected:
    i=fact['identity'];first=fact['first_legal'];anchor=first//12*12;rec=records[i];truth,mask=labels(rec,torch.tensor([anchor]),24);error=None
    try:validate_training_anchors(rec,torch.tensor([anchor]))
    except ValueError as e:error=str(e)
    seen=coverage[i]['anchor_ticks'];phase.append(dict(round=round,role=role,identity=i,first_legal=first,native_onset_anchor=anchor,first_legal_slot=first-anchor,selected_epoch=saved['epoch'],visits=coverage[i]['visits'],onset_anchor_allowed_by_target_window=anchor in planning_anchors(rec).tolist(),native_onset_loss_slots=mask.nonzero()[:,1].tolist(),onset_anchor_ever_selected=anchor in seen,minimum_actual_training_anchor=min(seen),training_validation_error=error))
 write('onset_phase_gap.json',dict(nonboundary_episodes=sum(x['first_legal']%12!=0 for x in facts),total_episodes=len(facts),rows=phase))
 p=RUN/'round02/disk_recovery04/B_interrupted_last.pt';saved=torch.load(p,map_location='cpu',weights_only=False,mmap=True);model=load_policy(saved,'cpu');averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75);opt=torch.optim.AdamW(model.parameters(),lr=1.5e-5,betas=(.95,.999),weight_decay=1e-6);schedule=torch.Generator().manual_seed(1);noise=torch.Generator(device='cuda').manual_seed(1)
 restore_training(saved,model,averaged,opt,ema,schedule,noise)
 check=dict(raw_model=same(saved['model'],model.state_dict()),ema_model=same(saved['ema'],averaged.state_dict()),optimizer=same(saved['optimizer'],opt.state_dict()),ema_step=ema.optimization_step==saved['ema_optimization_step'],schedule_rng=same(schedule.get_state(),saved['schedule_rng']),noise_rng=same(noise.get_state(),saved['noise_rng']),torch_rng=same(torch.get_rng_state(),saved['torch_rng']),cuda_rng=same(torch.cuda.get_rng_state_all(),saved['cuda_rng']))
 cfg=saved['configuration'];dr=RUN/'round02/learning/B/state_data';data=read(dr/'data.json');deck=episode_deck(data['splits']['train'],cfg['episode_multiplicities'],schedule);registered=read(RUN/'round02/learning/B/training/deck_002.json');check['next_epoch_deck_exact']=deck==registered['order'];states=[int(v['step']) for v in opt.state.values()];receipt=read(SRC/'research_docs/evidence/tight_adaptive_dagger_20261119_01/round02/disk_recovery04_first_durable.json')
 write('restore_verification.json',dict(path=str(p),sha256=sha256(p),epoch=saved['epoch'],updates=saved['updates'],optimizer_steps_min=min(states),optimizer_steps_max=max(states),optimizer_lr=[x['lr'] for x in opt.param_groups],plateau_state=saved['plateau_state'],checks=check,next_epoch_deck_sha256=hashlib.sha256(json.dumps(deck,separators=(',',':')).encode()).hexdigest(),first_durable_receipt=receipt,limitation='Epoch2 numeric checkpoint was replaced by later durable snapshots; receipt proves cursor/hash, but cannot independently compare its tensors or original ephemeral gradients. Current restore function and epoch2 deck independently reproduced.'))
 # Source identity follows each actual immutable plan, not the experiment HEAD alone.
 plans=list((RUN/'round01').glob('collection*/plan.json'))+list((RUN/'round02').glob('collection*/plan.json'))+list((RUN/'final_test').glob('paired2_wave*/plan.json'));sourcecache={};planrows=[];pairs={}
 for p in plans:
  plan=read(p);mismatches=[];missing=[]
  for name,expected in plan.get('source_sha256',{}).items():
   path=SRC/name
   if not path.is_file():missing.append(name);continue
   if name not in sourcecache:sourcecache[name]=sha256(path)
   if sourcecache[name]!=expected:mismatches.append(name)
  for role,pair in plan['pairs'].items():
   for field in ('frontend','backend'):
    path=pair[field];pairs[path]=dict(actual_sha256=sha256(Path(path)),expected=pair[field+'_sha256']) if path not in pairs else pairs[path]
  planrows.append(dict(path=str(p),source_commit=plan.get('source_commit'),files=len(plan.get('source_sha256',{})),mismatches=mismatches,missing=missing,kind=plan['kind'],cases=len(plan['cases'])))
 owners=[]
 for p in (RUN/'owners').glob('*/project_job_started.json'):
  x=read(p);owners.append(dict(path=str(p),record=x))
 write('source_and_launch.json',dict(plans=planrows,model_identities=pairs,owners=owners));print('phase',len(phase),'restore',check,'source plans',len(plans),'mismatches',sum(len(x['mismatches']) for x in planrows),'missing',sum(len(x['missing']) for x in planrows))
if __name__=='__main__':main()
