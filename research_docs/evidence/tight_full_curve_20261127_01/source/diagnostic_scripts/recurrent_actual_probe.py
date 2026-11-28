from pathlib import Path
import copy,gc,json,time,hashlib
import torch
from tools.tight_fresh_train import inputs,new_model,fresh_objective
from visuotactile.native_data import Episode
from visuotactile.fresh_visit_sampling import load_profile,visit
from visuotactile.frozen_visual_io import load_profile as load_io
from visuotactile.recurrent_execution import profile
from visuotactile.trajectory_training import execute_update
from visuotactile.models.vendor.ema_model import EMAModel
from visuotactile.precision import state_fp32
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_full_curve_20261127_01/old_only').resolve();D=Path('outputs/diagnostics/tight_full_curve_20261127_01');read=lambda p:json.loads(p.read_text())
state_fp32();torch.set_num_threads(2);reports=[]
for role in 'AB':
 dr,data,all_records,stats=inputs(R,role)
 order=read(R/'A/training/deck_001.json')['order'];ids=order[:2]
 if role=='B':ids=[i for i in ids if i in all_records]
 assert len(ids)==2
 records={i:all_records[i] for i in ids};sampling=load_profile(R);io=load_io(R)
 objective,_,_=fresh_objective(R,role,data,records,geometry_sampling_profile=sampling['geometry'])
 episodes=[Episode(dr,i,'I-HistoryDP',frozen_visual_io=io) for i in ids]
 source=torch.load(read(R/role/'continuation.json')['path'],map_location='cpu',weights_only=False,mmap=True);state=source['ema']
 models=[];times=[];losses=[];noise_states=[];peaks=[]
 # Reverse order after warm-up; each disposable update starts from identical weights.
 for method in ['original','packed','packed','original']:
  model=new_model(role,stats);model.load_state_dict(state)
  if method=='packed':model._offline_recurrent_execution=profile()
  averaged=copy.deepcopy(model).eval().requires_grad_(False);ema=EMAModel(averaged,power=.75)
  opt=torch.optim.AdamW(model.parameters(),lr=1.5e-5,betas=(.95,.999),weight_decay=1e-6)
  noise=torch.Generator(device='cuda').manual_seed(2026111403)
  sampled=[visit(records[i],sampling,identity=i,epoch=1,visit_index=j+1) for j,i in enumerate(ids)]
  torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
  result=execute_update(model,opt,ema,objective,ids,[records[i] for i in ids],episodes,noise,
      selected_anchors=[v[0] for v in sampled],anchor_receipts=[v[1] for v in sampled],geometry_generators=[v[2] for v in sampled],geometry_sampling_profile=sampling['geometry'])
  torch.cuda.synchronize();seconds=time.perf_counter()-began
  peak=torch.cuda.max_memory_allocated();fingerprint=hashlib.sha256(noise.get_state().cpu().numpy().tobytes()).hexdigest()
  models.append({n:p.detach().cpu().clone() for n,p in model.named_parameters()});times.append(dict(method=method,seconds=seconds,loss=result['loss'],peak_allocated=peak));noise_states.append(fingerprint)
  print(json.dumps(dict(role=role,**times[-1])),flush=True)
  del model,averaged,ema,opt,sampled,result;gc.collect();torch.cuda.empty_cache()
 differences=[max(float((models[0][n]-models[j][n]).abs().max()) for n in models[0]) for j in [1,2,3]]
 reports.append(dict(role=role,identities=ids,lengths=[len(records[i]['time']) for i in ids],rows=times,
   maximum_updated_parameter_errors_vs_original=differences,identical_noise_consumption=len(set(noise_states))==1,
   scope='Four disposable one-step actual-data updates, identical original parent; excluded from main training age and no physics'))
 del source,state,models,episodes,objective,all_records,records;gc.collect();torch.cuda.empty_cache()
atomic_json(D/'recurrent_actual_probe.json',dict(utc=time.time(),reports=reports));print('COMPLETE',flush=True)
