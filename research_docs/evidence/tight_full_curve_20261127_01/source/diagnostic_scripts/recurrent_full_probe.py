from pathlib import Path
import copy,gc,json,time,runpy
import torch
from visuotactile.trajectory_batch import encode_trajectories
from outputs.diagnostics.tight_full_curve_20261127_01.recurrent_candidate import candidate
from tools.tight_fresh_train import new_model
from visuotactile.precision import state_fp32
from visuotactile.training_resources import atomic_json
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01'
torch.set_num_threads(2);state_fp32()
factory=runpy.run_path('tests/test_trajectory_batch.py')['episode']
pairs=[factory(25,432),factory(37,433)];records,episodes=zip(*pairs)
stats=torch.load(R/'old_only/A/state_data/statistics.pt',map_location='cpu',weights_only=False)
model=new_model('A',stats);source=torch.load(json.loads((R/'registration.json').read_text())['starting_pairs']['A']['frontend'],map_location='cpu',weights_only=False,mmap=True);model.load_state_dict(source['ema']);del source
state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};rows=[];outcomes=[];gradient=[];parameters=[]
for method,fun in [('original',encode_trajectories),('packed',candidate())]:
 model.load_state_dict(state);model.train();model.zero_grad(set_to_none=True)
 optimizer=torch.optim.AdamW(model.parameters(),lr=1.5e-5,betas=(.95,.999),weight_decay=1e-6)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();began=time.perf_counter()
 values=fun(model,records,episodes);loss=sum(c.square().mean() for c,_ in values)/2;loss.backward()
 torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();torch.cuda.synchronize()
 rows.append(dict(method=method,forward_backward_update_s=time.perf_counter()-began,loss=float(loss.detach()),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
 outcomes.append([c.detach().cpu().clone() for c,_ in values]);gradient.append({n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None});parameters.append({n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None})
 del values,loss,optimizer;model.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache()
report=dict(rows=rows,maximum_condition_error=max(float((a-b).abs().max()) for a,b in zip(*outcomes)),maximum_gradient_error=max(float((gradient[0][n]-gradient[1][n]).abs().max()) for n in gradient[0]),maximum_adam_parameter_error=max(float((parameters[0][n]-parameters[1][n]).abs().max()) for n in parameters[0]),gradient_keys_equal=set(gradient[0])==set(gradient[1]),histories=[25,37],scope='Same real architecture, fixed source weights, synthetic observations, one temporary optimization per method; no main-run update or deployment')
atomic_json(D/'recurrent_full_probe.json',report);print(json.dumps(report),flush=True)
