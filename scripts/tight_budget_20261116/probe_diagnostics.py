"""Read-only epoch8 diagnostics and training RNG isolation check before updates."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from tools.tight_fresh_train import inputs,new_model,fresh_objective
from visuotactile.bounded_extension import dp_diagnostics
from visuotactile.frozen_visual_io import mapped_profile
from visuotactile.offline_planning import profile
from visuotactile.precision import state_fp32
from visuotactile.fresh_visit_sampling import load_profile
from visuotactile.training_resources import atomic_json
r=Path('outputs/visuotactile/run/tight_budget_20261116_01').resolve();o=r.parent/'tight_fresh_20261116_04/round_19';torch.set_num_threads(2);state_fp32()
class Lease:
 def check(self):assert torch.cuda.mem_get_info()[0]>4*2**30
for role in 'AB':
 if (r/role/'baseline_epoch8_diagnostics.json').exists():continue
 dr,data,records,stats=inputs(r,role);m=new_model(role,stats);s=torch.load(o/role/'training/last.pt',map_location='cpu',mmap=True,weights_only=False);m.load_state_dict(s['ema'])
 objective,_,_=fresh_objective(r,role,data,records,geometry_sampling_profile=load_profile(r)['geometry'] if role=='A' else None)
 before=torch.get_rng_state();cuda=torch.cuda.get_rng_state();start=time.time()
 d=dp_diagnostics(m,records,data,objective,mapped_profile(),profile(32),Lease())
 assert torch.equal(before,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state())
 atomic_json(r/role/'baseline_epoch8_diagnostics.json',dict(epoch=8,checkpoint='last',diagnostics=d,seconds=time.time()-start,rng_unchanged=True))
 print(role,'baseline diagnostics',time.time()-start,flush=True)
 del m,s,records,objective;torch.cuda.empty_cache()
