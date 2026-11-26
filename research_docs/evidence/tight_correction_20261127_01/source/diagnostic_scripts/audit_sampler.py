"""Reconstruct actual native diffusion draws from recorded per-case RNG states."""
from pathlib import Path
import json,csv,time,hashlib
import torch,numpy as np
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01');read=lambda p:json.loads(p.read_text());rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for row in csv.DictReader((D/'all_completed_development_cases.csv').open()):
 directory=Path(row['source']).parent;states={};receipts={}
 for stage in ['start','end']:
  receipt=read(directory/f'sampler_rng_{stage}.json');p=directory/receipt['state_path'];assert sha(p)==receipt['state_sha256'];assert receipt['initial_seed']==receipt['checkpoint_sampler_seed']==7091 and receipt['extra_random_draws']==0
  states[stage]=torch.load(p,map_location='cpu',weights_only=False)['generator_state'];receipts[stage]=receipt
 with np.load(directory/'plans.npz') as plans:
  assert len(plans.files)==1,plans.files
  n=plans[plans.files[0]].shape[0]
 assert n==(int(row['steps'])+11)//12,(directory,n,row['steps'])
 generator=torch.Generator(device='cuda:0').manual_seed(7091);initial=torch.equal(generator.get_state().cpu(),states['start']);generator.set_state(states['start'])
 for _ in range(n):torch.randn((1,24,29),generator=generator,device='cuda:0')
 exact=torch.equal(generator.get_state().cpu(),states['end']);assert exact,directory
 rows.append(dict(role=row['role'],seed=int(row['seed']),planning_calls=n,start_equals_fresh7091=initial,actual_consumption_matches=exact,start_state_sha256=receipts['start']['state_sha256'],end_state_sha256=receipts['end']['state_sha256'],path=str(directory)))
assert len(rows)==64 and all(r['start_equals_fresh7091'] for r in rows)
atomic_json(D/'sampler_consumption_audit.json',dict(rows=rows,utc=time.time(),all64_fresh7091=True,all64_native_draws_match=True,draw_shape=[1,24,29],hidden_physics_random_stream_claim=False))
print('64 actual start/end RNG reconstructions passed; each begins fresh7091')
