"""Close diagnostics with model/data identity and real replay checks."""
import json,torch,numpy as np,h5py,subprocess
from .common import *
from visuotactile.compact import sha256
from scripts.tight_layout_20261116.publication import write

def main():
 pairs=read(ROOT/'capture/plan.json')['pairs'];models={};files={}
 for role,p in pairs.items():
  for k in ['frontend','backend']:
   assert sha256(p[k])==p[k+'_sha256'];files[p[k]]=p[k+'_sha256']
  dp=torch.load(p['frontend'],map_location='cpu',weights_only=False,mmap=True);rear=torch.load(p['backend'],map_location='cpu',weights_only=False,mmap=True)
  assert rear['configuration']['base_sha256']==p['frontend_sha256']
  models[role]=dict(pair=p,dp_epoch=dp['epoch'],dp_updates=dp['updates'],rear_epoch=rear['epoch'],rear_updates=rear['updates'],dp_continuation=dp['configuration'].get('continuation'),rear_continuation=rear.get('continuation_from'),normalization_sha256=rear['configuration']['statistics_sha256'])
  if role in 'AB':
   assert dp['configuration']['continuation']['sha256']==pairs[role+'_baseline']['frontend_sha256']
   assert rear['continuation_from']['sha256']==pairs[role+'_baseline']['backend_sha256']
 data={}
 for role in 'AB':
  p=OLD/role/'state_data/data.json';d=read(p);assert len(d['splits']['train'])==76
  data[role]=dict(manifest_sha256=sha256(p),train_ids=d['splits']['train'])
 assert set(data['A']['train_ids'])==set(data['B']['train_ids'])
 replay=read(OUT/'same_input.json');selfs=[r for r in replay['self_replays'] if r['self_replay']]
 assert len(selfs)==4
 for s in selfs:
  e=s['max_abs_error'];assert all(e[k]==0 for k in ['robot','native','visual_memory','visual_read','generator'])
  # Report raw measured errors; this is numerical validation, not bitwise replay.
  assert e['command']<1e-6 and e['nominal']<2e-6 and e['delta']<3e-6
 captures=[]
 for p in sorted((ROOT/'capture').glob('*/numeric.h5')):
  with h5py.File(p) as f:
   o=f['episode/obs'];t=f['episode/transition'];n=len(o['step_id']);assert n==121
   assert np.array_equal(o['step_id'][:],np.arange(n))
   assert np.array_equal(t['position_requested_rad'][:],t['position_applied_rad'][:])
   assert np.array_equal(o['actuator_target_rad'][1:],t['position_applied_rad'][:-1])
  captures.append(dict(path=str(p),sha256=sha256(p),steps=n))
 assert len(captures)==8
 roots=[ROOT,ROOT/'replay',ROOT/'replay_fp32'];owners={str(p):read(p/'owner_exit.json') for p in roots};assert all(r['children_reaped'] for r in owners.values())
 for p in roots:
  monitor=read(p/'monitor_exit.json');assert monitor
 write(OUT/'closure.json',dict(models=models,data=data,protected_model_hashes=files,capture=captures,owners=owners,capture_complete=read(ROOT/'capture_complete.json'),initialization=read(ROOT/'capture/initialization.json'),capture_storage=read(ROOT/'capture_progress.json'),replay_seconds=replay['seconds'],new_optimizer_updates=0,new_dagger=0,full_task_evaluations=0,physical_starts=8,control_ticks=968,initial_prephysics_failure=str(RUNS/'tight_fork_diagnosis_20261117_01'),numerical_replay_note='First offline attempt omitted cudnn FP32 and is preserved locally as replay/result.json. Corrected replay_fp32 uses existing state_fp32; no production change. Command self-error below1e-6rad is measured acceptance, not a preregistered performance threshold.',source_checks={str(p.relative_to(TREE)):sha256(p) for p in (TREE/'scripts/tight_fork_20261117').glob('*.py')}))
 print('8 protected model hashes;4 complete-pair bindings;76 common;8 captures;4 self-replays;3 owner exits verified')
if __name__=='__main__':main()
