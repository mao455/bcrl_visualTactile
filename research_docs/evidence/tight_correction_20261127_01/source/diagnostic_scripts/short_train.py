from pathlib import Path
import argparse,json,gc,sys,time
import torch
from visuotactile.training_resources import atomic_json
from outputs.diagnostics.tight_correction_20261127_01.fit_worker import final

def worker(root):
 from tools.tight_fresh_train import train
 from tools.tight_fresh_backend import prepare
 from tools.tight_finish import residual_cache
 from visuotactile.anchor_continue import residual_train
 from visuotactile.native_data import cache_visual_pair
 from visuotactile.trajectory_training import profile
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2)
 from visuotactile.frozen_payload_audit import FrozenPayloadAudit
 audit=FrozenPayloadAudit()
 arithmetic=json.loads((root/'visual_arithmetic.json').read_text())
 for role in 'AB':
  assert json.loads((root/role/'visual_arithmetic.json').read_text())==arithmetic, 'Explicit child cache arithmetic required before preparation'
  cache_visual_pair(root/role,reuse_index=json.loads((root/role/'continuation.json').read_text())['reuse_index'],payload_audit=audit)
  gc.collect();torch.cuda.empty_cache()
 for role in 'AB':
  train(root,role,1,offline_anchor_batch=64,trajectory_batch_profile=profile());final(root/role/'training');gc.collect();torch.cuda.empty_cache()
 for role in 'AB':
  prepare(root,role,1,continue_rear=True,offline_anchor_batch=64,frozen_visual_io=json.loads((root/'frozen_visual_io.json').read_text()),paired_group_deck=json.loads((root/f'paired_group_deck_{role}.json').read_text()))
  residual_cache(root/role/'backend');residual_train(root/role/'backend'/role);final(root/role/'backend'/role/'training');gc.collect();torch.cuda.empty_cache()
 atomic_json(root/'short_complete.json',dict(utc=time.time(),selected='fixed_final',main_pool_changed=False))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  # A concurrent invocation joins the identical already-authorized finite job.
  # It never starts a second optimizer or restarts an consumed owner.
  job_path=Path(sys.argv[sys.argv.index('--job')+1]);job=json.loads(job_path.read_text());owner=Path(job['resource_root'])
  if (owner/'project_job_started.json').exists():
   started=json.loads((owner/'project_job_started.json').read_text());assert started['job']==job,'Different job cannot join existing owner'
   deadline=json.loads((owner/'session.json').read_text())['deadline']
   while not (owner/'project_complete.json').exists():
    if (owner/'project_failure.json').exists():raise RuntimeError('Joined finite job failed; preserve original attempt and inspect')
    if time.time()>=deadline:raise TimeoutError('Joined finite job expired')
    time.sleep(5)
   target=Path(job['stages'][0]['args'][1]);assert (target/'short_complete.json').exists()
   atomic_json(owner/'joined_dispatch.json',dict(utc=time.time(),job=str(job_path),completed_existing_job=True,additional_optimizer_updates=0))
   print('Joined identical completed finite training job;0additional updates');sys.exit(0)
  from tools import tight_project
  tight_project.ENTRIES['correction_train']=('train','outputs.diagnostics.tight_correction_20261127_01.short_train',[]);sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);worker(p.parse_args().root)
