"""Finite existingB72 fitting dispatch; production training/history consumers only."""
from pathlib import Path
import argparse,json,shutil,sys,time,gc
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256

def read(p):return json.loads(p.read_text())

def final(directory):
 old=directory/'best.pt'
 if old.exists():old.rename(directory/'validation_best.pt')
 shutil.copyfile(directory/'last.pt',old)
 saved=torch.load(old,map_location='cpu',weights_only=False,mmap=True)
 atomic_json(directory/'fixed_final_selection.json',dict(epoch=saved.get('epoch'),updates=saved['updates'],sha256=sha256(old),selection='Preregistered final checkpoint; validation never selects this fit model'))

def worker(root):
 from tools.tight_fresh_train import train
 from tools.tight_fresh_backend import prepare
 from tools.tight_finish import residual_cache
 from visuotactile.anchor_continue import residual_train
 from visuotactile.trajectory_training import profile
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2)
 for role in 'AB':
  if not (root/role/'training/fixed_final_selection.json').exists():
   train(root,role,2,offline_anchor_batch=64,trajectory_batch_profile=profile());final(root/role/'training')
  gc.collect();torch.cuda.empty_cache()
 for role in 'AB':
  prepare(root,role,2,continue_rear=True,offline_anchor_batch=64,frozen_visual_io=read(root/'frozen_visual_io.json'))
  residual_cache(root/role/'backend')
  residual_train(root/role/'backend'/role);final(root/role/'backend'/role/'training')
  gc.collect();torch.cuda.empty_cache()
 atomic_json(root/'fit_complete.json',dict(utc=time.time(),selected='final',main_pool_changed=False))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['b72_fit']=('train','outputs.diagnostics.tight_correction_20261127_01.fit_worker',[])
  sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker']);p.add_argument('--root',type=Path,required=True);a=p.parse_args();worker(a.root)
