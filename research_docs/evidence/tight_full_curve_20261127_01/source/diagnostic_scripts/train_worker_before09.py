from pathlib import Path
import argparse,copy,gc,json,os,shutil,sys,time
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
read=lambda p:json.loads(Path(p).read_text())
def final(directory):
 from visuotactile.checkpointing import promote_snapshot
 if (directory/'best.pt').exists():promote_snapshot(directory/'best.pt',directory/'validation_best.pt')
 promote_snapshot(directory/'last.pt',directory/'best.pt')
 saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False,mmap=True)
 atomic_json(directory/'fixed_final_selection.json',dict(epoch=saved['epoch'],updates=saved['updates'],sha256=sha256(directory/'best.pt'),selection='Preregistered milestone final checkpoint; validation never selects deployment'))
def frontend(root):
 from tools.tight_fresh_train import train
 from visuotactile.trajectory_training import profile
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2)
 for role in 'AB':
  train(root,role,60,offline_anchor_batch=64,trajectory_batch_profile=profile());final(root/role/'training');gc.collect();torch.cuda.empty_cache()
 atomic_json(root/'frontend_complete.json',dict(utc=time.time()))
def rear(root,epochs):
 from tools.tight_fresh_backend import prepare
 from tools.tight_finish import residual_cache
 from visuotactile.anchor_continue import residual_train
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2)
 work=root/f'rear{epochs}'
 if not work.exists():
  work.mkdir()
  s=copy.deepcopy(read(root/'session.json'));s['resource_root']=str(work);atomic_json(work/'session.json',s)
  (work/'shared_resources.json').symlink_to(root/'shared_resources.json');(work/'resources').symlink_to(root/'resources',target_is_directory=True)
  for name in ['frozen_visual_io.json','visual_arithmetic.json','family_phase_diagnostics.json']:
   shutil.copyfile(root/name,work/name)
  atomic_json(work/'matched_backend_learning_rates.json',{r:read(root/'matched_backend_learning_rates.json')[r][:epochs] for r in 'AB'})
  for role in 'AB':
   child=work/role;child.mkdir();(child/'state_data').symlink_to(root/role/'state_data',target_is_directory=True)
   shutil.copyfile(root/role/'cache_index.json',child/'cache_index.json');shutil.copyfile(root/role/'continuation.json',child/'continuation.json')
   (child/'training').mkdir();source=root/role/'training'/f'milestone_epoch_{epochs}.pt';os.link(source,child/'training/best.pt')
   atomic_json(child/'front_source_identity.json',dict(source=str(source),sha256=sha256(source),immutable_snapshot_hardlinked=True,inode=source.stat().st_ino))
   atomic_json(child/'session.json',s);(child/'shared_resources.json').symlink_to(root/'shared_resources.json')
 else:
  for role in 'AB':
   child=work/role;source=root/role/'training'/f'milestone_epoch_{epochs}.pt'
   assert sha256(child/'training/best.pt')==sha256(source)
   assert (child/'state_data').resolve()==(root/role/'state_data').resolve()
 for role in 'AB':
  backend=work/role/'backend';training=backend/role/'training'
  if not backend.exists():
   prepare(work,role,epochs,continue_rear=True,offline_anchor_batch=64,frozen_visual_io=read(root/'frozen_visual_io.json'),paired_group_deck=read(root/f'paired_group_deck_{role}_{epochs}.json'))
  if not (training/'fixed_final_selection.json').exists():
   residual_cache(backend);residual_train(backend/role);final(training);gc.collect();torch.cuda.empty_cache()
  assert read(training/'fixed_final_selection.json')['epoch']==epochs
 atomic_json(root/f'rear{epochs}_complete.json',dict(utc=time.time(),epochs=epochs))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['curve_train']=('train','outputs.diagnostics.tight_full_curve_20261127_01.train_worker',[])
  sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('mode',choices=['frontend','rear']);p.add_argument('--root',type=Path,required=True);p.add_argument('--epochs',type=int,choices=[30,60]);a=p.parse_args()
  if a.mode=='frontend':frontend(a.root)
  else:rear(a.root,a.epochs)
