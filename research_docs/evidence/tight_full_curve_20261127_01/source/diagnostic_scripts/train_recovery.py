"""Resume existing scientific state; only resource admission waits are added."""
from pathlib import Path
from contextlib import contextmanager
import argparse,gc,hashlib,json,os,sys,time
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile import state_resources
from outputs.diagnostics.tight_full_curve_20261127_01.train_worker import final,frontend,rear
D=Path('outputs/diagnostics/tight_full_curve_20261127_01');read=lambda p:json.loads(Path(p).read_text())
original=state_resources.FullLease.block
@contextmanager
def bounded_admission(self,**kwargs):
 while True:
  self.check(kwargs.get('deadline'))
  context=original(self,**kwargs)
  try:context.__enter__()
  except RuntimeError as error:
   if str(error) not in {'RAM reserve','VRAM reserve; no OOM retry','Stale resource telemetry'}:raise
   with (D/f'resource_wait_{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(dict(utc=time.time(),reason=str(error),model_update=False,random_draws=0))+'\n')
   time.sleep(2);continue
  try:yield
  except BaseException:
   if not context.__exit__(*sys.exc_info()):raise
  else:context.__exit__(None,None,None)
  return
state_resources.FullLease.block=bounded_admission

def fingerprint(value):
 h=hashlib.sha256()
 def visit(x):
  if isinstance(x,torch.Tensor):
   y=x.detach().cpu().contiguous();h.update(str((y.dtype,tuple(y.shape))).encode());h.update(y.numpy().tobytes())
  elif isinstance(x,dict):
   for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
  elif isinstance(x,(list,tuple)):
   for y in x:visit(y)
  else:h.update(repr(x).encode())
 visit(value);return h.hexdigest()

def resume_b(root):
 from tools import tight_fresh_train as trainer
 from visuotactile.trajectory_training import profile
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2)
 protected=read(root/'A/training/fixed_final_selection.json');assert sha256(root/'A/training/best.pt')==protected['sha256']
 p=root/'B/training/last.pt';before_hash=sha256(p);v=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 before={k:v[k] for k in ['epoch','updates','ema_optimization_step','visits_seen']};assert before['updates']==before['epoch']*206
 atomic_json(D/'frontend_B_resume_registration.json',dict(utc=time.time(),checkpoint=str(p),sha256=before_hash,**before,protected_A=protected,resource_wait_only=True,original_deadline=read(root/'session.json')['deadline']))
 del v;gc.collect()
 restore=trainer.restore_training
 def verified(saved,model,averaged,optimizer,ema,generator,noise):
  restore(saved,model,averaged,optimizer,ema,generator,noise)
  pairs=[('model',model.state_dict(),saved['model']),('ema',averaged.state_dict(),saved['ema']),('optimizer',optimizer.state_dict(),saved['optimizer']),('schedule_rng',generator.get_state(),saved['schedule_rng']),('noise_rng',noise.get_state(),saved['noise_rng']),('torch_rng',torch.get_rng_state(),saved['torch_rng']),('cuda_rng',torch.cuda.get_rng_state_all(),saved['cuda_rng'])]
  results={k:dict(actual=fingerprint(a),expected=fingerprint(b)) for k,a,b in pairs};assert all(x['actual']==x['expected'] for x in results.values());assert ema.optimization_step==saved['updates']
  atomic_json(D/'frontend_B_restored_state_verified.json',dict(utc=time.time(),checkpoint_sha256=before_hash,epoch=saved['epoch'],updates=saved['updates'],ema_optimization_step=ema.optimization_step,states=results,first_new_update=saved['updates']+1,all_equal=True))
 trainer.restore_training=verified
 trainer.train(root,'B',60,offline_anchor_batch=64,trajectory_batch_profile=profile());final(root/'B/training')
 assert sha256(root/'A/training/best.pt')==protected['sha256']
 atomic_json(root/'frontend_complete.json',dict(utc=time.time(),recovered_role='B',preserved_complete_A=True))

if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['curve_recovery']=('train','outputs.diagnostics.tight_full_curve_20261127_01.train_recovery',[])
  sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('mode',choices=['resume_B','frontend','rear']);p.add_argument('--root',type=Path,required=True);p.add_argument('--epochs',type=int);a=p.parse_args()
  if a.mode=='resume_B':resume_b(a.root)
  elif a.mode=='frontend':frontend(a.root)
  else:rear(a.root,a.epochs)
