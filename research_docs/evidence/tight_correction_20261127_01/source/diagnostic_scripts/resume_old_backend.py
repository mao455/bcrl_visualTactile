from pathlib import Path
import sys,argparse,json,time,gc
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from outputs.diagnostics.tight_correction_20261127_01.fit_worker import final
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';D=P/'outputs/diagnostics/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127';read=lambda p:json.loads(p.read_text())
def prepare():
 root=R/'short_old_only';prior=R/'owners/short_old_only_retry02';assert read(prior/'project_exit.json')['children_reaped']
 paths=[root/'A/training/best.pt',root/'B/training/best.pt',root/'A/backend/A/training/best.pt'];keep={str(p):sha256(p) for p in paths}
 for p,expected in zip(paths,[192,192,668]):
  state=torch.load(p,map_location='cpu',weights_only=False,mmap=True);assert state['updates']==expected;del state
 assert not (root/'B/backend/B/training/loss.jsonl').exists(),'Existing B updates require explicit exact resume, not restart'
 owner=R/'owners/short_old_backend_retry03';owner.mkdir(parents=True);(owner/'resources').mkdir();s=read(root/'session.json');s.update(resource_root=str(owner),owned_worker_roots=[str(root),str(owner)],recovery_registered_utc=time.time());atomic_json(owner/'session.json',s)
 archive=root/'attempt_vram03';archive.mkdir()
 for name in ['resources','shared_resources.json']:
  p=root/name;assert p.is_symlink();p.rename(archive/(name+'.prior_link'));p.symlink_to(owner/name,target_is_directory=name=='resources')
 atomic_json(root/'backend_only_recovery.json',dict(utc=time.time(),preserved_models=keep,frontend_updates_repeated=0,completed_A_rear_updates_repeated=0,B_rear_committed_updates_before=0,original_deadline=s['deadline'],reason='VRAM reserve in B nominal cache; keep completed cache entries and only finish B'))
 atomic_json(C/'short_old_backend_retry03.json',dict(version='tight_project_job_v1',operation='train',resource_root=str(owner),stages=[dict(entry='resume_old_rear',args=['--root',str(root)])]));print(owner)
def worker(root):
 from tools.tight_finish import residual_cache
 from visuotactile.anchor_continue import residual_train
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2);proof=read(root/'backend_only_recovery.json')
 for p,digest in proof['preserved_models'].items():assert sha256(p)==digest
 residual_cache(root/'B/backend');gc.collect();torch.cuda.empty_cache()
 residual_train(root/'B/backend/B');final(root/'B/backend/B/training');gc.collect();torch.cuda.empty_cache()
 for p,digest in proof['preserved_models'].items():assert sha256(p)==digest
 atomic_json(root/'short_complete.json',dict(utc=time.time(),selected='fixed_final',main_pool_changed=False,recovery='Only B cache/rear;192+192+668 preserved exactly'))
if __name__=='__main__':
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['resume_old_rear']=('train','outputs.diagnostics.tight_correction_20261127_01.resume_old_backend',[]);sys.argv.pop(1);tight_project.main()
 else:
  p=argparse.ArgumentParser();p.add_argument('--root',type=Path);a=p.parse_args();worker(a.root) if a.root else prepare()
