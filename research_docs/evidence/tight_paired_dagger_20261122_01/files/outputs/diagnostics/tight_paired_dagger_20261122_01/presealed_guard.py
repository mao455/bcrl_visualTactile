"""AUDIT-only, read-only deployment audit before development/sealed evaluation.
Never reads sealed trajectories or launches GPUs. No signal/stop operations.
"""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='2'
import argparse,importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('paired_report_assembler',HERE/'final_report_assembler.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
read,require=base.read,base.require
ROOT=base.ROOT

def validate_pool(data,role,collection):
 p=data['pool_partitions'];original=set(data['paired_pool']['original_ids']);full=set(p['new_full']);own=set(p['own_new_correction'])
 require(len(original)==180 and len(full)==24,'Retained180/shared24 required')
 require(not original&full and not original&own and not full&own,'Pool partitions overlap')
 require(set(p['old_full'])|set(p['old_correction'])==original,'Original pool partition changed')
 require(set(data['splits']['train'])==original|full|own and len(data['splits']['train'])==len(original|full|own),'Pool train population differs')
 require(len(own)==collection['roles'][role]['valid_trajectories'],'Actual admitted own count differs')
 for i in own:require(data['entries'][i].get('source_group')==role,'Cross-role correction rejected')
 require(data['aggregation']['student_prefix_supervised'] is False,'Student prefix supervision forbidden')
 require(set(data['aggregation']['new_training_ids'])==own,'Correction aggregation population differs')
 return dict(original=sorted(original),shared=sorted(full),own=sorted(own),full_families=sorted(data['entries'][i]['family'] for i in full))

def audit(root,mode):
 scopes=('common180_parents',) if mode=='parents' else ('common180_parents','learning')
 missing=[]
 for scope in scopes:
  for role in 'AB':
   for sub in ('training',f'backend/{role}/training'):
    for name in ('deployment.json','finished.json','best.pt','family_phase_curves.csv'):
     p=root/scope/role/sub/name
     if not p.exists():missing.append(str(p))
 for p in [root/'actual_parent_ages.json',root/'parent_pairs.json']+([root/'collection/collection_summary.json',root/'learning/budget_registration.json',root/'learning/A/state_data/data.json',root/'learning/B/state_data/data.json'] if mode=='main' else []):
  if not p.exists():missing.append(str(p))
 if missing:return dict(status='pending',missing=missing,checkpoint_reads=0)
 try:
  import torch
  torch.set_num_threads(2);torch.set_num_interop_threads(2)
  from visuotactile.effective_exposure import Ledger
  pairs=read(root/'parent_pairs.json');ages=read(root/'actual_parent_ages.json');actual={};pools={}
  for scope in scopes:
   actual[scope]={}
   for role in 'AB':
    actual[scope][role]={}
    for comp,sub in [('frontend','training'),('backend',f'backend/{role}/training')]:actual[scope][role][comp]=base.actual_component(root/scope/role/sub,comp,torch,Ledger)
    front=actual[scope][role]['frontend'];rear=actual[scope][role]['backend']
    require(rear['base_sha256']==front['checkpoint_sha256'],'Rear not bound to own DP')
    if scope=='common180_parents':
     for comp in ('frontend','backend'):
      x=actual[scope][role][comp]
      require(pairs[role][comp+'_sha256']==x['checkpoint_sha256'],'Parent active identity differs')
      require(ages['roles'][role][comp+'_epochs']==60 and ages['roles'][role][comp+'_updates']==x['updates'],'Parent age receipt differs')
   for comp in ('frontend','backend'):require(actual[scope]['A'][comp]['updates']==actual[scope]['B'][comp]['updates'],'A/B actual successful updates differ '+scope+comp)
  if mode=='main':
   collection=read(root/'collection/collection_summary.json');budget=read(root/'learning/budget_registration.json')
   require(collection['opportunities']==96 and all(collection['roles'][r]['opportunities']==48 for r in 'AB'),'96/48 opportunity denominator required')
   require(budget['epochs']==budget['deploy_epoch']==60 and budget['fixed_selection'],'Fixed60 registration differs')
   for r in 'AB':
    pools[r]=validate_pool(read(root/'learning'/r/'state_data/data.json'),r,collection)
    require(actual['learning'][r]['frontend']['updates']==budget['frontend']['roles'][r]['planned_optimizer_updates'],'DP actual vs budget differs')
    require(actual['learning'][r]['backend']['updates']==60*budget['backend_updates_per_epoch'][r],'Rear actual vs budget differs')
   require(pools['A']['original']==pools['B']['original'] and pools['A']['shared']==pools['B']['shared'] and pools['A']['full_families']==pools['B']['full_families'],'Shared pools differ')
   require(not set(pools['A']['own'])&set(pools['B']['own']),'Own new pools overlap')
  require(not torch.cuda.is_initialized(),'CPU guard initialized CUDA')
  return dict(status='passed',mode=mode,checkpoint_reads=4*len(scopes),actual=actual,pools=pools,cuda_initialized=False,scope='Actual deployment age/SHA/binding/exposure/pool guard. No closed-loop or performance claim; does not release driver automatically.')
 except Exception as e:return dict(status='failed',mode=mode,error=repr(e),required_action='Do not open evaluation with unverified final students; root must review. This read-only script sends no signals.')

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--mode',choices=['parents','main'],default='main');p.add_argument('--output',type=Path);a=p.parse_args()
 out=(a.output or HERE/('presealed_guard_'+a.mode+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+str(os.getpid())+'.json')).resolve();require(out.is_relative_to(HERE) and not out.exists(),'New AUDIT output required')
 result=audit(a.root.resolve(),a.mode);result.update(observed_utc=time.time(),root=str(a.root.resolve()),gpu_or_kit_started=False,readonly=True,script_sha256=base.digest(Path(__file__)))
 out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(status=result['status'],output=str(out),missing=len(result.get('missing',[]))),ensure_ascii=False))
 raise SystemExit(0 if result['status']=='passed' else 2)
if __name__=='__main__':main()
