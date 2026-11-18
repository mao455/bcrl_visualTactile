"""Recover closed pilot captures after classification-only error; no Kit rerun."""
import time,shutil
from pathlib import Path
from scripts.tight_recovery_20261118.run import ROOT,OUT,read,write
from tools.tight_fresh_batch_run import admit_capture
from visuotactile.state_benchmark import DaggerCaptureAuditMemo

def main():
 src=ROOT/'pilot';dst=ROOT/'pilot_admission_recovery02';dst.mkdir(exist_ok=False);plan=read(src/'plan.json');assert read(src/'owner_exit.json')['children_reaped']
 write(dst/'plan.json',plan)
 start=time.time();entries={};results=[];memo=DaggerCaptureAuditMemo(max_payloads=32,max_pairs=4)
 for c in plan['cases']:
  a=src/c['name'];b=dst/c['name'];b.mkdir();assert read(a/'result.json')['software_complete']
  for p in a.iterdir():
   if p.name in ('result_before_admission.json','capture_audit.json','contact_height_stage_audit.json'):continue
   if p.name=='result.json':shutil.copyfile(p,b/p.name)
   else:(b/p.name).symlink_to(p,target_is_directory=p.is_dir())
  r=read(b/'result.json');admit_capture(dst,c,r,plan['pairs'][c['role']],plan,entries,audit_memo=memo)
  results.append(dict(role=c['role'],case=c['name'],result=r,timing=dict(interpretation='Existing actual pilot lane; CPU admission replay only')))
 for n in ['plan.json','session.json','initialization.json','paired_initial_check.json','videos.json']:
  if (src/n).exists():shutil.copyfile(src/n,dst/n)
 write(dst/'results.json',results);elapsed=read(src/'owner_exit.json')['seconds'];write(dst/'result.json',dict(software_complete=True,worker_error=None,cases=len(results),lanes=8,elapsed_seconds=elapsed,accepted_training_trajectories=len(entries),classification_recovered=True,original_failed_owner=str(src)))
 write(dst/'owner_exit.json',dict(children_reaped=True,utc=time.time(),no_simulation=True,seconds=time.time()-start));write(dst/'owner_complete.json',dict(seconds=elapsed,utc=time.time(),physical_source=str(src)));write(dst/'recovery.json',dict(utc=time.time(),reason='New trigger classification omitted non-delay branch; all8 execution_complete/results already saved. Only CPU admission replay, no optimizer/Kit/action rerun.',source=str(src),seconds=time.time()-start,source_failure=read(src/'owner_failure.json')))
 write(OUT/'pilot_recovery.json',read(dst/'recovery.json'));print(len(entries))
if __name__=='__main__':main()
