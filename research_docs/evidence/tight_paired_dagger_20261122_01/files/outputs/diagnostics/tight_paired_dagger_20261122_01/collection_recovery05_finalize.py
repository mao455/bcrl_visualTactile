"""CPU-only immutable overlays for interrupted collection; never restarts a case."""
import argparse,copy,importlib.util,json,shutil,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collection_recovery05 import census,read,write,digest,PROJECT
sys.path.insert(0,str(PROJECT))
def overlay(source,destination):
 source,destination=Path(source),Path(destination);destination.mkdir(parents=True,exist_ok=False)
 for p in source.iterdir():
  if p.suffix=='.json':shutil.copy2(p,destination/p.name)
  else:(destination/p.name).symlink_to(p.resolve(),target_is_directory=p.is_dir())
def recover_closed_execution(directory):
 """Reverify already-closed bytes. Does not recover missing/corrupt/open HDF5."""
 import h5py,numpy as np
 from visuotactile.async_recording import _verify_closed_capture
 from visuotactile.state_benchmark import dagger_recovery_coverage
 directory=Path(directory);result=read(directory/'execution_complete.json');path=directory/'demonstration/data.h5'
 if (directory/'result.json').exists():raise ValueError('Already has result; no replacement')
 if not result.get('software_complete') or not result.get('legal'):raise ValueError('Not a closed legal execution')
 with h5py.File(directory/'numeric.h5','r') as f:
  if f.attrs.get('async_recording_status')!='complete':raise ValueError('Numeric writer never committed complete')
  with np.load(directory/'trace.npz') as z:arrays={k:z[k] for k in z.files}
  if any(len(v)!=result['steps'] for v in arrays.values()):raise ValueError('Trace length mismatch')
  result['dagger']['recovery_coverage']=dagger_recovery_coverage(arrays,f['episode'])
 with h5py.File(path,'r') as f:
  embedded=json.loads(f.attrs['result_json'])
  if embedded!=read(directory/'execution_complete.json'):raise ValueError('Compact terminal metadata mismatch')
  if result['identity'] not in f['episodes']:raise ValueError('Compact identity mismatch')
 binding=dict(path=str(path.resolve()),checkpoint_sha256=result['checkpoint_sha256'],trajectory_kind='expert_takeover_v1',accepted_success=bool(result['dagger']['expert_takeover_success'] and result['dagger']['expert_valid_ticks']>0))
 receipt=dict(numeric_path=str(directory/'numeric.h5'),dataset_capture=binding,steps=result['steps'],profile=dict(worker_verify_s=0.,worker_hash_s=0.,scope='New independent CPU verification only; original write/drain timings unavailable'),recording_contract='recovered_closed_async_capture_v1',recovered_after_process_loss=True)
 # The production verifier checks every saved source-field digest and whole-file SHA.
 verified=_verify_closed_capture(receipt)
 result['dataset_capture']=verified['dataset_capture'];result['async_capture']=verified
 result['recovery_verification']=dict(original_execution_sha256=digest(directory/'execution_complete.json'),new_verification=True,physics_steps_added=0,original_timing_reconstructed=False,utc=time.time())
 write(directory/'result.json',result)
 return result

def interrupted_result(case,source):
 source=Path(source);collision=read(source/'initial_collision.json') if (source/'initial_collision.json').exists() else {}
 return dict(**case['spec'],software_complete=False,legal=collision.get('legal'),success=False,accepted_for_training=False,worker_error='Interrupted process; original opportunity consumed; never retried',recovery_status='assigned_software_incomplete',assignment_consumed=True,source_attempt=str(source),unknown_legality=not bool(collision))

def finalize_original(original,output):
 """Creates overlays + admits verifiable closed suffixes; no originals mutated."""
 from tools import tight_paired_collection as tc,tight_fresh_batch_run as runner
 original,output=Path(original).resolve(),Path(output).resolve();state=census(original);plan=read(original/'plan.json');tc.validate_plan(plan)
 output.mkdir(exist_ok=False);write(output/'interruption_census.json',state)
 (output/'plan.json').write_bytes((original/'plan.json').read_bytes());entries={};rows=[]
 byname={c['name']:c for c in plan['cases']}
 oldclassify=runner.classify_capture;runner.classify_capture=tc.classify_capture
 try:
  for row in state['rows']:
   case=byname[row['case']];src=original/case['name'];dst=output/case['name'];overlay(src,dst)
   if row['status']=='closed':result=read(dst/'result.json')
   elif row['status']=='execution_pending_verification':
    try:result=recover_closed_execution(dst)
    except Exception as exc:
     write(dst/'recovery_verification_failure.json',dict(error=repr(exc),raw_files_preserved=True));result=interrupted_result(case,src)
   else:result=interrupted_result(case,src)
   if result.get('software_complete'):
    with tc.trigger_scope():runner.admit_capture(output,case,result,plan['pairs'][case['role']],plan,entries)
    if case['name'] in entries:
     receipt=result['dagger']['trigger_receipt'];entries[case['name']].update(opportunity=case['opportunity'],paired_collection_contract=tc.CONTRACT,ever_entered_15mm_at_takeover=receipt['ever_entered_15mm'],ever_entered_then_left_15mm_at_takeover=receipt['ever_entered_then_left_15mm'],never_entered_15mm_at_takeover=receipt['never_entered_15mm'])
   else:
    # This is an outcome ledger, not a fabricated execution result or training label.
    write(dst/'interrupted_outcome.json',result)
   rows.append(dict(case=case['name'],role=case['role'],result=result,recovery_original_status=row['status']))
  write(output/'recovered_results.json',rows)
  # admit_capture atomically maintains correction_entries; write only if no admission ran.
  if not (output/'correction_entries.json').exists():write(output/'correction_entries.json',entries)
  else:(output/'correction_entries.json').write_text(json.dumps(entries,indent=2))
 finally:runner.classify_capture=oldclassify
 return dict(consumed=len(rows),software_complete=sum(bool(r['result'].get('software_complete')) for r in rows),admitted=len(entries),remaining=state['remaining'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(finalize_original(a.original,a.output),indent=2))
if __name__=='__main__':main()
