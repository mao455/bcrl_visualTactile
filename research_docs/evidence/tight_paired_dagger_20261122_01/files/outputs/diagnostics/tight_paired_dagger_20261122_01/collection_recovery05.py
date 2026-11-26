"""AUDIT-only no-retry continuation for the frozen 96-opportunity collection."""
import argparse, copy, hashlib, json, os, sys, time
from pathlib import Path
from contextlib import contextmanager
PROJECT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PROJECT))
def read(p):return json.loads(Path(p).read_text())
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def census(original):
 original=Path(original).resolve();plan=read(original/'plan.json');assign=read(original/'paired_queue_assignments.json')
 names=[c['name'] for c in plan['cases']];assigned=[a['case'] for a in assign]
 if names[:len(assigned)]!=assigned or len(set(assigned))!=len(assigned):raise ValueError('Assignments must be a unique frozen prefix')
 # Existing case directories beyond assignment receipt imply ambiguous initiation. Never replay.
 unknown=[n for n in names[len(assigned):] if (original/n).exists()]
 if unknown:raise ValueError('Unrecorded initiated directories: '+str(unknown))
 rows=[]
 for name in assigned:
  d=original/name
  status='closed' if (d/'result.json').is_file() else 'execution_pending_verification' if (d/'execution_complete.json').is_file() else 'software_incomplete'
  rows.append(dict(case=name,status=status,started=(d/'started.json').exists()))
 return dict(version='paired_no_retry_recovery05_v1',original=str(original),original_plan_sha256=digest(original/'plan.json'),assignment_sha256=digest(original/'paired_queue_assignments.json'),assigned=len(assigned),remaining=len(names)-len(assigned),rows=rows,remaining_cases=names[len(assigned):],utc=time.time())
def prepare(original,output):
 original=Path(original).resolve();output=Path(output).resolve();c=census(original)
 output.mkdir(exist_ok=False)
 for file in ('plan.json','session.json'):(output/file).write_bytes((original/file).read_bytes())
 registration=read(original/'registration.json');registration.update(recovery_adapter=str(Path(__file__).resolve()),recovery_adapter_sha256=digest(__file__),original_output=str(original),no_rerun_initiated=True)
 write(output/'registration.json',registration);write(output/'recovery05_binding.json',c)
 return c

def worker(output):
 """Run only previously unassigned suffix. Root must invoke under registered owner/watchers."""
 output=Path(output).resolve();binding=read(output/'recovery05_binding.json');original=Path(binding['original'])
 actual=census(original)
 for key in ('original_plan_sha256','assignment_sha256','assigned','remaining_cases'):
  if actual[key]!=binding[key]:raise ValueError('Original interruption changed: '+key)
 reg=read(output/'registration.json')
 if reg['recovery_adapter_sha256']!=digest(__file__):raise ValueError('Recovery adapter changed')
 from tools import tight_paired_collection as tc,tight_fresh_batch_run as runner
 full=read(output/'plan.json');tc.validate_plan(full)
 selected=copy.deepcopy(full);selected['cases']=full['cases'][binding['assigned']:]
 if not selected['cases']:raise ValueError('No unassigned physical opportunity remains')
 oldread,oldvalidate,oldpairing=runner.read,tc.validate_plan,tc.pairing_scope
 def runtime_read(path):
  return copy.deepcopy(selected) if Path(path).resolve()==output/'plan.json' else oldread(path)
 def validate(plan):
  if plan==selected:return oldvalidate(full)
  return oldvalidate(plan)
 @contextmanager
 def pairing(plan,out):
  if plan!=full:raise ValueError('Unexpected pairing plan')
  with oldpairing(selected,out):yield
 runner.read,tc.validate_plan,tc.pairing_scope=runtime_read,validate,pairing
 write(output/'recovery05_runtime_scope.json',dict(original_opportunities=96,continued=len(selected['cases']),excluded_initiated=binding['assigned'],full_plan_checked=True,scientific_configuration_changed=False,original_plan_sha256=binding['original_plan_sha256'],limitations=['First remaining case may have an interrupted counterpart; do not claim all 48 observable pairs passed.','New indexed-reset batch changes scheduling and does not reproduce hidden state or future streams.']))
 try:tc.worker(output)
 finally:runner.read,tc.validate_plan,tc.pairing_scope=oldread,oldvalidate,oldpairing

MODULE='outputs.diagnostics.tight_paired_dagger_20261122_01.collection_recovery05'
def prepare_project(original,output):
 result=prepare(original,output);output=Path(output).resolve();plan=read(output/'plan.json');root=Path(plan['resource_root']);owner=root/'owners/collection_recovery05'
 owner.mkdir(exist_ok=False);session=read(root/'session.json');session['resource_root']=str(owner)
 write(owner/'session.json',session);write(owner/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
 job=root/'collection_recovery05_job.json';write(job,dict(version='tight_project_job_v1',operation='dagger',resource_root=str(owner),stages=[dict(entry='paired_collection_recovery05',args=['--output',str(output)])]))
 registration=read(output/'registration.json');registration.update(output=str(output),owner=str(owner),job=str(job),command=f'python -m {MODULE} project dagger --job {job}',recovery_adapter_sha256=digest(__file__),source_hash_binding_unchanged=True)
 (output/'registration.json').write_text(json.dumps(registration,indent=2))
 return dict(**result,owner=str(owner),job=str(job),module=MODULE,output=str(output),worker_command=[sys.executable,'-m',MODULE,'worker','--output',str(output)],project_command=[sys.executable,'-m',MODULE,'project','dagger','--job',str(job)])
def main():
 if len(sys.argv)>1 and sys.argv[1]=='project':
  from tools import tight_project
  tight_project.ENTRIES['paired_collection_recovery05']=('dagger',MODULE,['worker'])
  sys.argv.pop(1);return tight_project.main()
 p=argparse.ArgumentParser();p.add_argument('command',choices=['census','prepare','prepare-project','worker']);p.add_argument('--original');p.add_argument('--output');a=p.parse_args()
 if a.command=='census':print(json.dumps(census(a.original),indent=2))
 elif a.command=='prepare':print(json.dumps(prepare(a.original,a.output),indent=2))
 elif a.command=='prepare-project':print(json.dumps(prepare_project(a.original,a.output),indent=2))
 else:worker(a.output)
if __name__=='__main__':main()
