"""Bound summary dispatch for the complete truthful no-retry opportunity ledger."""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from collection_recovery05 import read,digest,PROJECT
sys.path.insert(0,str(PROJECT))
def validated_summary(collection):
 collection=Path(collection).resolve();pointer=read(collection/'recovery05_active_aggregate.json') if (collection/'recovery05_active_aggregate.json').exists() else None
 aggregate=Path(pointer['path']).resolve() if pointer else collection;binding=read(aggregate/'recovery05_aggregate_binding.json')
 if pointer and pointer['binding_sha256']!=digest(aggregate/'recovery05_aggregate_binding.json'):raise ValueError('Aggregate pointer changed')
 for name,value in binding['artifact_sha256'].items():
  if digest(aggregate/name)!=value:raise ValueError('Aggregate artifact changed: '+name)
 if pointer:
  for name in pointer['indexed_files']:
   if (collection/name).read_bytes()!=(aggregate/name).read_bytes():raise ValueError('Published index differs: '+name)
 from tools import tight_paired_collection as tc
 plan=read(collection/'plan.json');tc.validate_plan(plan)
 if digest(collection/'plan.json')!=binding['original_plan_sha256']:raise ValueError('Original frozen plan changed')
 rows=read(aggregate/'results.json');entries=read(aggregate/'correction_entries.json');summary=read(aggregate/'collection_summary.json');checks=read(aggregate/'paired_initial_check.json');assign=read(aggregate/'paired_queue_assignments.json')
 if len(rows)!=96 or [r['case'] for r in rows]!=[c['name'] for c in plan['cases']] or [a['case'] for a in assign]!=[c['name'] for c in plan['cases']]:raise ValueError('Original96 denominator/order differs')
 accepted={r['case'] for r in rows if r['result'].get('accepted_for_training')}
 if accepted!=set(entries):raise ValueError('Admission manifest differs')
 if len(checks['rows'])!=47 or len(checks['unverified_rows'])!=1 or checks['passed'] is not False or not all(r['passed'] for r in checks['rows']):raise ValueError('Truthful47verified/1unverified paired checks required')
 for role in 'AB':
  own=[r for r in rows if r['role']==role];s=summary['roles'][role]
  if len(own)!=48 or s['opportunities']!=48:raise ValueError('Own48 denominator changed')
  if s['software_incomplete']!=sum(not r['result'].get('software_complete',False) for r in own):raise ValueError('Incomplete count differs')
  if s['valid_trajectories']!=sum(r['case'] in entries for r in own):raise ValueError('Admitted count differs')
  for row in own:
   r=row['result']
   if row['case'] in entries and (entries[row['case']]['source_group']!=role or not all(r.get(k) for k in ('software_complete','legal','success')) or not r.get('dagger',{}).get('expert_takeover_success')):raise ValueError('Unsafe admitted suffix')
  expected=sum(bool(r['result'].get('software_complete') and r['result'].get('legal') and r['result'].get('dagger',{}).get('recovery_coverage',{}).get('switch_tick') is None) for r in own)
  if s['no_trigger']!=expected:raise ValueError('Incomplete/illegal outcomes counted as no-trigger')
 if summary['opportunities']!=96:raise ValueError('Full denominator missing')
 return summary
@contextmanager
def summary_scope(collection):
 from tools import tight_paired_collection as tc
 collection=Path(collection).resolve();validated_summary(collection);old=tc.summarize
 def bound(output):return validated_summary(collection) if Path(output).resolve()==collection else old(output)
 tc.summarize=bound
 try:yield
 finally:tc.summarize=old
def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=['validate','prepare-learning']);p.add_argument('--collection',type=Path,required=True);a=p.parse_args()
 if a.command=='validate':print(json.dumps(validated_summary(a.collection),indent=2));return
 from tools import tight_paired_prepare
 with summary_scope(a.collection):tight_paired_prepare.prepare_learning()
if __name__=='__main__':main()
