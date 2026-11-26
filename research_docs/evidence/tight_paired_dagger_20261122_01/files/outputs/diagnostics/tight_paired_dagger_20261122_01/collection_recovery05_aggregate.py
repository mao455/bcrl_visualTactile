"""CPU-only exact96 opportunity assembly across an interruption, never retries."""
import argparse,json,sys
sys.path.insert(0,str(__import__("pathlib").Path(__file__).resolve().parent))
from pathlib import Path
from collection_recovery05 import read,write,census,digest,PROJECT
sys.path.insert(0,str(PROJECT))
def aggregate(original,recovered,continued,output):
 original,recovered,continued,output=map(lambda p:Path(p).resolve(),(original,recovered,continued,output))
 plan=read(original/'plan.json');state=census(original);old=read(recovered/'recovered_results.json');new=read(continued/'results.json')
 if read(continued/'recovery05_binding.json')['assignment_sha256']!=state['assignment_sha256']:raise ValueError('Continuation binding differs')
 if {r['case'] for r in old}!={r['case'] for r in state['rows']} or len(old)!=state['assigned']:raise ValueError('Consumed opportunity census differs')
 if {r['case'] for r in new}!=set(state['remaining_cases']) or len(new)!=state['remaining']:raise ValueError('Continuation is not exactly the unassigned suffix')
 rows=old+new;byname={r['case']:r for r in rows}
 if len(byname)!=96 or len(rows)!=96:raise ValueError('Exactly96 unique original opportunities required')
 ordered=[];cases={c['name']:c for c in plan['cases']};entries={}
 for root in (recovered,continued):
  for name,entry in read(root/'correction_entries.json').items():
   if name in entries:raise ValueError('Duplicate correction admission')
   entries[name]=entry
 for case in plan['cases']:
  row=byname[case['name']];r=row['result']
  if row['role']!=case['role'] or r.get('identity')!=case['name']:raise ValueError('Role/identity differs')
  if bool(r.get('accepted_for_training'))!=(case['name'] in entries):raise ValueError('Admission result differs from manifest')
  if case['name'] in entries and (entries[case['name']]['source_group']!=case['role'] or not all(r.get(k) for k in ('legal','software_complete','success'))):raise ValueError('Unsafe correction admission')
  row=dict(row,evidence_directory=str((recovered if case['name'] in {r['case'] for r in old} else continued)/case['name']));ordered.append(row)
 assignments=[]
 for attempt,root in [('original',original),('continued',continued)]:
  for a in read(root/'paired_queue_assignments.json'):
   a=dict(a,attempt=attempt,attempt_ordinal=a['ordinal'],ordinal=len(assignments));assignments.append(a)
 if [a['case'] for a in assignments]!=[c['name'] for c in plan['cases']]:raise ValueError('Assignments changed original frozen order')
 checkrows=[]
 for attempt,root in [('original',original),('continued',continued)]:
  x=read(root/'paired_initial_check.json')
  if not x['passed'] or not all(r['passed'] for r in x['rows']):raise ValueError('A recorded paired initial check failed')
  checkrows.extend(dict(r,attempt=attempt) for r in x['rows'])
 expected={(c['spec']['family'],c['opportunity']) for c in plan['cases']};checked={(r['family'],r['opportunity']) for r in checkrows}
 if len(checked)!=len(checkrows) or not checked<=expected:raise ValueError('Duplicate or unregistered paired check')
 missing=[dict(family=f,opportunity=o,passed=None,reason='Cross-attempt counterpart lacks complete saved pre-action fields; not asserted equal') for f,o in sorted(expected-checked)]
 summary=dict(version='paired_no_retry_recovery05_v1',opportunities=96,roles={},expert_recovery_is_autonomous_success=False,attempts=[str(original),str(continued)],observable_pair_checks=dict(passed=len(checkrows),unverified=len(missing),failed=0,all_verified=not missing))
 for role in 'AB':
  selected=[r for r in ordered if r['role']==role]
  if len(selected)!=48:raise ValueError('Each own student must retain48 original chances')
  totals=dict(opportunities=48,invalid_initialization=0,unknown_initialization=0,software_incomplete=0,no_trigger=0,takeovers=0,expert_recovered=0,student_success_without_takeover=0,valid_trajectories=0,legal_expert_ticks=0,admitted_legal_expert_ticks=0,stages={})
  for row in selected:
   r=row['result'];d=r.get('dagger',{});receipt=d.get('trigger_receipt',{});complete=bool(r.get('software_complete'));switched=d.get('recovery_coverage',{}).get('switch_tick') is not None
   totals['invalid_initialization']+=r.get('legal') is False;totals['unknown_initialization']+=r.get('legal') is None;totals['software_incomplete']+=not complete
   totals['no_trigger']+=bool(complete and r.get('legal') and not switched);totals['takeovers']+=switched;totals['expert_recovered']+=bool(switched and d.get('expert_takeover_success'));totals['student_success_without_takeover']+=bool(complete and not switched and d.get('student_autonomous_success'));totals['valid_trajectories']+=row['case'] in entries;totals['legal_expert_ticks']+=int(d.get('expert_valid_ticks',0));totals['admitted_legal_expert_ticks']+=int(d.get('expert_valid_ticks',0)) if row['case'] in entries else 0
   stage=receipt.get('reason','software_incomplete' if not complete else 'no_trigger');s=totals['stages'].setdefault(stage,dict(opportunities=0,takeovers=0,recovered=0,legal_ticks=0,never_entered=0,entered_then_left=0));s['opportunities']+=1;s['takeovers']+=switched;s['recovered']+=bool(switched and d.get('expert_takeover_success'));s['legal_ticks']+=int(d.get('expert_valid_ticks',0));s['never_entered']+=bool(switched and receipt.get('never_entered_15mm'));s['entered_then_left']+=bool(switched and receipt.get('ever_entered_then_left_15mm'))
  summary['roles'][role]=totals
 output.mkdir(exist_ok=False)
 for p in ('plan.json','session.json'):(output/p).write_bytes((original/p).read_bytes())
 for case in plan['cases']:
  source=recovered if case['name'] in {r['case'] for r in old} else continued
  (output/case['name']).symlink_to(source/case['name'],target_is_directory=True)
 write(output/'results.json',ordered);write(output/'correction_entries.json',entries);write(output/'collection_summary.json',summary);write(output/'paired_queue_assignments.json',assignments)
 write(output/'paired_initial_check.json',dict(rows=checkrows,unverified_rows=missing,passed=not missing,all_recorded_checks_passed=True,full48_verified=not missing))
 write(output/'result.json',dict(software_complete=False,opportunity_ledger_complete=True,opportunities_accounted=96,software_incomplete=sum(x['software_incomplete'] for x in summary['roles'].values()),note='Full finite opportunity ledger closed; interrupted physical opportunities were retained as failures and never retried.'))
 write(output/'recovery05_aggregate_binding.json',dict(original_plan_sha256=digest(original/'plan.json'),recovered_results_sha256=digest(recovered/'recovered_results.json'),continued_results_sha256=digest(continued/'results.json'),source_directories=[str(original),str(recovered),str(continued)],no_initiated_opportunity_reexecuted=True,artifact_sha256={name:digest(output/name) for name in ('results.json','correction_entries.json','collection_summary.json','paired_queue_assignments.json','paired_initial_check.json')}))
 return summary
def publish_index(original,aggregate_root):
 original,aggregate_root=Path(original).resolve(),Path(aggregate_root).resolve()
 for name in ('results.json','correction_entries.json','collection_summary.json','recovery05_aggregate_binding.json'):
  if (original/name).exists():raise FileExistsError('Never overwrite original evidence: '+name)
 for name in ('results.json','correction_entries.json','collection_summary.json','recovery05_aggregate_binding.json'):
  with (original/name).open('xb') as f:f.write((aggregate_root/name).read_bytes())
 write(original/'recovery05_active_aggregate.json',dict(path=str(aggregate_root),binding_sha256=digest(aggregate_root/'recovery05_aggregate_binding.json'),indexed_files=['results.json','correction_entries.json','collection_summary.json']))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for key in ('original','recovered','continued','output'):p.add_argument('--'+key,required=True)
 p.add_argument('--publish-index',action='store_true');a=p.parse_args();result=aggregate(a.original,a.recovered,a.continued,a.output)
 if a.publish_index:publish_index(a.original,a.output)
 print(json.dumps(result,indent=2))
