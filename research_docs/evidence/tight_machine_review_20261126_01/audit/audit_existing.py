import json,csv,collections,datetime,traceback
from pathlib import Path
import h5py,numpy as np
from visuotactile.adaptive_takeover import AdaptiveTrigger,profile
from visuotactile.compact import sha256
from scripts.tight_layout_20261116.pilot import reset_configuration,displacement
R=Path.cwd();O=R/'outputs/diagnostics/tight_machine_review_20261126_01';P=R/'outputs/visuotactile/run/tight_paired_dagger_20261122_01';E=P/'evaluation/test';TZ=datetime.timezone(datetime.timedelta(hours=8))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False))
def timestamp(t):return datetime.datetime.fromtimestamp(t,TZ).isoformat() if t else None
def csvout(p,rows):
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def status():
 exits={};logs={}
 for owner in [E,P/'owners/test_fragment13',P/'owners/test_continuation15']:
  if (owner/'project_job_started.json').exists():
   job=read(owner/'project_job_started.json')['job']
   for i,s in enumerate(job['stages']):
    args=s['args']
    if '--output' in args:logs[args[args.index('--output')+1]]=owner/f'project_stage_{i:02d}.log'
  if (owner/'stage_exits').exists():
   for p in (owner/'stage_exits').glob('wave*.json'):exits[p.stem]=read(p)
 bad={12:'native_startup_stall',16:'native_SIGABRT',21:'native_startup_stall',22:'native_startup_stall',31:'native_startup_stall',41:'native_startup_stall',58:'runtime_RAM_reserve_then_cleanup_stall',59:'native_startup_stall',60:'native_startup_stall'}
 waves=[];rows=[]
 for i in range(64):
  w=E/f'wave{i:02d}';plan=read(w/'plan.json');start=read(w/'run_started.json');ex=exits.get(w.name,{})
  end=ex.get('utc')
  if not end and (w/'result.json').exists():end=(w/'result.json').stat().st_mtime
  log=logs.get(str(w));text=log.read_text(errors='replace') if log and log.exists() else ''
  # Keep all first error-looking lines, not a claim that every logged warning caused termination.
  errors=[dict(line=j+1,text=l) for j,l in enumerate(text.splitlines()) if any(k in l for k in ('[Error]','Fatal Python','RuntimeError:','Traceback (most recent','Aborted','timeout reached'))]
  tails=text.splitlines()[-65:]
  trace=(w/'startup_tracebacks.log').read_text(errors='replace') if (w/'startup_tracebacks.log').exists() else ''
  frames=[l.strip() for l in trace.splitlines() if any(k in l for k in ('simulation_context.py','simulation_manager.py','native_evaluation.py','_environment','play','app.update'))]
  row=dict(wave=i,seed=plan['cases'][0]['spec']['seed'],pid=start['pid'],start_utc=start['utc'],start_local=timestamp(start['utc']),end_utc=end,end_local=timestamp(end),elapsed_s=end-start['utc'] if end else None,
   cause=bad.get(i,'native_physical_outcome'),phase='before_controls' if i in bad and i!=58 else 'controls_started',
   last_progress=read(w/'batch_progress.json') if (w/'batch_progress.json').exists() else None,
   exit=ex,log_path=str(log) if log else None,first_error_lines=errors[:8],last_log_lines=tails if i in bad else [],startup_current_frames=frames[:18],
   qualification='SIGABRT underlying C++ cause unestablished; startup stacks locate wait, not root cause. Shared worker failure is one technical event for four opportunities.')
  waves.append(row)
  for case in plan['cases']:
   d=w/case['name'];p=d/'result.json';res=read(p) if p.exists() else {};complete=bool(res.get('software_complete'))
   reason='technical_unfinished:'+bad[i] if not complete else 'strict_success' if res.get('success') else '|'.join(k for k,v in res['reasons'].items() if v)
   rows.append(dict(role=case['role'],family=case['spec']['family'],seed=case['spec']['seed'],wave=i,lane=plan['cases'].index(case),worker_pid=start['pid'],
    start_local=row['start_local'],end_local=row['end_local'],software_complete=complete,legal=res.get('legal'),success=res.get('success',False),
    physical_steps=res.get('steps'),reason=reason,worker_cause=row['cause'],source_result=str(p),first_error_source=str(log) if log else None))
 csvout(O/'evaluation256_status.csv',rows);write(O/'wave64_status.json',waves)
 write(O/'completion_summary.json',{r:dict(denominator=64,complete=sum(x['software_complete'] for x in rows if x['role']==r),
    reasons=dict(collections.Counter(x['reason'] for x in rows if x['role']==r))) for r in dict.fromkeys(x['role'] for x in rows)})
 session=read(P/'session.json');cohorts=read(P/'cohorts.json');layout=[]
 for cohort,items in cohorts.items():
  for item in items:
   expected=reset_configuration(session['domain'],item['seed'],.1);actual=item['narrow_reset']
   mismatch=float(np.linalg.norm(np.array(actual['fixed_start_pose'][:2])-expected['fixed_start_pose'][:2]))
   layout.append(dict(cohort=cohort,seed=item['seed'],actual_pose=actual['fixed_start_pose'],historical_contract_pose=expected['fixed_start_pose'],peg_xy_discrepancy_m=mismatch,
    hole_unchanged=actual['hole_x_range']==expected['hole_x_range'] and actual['hole_y_range']==expected['hole_y_range']))
 write(O/'layout_contract_comparison.json',dict(rows=layout,displacement_xy_m=displacement(session['domain'],.1).tolist(),
   actual_new_families=len(layout),mismatched_families=sum(x['peg_xy_discrepancy_m']>1e-6 for x in layout),
   earliest_source='tools.tight_paired_dagger.prepare calls direct_configuration while historical scripts.tight_entry_final24.prepare calls reset_configuration(...,.1)',
   runtime_consumer='narrow_data.configure_direct_reset[_lanes] reads narrow_reset only. layout metadata is not a state writer.',
   inferential_limit='Existing0/64 remains true for executed unshifted cohort. It is not a test of the promised historical100mm displaced distribution. Rebuilt parents also differ in weights/training age from historical130-50.'))
 print('status/layout written',flush=True)

def replay(path,group='episode'):
 with h5py.File(path) as f:
  tr=f[group+'/transition'];a={name:tr[name][:] for name in ['diagnostics_after/object_pose_w','diagnostics_after/palm_body_pose_w','object_contact_force_w','native_keypoints_max_dist','diagnostics_before/successes','diagnostics_after/successes','diagnostics_before/goal_pose_w','diagnostics_after/goal_pose_w']}
  sources=tr['dagger/control_source'][:] if 'dagger/control_source' in tr else None
 # Stage census for corrections describes actual student prefix, never expert rescue suffix.
 if sources is not None and (sources!=0).any():
  stop=int(np.flatnonzero(sources!=0)[0]);a={k:v[:stop] for k,v in a.items()};sources=sources[:stop]
 obj=a['diagnostics_after/object_pose_w'];key=a['native_keypoints_max_dist'];before=a['diagnostics_before/successes'];after=a['diagnostics_after/successes'];n=len(key)
 contacts=(np.linalg.norm(a['object_contact_force_w'],axis=-1)>.1).sum(-1).min(-1);held=np.zeros(n,bool)
 if n>=48:
  good=(contacts>=2)&(obj[:,2]>=np.median(obj[24:48,2])+.02);run=0
  for t,g in enumerate(good):run=run+1 if g else 0;held[t]=run>=12 and t>=47
 root=np.linalg.norm(obj[:,:3]-a['diagnostics_before/goal_pose_w'][:,:3],axis=-1)
 gate=(key<=.015)&(before==0);ever=np.maximum.accumulate(gate);near=held&(root<=.09)&(before==0)
 category=('post_subgoal' if after.max()>0 else 'entered_unconfirmed' if gate.any() else 'held_near_never_gate' if near.any() else 'held_not_near' if held.any() else 'grasp_not_held')
 # A standalone opt-in history filter is replayed causally; no future final outcome enters it.
 trigger=AdaptiveTrigger(profile('alignment'),10**9);hit=None;oldhit=None;eligible=[]
 for t in range(n):
  trigger.observe(t,obj[t],a['diagnostics_after/palm_body_pose_w'][t],a['object_contact_force_w'][t],
   keypoint_max_distance_m=float(key[t]),goal_index_before=int(before[t]),goal_index_after=int(after[t]),
   goal_pose_before=a['diagnostics_before/goal_pose_w'][t],goal_pose_after=a['diagnostics_after/goal_pose_w'][t])
  ready=trigger.confirm>=2
  if ready:
   if oldhit is None:oldhit=t+1
   if not ever[t]:
    eligible.append(t+1)
    if hit is None:hit=t+1
  trigger.scheduled_tick=10**9;trigger.trigger_completed_tick=None;trigger.at_trigger=None
 return dict(ticks=n,category=category,held12=bool(held.any()),near_held=bool(near.any()),ever15mm=bool(gate.any()),native_first_subgoal=bool(after.max()>0),
  old_alignment_first=oldhit,never_alignment_first=hit,eligible_completed_ticks=eligible,eligibility_count=len(eligible),student_prefix_ticks=int((sources==0).sum()) if sources is not None else n)

def census():
 old=read(R/'outputs/diagnostics/tight_premature_scope_20261122_01/train_dev/trajectories.json');rows=[];missing=[]
 for r in old:
  p=Path(r['path'])
  if not p.exists():
   restored=O/'reconstructed_numeric/development'/(r['identity']+'.h5')
   if restored.exists():p=restored
   else:missing.append(str(p));continue
  try:val=replay(p,r['group'])
  except KeyError:
   saved={x['identity']:x for x in read(R/'outputs/diagnostics/tight_premature_scope_20261122_01/train_dev/never_gate_stratification/trajectories.json')}[r['identity']]
   hit=saved['alignment_screen_scheduled_next_action_tick'];gate=r['gate_first_entry_tick']
   neverhit=hit if hit is not None and (gate is None or hit-1<gate) else None
   val=dict(ticks=r['ticks'],category='post_subgoal' if r['native_first_subgoal_tick'] is not None else 'entered_unconfirmed' if not saved['never15mm'] else 'held_near_never_gate' if saved['held12_near90_ever'] else 'held_not_near' if saved['stable_held12_ever'] else 'grasp_not_held',held12=saved['stable_held12_ever'],near_held=saved['held12_near90_ever'],ever15mm=not saved['never15mm'],native_first_subgoal=r['native_first_subgoal_tick'] is not None,old_alignment_first=hit,never_alignment_first=neverhit,eligible_completed_ticks=[neverhit] if neverhit else [],eligibility_count=None,student_prefix_ticks=None,screen_evidence='Existing full-source verified receipt; compressed retired numeric lacks active goal poses. No reconstructed or invented goal labels.')
  except Exception as e:missing.append([str(p),repr(e)]);continue
  rows.append(dict(cohort='historical_'+r['cohort'],identity=r['identity'],family=r['family'],policy=r['policy'],success=r['native_success'],path=str(p),**val))
 for wave in sorted((P/'evaluation/development').glob('wave*')):
  if not (wave/'plan.json').exists():continue
  plan=read(wave/'plan.json')
  for c in plan['cases']:
   d=wave/c['name']
   if not (d/'result.json').exists():
    missing.append(str(d));continue
   r=read(d/'result.json')
   if not r.get('legal'):continue
   rows.append(dict(cohort='paired_development_unshifted',identity=c['name'],family=c['spec']['family'],policy=c['role'],success=r.get('success'),path=str(d/'numeric.h5'),**replay(d/'numeric.h5')))
 write(O/'train_development_stage_census.json',dict(rows=rows,missing=missing,terminal64_read_for_selection=False))
 keys=list(dict.fromkeys(k for x in rows for k in x if k!='eligible_completed_ticks'));flat=[{k:x.get(k) for k in keys} for x in rows];csvout(O/'train_development_stage_census.csv',flat)
 summaries={}
 for cohort in dict.fromkeys(r['cohort'] for r in rows):
  selected=[r for r in rows if r['cohort']==cohort]
  summaries[cohort]=dict(n=len(selected),categories=dict(collections.Counter(r['category'] for r in selected)),
   eligible_episodes=sum(r['never_alignment_first'] is not None for r in selected),successes=sum(r['success'] for r in selected),
   successful_episodes_with_eligible_screen=sum(r['success'] and r['never_alignment_first'] is not None for r in selected),
   warning='Trigger can intercept eventual successes: this is a collection screen, not a failure oracle')
 write(O/'train_development_stage_summary.json',summaries);print('census',summaries,'missing',len(missing),flush=True)

if __name__=='__main__':status();census()
