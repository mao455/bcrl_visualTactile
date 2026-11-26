from pathlib import Path
import sys,json,csv,math
import numpy as np,h5py
from visuotactile.training_resources import atomic_json
sys.path.insert(0,str(Path('outputs/diagnostics/tight_machine_review_20261126_01').resolve()))
from audit_existing import replay
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01');read=lambda p:json.loads(p.read_text())
def csvout(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
def wilson(k,n):
 z=1.959963984540054;p=k/n;den=1+z*z/n;middle=(p+z*z/(2*n))/den;half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
 return [max(0,middle-half),min(1,middle+half)]
def paired(a,b):
 a=np.asarray(a,int);b=np.asarray(b,int);assert len(a)==len(b)==8;delta=a-b;generator=np.random.default_rng(20261127);samples=delta[generator.integers(0,len(a),size=(200000,len(a)))].mean(-1)*100
 return dict(n=8,a_success=int(a.sum()),b_success=int(b.sum()),difference_pp=float(delta.mean()*100),bootstrap95_pp=np.quantile(samples,[.025,.975]).tolist(),retained=int(((a==1)&(b==1)).sum()),added=int(((a==1)&(b==0)).sum()),lost=int(((a==0)&(b==1)).sum()),both_failed=int(((a==0)&(b==0)).sum()),degenerate_all_zero_differences=bool(not delta.any()),conservative_simultaneous95_pp=([-100*(1-.025**(1/len(delta))),100*(1-.025**(1/len(delta)))] if not delta.any() else None),interpretation='Pair resamples by existing development family; reused for correction, not independent test; a zero-width all-zero interval does not prove equivalence')
before=read(D/'development_before_cases.json');assert len(before)==32 and all(r['complete'] for r in before)
rows=before[:]
for role in 'AB':
 wave=R/('development_after_'+role)/'rollouts';plan=read(wave/'plan.json');assert read(wave/'result.json')['software_complete']
 for c in plan['cases']:
  directory=wave/c['name'];res=read(directory/'result.json');row=dict(wave=wave.name,role=c['role'],family=c['spec']['family'],seed=c['spec']['seed'],complete=res['software_complete'],success=res['success'],steps=res['steps'],reasons='|'.join(k for k,v in res['reasons'].items() if v),source=str(directory/'result.json'))
  if res.get('legal'):
   row.update(replay(directory/'numeric.h5'));row.pop('eligible_completed_ticks',None)
   receipt=read(directory/'observe_only_trigger.json');row['live_first_eligible_completed_tick']=receipt.get('first_eligible_completed_tick');row['live_matches_replay']=row['live_first_eligible_completed_tick']==row['never_alignment_first']
  rows.append(row)
assert len(rows)==64 and all(r['complete'] for r in rows)
roles=list(dict.fromkeys(r['role'] for r in rows));summary={};outcomes={}
for role in roles:
 rr=sorted([r for r in rows if r['role']==role],key=lambda x:x['seed']);assert [r['seed'] for r in rr]==list(range(28911000,28911008));k=sum(r['success'] for r in rr);outcomes[role]=[int(r['success']) for r in rr]
 summary[role]=dict(denominator=8,complete=sum(r['complete'] for r in rr),strict_success=k,success_rate=k/8,wilson95=wilson(k,8),held12=sum(r.get('held12',False) for r in rr),near_held=sum(r.get('near_held',False) for r in rr),first15mm=sum(r.get('ever15mm',False) for r in rr),first_subgoal=sum(r.get('native_first_subgoal',False) for r in rr),never_stall=sum(r.get('never_alignment_first') is not None for r in rr),categories={category:sum(r.get('category')==category for r in rr) for category in {r.get('category') for r in rr}})
comparisons={}
for role in 'AB':
 for a,b,label in [(role+'_new',role+'_parent','before_new_minus_actual_parent'),(role+'_with_corrections',role+'_new','correction_minus_before'),(role+'_old_only',role+'_new','old_only_minus_before'),(role+'_with_corrections',role+'_old_only','correction_minus_old_only')]:comparisons[role+'_'+label]=paired(outcomes[a],outcomes[b])
comparisons['after_A_minus_B']=paired(outcomes['A_with_corrections'],outcomes['B_with_corrections']);comparisons['before_A_minus_B']=paired(outcomes['A_new'],outcomes['B_new'])
wide=[]
for seed in range(28911000,28911008):
 row=dict(seed=seed,family=next(r['family'] for r in rows if r['seed']==seed))
 for r in rows:
  if r['seed']==seed:
   for key in ['success','steps','held12','near_held','ever15mm','native_first_subgoal','category','never_alignment_first']:row[r['role']+'_'+key]=r.get(key)
 wide.append(row)
csvout(D/'per_family_before_after.csv',wide);csvout(D/'all_completed_development_cases.csv',rows)
# Actual pre-action observable start comparison across stages; no hidden-state claim.
def initial(path):
 with h5py.File(path) as f:
  tr=f['episode/transition'];b=tr['diagnostics_before'];base=b['base_pose_w'][0,:3];v={k:b[k][0] for k in ['joint_position_rad','joint_velocity_rad_s','object_velocity_w_m_s_rad_s','goal_velocity_w_m_s_rad_s']};v['actuator_target_rad']=f['episode/obs/actuator_target_rad'][0]
  for k in ['object_pose_w','hole_pose_w','goal_pose_w','table_pose_w','palm_body_pose_w']:
   value=b[k][0].copy();value[:3]-=base;v[k]=value
  return v
initials={};checks=[]
for r in rows:
 if not Path(r['source']).with_name('numeric.h5').exists():continue
 v=initial(Path(r['source']).with_name('numeric.h5'));key=(r['seed'],r['role'][0])
 if key not in initials:initials[key]=(r['role'],v);continue
 tag,reference=initials[key];diff={}
 for k,value in v.items():
  if k.endswith('pose_w'):
   diff[k+'_xyz_m']=float(np.max(np.abs(value[:3]-reference[k][:3])));diff[k+'_quaternion']=float(min(np.max(np.abs(value[3:]-reference[k][3:])),np.max(np.abs(value[3:]+reference[k][3:]))))
  else:diff[k]=float(np.max(np.abs(value-reference[k])))
 checks.append(dict(seed=r['seed'],reference=tag,policy=r['role'],differences=diff,within1e5=max(diff.values())<=1e-5))
atomic_json(D/'cross_phase_initial_observables.json',dict(rows=checks,all_within1e5=all(x['within1e5'] for x in checks),hidden_state_or_future_random_stream_claim=False))
atomic_json(D/'primary_results.json',dict(policies=summary,comparisons=comparisons,bootstrap_replicates=200000,seed=20261127,original64_reused=False,denominator_per_policy=8));print(json.dumps(summary))
