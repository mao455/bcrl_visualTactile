from pathlib import Path
import collections,csv,json,time
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.latest_takeover_sampling import sampling_plan
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';read=lambda p:json.loads(p.read_text())
def csvout(path,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
reg=read(R/'registration.json');collection=read(R/'full_expert_collection.json');entries=collection['entries'];assert len(entries)==24
admissions=[]
for identity,e in entries.items():
 directory=Path(e['path']).parents[1];v=read(directory/'actual_layout_admission.json');assert v['passed']
 result=read(directory/'result.json');assert result['software_complete'] and result['legal'] and result['success']
 assert not result.get('expert_shadow_placeholder_zero')
 for row in v['rows']:
  assert row['passed'] and max(row['errors'].values())<=1e-5
  admissions.append(dict(identity=identity,seed=e['seed'],ticks=e['length'],object_x=row['object_xy_m'][0],object_y=row['object_xy_m'][1],hole_x=row['hole_xy_m'][0],hole_y=row['hole_xy_m'][1],displacement_x=row['actual_minus_unshifted_xy_m'][0],displacement_y=row['actual_minus_unshifted_xy_m'][1],maximum_error_m=max(row['errors'].values()),source=str(directory)))
rows=[];pools=[]
for condition in reg['conditions']:
 root=R/condition;budget=read(root/'budget_registration.json')
 for role in 'AB':
  child=root/role;data=read(child/'state_data/data.json');cont=read(child/'continuation.json');spec,counts=sampling_plan(data,cont['sampling_contract']);rear=read(root/f'paired_group_deck_{role}_60.json')
  assert cont['path']==reg['starting_pairs'][role]['frontend'] and cont['sha256']==reg['starting_pairs'][role]['frontend_sha256']
  assert cont['previous_backend']==reg['starting_pairs'][role]['backend']
  assert sum(counts.values())==412 and rear['updates_per_epoch']==812
  new=set(data['full_curve']['corrected_full_ids']);assert len(new)==(24 if condition=='plus_corrected24' else 0)
  blocked=set(reg['development_families']+reg['sealed_historical64']+reg['sealed_new_final_families'])
  assert all(data['entries'][i].get('seed') not in blocked for i in data['splits']['train'])
  train_families={data['entries'][i]['family'] for i in data['splits']['train']};val_families={data['entries'][i]['family'] for i in data['splits']['validation']};assert not (train_families&val_families)
  membership={i:k for k,ids in data['pool_partitions'].items() for i in ids};assert len(membership)==len(data['splits']['train'])
  grouped=collections.defaultdict(lambda:dict(trajectories=0,ticks=0,dp_visits_per_epoch=0,rear_base_visits_per_epoch=0))
  for i in data['splits']['train']:
   e=data['entries'][i];partition='corrected_full24' if i in new else membership[i];group=grouped[partition]
   for k,n in [('trajectories',1),('ticks',e['length']),('dp_visits_per_epoch',counts[i]),('rear_base_visits_per_epoch',rear['base_multiplicities'][i])]:group[k]+=n
   rows.append(dict(condition=condition,role=role,identity=i,family=e['family'],seed=e.get('seed'),partition=partition,length=e['length'],dp_visits_per_epoch=counts[i],rear_base_visits_per_epoch=rear['base_multiplicities'][i],actual_layout='verified100mm' if i in new else 'historical; layout annotation retained',source=e['path'],source_sha256=e['source_sha256']))
  pools.append(dict(condition=condition,role=role,entries=len(data['entries']),training=len(counts),validation=len(data['splits']['validation']),families=len(data['families']),training_families=len(train_families),validation_families=len(val_families),grouped=dict(grouped),hashes={name:sha256(child/'state_data'/name) for name in ['data.json','numeric.pt','statistics.pt']},shared_current_corrected_full_ids=sorted(new),parent_frontend_sha256=cont['sha256'],parent_backend_sha256=cont['previous_backend_sha256']))
original=set(collection['original24_requested']);accepted={e['seed'] for e in entries.values()}
summary=dict(utc=time.time(),full_success=24,legal_ticks=sum(e['length'] for e in entries.values()),attempted_resets=len(collection['attempts']),unique_attempted_families=len({a['family'] for a in collection['attempts']}),registered_families=collection['registered_families'],pre_action_rejections=sum(not a['accepted'] for a in collection['attempts']),original24_accepted=len(original&accepted),original24_unavailable=sorted(original-accepted),extra_accepted_families=sorted(accepted-original),maximum_layout_error_m=max(r['maximum_error_m'] for r in admissions),pools=pools,old64_trajectories_read=0,pose_auxiliary_enabled=False,architecture_changed=False)
csvout(D/'corrected_full_layout_admissions.csv',admissions);csvout(D/'pool_sampling.csv',rows);atomic_json(D/'data_contract_audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='pools'}),flush=True)
