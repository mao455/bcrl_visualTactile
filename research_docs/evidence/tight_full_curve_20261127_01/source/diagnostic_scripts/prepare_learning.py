from pathlib import Path
import copy,json,shutil,time
import torch
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.paired_training_decks import build_pair,validate_pool_pair
from visuotactile.latest_takeover_sampling import sampling_plan
from visuotactile.trajectory_training import profile,budget
from visuotactile.paired_curve_diagnostics import profile as diagnostic_profile
from visuotactile.narrow_data import validate_aggregate_families
from tools.tight_paired_pool import full_record
from visuotactile.state_image_loss import geometry_loss_context
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';C=P/'configs/experiments/full_curve_20261127';PREV=P/'outputs/visuotactile/run/tight_correction_20261127_01';OLD=P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning';read=lambda p:json.loads(p.read_text())
registration=read(R/'registration.json');entries={};attempts=[];registered={}
for shard in range(3):
 d=R/f'experts_{shard:02d}';assert read(d/'result.json')['software_complete'];assert read(d/'project_exit.json')['children_reaped'];entries.update(read(d/'expert_entries.json'));attempts.extend(read(d/'results.json'))
 registered.update({c['seed']:c for c in read(d/'plan.json')['cases']})
# The reset families are historical24, not24additional independent families.
atomic_json(R/'full_expert_collection.json',dict(entries=entries,attempts=attempts,accepted_families=len({e['seed'] for e in entries.values()}),registered_families=len(registered),original24_requested=registration['corrected_demo_families'],complete=True))
assert len(entries)==24,'Do not silently replace a collision/failed family; inspect preserved attempts'
roots={};data_all={};records_all={};sampling={};raw_new={};contexts_new={}
for i,e in entries.items():
 raw_new[i],audit=full_record(e,registered[e['seed']]);contexts_new[i]=geometry_loss_context(e)
 atomic_json(R/(i+'_label_audit.json'),audit)
reuse=copy.deepcopy(read(PREV/'short_with_corrections/A/cache_index.json'));reuse['entries'].update(read(PREV/'short_with_corrections/B/cache_index.json')['entries']);atomic_json(R/'reusable_visual_index.json',reuse)
for condition in registration['conditions']:
 root=R/condition;root.mkdir();roots[condition]=root;data_all[condition]={};records_all[condition]={};sampling[condition]={}
 s=copy.deepcopy(read(R/'session.json'));s['resource_root']=str(root);atomic_json(root/'session.json',s)
 for name in ['metadata.json','frozen_visual_io.json','frozen_visual_hotset.json','cuda_allocator_policy.json','training_sampling.json','visual_arithmetic.json','effective_exposure.json','entry_exposure.json']:
  shutil.copyfile(OLD/name,root/name)
 atomic_json(root/'family_phase_diagnostics.json',diagnostic_profile());atomic_json(root/'training_milestones.json',[30,60])
 atomic_json(root/'gate.json',dict(passed=True,inherited_input_gate=str(OLD/'gate.json'),new_full_layout_audits=str(R/'full_expert_collection.json'),affine_unchanged=True))
 for role in 'AB':
  source=PREV/'short_with_corrections'/role/'state_data';child=root/role;sd=child/'state_data';sd.mkdir(parents=True)
  data=copy.deepcopy(read(source/'data.json'));raw=torch.load(source/'numeric.pt',map_location='cpu',weights_only=False);records=raw['records'];contexts=torch.load(source/'loss_context.pt',map_location='cpu',weights_only=False) if role=='A' else None
  added=copy.deepcopy(entries) if condition=='plus_corrected24' else {}
  family_aliases={}
  for i,e in added.items():
   captured=e['family'];canonical=next((data['entries'][j]['family'] for j in data['paired_pool']['shared_full_ids'] if data['entries'][j]['seed']==e['seed']),f'full_curve_s{e["seed"]}')
   e.update(family=canonical,capture_family=captured,actual_layout='verified100mm');family_aliases[captured]=canonical
  known=set(data['families'])
  validate_aggregate_families(data,{i:e for i,e in added.items() if e['family'] in known})
  for i,e in added.items():
   assert e['seed'] in registered and registered[e['seed']]['seed']==e['seed']
   assert e['seed'] not in set(registration['development_families']+registration['sealed_historical64']+registration['sealed_new_final_families'])
   assert e['split']=='train' and read(Path(e['path']).parents[1]/'actual_layout_admission.json')['passed']
   assert e['family'] not in {data['entries'][j]['family'] for k,ids in data['splits'].items() if k!='train' for j in ids}
  new_family_count=len({e['family'] for e in added.values()}-set(data['families']))
  for i,e in added.items():
   records[i]=raw_new[i];data['entries'][i]=e;data['splits']['train'].append(i)
   if contexts is not None:contexts[i]=contexts_new[i]
  data['families'].update({e['family']:'train' for e in added.values()});data['paired_pool']['shared_full_ids']+=list(added);data['aggregation']['shared_full_training_ids']+=list(added);data['pool_partitions']['new_full']+=list(added);data['paired_pool_provenance']['shared_full_ids']+=list(added)
  data['full_curve']=dict(condition=condition,source_pool=str(source/'data.json'),corrected_full_ids=list(added),family_aliases=family_aliases,physical_new_families=new_family_count,old_shared24_actual_layout='unshifted; retained unchanged',old_corrections_preserved=True,terminal64_used=False)
  atomic_json(sd/'data.json',data);torch.save(dict(raw,records=records,data_sha256=sha256(sd/'data.json')),sd/'numeric.pt');shutil.copyfile(source/'statistics.pt',sd/'statistics.pt')
  if contexts is not None:torch.save(contexts,sd/'loss_context.pt')
  atomic_json(child/'session.json',s);(child/'shared_resources.json').symlink_to(root/'shared_resources.json');(child/'resources').symlink_to(root/'resources',target_is_directory=True);shutil.copyfile(root/'visual_arithmetic.json',child/'visual_arithmetic.json')
  pair=registration['starting_pairs'][role]
  atomic_json(child/'continuation.json',dict(path=pair['frontend'],sha256=pair['frontend_sha256'],statistics_sha256=sha256(sd/'statistics.pt'),target_window_contract='recovery_onset_commit12_v1',learning_rate=1.5e-5,initialization='Own paired60 selected EMA; fresh AdamW/EMA; same across conditions',previous_backend=pair['backend'],previous_backend_sha256=pair['backend_sha256'],reuse_index=str(R/'reusable_visual_index.json')))
  data_all[condition][role]=data;records_all[condition][role]=records
  sampling[condition][role]=dict(version='paired_dagger_replay_v1',extra_replays=dict.fromkeys(data['splits']['train'],0))
 validate_pool_pair(data_all[condition]['A'],data_all[condition]['B'],role='A',enabled=True)
visits=max(sum(sampling_plan(data_all[c][r],sampling[c][r])[1].values()) for c in roots for r in 'AB');visits+=visits%2
for c,root in roots.items():
 reg=dict(profile=profile(),roles={})
 for role in 'AB':
  data=data_all[c][role];spec=sampling[c][role];counts=sampling_plan(data,spec)[1];ids=data['paired_pool']['original_ids']
  for k in range(visits-sum(counts.values())):spec['extra_replays'][ids[k%len(ids)]]+=1
  spec,counts=sampling_plan(data,spec);cont=read(root/role/'continuation.json');cont['sampling_contract']=spec;atomic_json(root/role/'continuation.json',cont)
  reg['roles'][role]=dict(**budget(sum(counts.values()),60),data_sha256=sha256(root/role/'state_data/data.json'),episode_multiplicities=counts)
 atomic_json(root/'full_trajectory_batch.json',reg)
 atomic_json(root/'matched_learning_rates.json',{r:[1.5e-5]*48+[7.5e-6]*12 for r in 'AB'})
 # Rear uses original partition weights; its equalization is independent of DP.
 rear={r:dict(version='paired_dagger_replay_v1',extra_replays=dict.fromkeys(data_all[c][r]['splits']['train'],0)) for r in 'AB'}
 atomic_json(root/'backend_episode_sampling.json',rear)
pre={c:build_pair(data_all[c],records_all[c],{r:sha256(roots[c]/r/'state_data/data.json') for r in 'AB'},sampling_by_role=read(roots[c]/'backend_episode_sampling.json'),epochs=60,curve=True) for c in roots}
rear_budget=max(d['updates_per_epoch'] for ds in pre.values() for d in ds.values())
for c,root in roots.items():
 for epochs in (30,60):
  decks=build_pair(data_all[c],records_all[c],{r:sha256(root/r/'state_data/data.json') for r in 'AB'},sampling_by_role=read(root/'backend_episode_sampling.json'),epochs=epochs,curve=True,minimum_updates_per_epoch=rear_budget)
  for r in 'AB':atomic_json(root/f'paired_group_deck_{r}_{epochs}.json',decks[r])
 atomic_json(root/'matched_backend_learning_rates.json',{r:[1e-4]*48+[5e-5]*12 for r in 'AB'})
 atomic_json(root/'budget_registration.json',dict(epochs=60,milestones=[30,60],frontend_updates_per_epoch=visits//2,backend_updates_per_epoch=rear_budget,condition=c,weights='Original30 full3, each shared full3, each legal correction1 per base epoch; explicit replay ledger aligns real updates across four models; actual rear singleton matching separate',selection='fixed30/60; no development-based selection',new_full_ids=list(entries) if c=='plus_corrected24' else []))
atomic_json(R/'budget_registration.json',dict(utc=time.time(),frontend_updates_each=visits//2*60,backend_final_updates_each=rear_budget*60,backend_midpoint_extra_updates_each=rear_budget*30,epochs=60,fixed_milestones=[0,30,60],conditions=list(roots),new_full_families=24,physical_families_reused=True))
print(read(R/'budget_registration.json'),flush=True)
