from pathlib import Path
import json,copy,shutil,time
import torch
from tools.tight_machine_review import session_at
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.narrow_data import dagger_record,validate_aggregate_families
from visuotactile.latest_takeover_sampling import sampling_plan
from visuotactile.trajectory_training import profile,budget
from visuotactile.paired_training_decks import build_pair,validate_pool_pair
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';O=P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning';C=P/'configs/experiments/correction_20261127';read=lambda p:json.loads(p.read_text())
entries={}
for directory in sorted((R/'current_takeovers').glob('*_*')):
 if not directory.is_dir() or not (directory/'plan.json').exists():continue
 assert read(directory/'result.json')['software_complete']
 entries.update(read(directory/'correction_entries.json'))
assert entries,'Require actual current-own expert rescues before targeted training'
all_data={};all_records={};sampling={};roots={};configs={};pairs=read(R/'registration.json')['pairs']
reuse=copy.deepcopy(read(O/'A/cache_index.json'));reuse['entries'].update(read(O/'B/cache_index.json')['entries']);atomic_json(R/'reusable_visual_index.json',reuse)
for condition in ['old_only','with_corrections']:
 root=R/('short_'+condition);s=session_at(root,duration=12*3600);s['cache_deadline']=s['deadline'];s['resources']['min_ram_gib']=6
 s['authorization']='Explicit finite current-own correction versus own same-start old-data-only control,1complete registered epoch, matched actual component update budgets, fixed-final deployment; no terminal64'
 atomic_json(root/'session.json',s);roots[condition]=root;all_data[condition]={};all_records[condition]={};sampling[condition]={}
 for name in ['metadata.json','frozen_visual_io.json','frozen_visual_hotset.json','training_sampling.json','visual_arithmetic.json','cuda_allocator_policy.json']:
  shutil.copyfile(O/name,root/name)
 atomic_json(root/'gate.json',dict(passed=True,inherited_input_gate=str(O/'gate.json'),own_capture_audits=str(R/'current_takeovers'),fixed_affine=True))
 for role in 'AB':
  child=root/role;sd=child/'state_data';sd.mkdir(parents=True);data=copy.deepcopy(read(O/role/'state_data/data.json'));raw=torch.load(O/role/'state_data/numeric.pt',map_location='cpu',weights_only=False);records=raw['records'];contexts=torch.load(O/role/'state_data/loss_context.pt',map_location='cpu',weights_only=False) if role=='A' else None
  added={i:e for i,e in entries.items() if e['source_group']==role} if condition=='with_corrections' else {}
  validate_aggregate_families(data,added)
  for i,e in added.items():
   row,touch,context=dagger_record(e,include_geometry=role=='A');records[i]=row
   if contexts is not None:contexts[i]=context
   data['entries'][i]=e;data['splits']['train'].append(i)
  own=data['paired_pool']['own_correction_ids']+list(added);data['paired_pool']['own_correction_ids']=own;data['pool_partitions']['own_new_correction']=own;data['aggregation']['new_training_ids']=own;data['paired_pool_provenance']['own_correction_ids']=own
  data['targeted_correction']=dict(condition=condition,new_own_ids=list(added),old_pool_manifest=str(O/role/'state_data/data.json'),old_pool_preserved=True,original64_used=False,old_shared24_layout='Executed historical unshifted layout retained; new captures enforce actual corrected100mm reset')
  atomic_json(sd/'data.json',data);torch.save(dict(raw,records=records,data_sha256=sha256(sd/'data.json')),sd/'numeric.pt');shutil.copyfile(O/role/'state_data/statistics.pt',sd/'statistics.pt')
  if contexts is not None:torch.save(contexts,sd/'loss_context.pt')
  shutil.copyfile(root/'visual_arithmetic.json',child/'visual_arithmetic.json')
  atomic_json(child/'session.json',s);(child/'shared_resources.json').symlink_to(root/'shared_resources.json');(child/'resources').symlink_to(root/'resources',target_is_directory=True)
  pair=pairs[role+'_new'];atomic_json(child/'continuation.json',dict(path=pair['frontend'],sha256=pair['frontend_sha256'],statistics_sha256=sha256(sd/'statistics.pt'),target_window_contract='recovery_onset_commit12_v1',learning_rate=1.5e-5,initialization='Own current selected EMA; fresh AdamW/EMA, both conditions same start',previous_backend=pair['backend'],previous_backend_sha256=pair['backend_sha256'],reuse_index=str(R/'reusable_visual_index.json')))
  all_data[condition][role]=data;all_records[condition][role]=records
  extras=dict.fromkeys(data['splits']['train'],0)
  for i in added:extras[i]=15
  sampling[condition][role]=dict(version='paired_dagger_replay_v1',extra_replays=extras)
 validate_pool_pair(all_data[condition]['A'],all_data[condition]['B'],role='A',enabled=True)
maximum=max(sum(sampling_plan(all_data[c][r],sampling[c][r])[1].values()) for c in roots for r in 'AB');maximum+=(maximum%2)
for condition,root in roots.items():
 reg=dict(profile=profile(),roles={})
 for role in 'AB':
  data=all_data[condition][role];spec=sampling[condition][role];counts=sampling_plan(data,spec)[1];missing=maximum-sum(counts.values());old_ids=data['paired_pool']['original_ids'];old_ids=sorted(old_ids,key=lambda i:(int(all_records[condition][role][i]['supervision_mask'].sum()),i))
  for k in range(missing):spec['extra_replays'][old_ids[k%len(old_ids)]]+=1
  spec,counts=sampling_plan(data,spec);cont=read(root/role/'continuation.json');cont['sampling_contract']=spec;atomic_json(root/role/'continuation.json',cont)
  reg['roles'][role]=dict(**budget(sum(counts.values()),1),data_sha256=sha256(root/role/'state_data/data.json'),episode_multiplicities=counts)
 atomic_json(root/'full_trajectory_batch.json',reg);atomic_json(root/'backend_episode_sampling.json',sampling[condition]);configs[condition]=reg
# Plan all four rear conditions before updates, matching real complete-group updates.
pre={c:build_pair(all_data[c],all_records[c],{r:sha256(roots[c]/r/'state_data/data.json') for r in 'AB'},sampling_by_role=sampling[c],epochs=1,finite=True) for c in roots}
rear_maximum=max(d['updates_per_epoch'] for ds in pre.values() for d in ds.values())
for condition,root in roots.items():
 decks=build_pair(all_data[condition],all_records[condition],{r:sha256(root/r/'state_data/data.json') for r in 'AB'},sampling_by_role=sampling[condition],epochs=1,finite=True,minimum_updates_per_epoch=rear_maximum)
 for role in 'AB':atomic_json(root/f'paired_group_deck_{role}.json',decks[role])
 atomic_json(root/'registration.json',dict(utc=time.time(),epochs=1,selected='fixed_final',condition=condition,frontend_updates=maximum//2,backend_updates=rear_maximum,frontend=configs[condition],new_own_ids={r:all_data[condition][r]['targeted_correction']['new_own_ids'] for r in 'AB'},original64_used=False,main_pools_preserved=True))
 atomic_json(C/('short_'+condition+'.json'),dict(version='tight_project_job_v1',operation='train',resource_root=str(root),stages=[dict(entry='correction_train',args=['--root',str(root)])]))
atomic_json(R/'short_budget_registration.json',dict(utc=time.time(),conditions=list(roots),epochs=1,frontend_updates_each=maximum//2,backend_updates_each=rear_maximum,new_captures=len(entries),selection='Preregistered final1epoch; no development outcome used for checkpoint selection',fifty_step_claim=False))
print(read(R/'short_budget_registration.json'))
