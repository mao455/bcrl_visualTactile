from pathlib import Path
import json,copy,shutil,time
import torch
from tools.tight_machine_review import session_at
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.trajectory_training import profile,budget
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';D=P/'outputs/diagnostics/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127'
O=P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01';F=P/'outputs/visuotactile/run/tight_machine_review_20261126_01/fresh_rescue_history02'
read=lambda p:json.loads(p.read_text())
owner=R/'b72_fit';s=session_at(owner,duration=12*3600)
s.update(authorization='Explicit new authorization: actual isolated optimization from currentA/B on two existingB72 expert rescue suffixes; no main-pool change or autonomous rescue claim',cache_deadline=s['deadline'])
atomic_json(owner/'session.json',s)
for name in ['metadata.json','frozen_visual_io.json','frozen_visual_hotset.json','training_sampling.json','visual_arithmetic.json']:
 shutil.copyfile(O/'learning'/name,owner/name)
atomic_json(owner/'gate.json',dict(passed=True,inherited_input_gate=str(O/'learning/gate.json'),fresh_rescue_audits=str(F/'data_preparation.json'),scope='Exact canonical numeric/visual inputs from passed real handoff audits'))
sampling=dict(version='latest_takeover_repeat_v1',latest_repeat=16)
atomic_json(owner/'backend_episode_sampling.json',sampling)
atomic_json(owner/'full_trajectory_batch.json',dict(profile=profile(),roles={r:budget(62,2) for r in 'AB'}))
freeze=read(R/'registration.json')['pairs'];reg={}
for role in 'AB':
 oldroot=O/f'common180_parents/{role}';old=read(oldroot/'state_data/data.json');fresh=read(F/f'{role}/state_data/data.json')
 train=list(old['normalization_train_ids']);val=list(old['splits']['validation']);ids=train+val+list(fresh['entries'])
 data=copy.deepcopy(old);data.update(entries={i:copy.deepcopy((old if i in old['entries'] else fresh)['entries'][i]) for i in ids},splits=dict(train=train+list(fresh['entries']),validation=val,test=[]),normalization_scope='unchanged original training affine; no aggregate refit',aggregation=dict(version='expert_takeover_aggregate_v1',student_prefix_supervised=False,new_training_ids=list(fresh['entries']),old_training_ids=train),diagnostic_only=True,training_forbidden=False,authorization='User20261127 existingB72 isolated optimization; samples stay labeled existingB72')
 for i in fresh['entries']:data['entries'][i].update(split='train',round_index=1,source_policy='historical_B72',fit_only=True)
 child=owner/role;sd=child/'state_data';(sd/'perception').mkdir(parents=True)
 atomic_json(sd/'data.json',data)
 oldnum=torch.load(oldroot/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records'];newnum=torch.load(F/f'{role}/state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
 records={i:(oldnum if i in oldnum else newnum)[i] for i in ids}
 torch.save(dict(data_sha256=sha256(sd/'data.json'),records=records),sd/'numeric.pt')
 shutil.copyfile(oldroot/'state_data/statistics.pt',sd/'statistics.pt')
 ci=read(oldroot/'cache_index.json');fi=read(F/f'{role}/cache_index.json');ci.update(data_sha256=sha256(sd/'data.json'),entries={i:copy.deepcopy((ci if i in ci['entries'] else fi)['entries'][i]) for i in ids},complete=True)
 for i in ids:
  source=(oldroot/'state_data/perception'/(i+'.pt') if i in old['entries'] else F/f'{role}/state_data/perception'/(i+'.pt')).resolve()
  assert source.exists(),source
  (sd/'perception'/(i+'.pt')).symlink_to(source)
 atomic_json(child/'cache_index.json',ci);atomic_json(child/'session.json',s)
 (child/'shared_resources.json').symlink_to(owner/'shared_resources.json')
 pair=freeze[f'{role}_new']
 atomic_json(child/'continuation.json',dict(path=pair['frontend'],sha256=pair['frontend_sha256'],statistics_sha256=sha256(sd/'statistics.pt'),target_window_contract='recovery_onset_commit12_v1',learning_rate=1.5e-5,initialization='Current new full-policy EMA; isolated fresh AdamW/EMA, B72 labels never called own-student',sampling_contract=sampling,previous_backend=pair['backend'],previous_backend_sha256=pair['backend_sha256'],reuse_index=str(child/'cache_index.json')))
 reg[role]=dict(initial_pair=pair,train=len(train)+2,old_full=len(train),existing_B72_families=[28817004,28817008],existing_B72_legal_ticks=[92,116],epochs=2,latest_repeat=16,frontend_updates=62,statistics_sha256=sha256(sd/'statistics.pt'))
atomic_json(owner/'registration.json',dict(utc=time.time(),roles=reg,main_training_pool_modified=False,deploy_selection='fixed final update; validation does not choose',backend='same population/order/epochs/microbatch; own selected DP nominal cache refreshed before rear updates'))
atomic_json(C/'b72_fit.json',dict(version='tight_project_job_v1',operation='train',resource_root=str(owner),stages=[dict(entry='b72_fit',args=['worker','--root',str(owner)])]))
print(owner)
