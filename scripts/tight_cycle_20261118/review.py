"""Compact evidence export for this single cycle; no new optimization or rollout."""
import collections,csv,gzip,json,shutil,time
from pathlib import Path
import h5py,numpy as np,torch
from scripts.tight_cycle_20261118.cycle import ROOT,DATA,PARENT,OUT,read,write
from scripts.tight_balance_20261117.diagnose import phases
from scripts.tight_layout_20261116.publication import table
from visuotactile.compact import sha256
from visuotactile.geometry_sampling import STAGES

def collection():
 from scripts.tight_layout_20261116.phase_report import summarize
 d=ROOT/'collection';out=OUT/'collection';out.mkdir(exist_ok=True)
 summarize(d,out/'summary.json');s=read(out/'summary.json');entries=read(d/'correction_entries.json');old=read(DATA/'A/state_data/data.json');parent=read(PARENT/'A/state_data/data.json');coverage=[]
 for i,e in entries.items():
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];mask=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool);p=phases(t);valid=np.flatnonzero(mask);first=int(valid[0])
   q=t['diagnostics_before/joint_position_rad'][first];pose=t['diagnostics_before/object_pose_w'][first];palm=t['diagnostics_before/palm_body_pose_w'][first];vel=t['diagnostics_before/object_velocity_w_m_s_rad_s'][first]
   coverage.append(dict(identity=i,source=e['source_group'],family=e['family'],legal_ticks=int(mask.sum()),history_ticks=len(mask),first_legal_tick=first,entry_phase=STAGES[int(p[first])],phase_ticks={STAGES[k]:int(((p==k)&mask).sum()) for k in range(len(STAGES))},object_palm_m=(pose[:3]-palm[:3]).tolist(),object_rotation=pose[3:].tolist(),object_velocity=vel.tolist(),actual_joint_rad=q.tolist(),source_sha256=e['source_sha256']))
 write(out/'coverage.json',dict(rows=coverage,new_trajectories=len(entries),new_independent_families=len({e['family'] for e in entries.values()}-{e['family'] for e in old['entries'].values()}),existing_pool=len(old['splits']['train']),old_pool_new_to_parent=sorted(set(old['splits']['train'])-set(parent['splits']['train'])),new_to_parent_count=len(set(old['splits']['train'])-set(parent['splits']['train']))+len(entries)))
 table(out/'opportunities.csv',s['rows']);table(out/'coverage.csv',coverage)
 for n in ['plan.json','result.json','correction_entries.json','initialization.json','owner_complete.json','owner_exit.json','paired_initial_check.json','wave_profiles.json','capture_audit_memo.json']:
  if (d/n).exists():shutil.copyfile(d/n,out/n)
 print(json.dumps(dict(counts=s['counts'],stages=s['requested_stages'],new_data=len(entries))))

def learning():
 from visuotactile.anchor_sampling import supervision_rows
 from visuotactile.target_windows import window_mask
 d=ROOT/'learning';out=OUT/'learning';out.mkdir(exist_ok=True)
 curves=[];models=[];performance=[];exposure=[];episode_rows=[]
 baseids=set(read(PARENT/'A/state_data/data.json')['splits']['train']);oldids=set(read(DATA/'A/state_data/data.json')['splits']['train'])
 for role in 'AB':
  data=read(d/role/'state_data/data.json');records=torch.load(d/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
  dp=torch.load(d/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True);rear=torch.load(d/role/'backend'/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True)
  assert rear['configuration']['base_sha256']==sha256(d/role/'training/best.pt')
  models.append(dict(role=role,dp_sha256=sha256(d/role/'training/best.pt'),backend_sha256=sha256(d/role/'backend'/role/'training/best.pt'),dp_selected_updates=dp['updates'],rear_selected_updates=rear['updates'],dp_selected_epoch=dp['epoch'],rear_selected_epoch=rear['epoch'],rear_selected_supervised_ticks=rear['supervised_ticks'],rear_selected_episode_counts=rear['episode_counts'],dp_parent=dp['configuration']['continuation'],rear_parent=rear['continuation_from'],data_sha256=dp['configuration']['data_sha256'],statistics_sha256=dp['configuration']['statistics_sha256']))
  phase={};masks={};groups={}
  for i in data['splits']['train']:
   e=data['entries'][i];masks[i]=supervision_rows(records[i]);groups[i]='original30' if e.get('trajectory_kind')!='expert_takeover_v1' else 'new_cycle' if i not in oldids else 'old_pool_unseen4' if i not in baseids else 'historical_correction42'
   with h5py.File(e['path']) as f:phase[i]=phases(f['episodes'][e['episode']]['transition'])
  counts=collections.defaultdict(collections.Counter);unique=collections.defaultdict(set)
  for line in (d/role/'training/learning.jsonl').open():
   row=json.loads(line)
   for v in row['visits']:
    i=v['episode'];mask,seg=masks[i];a=torch.tensor(v['anchor_sampling']['selected_anchors']);valid=window_mask(mask,seg,a,24,'recovery_onset_commit12_v1');ticks=(a[:,None]+torch.arange(24)[None])[valid].numpy();assert len(ticks)==v['target_ticks'];start=int(mask.nonzero()[0]);early=int((ticks<150).sum()) if groups[i]=='original30' else 0
    for scope in ['actual']+(['selected'] if row['updates']<=dp['updates'] else []):
     c=counts[scope,i];c.update(visits=1,windows=len(a),slots=len(ticks),early_slots=early,base_objective_mass=v['loss_weight'],early_base_mass=v['loss_weight']*early/len(ticks),onset24_slots=int((ticks<start+24).sum()) if start else 0)
     for k,n in enumerate(STAGES):c[n]+=int((phase[i][ticks]==k).sum())
     unique[scope,i].update(ticks.tolist())
  for (scope,i),c in counts.items():episode_rows.append(dict(role=role,scope=scope,group=groups[i],identity=i,family=data['entries'][i]['family'],new_to_parent=i not in baseids,unique_supervised_ticks=len(unique[scope,i]),**c))
  for scope in ['actual','selected']:
   for g in sorted(set(groups.values())):
    ids=[i for i in groups if groups[i]==g];c=collections.Counter()
    for i in ids:c.update(counts[scope,i])
    exposure.append(dict(role=role,scope=scope,group=g,trajectories=len(ids),**c))
  for comp,folder in [('DP',d/role/'training'),('Transformer',d/role/'backend'/role/'training')]:
   done=read(folder/'finished.json');dest=out/role/comp;dest.mkdir(parents=True,exist_ok=True)
   if comp=='DP':
    vals=[json.loads(l) for l in (folder/'validation.jsonl').open()];train=sum(r['training_epoch_io']['wall_seconds'] for r in vals)
    performance.append(dict(role=role,component=comp,updates=done['updates'],stage_seconds=done['wall_seconds'],train_seconds=train,updates_per_train_second=done['updates']/train,validation_seconds=sum(r['validation']['offline_planning']['seconds'] for r in vals),train_diagnostic_seconds=sum(r['train']['offline_planning']['seconds'] for r in vals),physical_read_bytes=done['stage_process_io']['read_bytes'],hotset=done['frozen_visual_hotset'],peak_cuda_allocated_bytes=done['peak_cuda_allocated_bytes'],peak_cuda_reserved_bytes=done['peak_cuda_reserved_bytes']))
    for r in vals:curves.append(dict(role=role,component=comp,epoch=r['epoch'],updates=r['updates'],validation=r['validation']['selection'],train_fit=r['train']['selection'],train_loss=r['loss']))
   else:
    performance.append(dict(role=role,component=comp,updates=done['updates'],stage_seconds=done['wall_s'],train_seconds=done['training_wall_s'],setup_seconds=done['setup_wall_s']))
    for r in read(folder/'validation.json'):curves.append(dict(role=role,component=comp,epoch=r['epoch'],updates=r['update'],validation=r['validation']['all']['normalized_equal_mse'],train_fit=r['train']['all']['normalized_equal_mse'],train_loss=None))
   for name in ['configuration.json','finished.json','validation.json','validation.jsonl','loss.jsonl']:
    if (folder/name).exists():
     if name.endswith('jsonl'):
      with gzip.GzipFile(filename=str(dest/(name+'.gz')),mode='wb',mtime=0) as f:f.write((folder/name).read_bytes())
     else:shutil.copyfile(folder/name,dest/name)
  shutil.copyfile(d/role/'state_data/data.json',out/role/'data.json')
 from collections import defaultdict
 backend_exposure=[]
 for role in 'AB':
  folder=d/role/'backend'/role/'training'
  done=read(folder/'finished.json')
  backend_exposure.append(dict(role=role,selected_updates=next(m['rear_selected_updates'] for m in models if m['role']==role),actual_updates=done['updates'],actual_supervised_ticks=done['supervised_ticks'],selected_supervised_ticks=next(m['rear_selected_supervised_ticks'] for m in models if m['role']==role),supervised_population=done['supervised_tick_population'],control_tick_epochs=done['control_tick_epochs'],minimum_exposure_reached=done['minimum_exposure_reached']))
 write(out/'backend_exposure.json',backend_exposure)
 for name,v in [('models',models),('performance',performance),('exposure',exposure),('episode_exposure',episode_rows),('curves',curves)]:write(out/(name+'.json'),v);table(out/(name+'.csv'),v)
 write(out/'exposure_scope.json',dict(note='Window slots overlap and repeat; objective mass is base-loss coefficient, NOT gradient mass; geometry excluded. Actual run updates and selected checkpoint updates are separate; new_cycle and old_pool_unseen4 are jointly new to72 parent.'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(11,6))
 for row,role in enumerate('AB'):
  for col,comp in enumerate(['DP','Transformer']):
   rs=[r for r in curves if r['role']==role and r['component']==comp];ax=axes[row,col];ax.plot([r['updates'] for r in rs],[r['validation'] for r in rs],marker='.',label='Independent validation');ax.plot([r['updates'] for r in rs],[r['train_fit'] for r in rs],label='Fixed train2 fitting diagnostic');selected=next(m for m in models if m['role']==role)['dp_selected_updates' if comp=='DP' else 'rear_selected_updates'];ax.axvline(selected,color='green',linestyle='--',label='Selected checkpoint');ax.set_title(role+' '+comp);ax.set_xlabel('Optimizer updates in this stage');ax.set_ylabel('Normalized MSE (same target domain)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(out/'validation_curves.png',dpi=140);plt.close(fig)
 print(json.dumps(models))

def evaluation(parts=2):
 from scripts.tight_layout_20261116.phase_report import summarize
 out=OUT/'evaluation';out.mkdir(exist_ok=True);rows=[]
 for part in range(parts):
  d=ROOT/f'comparison{part:02d}';summarize(d,out/f'part{part}.json');s=read(out/f'part{part}.json');cases={c['name']:c for c in read(d/'plan.json')['cases']}
  for r in s['rows']:r['cohort']=cases[r['case']]['diagnostic_cohort'];rows.append(r)
  dst=out/f'part{part}';dst.mkdir(exist_ok=True)
  for n in ['plan.json','result.json','paired_initial_check.json','initialization.json','owner_complete.json','owner_exit.json','wave_profiles.json','videos.json']:shutil.copyfile(d/n,dst/n)
 totals={}
 for co in ['training','new_development']:
  for role in ['A_baseline','A_candidate','B_baseline','B_candidate']:
   rr=[r for r in rows if r['cohort']==co and r['role']==role];totals[co+'_'+role]=dict(n=len(rr),success=sum(r['success'] for r in rr),hold60=sum(r['hold60']['sustained'] for r in rr),hold120=sum(r['hold120']['sustained'] for r in rr),first_subgoal=sum(r['native_subgoals']>0 for r in rr))
 write(out/'results.json',dict(counts=totals,rows=rows,scope='Same-wave actual paired complete students; previously opened development, one candidate/training seed, small sample, not strict funnel or significance claim. Capability baseline run is separate batch context and not pooled.'));table(out/'per_case.csv',rows);print(json.dumps(totals))
if __name__=='__main__':globals()[__import__('sys').argv[1]]()
