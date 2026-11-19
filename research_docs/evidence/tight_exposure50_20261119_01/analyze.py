"""Export actual selected exposure and prepare fixed paired evaluation.

Kept outside the trainer source snapshot while that immutable job is running.
"""
import argparse,collections,copy,csv,gzip,json,shutil,time
from pathlib import Path
import numpy as np
import torch
from tools.tight_frozen_frontend import PROJECT,read,write,sources
from tools.tight_frozen_comparison import table,predictions
from visuotactile.compact import sha256
from visuotactile.effective_exposure import Ledger

CONFIG=PROJECT/'configs/experiments/exposure50/experiment.json'

def paths():
 c=read(CONFIG);root=PROJECT/c['root'];parent=PROJECT/c['parent'];history=PROJECT/c['history'];public=Path(__file__).resolve().parent
 return c,root,parent,history,public


def training():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 c,root,parent,history,public=paths();summary=[];visits=[];curves=[];histograms={}
 with gzip.open(public/'selected_tick_exposure.csv.gz','wt',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=['model','identity','tick','legal','optimizer_steps']);writer.writeheader()
  for role in 'AB':
   for component in ('DP','Transformer'):
    folder=root/role/('training' if component=='DP' else f'backend/{role}/training')
    selected=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    state=selected['effective_exposure_state'];ledger=Ledger(state['legal'],state)
    counts=selected['sampling_exposure' if component=='DP' else 'episode_counts']
    receipt=ledger.summary(counts);assert receipt['eligible']
    assert selected['updates']==state['updates']
    # Independent reconstruction from successful optimizer transaction receipts.
    replay=Ledger(state['legal']);log=folder/('learning.jsonl' if component=='DP' else 'loss.jsonl')
    visit_replay=collections.Counter()
    if component=='DP':
     from visuotactile.native_data import visual_records
     from visuotactile.effective_exposure import dp_ticks
     contract=read(root/role/'continuation.json').get('target_window_contract')
     records=visual_records(root/role/'state_data',list(state['legal']),'G00',target_window_contract=contract)
    with log.open() as f:
     for line in f:
      r=json.loads(line);update=r['updates' if component=='DP' else 'update']
      if update>selected['updates']:break
      if component=='DP':
       actual=collections.defaultdict(set)
       for visit in r['visits']:
        identity=visit['episode'];visit_replay[identity]+=1
        actual[identity].update(dp_ticks(records[identity],visit['anchor_sampling']['selected_anchors']))
       assert {i:sorted(t) for i,t in actual.items()}==r['effective_tick_contributions']
      replay.step(update,list(r['effective_tick_contributions'].items()))
    for i in ledger.legal:assert torch.equal(ledger.counts[i],replay.counts[i])
    if component=='DP':assert dict(visit_replay)==counts
    name=role+'-'+component
    summary.append(dict(model=name,actual_epochs=last['epoch'],actual_updates=last['updates'],selected_epoch=selected['epoch'],
     selected_updates=selected['updates'],minimum_visits=receipt['minimum_visits'],minimum_tick_steps=receipt['minimum_tick_steps'],
     legal_ticks=receipt['legal_ticks'],eligible=True,selected_sha256=sha256(folder/'best.pt'),
     learning_rates=[g['lr'] for g in selected['optimizer']['param_groups']],receipt_replay_exact=True,
     actual_anchor_mask_recomputed=component=='DP',sha_scope='Selected complete resumable checkpoint'))
    histograms[name]=torch.cat([ledger.counts[i][m] for i,m in ledger.legal.items()]).numpy()
    for i,m in ledger.legal.items():
     x=ledger.counts[i];visits.append(dict(model=name,identity=i,new_visits=counts[i],legal_ticks=int(m.sum()),
      minimum_tick_steps=int(x[m].min()),median_tick_steps=float(x[m].double().median()),maximum_tick_steps=int(x[m].max())))
     for tick in range(len(m)):writer.writerow(dict(model=name,identity=i,tick=tick,legal=bool(m[tick]),optimizer_steps=int(x[tick])))
    for r in last['history']:
     curves.append(dict(model=name,epoch=r['epoch'],update=r['updates' if component=='DP' else 'update'],
      train=r['train']['selection'] if component=='DP' else r['train']['all']['normalized_equal_mse'],
      validation=r['validation']['selection'] if component=='DP' else r['selection_score'],
      eligible=r['selection_eligible'],minimum_tick_steps=r['effective_exposure']['minimum_tick_steps'],
      learning_rates=r.get('learning_rates'),training_objective=r.get('loss')))
    del selected,last,ledger,replay
 table(public/'training_completion.csv',summary);table(public/'selected_episode_exposure.csv',visits);table(public/'training_curves.csv',curves)
 write(public/'training_completion.json',dict(components=summary,scope='Only newly committed optimizer updates of each selected checkpoint; ancestors excluded, same-step duplicates removed',histograms={k:dict(zip(*[x.tolist() for x in np.unique(v,return_counts=True)])) for k,v in histograms.items()}))
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for ax,name in zip(axes.flat,histograms):
  rr=[r for r in curves if r['model']==name];ax.plot([r['update'] for r in rr],[r['train'] for r in rr],label='train2 diagnostic')
  ax.plot([r['update'] for r in rr],[r['validation'] for r in rr],label='validation6');ax.set_title(name);ax.set_xlabel('New optimizer updates');ax.set_ylabel('Normalized MSE');ax.legend();ax.grid(alpha=.25)
 fig.tight_layout();fig.savefig(public/'training_curves.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7))
 for ax,(name,x) in zip(axes.flat,histograms.items()):
  ax.hist(x,bins=35);ax.axvline(50,color='red',linestyle='--');ax.set_title(name+f' minimum={x.min()}');ax.set_xlabel('Distinct optimizer steps per legal tick');ax.set_ylabel('Ticks')
 fig.tight_layout();fig.savefig(public/'selected_exposure_histograms.png',dpi=160);plt.close(fig)
 print(json.dumps(summary),flush=True)


def evaluation():
 from tools.tight_fresh_rollouts import completed_pair
 c,root,parent,history,public=paths();training()
 assert (root/'project_complete.json').exists()
 pairs={name:completed_pair(p,role) for name,p,role in [('A72',parent,'A'),('B72',parent,'B'),('A130_50',root,'A'),('B130_50',root,'B')]}
 old=read(history.parent/'capability/plan.json');specs={x['spec']['seed']:x for x in old['cases'] if x['role']=='A'}
 dest=root/'evaluation';dest.mkdir();session=read(root/'session.json');session.update(resource_root=str(dest),owned_worker_roots=[str(dest)])
 assert session['deadline']>time.time();write(dest/'session.json',session);write(dest/'cuda_allocator_policy.json',dict(version='reclaim_unused_on_low_headroom_v1'))
 stages=[]
 for part,seed in enumerate([x for pair in zip(c['training_seeds'],c['development_seeds']) for x in pair]):
  wave=dest/f'wave{part:02d}';wave.mkdir();p=copy.deepcopy(old);p.update(pairs=pairs,cases=[],resource_root=str(dest),source_sha256=sources())
  names=list(pairs);names=names[part%4:]+names[:part%4]
  for name in names:
   x=copy.deepcopy(specs[seed]);x.update(role=name,name=name+'_'+x['spec']['identity'],video=seed in (28817004,28826004));p['cases'].append(x)
  p['batch']['lanes']=4
  p['evaluation_scope']=dict(protocol='Four complete A/B old/new policies paired within each wave, fixed train8/dev8',
   inference_rng='Native per-lane sampler7093; retained reset/event seeds',test_used=False,
   limitation='Opened development, observable initial state pairing; hidden physics and future randomness are not exact replay')
  write(wave/'session.json',session);write(wave/'plan.json',p);stages.append(dict(entry='evaluate',args=['--output',str(wave)]))
 job=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(dest),stages=stages)
 write(PROJECT/c['job_directory']/'evaluate.json',job);write(public/'evaluation_registration.json',dict(pairs=pairs,job=job,config=c))
 print('evaluation registered64',flush=True)


def report():
 import h5py
 from scripts.tight_layout_20261116.phase_report import summarize
 from visuotactile.grasp_curriculum import holding_metrics
 c,root,parent,history,public=paths();dest=root/'evaluation';assert (dest/'project_complete.json').exists()
 rows=[];pairing=[]
 for wave in sorted(dest.glob('wave*')):
  out=public/'evaluation'/wave.name;out.mkdir(parents=True,exist_ok=True);summarize(wave,out/'summary.json')
  for r in read(out/'summary.json')['rows']:
   folder=wave/r['case'];result=read(folder/'result.json');r['cohort']='training' if r['seed'] in c['training_seeds'] else 'development'
   assert r['legal'] and r['software_complete']
   with h5py.File(folder/'numeric.h5') as f:
    t=f['episode/transition'];r['hold30']=holding_metrics(t['diagnostics_after/object_pose_w'][:],t['object_contact_force_w'][:],window=30)
    command=t['target_after_rad'][:];position=t['diagnostics_after/joint_position_rad'][:]
    assert np.isfinite(command).all() and np.isfinite(position).all()
    delta=np.diff(command,axis=0);tracking=command-position
    r['execution_diagnostics']=dict(command_step_arm_max_rad=float(np.abs(delta[:,:7]).max()) if len(delta) else 0.,
     command_step_hand_max_rad=float(np.abs(delta[:,7:]).max()) if len(delta) else 0.,
     tracking_arm_rmse_rad=float(np.sqrt(np.mean(tracking[:,:7]**2))),
     tracking_hand_rmse_rad=float(np.sqrt(np.mean(tracking[:,7:]**2))),
     clipped_ticks=int(t['position_clip_mask'][:].reshape(len(command),-1).any(-1).sum()),
     scope='Observed command increments, clipping and actual tracking; not counterfactual expert error')
   r['stable_grasp']=result.get('contact',{}).get('stable_grasp_diagnostic')
   r['failure_stage']='success' if r['success'] else 'after_first_subgoal' if r['native_subgoals'] else 'after_lift' if r['native_lift'] else 'before_lift'
   rows.append(r)
  pairing.extend(read(wave/'paired_initial_check.json')['rows'])
  for name in ('plan.json','result.json','paired_initial_check.json','videos.json','wave_profiles.json'):shutil.copyfile(wave/name,out/name)
 assert len(rows)==64 and len({(r['role'],r['seed']) for r in rows})==64
 counts=[];changes=[]
 for cohort in ('training','development'):
  for name in c['evaluation_models']:
   rr=[r for r in rows if r['cohort']==cohort and r['role']==name];assert len(rr)==8
   counts.append(dict(cohort=cohort,model=name,n=len(rr),success=sum(r['success'] for r in rr),lift=sum(bool(r['native_lift']) for r in rr),
    grasp=sum(bool(r['stable_grasp']) for r in rr),**{f'hold{w}':sum(r[f'hold{w}']['sustained'] for r in rr) for w in (30,60,120)},
    first_subgoal=sum(r['native_subgoals']>0 for r in rr),insertion_subgoal=sum(r['native_subgoals']>=2 for r in rr)))
 for role in 'AB':
  for seed in c['training_seeds']+c['development_seeds']:
   a=next(r for r in rows if r['role']==role+'72' and r['seed']==seed);b=next(r for r in rows if r['role']==role+'130_50' and r['seed']==seed)
   changes.append(dict(role=role,seed=seed,cohort=a['cohort'],old_success=a['success'],new_success=b['success'],retained=a['success'] and b['success'],
    lost=a['success'] and not b['success'],gained=not a['success'] and b['success'],
    old_lift=a['native_lift'],new_lift=b['native_lift'],old_grasp=a['stable_grasp'],new_grasp=b['stable_grasp'],
    **{f'{when}_hold{w}':x[f'hold{w}']['sustained'] for when,x in [('old',a),('new',b)] for w in (30,60,120)}))
 table(public/'per_case.csv',rows);table(public/'counts.csv',counts);table(public/'paired_changes.csv',changes)
 write(public/'results.json',dict(counts=counts,changes=changes,pairing=pairing,hold60_is_success_requirement=False))
 print(json.dumps(counts),flush=True)


def videos():
 import av
 c,root,parent,history,public=paths();dest=public/'videos';dest.mkdir(exist_ok=True)
 rows=[];index=['# 预登记闭环视频','',
  '训练28817004、开发28826004在结果产生前登记；四个完整策略各两例。真实三视角逐控制tick录制，完整自然结束，不增加评测分母。','']
 for wave in sorted((root/'evaluation').glob('wave*')):
  for row in read(wave/'videos.json')['rows']:
   source=wave/row['case']/row['file'];target=dest/(row['case']+'.mp4')
   with av.open(str(source)) as movie:
    frames=sum(1 for _ in movie.decode(video=0))
   assert frames==row['frames']==row['decode']['decoded_frames']
   digest=sha256(source)
   if target.exists():assert sha256(target)==digest
   else:shutil.copyfile(source,target)
   assert sha256(target)==digest
   rows.append(dict(case=row['case'],success=row['success'],frames=frames,sha256=digest,source=str(source),file=str(target.relative_to(public))))
   index.append(f"- [{target.name}]({target.name})：成功={row['success']}，{frames}帧，完整解码通过。")
 assert len(rows)==8
 write(public/'video_receipts.json',dict(rows=rows,selection='Two preregistered seeds; not selected after viewing outcomes'))
 (dest/'README.md').write_text('\n'.join(index)+'\n')


@torch.no_grad()
def offline():
 from tools.tight_representation_probe import source
 from visuotactile.models.native_dp import load_residual
 from visuotactile.geometry_sampling import STAGES
 c,root,parent,history,public=paths();rows=[]
 data=read(root/'B/state_data/data.json');original=set(data['normalization_train_ids']);old=set(read(parent/'B/state_data/data.json')['splits']['train']);val=set(data['splits']['validation'])
 for role in 'AB':
  for label,tree in [('72',parent),('130_50',root)]:
   model=load_residual(torch.load(tree/role/f'backend/{role}/training/best.pt',map_location='cpu',weights_only=False)).eval()
   cache=tree/role/f'backend/{role}/nominal';index=read(cache/'index.json')
   ids=read(tree/role/'state_data/data.json')['splits'];ids=ids['train']+ids['validation']
   for j,i in enumerate(ids):
    p=cache/(i+'.pt');assert sha256(p)==index['episodes'][i]['sha256'];r=torch.load(p,map_location='cpu',weights_only=False)
    raw=source(data['entries'][i]);truth=torch.from_numpy(raw['command']);actual=torch.from_numpy(raw['q_reached'])
    assert np.array_equal(r['supervision_mask'].numpy(),raw['legal'])
    assert torch.equal(r['target'][r['supervision_mask']],truth[r['supervision_mask']])
    out=predictions(model,r);masks=dict(legal=raw['legal'],first24=raw['onset']);masks.update({name:raw['legal']&(raw['phase']==k) for k,name in enumerate(STAGES)})
    cohort='validation6' if i in val else 'original30' if i in original else 'old42' if i in old else 'new58'
    for stage,mask in masks.items():
     if not mask.any():continue
     for component in ('dp_own_target','dp_command','anchor','correction','final'):
      pred=out['dp'] if component.startswith('dp_') else out[component]
      target=actual if component=='dp_own_target' and role=='A' else truth-out['anchor'] if component=='correction' else truth
      error=(pred-target)[mask];span=out['span'][mask]
      rows.append(dict(model=role+label,identity=i,family=data['entries'][i]['family'],cohort=cohort,stage=stage,component=component,ticks=len(error),
       arm_squared_sum=float(error[:,:7].square().sum()),hand_squared_sum=float(error[:,7:].square().sum()),
       arm_span_squared_sum=float((error[:,:7]/span[:,:7]).square().sum()),hand_span_squared_sum=float((error[:,7:]/span[:,7:]).square().sum())))
    if j%20==0:print('offline',role+label,j,flush=True)
 table(public/'offline_per_episode.csv',rows);totals=collections.defaultdict(collections.Counter)
 for r in rows:totals[r['model'],r['cohort'],r['stage'],r['component']].update({k:r[k] for k in ('ticks','arm_squared_sum','hand_squared_sum','arm_span_squared_sum','hand_span_squared_sum')})
 out=[]
 for (model,cohort,stage,component),v in totals.items():
  out.append(dict(model=model,cohort=cohort,stage=stage,component=component,ticks=v['ticks'],
   arm_rmse_rad=(v['arm_squared_sum']/v['ticks']/7)**.5,hand_rmse_rad=(v['hand_squared_sum']/v['ticks']/22)**.5,
   arm_span_rmse=(v['arm_span_squared_sum']/v['ticks']/7)**.5,hand_span_rmse=(v['hand_span_squared_sum']/v['ticks']/22)**.5))
 table(public/'offline_summary.csv',out)
 write(public/'offline_scope.json',dict(old_model_episodes=78,new_model_episodes=136,comparison='Shared original30/old42/validation6 only; new58 has no baseline recomputation',
  teacher_forcing='Actual recorded histories, legal expert mask only; no counterfactual labels for student prefixes',
  dp_A_target='q_reached',dp_B_target='u_expert',final_target='u_expert',new58_is_test=False))


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['training','evaluation','report','offline','videos']);a=p.parse_args();globals()[a.mode]()
