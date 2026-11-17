"""Read-only lightweight publication export for the completed Nov17 study.

No model load, optimizer, simulation, raw image export, or source mutation.
Run with Isaac python for h5py; output is repository evidence only.
"""
import argparse, collections, csv, datetime, gzip, hashlib, json, pathlib, shutil, subprocess
P=pathlib.Path
BASE='14815ae5ded77a590a5baf16307866baf5b1aef7'
def read(p):return json.loads(P(p).read_text())
def sha(p):return hashlib.sha256(P(p).read_bytes()).hexdigest()
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def table(p,rows):
 p.parent.mkdir(parents=True,exist_ok=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=keys,lineterminator="\n");w.writeheader();w.writerows([{k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])
def iso(t):return datetime.datetime.fromtimestamp(t,datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def main(root,out):
 import h5py,numpy as np
 out.mkdir(parents=True,exist_ok=True);inventory=[]
 def copy(p,q):
  p=P(p);q=out/q;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
  inventory.append(dict(source=str(p),published=str(q),source_sha256=sha(p),bytes=p.stat().st_size,kind='verbatim'))
 stages={'midpoint':root/'midpoint','new65':root/'round04_pool/learning','old56':root/'matched_old56','round05':root/'round05/learning','round06':root/'round06/learning'}
 manifests={};models=[];timing=[];io=[];visits=[];matrix=[]
 # Freeze actual manifests, configurations, raw numeric curves and update exposure.
 for stage,folder in stages.items():
  for role in 'AB':
   d=read(folder/role/'state_data/data.json');manifests[stage,role]=d
   for rel in ['state_data/data.json','continuation.json','backend_continuation.json','training/configuration.json','training/finished.json','training/extension_restore.json','training/schedule_restart.json','backend/configuration.json','backend/base_identities.json','backend/'+role+'/training/configuration.json','backend/'+role+'/training/finished.json','backend/'+role+'/training/selected.json','backend/'+role+'/nominal/index.json','cache_index.json']:
    p=folder/role/rel
    if p.exists():copy(p,P('runs')/stage/role/rel)
   for component,train in [('DP',folder/role/'training'),('Transformer',folder/role/'backend'/role/'training')]:
    finished=read(train/'finished.json');actual=finished['updates'];rows=[];counts=collections.Counter()
    path=train/('learning.jsonl' if component=='DP' else 'loss.jsonl')
    for line in path.open():
     x=json.loads(line)
     if component=='DP':
      # Keep every scalar metric, exact sampled anchors and visits; omit large geometry debug arrays only.
      for v in x.get('visits',[x]):
       counts[v['episode']]+=1
       v.pop('geometry',None)
     rows.append(x)
    q=out/'curves'/f'{stage}_{role}_{component}_updates.jsonl.gz';q.parent.mkdir(exist_ok=True)
    with gzip.GzipFile(filename=str(q),mode='wb',mtime=0) as f:
     for x in rows:f.write((json.dumps(x,separators=(',',':'))+'\n').encode())
    inventory.append(dict(source=str(path),published=str(q),source_sha256=sha(path),bytes=q.stat().st_size,kind='all update rows; DP per-visit geometry debug arrays omitted'))
    if component=='DP':
     assert set(counts)==set(d['splits']['train'])
     visits.extend(dict(stage=stage,role=role,trajectory_id=i,actual_visits=n) for i,n in counts.items())
     vals=[json.loads(s) for s in (train/'validation.jsonl').read_text().splitlines()]
     # Full-state resume includes old epochs in these logs. Separate newly measured interval.
     start=read(folder/'logs'/f'dp_{role}_started.json')['utc']
     for x in vals:
      e=x.get('training_epoch_io',{})
      if e.get('started_utc',0)>=start:
       io.append(dict(stage=stage,role=role,epoch=x['epoch'],updates=x['updates'],**e))
    else:vals=read(train/'validation.json')
    q=out/'curves'/f'{stage}_{role}_{component}_validation.json.gz'
    with gzip.GzipFile(filename=str(q),mode='wb',mtime=0) as f:f.write(json.dumps(vals,separators=(',',':')).encode())
   binding=read(folder/role/'backend'/role/'nominal/index.json');sel=read(folder/role/'backend'/role/'training/selected.json')
   assert binding['base_sha256']==sel['base_checkpoint_sha256']
   assert set(binding['episodes'])>=set(d['splits']['train'])
   models.append(dict(stage=stage,role=role,policy_id=binding['base_sha256'][:12]+'+'+sel['checkpoint_sha256'][:12],frontend_sha256=binding['base_sha256'],backend_sha256=sel['checkpoint_sha256'],cache_base_sha256=binding['base_sha256'],source_root=str(folder),parent=read(folder/role/'continuation.json'),backend_parent=(read(folder/role/'backend_continuation.json') if (folder/role/'backend_continuation.json').exists() else read(folder/role/'continuation.json').get('previous_backend')),source_session_commit=read(folder/'session.json').get('source_commit')))
  assert manifests[stage,'A']['splits']==manifests[stage,'B']['splits']
  for name in ['session.json','prepared.json','training_sampling.json','full_trajectory_batch.json','bounded_extension.json','frozen_visual_io.json','complete.json']:
   if (folder/name).exists():copy(folder/name,P('runs')/stage/name)
 table(out/'training_visits.csv',visits);write(out/'models.json',models);write(out/'epoch_io.json',io)
 # Exact supervision boundaries from small HDF5 provenance arrays (no RGB-D read).
 entries=manifests['round06','A']['entries'];records=[]
 for identity,e in entries.items():
  correction=e.get('trajectory_kind')=='expert_takeover_v1';directory=P(e['path']).parent.parent
  row=dict(trajectory_id=identity,**e,source_role=e.get('source_group','original_expert'),source_complete_policy=None)
  with h5py.File(e['path']) as h:
   g=h['episodes'][e.get('episode',identity)]
   if correction:
    mask=g['transition/dagger/expert_label_valid'][:];control=g['transition/dagger/control_source'][:];clean=g['transition/dagger/switch_clean'][:]
    assert not np.any(mask & ((control!=1)|~clean))
   else:mask=np.ones(e['length'],dtype=bool)
  ids=np.flatnonzero(mask);bounds=[]
  for t in ids:
   if not bounds or t!=bounds[-1][1]:bounds.append([int(t),int(t)+1])
   else:bounds[-1][1]+=1
  row.update(supervision_valid_half_open_ranges=bounds,valid_expert_ticks=len(ids),mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),history_range=[0,e['length']],range_basis='HDF5 expert_label_valid' if correction else 'complete original expert demonstration')
  if correction:
   plan=read(directory.parent/'plan.json');pair=plan['pairs'][e['source_group']];row['source_complete_policy']=pair
   for name in ['capture_audit.json','result_before_admission.json','result.json']:
    if (directory/name).exists():copy(directory/name,P('provenance')/identity/name)
  records.append(row)
 write(out/'trajectory_manifest.json',records)
 for stage in ['midpoint','new65','round05','round06']:
  d=manifests[stage,'A'];n=len(d['splits']['train']);selected=[x for x in records if x['trajectory_id'] in d['splits']['train']]
  write(out/f'manifest_{n}.json',dict(stage=stage,splits=d['splits'],identical_actual_A_B_splits=True,records=selected,validation_ids=d['splits']['validation']))
  c=collections.Counter(x['source_role'] for x in selected)
  matrix.append(dict(pool=n,original=c['original_expert'],A_source=c['A'],B_source=c['B'],used_A=n,used_B=n))
 table(out/'source_matrix.csv',matrix)
 # Physical opportunities and actual full policy provenance, not successful suffixes alone.
 opportunities=[]
 for stage,folder in [('round04',root/'round04_pool/collection'),('round05',root/'round05/collection'),('round06',root/'round06/collection')]:
  plan=read(folder/'plan.json');report=read(out.parent/(stage+'_collection.json'))
  for name in ['plan.json','correction_entries.json','result.json','paired_initial_check.json']:
   copy(folder/name,P('collection')/stage/name)
  for x in report['rows']:
   pair=plan['pairs'][x['role']];opportunities.append(dict(round=stage,policy_id=pair['frontend_sha256'][:12]+'+'+pair['backend_sha256'][:12],**{k:x.get(k) for k in ['case','role','seed','source_family','requested_stage','legal','takeover_tick','takeover_phase','expert_rescue','admitted','valid_expert_ticks','steps','reasons']},trigger_reason=x.get('actual_trigger',{}).get('reason')))
 table(out/'collection_opportunities.csv',opportunities)
 # Every autonomous result, pairing receipts and full model plans.
 evaluations=[]
 for planfile in root.glob('**/plan.json'):
  folder=planfile.parent
  if folder.is_symlink() or not (folder/'results.json').exists():continue
  plan=read(planfile)
  if plan.get('kind')!='autonomous':continue
  rel=folder.relative_to(root)
  for name in ['plan.json','result.json','paired_initial_check.json','wave_profiles.json']:
   if (folder/name).exists():copy(folder/name,P('evaluations')/rel/name)
  for x in read(folder/'results.json'):
   z=x['result'];role=x['role'];pair=plan['pairs'][role]
   assert z['checkpoint_sha256']==pair['backend_sha256'], 'Executed rear differs from registered full pair'
   evaluations.append(dict(cohort=str(rel),case=x['case'],role=role,policy_id=pair['frontend_sha256'][:12]+'+'+pair['backend_sha256'][:12],initial_id=z.get('seed'),family=z.get('family'),legal=z.get('legal'),steps=z.get('steps'),success=z.get('success'),native_subgoals=z.get('native_subgoals'),reasons=z.get('reasons'),software_complete=z.get('software_complete'),video_local=str(folder/x['case']/'autonomous.mp4') if (folder/x['case']/'autonomous.mp4').exists() else None))
 table(out/'autonomous_all.csv',evaluations)
 final=[];events={x['case']:x for x in read(out.parent/'final_test.json')['rows']}
 for x in evaluations:
  if x['cohort'].startswith('final_test/part'):
   z=events[x['case']];final.append(dict(**x,hold120=z['hold120']['sustained'],hold120_first=z['hold120']['first_completed_tick'],first_contact=z['first_contact_completed_tick']))
 assert len(final)==40 and len({(r['role'],r['initial_id']) for r in final})==40
 table(out/'final40.csv',final)
 success={r:sorted(x['initial_id'] for x in final if x['role']==r and x['success']) for r in 'AB'}
 write(out/'final_events.json',dict(success_ids=success,intersection=sorted(set(success['A'])&set(success['B'])),success_without_hold120=[x['case'] for x in final if x['success'] and not x['hold120']],subgoal_without_hold120=[x['case'] for x in final if x['native_subgoals'] and not x['hold120']]))
 # Timeline receipts; interval union instead of sum of overlapping owner lifetimes.
 for p in root.glob('**/logs/*_started.json'):
  q=p.with_name(p.name.replace('_started.json','_complete.json'))
  if not q.exists():continue
  s,e=read(p),read(q);a,b=s.get('utc'),e.get('utc')
  if not isinstance(a,(int,float)) or not isinstance(b,(int,float)):continue
  stage=p.stem[:-8];category=('DP_train_validation' if stage.startswith('dp_') else 'Transformer_train_validation' if stage.startswith('rear_train') else 'Transformer_cache' if stage.startswith('rear_cache') else 'frozen_visual_cache' if stage=='cache' else 'autonomous_evaluation' if stage=='evaluation' or stage.startswith('part') or stage in ['run','batch'] else 'preparation_or_diagnostic')
  timing.append(dict(task=str(p.parent.parent.relative_to(root))+'/'+stage,category=category,start=a,end=b,start_plus08=iso(a),end_plus08=iso(b),seconds=b-a,exit_code=e.get('exit_code'),command=s.get('command'),source=str(p.relative_to(root))))
  copy(p,P('timing_receipts')/p.relative_to(root));copy(q,P('timing_receipts')/q.relative_to(root))
 # Collections have dedicated owner receipts; report elapsed is the physical loop+admission.
 for folder in [root/'round04_recovery01_part0',root/'round04_recovery01_part1',root/'round05',root/'round06']:
  s=read(folder/'collection/run_started.json');e=read(folder/'collection/result.json');exit_=read(folder/'collection_owner_exit.json')
  start=s.get('utc',s.get('started_utc'));end=exit_['utc']
  if start is not None:timing.append(dict(task=str(folder.relative_to(root))+'/collection',category='expert_collection',start=start,end=end,start_plus08=iso(start),end_plus08=iso(end),seconds=end-start,exit_code=0,source=str(folder/'collection/run_started.json')))
  for name in ['collection/run_started.json','collection/result.json','collection_owner_exit.json','collection/kit_started.json','collection/eval_startup.json']:
   if (folder/name).exists():copy(folder/name,P('timing_receipts')/folder.relative_to(root)/name)
 # Final test uses owner receipts rather than logs/*_complete.json.
 for name,category,startfile,endfile in [('final_test','final_test','owner_started.json','owner_exit.json'),('round04','software_recovery','owner_started.json','collection_owner_exit.json')]:
  a=read(root/name/startfile)['utc'];b=read(root/name/endfile)['utc']
  timing.append(dict(task=name,category=category,start=a,end=b,start_plus08=iso(a),end_plus08=iso(b),seconds=b-a,source=name+'/'+startfile))
  for rel in [startfile,endfile,'registration.json','prepared.json','logs/collection.log']:
   if (root/name/rel).exists():copy(root/name/rel,P('timing_receipts')/name/rel)
 table(out/'timeline.csv',timing)
 begin=read(root/'registration.json')['started_utc'] if 'started_utc' in read(root/'registration.json') else read(root/'midpoint/session.json')['started_utc']
 end=read(root/'final_test/owner_exit.json')['utc']
 points=sorted({begin,end}|{max(begin,min(end,r[k])) for r in timing for k in ['start','end']})
 totals=collections.Counter();overlaps=[]
 priority=['DP_train_validation','Transformer_train_validation','Transformer_cache','frozen_visual_cache','expert_collection','autonomous_evaluation','preparation_or_diagnostic','final_test','software_recovery']
 for a,b in zip(points,points[1:]):
  live=[r for r in timing if r['start']<=a and r['end']>=b]
  cat=min((r['category'] for r in live),key=priority.index) if live else 'unattributed_gap'
  totals[cat]+=b-a
  if len(live)>1:overlaps.append(dict(start=a,end=b,tasks=[r['task'] for r in live],assigned=cat))
 write(out/'time_account.json',dict(start_plus08=iso(begin),end_plus08=iso(end),wall_seconds=end-begin,exclusive_seconds=dict(totals),overlaps=overlaps,scope='Measured receipt intervals, overlapping owners not summed. Unattributed includes human analysis and uninstrumented setup; NOT all idle or I/O wait.'))
 write(out/'export_inventory.json',inventory)
 supplement(root,out)
 print(json.dumps(dict(models=len(models),trajectories=len(records),final=len(final),timeline=len(timing),time=dict(totals),matrix=matrix),indent=2))
def supplement(root,out):
 # Resolve each actual launcher tree; inherited session.source_commit can be stale.
 models=read(out/'models.json')
 for x in models:
  d=read(P(x['source_root'])/'logs'/('dp_'+x['role']+'_started.json'))
  tree=P(d['command'][0]).parents[2]
  x['actual_runtime_commit']=subprocess.check_output(['git','-C',str(tree),'rev-parse','HEAD'],text=True).strip()
  x['runtime_command']=d['command']
 write(out/'models.json',models)
 # Per-stage budget table uses existing audited checkpoint receipts, not reloaded weights.
 budgets=[]
 for stage,name,needle in [('baseline56','baseline_budget','round03/learning'),('midpoint','round04_budget','midpoint'),('old56','matched_old_budget','matched_old56'),('new65','round04_budget','round04_pool/learning'),('round05','round05_budget','round05/learning'),('round06','round06_budget','round06/learning')]:
  report=read(out.parent/(name+'.json'))
  for role,parts in report['roles'].items():
   for component,rows in parts.items():
    x=next(v for v in rows if '/'+needle+'/' in v['checkpoint'])
    budgets.append(dict(stage=stage,role=role,component=component,**x))
 table(out/'training_budget.csv',budgets)
 # Actual collection source plans for historical01-03 and current04-06.
 sources={}
 for row in read(out/'trajectory_manifest.json'):
  if row.get('trajectory_kind')=='expert_takeover_v1':
   p=P(row['path']).parent.parent.parent;sources[str(p)]=row['round_index']
 opportunities=[];pairs={}
 for p,n in sources.items():
  p=P(p);plan=read(p/'plan.json');pairs[str(n)]=plan['pairs']
  for x in read(p/'results.json'):
   r=x['result'];d=r.get('dagger',{});i=x['case'];role=x['role'];pair=plan['pairs'][role]
   opportunities.append(dict(round=n,case=i,role=role,policy_id=pair['frontend_sha256'][:12]+'+'+pair['backend_sha256'][:12],legal=r.get('legal'),takeover=bool(d.get('expert_ticks',0)),success=bool(d.get('expert_takeover_success')),valid_expert_ticks=d.get('expert_valid_ticks',0),admitted=r.get('accepted_for_training',False),source_results=str(p/'results.json')))
  dest=out/'collection'/('historical_'+str(n)+'_'+p.parent.name);dest.mkdir(parents=True,exist_ok=True)
  for name in ['plan.json','results.json','result.json','paired_initial_check.json','correction_entries.json']:
   if (p/name).exists():shutil.copyfile(p/name,dest/name)
 assert len({x['case'] for x in opportunities})==len(opportunities)
 table(out/'collection_all_rounds.csv',opportunities);write(out/'collection_source_models.json',pairs)
 summary=[]
 for n in sorted(set(x['round'] for x in opportunities)):
  for role in 'AB':
   a=[x for x in opportunities if x['round']==n and x['role']==role]
   summary.append(dict(round=n,role=role,opportunities=len(a),legal=sum(bool(x['legal']) for x in a),takeovers=sum(x['takeover'] for x in a),rescues=sum(x['success'] for x in a),valid_expert_ticks=sum(x['valid_expert_ticks'] for x in a)))
 table(out/'collection_counts.csv',summary)
 # Update and epoch time components are nested, never additional session wall time.
 io=read(out/'epoch_io.json');profile=[]
 for path in sorted((out/'curves').glob('*_DP_updates.jsonl.gz')):
  stage,role=path.name.rsplit('_DP_',1)[0].rsplit('_',1)
  starts=[x['started_utc'] for x in io if x['stage']==stage and x['role']==role];start=min(starts)
  rows=[json.loads(l) for l in gzip.open(path,'rt')];rows=[x for x in rows if x.get('utc',0)>=start]
  epochs=[x for x in io if x['stage']==stage and x['role']==role]
  profile.append(dict(stage=stage,role=role,new_updates=len(rows),training_epoch_seconds=sum(x['wall_seconds'] for x in epochs),physical_read_bytes=sum(x['process_io']['read_bytes'] for x in epochs),explicit_frozen_input_wait_seconds=sum(x.get('frozen_input_wait_s',0) for x in rows),update_host_seconds=sum(x.get('update_host_s',0) for x in rows),batch_elapsed_seconds=sum(x['seconds'] for x in rows),timing_scope='Nested counters. mmap page faults can occur inside update_host; explicit load wait does not capture all disk stalls.'))
 table(out/'training_io_summary.csv',profile)
 # Same observed initial states, all eight old/new matched results on one row.
 ev=list(csv.DictReader((out/'autonomous_all.csv').open()));matched={}
 for x in ev:
  if x['cohort'].startswith('matched_comparison'):
   matched.setdefault(x['initial_id'],{})[x['role']]=x['success']
 table(out/'matched8.csv',[dict(initial_id=k,**v) for k,v in sorted(matched.items())])
 # Full original videos are already only 5-10s; export three without re-encoding.
 videos=read(out.parent/'video_index.json')['videos'];media=[]
 for v in videos:
  p=P(v['path']);q=None
  if '/final_test/part0/' in str(p):q=out/'media'/('final_'+v['role']+'_28823000.mp4')
  elif 'phase_post_subgoal/student_then_expert.mp4' in str(p):q=out/'media'/'expert_rescue_not_autonomous.mp4'
  row=dict(v,remote_access='local research machine only; no public hosting',published=None)
  if q:
   q.parent.mkdir(exist_ok=True);shutil.copyfile(p,q);assert sha(q)==v['sha256'];row.update(published=str(q.relative_to(out)),remote_access='GitHub repository raw download',bytes=q.stat().st_size)
   sheet=v.get('contactsheet',v.get('export_metadata',{}).get('sheet'))
   if sheet and P(sheet).exists():shutil.copyfile(sheet,q.with_suffix('.jpg'));row['keyframes']=str(q.with_suffix('.jpg').relative_to(out))
  media.append(row)
 write(out/'media_index.json',media)
 # Convenient raw-number epoch table and event exceptions (diagnostics are not nested gates).
 epochs=[]
 for path in sorted((out/'curves').glob('*_validation.json.gz')):
  stage,role,component=path.name.split('_validation')[0].rsplit('_',2)
  for v in json.loads(gzip.open(path,'rt').read()):
   row=dict(stage=stage,role=role,component=component,epoch=v['epoch'],updates=v.get('updates',v.get('update')),train_loss=v.get('loss'),selection=v.get('selection_score',v.get('validation',{}).get('selection')))
   for cohort in ['train','validation','correction']:
    z=v.get(cohort,{})
    if 'all' in z:z=z['all']
    row.update({cohort+'_'+k:z.get(k) for k in ['arm_rmse_rad','hand_rmse_rad','normalized_equal_mse','ticks']})
   epochs.append(row)
 table(out/'epoch_curves.csv',epochs)
 f=read(out/'final_events.json');raw=read(out.parent/'final_test.json')['rows']
 f.update(success_without_native_lift=[x['case'] for x in raw if x['insertion'] and not x['native_lift']],interpretation='All six successes have hold120 and native subgoals, but none meets native_lift diagnostic. Do not impose a nested stage funnel or change native success.')
 write(out/'final_events.json',f)
 final=list(csv.DictReader((out/'final40.csv').open()))
 for r in final:
  r['video_remote']=('media/final_'+r['role']+'_28823000.mp4') if r['initial_id']=='28823000' else 'not_recorded'
  r['first_subgoal_reached']=int(r['native_subgoals'])>=1
 table(out/'final40.csv',final)
 wave_rows=[]
 for path in (out/'evaluations').glob('**/wave_profiles.json'):
  values=read(path)
  if isinstance(values,dict):values=values.get('waves',values.get('profiles',[]))
  for v in values:
   wave_rows.append(dict(cohort=str(path.parent.relative_to(out/'evaluations')),**{k:v.get(k) for k in ['wave','cases','seconds','control_ticks','controller_s','environment_s','expert_s','record_submit_s','finish_audit_s','reset_refill_s','admission_s','aggregate_control_ticks_per_second','active_lane_fraction']},allocated_lanes=8))
 table(out/'simulation_profiles.csv',wave_rows)
 # Whole base-commit evidence inventory and every absolute reference inside it.
 tracked=subprocess.check_output(['git','ls-tree','-r','--name-only',BASE,'research_docs/evidence/tight_long_20261117'],text=True).splitlines();old=[];refs=set()
 def strings(x):
  if isinstance(x,str):
   if x.startswith('/home/'):refs.add(x)
  elif isinstance(x,dict):
   for v in x.values():strings(v)
  elif isinstance(x,list):
   for v in x:strings(v)
 for name in tracked:
  raw=subprocess.check_output(['git','show',BASE+':'+name]);old.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),uploaded_at=BASE))
  if name.endswith('.json'):strings(json.loads(raw))
 table(out/'previous_upload_inventory.csv',old)
 reference=[]
 for name in sorted(refs):
  p=P(name);status='local_directory' if p.is_dir() else 'local_file' if p.is_file() else 'missing_or_nonpath_string'
  ignored=subprocess.run(['git','-C','/home/CNS2025331827/docs/bc+rl_RSS','check-ignore',str(p)],capture_output=True,text=True).returncode==0
  reference.append(dict(path=name,status=status,ignored_in_source_tree=ignored,remote_raw_payload=False))
 table(out/'previous_local_references.csv',reference)
 # Historical windows are context, not silently included in this session.
 previous=[]
 for name in ['tight_layout_20261116_01','tight_layout10_20261116_01','tight_layout10_20261116_learning01','tight_layout10_20261116_dagger01','tight_phase_20261116_01']:
  for rel in ['session.json','owner_started.json','owner_exit.json','complete.json']:
   p=root.parent/name/rel
   if p.exists():
    d=read(p);previous.append(dict(run=name,receipt=rel,timestamp=d.get('utc',d.get('started_utc')),note='session timestamp can be inherited; owner is actual launch'))
 write(out/'prior_windows.json',previous)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=P,required=True);p.add_argument('--output',type=P,required=True);a=p.parse_args();main(a.root,a.output)
