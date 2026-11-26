"""One small read-only night status: heartbeat, actual progress and resource headroom."""
import argparse,json,shutil,time
from pathlib import Path

def read(path):
 try:return json.loads(path.read_text())
 except FileNotFoundError:return None

def status(root, *, all_rounds=False):
 now=time.time();state=read(root/'night/state.json') or {}
 diagnostic=not state and (root/'fixed_fit_owner/state.json').exists()
 if diagnostic:state=read(root/'fixed_fit_owner/state.json')
 ledger=Path(state['stage_ledger']) if state.get('stage_ledger') else root/('fixed_fit_owner' if diagnostic else 'night')
 heartbeat=read(ledger/'owner_status.json') or {}
 resource=read(root/'shared_resources.json') or {}
 result=dict(root=str(root),observed_utc=now,stage=state.get('stage'),owner_pid=state.get('owner_pid'),
  safe_stop=state.get('safe_stop',False),error=state.get('error'),
  completed_rounds=[r['index'] for r in state.get('rounds',[]) if r.get('complete')],
  unattended_release_exists=(root/'night/supervised_release.json').exists(),
  stage_worker_pid=heartbeat.get('worker_pid'),
  heartbeat_age_seconds=now-heartbeat['heartbeat'] if 'heartbeat' in heartbeat else None,
  stale_seconds=heartbeat.get('stale_seconds'),deadline=state.get('deadline'),
  resources=dict(telemetry_age_seconds=now-resource['utc_epoch'] if 'utc_epoch' in resource else None,
   gpu_free_mib=resource.get('gpu_free_mib'),ram_available_gib=resource.get('ram_available_bytes',0)/2**30,
   disk_free_gib=shutil.disk_usage(root).free/2**30,pause_reasons=resource.get('pause_reasons')))
 result['scope']='fixed_fit_closed_loop_diagnosis' if diagnostic else 'supervised_grasp_night'
 result['owner_process_matches_root']=False
 if isinstance(state.get('owner_pid'),int):
  try:
   process=Path('/proc',str(state['owner_pid']))
   args=(process/'cmdline').read_bytes().decode().split('\0')
   result['owner_process_matches_root']=str(root) in args and (process/'stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
  except (OSError,IndexError):pass
 rows=[]
 for directory in sorted(root.glob('round_*')):
  if not directory.is_dir() or directory.name.endswith('_stages'):continue
  stage=read(directory.with_name(directory.name+'_stages')/'status.json')
  if stage:rows.append(dict(round=directory.name,phase=stage.get('stage'),error=stage.get('error')))
  if stage and stage.get('stage')=='complete' and not all_rounds:continue
  for role in 'AB':
   paths=[directory/role/'training/progress.json',directory/role/'backend/cache_progress.json',
          directory/role/'backend'/role/'training/progress.json']
   for path in paths:
    finished=read(path.with_name('finished.json')) if path.name=='progress.json' else None
    row=finished or read(path)
    if row:
     done=bool(finished) or (row.get('episodes') is not None and row.get('episodes')==row.get('total'))
     rows.append(dict(path=str((path.with_name('finished.json') if finished else path).relative_to(root)),
       state='complete' if done else 'running',age_seconds=None if done else now-path.stat().st_mtime,
       **{k:row[k] for k in ('epoch','epochs','updates','update','control_tick_epochs','supervised_ticks','episodes','total','elapsed_s','loss') if k in row}))
 for directory in sorted(root.glob('new_family_*'))+[root/'fixed_fit_source_evaluation',root/'fixed_fit_evaluation']:
  row=read(directory/'result.json') or read(directory/'batch_progress.json')
  if row:rows.append(dict(path=str(directory.relative_to(root)),
   **{k:row[k] for k in ('software_complete','worker_error','accepted_training_trajectories','elapsed_seconds',
    'tick','active_count','remaining_tasks','closed_results','verifying','active_lane_steps') if k in row}))
 fit_status=read(root/'fixed_fit_stages/owner_status.json')
 if fit_status:result['independent_fixed_fit']=fit_status
 for role in 'AB':
  path=root/'fixed_fit'/role/'progress.json';row=read(path)
  if row:rows.append(dict(path=str(path.relative_to(root)),**row))
 fit_finished=read(root/'fixed_fit_stages/finished.json')
 if fit_finished:
  result['fixed_fit_software_complete']=fit_finished.get('software_complete')
  result['independent_fixed_fit']=dict(stage='complete',utc=fit_finished.get('utc'),
   scope=fit_finished.get('scope'))
 result['progress']=rows;return result

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
 p.add_argument('--all-rounds',action='store_true');a=p.parse_args()
 print(json.dumps(status(a.root.resolve(),all_rounds=a.all_rounds),ensure_ascii=False,indent=2))
