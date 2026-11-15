"""One small read-only night status: heartbeat, actual progress and resource headroom."""
import argparse,json,shutil,time
from pathlib import Path

def read(path):
 try:return json.loads(path.read_text())
 except FileNotFoundError:return None

def status(root):
 now=time.time();state=read(root/'night/state.json') or {}
 ledger=Path(state['stage_ledger']) if state.get('stage_ledger') else root/'night'
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
 rows=[]
 for directory in sorted(root.glob('round_*')):
  if not directory.is_dir() or directory.name.endswith('_stages'):continue
  stage=read(directory.with_name(directory.name+'_stages')/'status.json')
  if stage:rows.append(dict(round=directory.name,phase=stage.get('stage'),error=stage.get('error')))
  for role in 'AB':
   paths=[directory/role/'training/progress.json',directory/role/'backend/cache_progress.json',
          directory/role/'backend'/role/'training/progress.json']
   for path in paths:
    row=read(path)
    if row:
     rows.append(dict(path=str(path.relative_to(root)),age_seconds=now-path.stat().st_mtime,
       **{k:row[k] for k in ('epoch','updates','supervised_ticks','episodes','total','elapsed_s','loss') if k in row}))
 for directory in sorted(root.glob('new_family_*')):
  row=read(directory/'result.json') or read(directory/'batch_progress.json')
  if row:rows.append(dict(path=str(directory.relative_to(root)),
   **{k:row[k] for k in ('software_complete','worker_error','accepted_training_trajectories','elapsed_seconds',
    'tick','active_count','remaining_tasks','closed_results','verifying','active_lane_steps') if k in row}))
 result['progress']=rows;return result

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 print(json.dumps(status(a.root.resolve()),ensure_ascii=False,indent=2))
