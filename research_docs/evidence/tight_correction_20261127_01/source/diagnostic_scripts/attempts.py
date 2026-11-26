from pathlib import Path
import json,csv,time
from visuotactile.training_resources import atomic_json
R=Path('outputs/visuotactile/run/tight_correction_20261127_01');D=Path('outputs/diagnostics/tight_correction_20261127_01');read=lambda p:json.loads(p.read_text());rows=[]
for start_file in sorted(R.rglob('project_job_started.json')):
 owner=start_file.parent
 begin=read(owner/'project_job_started.json');job=begin['job'];failed=read(owner/'project_failure.json') if (owner/'project_failure.json').exists() else None
 for index,stage in enumerate(job['stages']):
  log=owner/f'project_stage_{index:02d}.log';args=stage['args']
  if '--output' not in args or not log.exists():continue
  output=Path(args[args.index('--output')+1]);output=owner/'rollouts' if output.name=='rollouts' and (owner/'rollouts/plan.json').exists() else output;plan=read(output/'plan.json');text=log.read_text(errors='replace');errors=[dict(line=j+1,text=s) for j,s in enumerate(text.splitlines()) if any(k in s for k in ('RuntimeError:','ValueError:','AssertionError:','Traceback (most recent','Fatal Python'))]
  progress=read(output/'batch_progress.json') if (output/'batch_progress.json').exists() else {};start=read(output/'run_started.json') if (output/'run_started.json').exists() else read(output/'kit_started.json') if (output/'kit_started.json').exists() else {}
  for c in plan.get('cases',[]):
   directory=output/c['name'];res=read(directory/'result.json') if (directory/'result.json').exists() else {};started=(directory/'started.json').exists();complete=res.get('software_complete',False)
   rows.append(dict(owner=str(owner.relative_to(R)),stage=index,wave=output.name,case=c['name'],role=c.get('role','expert'),seed=c['spec']['seed'],family=c['spec']['family'],case_started_record=started,complete=complete,strict_success=res.get('success',False),steps=res.get('steps'),first_error=next((e['text'] for e in errors if 'Error:' in e['text']),errors[0]['text'] if errors else None),worker_last_tick=progress.get('tick'),worker_active_steps=progress.get('active_lane_steps'),owner_failure=failed,worker_start=start,source_result=str(directory/'result.json'),log=str(log)))
keys=list(rows[0]) if rows else []
with (D/'all_attempts.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
atomic_json(D/'all_attempts.json',dict(utc=time.time(),rows=rows,interpretation='One row per invoked worker planned case. No dispatch/start record is reported separately; primary8family denominators use designated completed technical retries, never replace completed physical outcomes.'))
print(len(rows),'invoked-worker planned case records')
