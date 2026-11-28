"""Export actual closed full-curve evidence; original payloads remain local."""
from pathlib import Path
import csv,gzip,hashlib,json,shutil
import h5py,numpy as np
P=Path.cwd();D=P/'outputs/diagnostics/tight_full_curve_20261127_01';R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';B=P/'research_docs/evidence/tight_full_curve_20261127_01'
assert (D/'audits_complete.json').exists(), 'Never publish partial training as a completed result'
assert json.loads((D/'primary_summary.json').read_text())['complete_opportunities']==224
B.mkdir(parents=True,exist_ok=True);origins={};local=[]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def target(p,d):
 q=B/d;q.parent.mkdir(parents=True,exist_ok=True);origins[str(d)]=str(p.relative_to(P));return q
def copy(p,d):
 q=target(p,d)
 if p.suffix=='.csv':q.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 else:shutil.copyfile(p,q)
def zipped(p,d):
 with p.open('rb') as f,gzip.open(target(p,d),'wb') as out:shutil.copyfileobj(f,out)
for p in sorted(D.iterdir()):
 if p.is_file() and p.suffix in {'.json','.csv','.gz','.png'} and p.name not in {'postprocess_wait.json'}:
  if p.suffix=='.csv' and p.stat().st_size>8*2**20:zipped(p,Path('audit')/(p.name+'.gz'))
  else:copy(p,Path('audit')/p.name)
for p in sorted((P/'configs/experiments/full_curve_20261127').glob('*.json')):copy(p,Path('config')/p.name)
for p in sorted(D.glob('*.py')):copy(p,Path('source/diagnostic_scripts')/p.name)
# Identity, mask, source and actual update receipts; optimizer/RGB-D/feature payloads stay local.
names={'worker_failure.json','watchdog_failure.json','watchdog_policy14.json','retry.json','actual_layout_admission.json','session.json','plan.json','registration.json','result.json','results.json','started.json','actual_layout_admission.json','initial_observable.json','sampler_rng_start.json','sampler_rng_end.json','sampler_episode_reset.json','observe_only_trigger.json','metadata.json','execution_complete.json','project_job_started.json','project_failure.json','project_complete.json','project_exit.json','project_children.json','project_stage_00.json','kit_started.json','run_started.json','joined_dispatch.json','restoration.json','restore.json','recovery.json','configuration.json','continuation.json','fixed_final_selection.json','best.json','finished.json','validation.json','initial_metrics.json','training_sampling.json','full_trajectory_batch.json','backend_episode_sampling.json','budget_registration.json','family_phase_definition.json','matched_learning_rates.json','matched_backend_learning_rates.json','front_source_identity.json','source_freeze.json','monitor_exit.json','recurrent_execution.json','frozen_visual_io.json','frozen_visual_hotset.json','visual_arithmetic.json','training_milestones.json'}
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.is_symlink() or 'resources' in p.parts:continue
 rel=p.relative_to(R);dest=Path('runs')/rel
 if p.suffix=='.json' and (p.name in names or p.name.endswith('_failure.json') or p.name.endswith('_complete.json') or p.name.endswith('_audit.json') or p.name.startswith('paired_group_deck_') or p.name in {'data.json','expert_entries.json','full_expert_collection.json'}):
  if p.stat().st_size>8*2**20:zipped(p,dest.with_suffix('.json.gz'))
  else:copy(p,dest)
 elif p.name in {'family_phase_curves.csv','family_objective_loss.csv','validation.jsonl','family_phase_events.jsonl','sampling_coverage.json','entry_exposure.json'}:zipped(p,Path(str(dest)+'.gz'))
 elif p.name in {'plans.npz','trace.npz'}:copy(p,dest)
 elif p.name.startswith('learning.abandoned.') or p.name.startswith('recovery.'):
  zipped(p,Path(str(dest)+'.gz'))
 elif p.name in {'learning.jsonl','loss.jsonl'}:
  # Keep step/episode/aggregate losses, omit redundant per-contact geometry arrays.
  q=target(p,Path(str(dest)+'.compact.gz'));count=0
  with p.open() as f,gzip.open(q,'wt') as out:
   for line in f:
    v=json.loads(line)
    def compact(o):
     if isinstance(o,dict):return {k:compact(x) for k,x in o.items() if k not in {'geometry','rows'}}
     if isinstance(o,list):return [compact(x) for x in o]
     return o
    out.write(json.dumps(compact(v),separators=(',',':'))+'\n');count+=1
  local.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p),published=str(q.relative_to(B)),rows=count,omitted='Per-contact geometry rows; originals retained locally'))
cores=[]
for p in sorted(R.rglob('numeric.h5')):
 arrays={}
 with h5py.File(p) as f:
  def visit(n,o):
   if isinstance(o,h5py.Dataset) and 'rng' not in n and 'expert_shadow' not in n and o.dtype.kind not in 'OSU':arrays[n]=o[:]
  f.visititems(visit)
 rel=Path('runs')/p.relative_to(R).with_name('core_numeric.npz');q=target(p,rel);np.savez_compressed(q,**arrays)
 cores.append(dict(source=str(p.relative_to(P)),sha256=sha(p),published=str(rel),fields=list(arrays),bytes=q.stat().st_size))
(B/'audit/core_numeric_index.json').write_text(json.dumps(cores,indent=2)+'\n')
(B/'audit/local_detailed_training_logs.json').write_text(json.dumps(local,indent=2)+'\n')
# Current canonical implementations carry git history; snapshot only experiment-specific scripts.
manifest={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.name!='MANIFEST.json':manifest[str(p.relative_to(B))]=dict(bytes=p.stat().st_size,sha256=sha(p),source=origins.get(str(p.relative_to(B)),'Generated evidence index or report'))
(B/'MANIFEST.json').write_text(json.dumps(dict(files=manifest,large_payloads='Original RGB-D, frozen features, models/optimizer checkpoints and full detailed geometry logs retained at registered local paths',numeric_exclusions=['RNG byte arrays: separate RNG receipts retained','Pure autonomous expert_shadow placeholders'],second_repository_writes=0),indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),numeric_histories=len(cores))))
