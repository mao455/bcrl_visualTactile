"""Export real small numeric artifacts; heavy RGB-D/features/models remain local."""
from pathlib import Path
import json,csv,shutil,hashlib,gzip
import h5py,numpy as np
P=Path.cwd();D=P/'outputs/diagnostics/tight_correction_20261127_01';R=P/'outputs/visuotactile/run/tight_correction_20261127_01';B=P/'research_docs/evidence/tight_correction_20261127_01';B.mkdir(parents=True,exist_ok=True)
origins={}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
def copy(p,d):
 q=B/d;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q);origins[str(d)]=str(p.relative_to(P))
for p in sorted(D.iterdir()):
 if p.suffix in ('.json','.csv','.gz','.png') and p.name not in ('ACTIVE_HANDOFF.md',):copy(p,Path('audit')/p.name)
for p in sorted((P/'configs/experiments/correction_20261127').glob('*.json')):copy(p,Path('config')/p.name)
names={'registration.json','retry_registration.json','session.json','plan.json','result.json','result_before_admission.json','started.json','actual_layout_admission.json','initial_observable.json','observe_only_trigger.json','targeted_stage_audit.json','capture_audit.json','capture_complete.json','execution_complete.json','metadata.json','restoration.json','sampler_rng_start.json','sampler_rng_end.json','project_job_started.json','project_failure.json','project_complete.json','project_children.json','project_exit.json','sampler_episode_reset.json','backend_only_recovery.json','joined_dispatch.json','project_stage_00.json','kit_started.json','run_started.json','short_complete.json','training.json','optimization.json','best.json','restore.json','recovery.json','full_trajectory_batch.json','backend_episode_sampling.json','short_budget_registration.json','visual_arithmetic.json','frozen_visual_io.json','frozen_visual_hotset.json','training_sampling.json','entry_same_history_plan.json','summary.json','complete.json','not_applicable.json','model_identity.json'}
for p in sorted(R.rglob('*')):
 if not p.is_file() or p.is_symlink() or 'resources' in p.parts:continue
 relative=p.relative_to(R)
 if p.suffix=='.json' and (p.name in names or p.name.endswith('_audit.json') or p.name.startswith('paired_group_deck_') or p.name=='continuation.json' or (p.name=='data.json' and 'state_data' in p.parts)):
  copy(p,Path('runs')/relative)
 elif p.name in ('per_slot_errors.csv','paired_errors.csv','trace.npz','plans.npz'):
  copy(p,Path('runs')/relative)
 elif p.name in ('learning.jsonl','loss.jsonl'):
  target=Path('runs')/relative.with_suffix('.jsonl.gz');q=B/target;q.parent.mkdir(parents=True,exist_ok=True)
  with p.open('rb') as f,gzip.open(q,'wb') as out:shutil.copyfileobj(f,out)
  origins[str(target)]=str(p.relative_to(P))
# Minimal complete numerical histories for physical diagnoses. Exclude only RNG byte dumps
# and placeholder expert_shadow. Real expert command/mask fields remain in captures.
cores=[]
for p in sorted(R.rglob('numeric.h5')):
 relative=p.relative_to(R)
 with h5py.File(p) as f:
  arrays={}
  def visit(n,o):
   if isinstance(o,h5py.Dataset) and 'rng' not in n and 'expert_shadow' not in n and o.dtype.kind not in ('O','S','U'):arrays[n]=o[:]
  f.visititems(visit)
 target=Path('runs')/relative.with_name('core_numeric.npz');q=B/target;q.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(q,**arrays);origins[str(target)]=str(p.relative_to(P));cores.append(dict(source=str(p.relative_to(P)),source_sha256=digest(p),published=str(target),fields=list(arrays),bytes=q.stat().st_size))
(B/'audit/core_numeric_index.json').write_text(json.dumps(cores,indent=2)+'\n')
for p in sorted((P/'scripts/tight_correction_20261127').glob('*.py')):copy(p,Path('source/scripts')/p.name)
for p in sorted(D.glob('*.py')):
 if p.name not in ('apply_capture.py','apply_finite.py','apply_queue.py'):copy(p,Path('source/diagnostic_scripts')/p.name)
# Final manifest is regenerated after README and final closure/export.
manifest={}
for p in sorted(B.rglob('*')):
 if p.is_file() and p.name!='MANIFEST.json':manifest[str(p.relative_to(B))]=dict(bytes=p.stat().st_size,sha256=digest(p),source=origins.get(str(p.relative_to(B)),'Generated public evidence'))
(B/'MANIFEST.json').write_text(json.dumps(dict(files=manifest,large_payloads='Original RGB-D, frozen features, optimizer/model checkpoints retained at registered local paths; no test64 copied or selected',numeric_exclusions=['RNG byte arrays (separate sampler receipts retained)','placeholder expert_shadow']),indent=2)+'\n')
print(json.dumps(dict(files=len(manifest),bytes=sum(x['bytes'] for x in manifest.values()),numeric_histories=len(cores))))
