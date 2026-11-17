"""One bounded fixed76 candidate, preserving selected72 full parents."""
import json,time,shutil,subprocess
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.anchor_sampling import onset_sampling_profile
from visuotactile.trajectory_training import profile,budget
from visuotactile.bounded_extension import phase_dagger_profile
TREE=Path.cwd();BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_long_20261117_01')
FAILED=BASE.parent/'tight_balance76_20261117_01'
ROOT=BASE.parent/'tight_balance76_20261117_01_recovery01'
def read(p):return json.loads(p.read_text())
def main():
 source=BASE/'round06/learning';parent=BASE/'round05/learning'
 assert not ROOT.exists();ROOT.mkdir();(ROOT/'logs').mkdir()
 s=read(source/'session.json');now=time.time()
 # Separate current authorization; historical deadlines and receipts remain intact.
 s.update(version='balanced76_bounded_cycle_v1',started_utc=now,deadline=now+6*3600,training_deadline=now+4*3600,cache_deadline=now+4*3600,offline_deadline=now+5*3600,collection_deadline=now+5*3600,resource_root=str(ROOT),owned_worker_roots=[str(ROOT)],source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),authorization='Fixed76 onset sampling candidate from protected72, formal4GiB immutable hotset, paired development, conditional targeted DAgger',reserve_bytes=12*1024**3)
 old=read(FAILED/'session.json')
 for key in ['started_utc','deadline','training_deadline','cache_deadline','offline_deadline','collection_deadline']:s[key]=old[key]
 atomic_json(FAILED/'preparation_failure.json',dict(error='Missing optional source_data.json in aggregate root; zero optimization/simulation',recovery=str(ROOT),utc=time.time()))
 s['resources']['min_disk_gib']=12
 assert shutil.disk_usage(ROOT).free>18*1024**3,'Need 18GiB: 6GiB bounded new outputs plus12GiB reserve'
 atomic_json(ROOT/'session.json',s)
 for name in ['metadata.json','source_data.json','selection.json','cuda_allocator_policy.json','frozen_visual_io.json']:
  if (source/name).exists():shutil.copyfile(source/name,ROOT/name)
 sampling=read(source/'training_sampling.json');sampling['anchors']=onset_sampling_profile();atomic_json(ROOT/'training_sampling.json',sampling)
 atomic_json(ROOT/'frozen_visual_hotset.json',dict(version='formal_immutable_hotset_v1',max_bytes=4*1024**3,minimum_free_bytes=8*1024**3))
 reg=dict(profile=profile(),roles={});hashes={};protected={}
 for role in 'AB':
  src=source/role;dest=ROOT/role;dest.mkdir();(dest/'state_data').symlink_to(src/'state_data',target_is_directory=True)
  for name in ['cache_index.json','continuation.json']:shutil.copyfile(src/name,dest/name)
  c=read(dest/'continuation.json');assert Path(c['path'])==parent/role/'training/best.pt'
  for k,h in [('path','sha256'),('previous_backend','previous_backend_sha256')]:
   assert sha256(c[k])==c[h];protected[c[k]]=c[h]
  protected[str(src/'state_data/data.json')]=sha256(src/'state_data/data.json');protected[str(src/'state_data/statistics.pt')]=sha256(src/'state_data/statistics.pt')
  (dest/'shared_resources.json').symlink_to(ROOT/'shared_resources.json');atomic_json(dest/'session.json',s)
  d=read(dest/'state_data/data.json');assert len(d['splits']['train'])==76 and len(d['splits']['validation'])==6
  reg['roles'][role]=dict(**budget(76,40),data_sha256=sha256(dest/'state_data/data.json'),episode_multiplicities=dict.fromkeys(d['splits']['train'],1));hashes[role]=sha256(dest/'cache_index.json')
 atomic_json(ROOT/'full_trajectory_batch.json',reg);atomic_json(ROOT/'bounded_extension.json',dict(policy=phase_dagger_profile(),roles={}))
 atomic_json(ROOT/'gate.json',dict(passed=True,inherited_input_gate=str(source/'gate.json'),inherited_input_gate_sha256=sha256(source/'gate.json')))
 atomic_json(ROOT/'prepared.json',dict(complete=True,continue_student_pair=True,reuse_frozen_cache=True,defer_development=True,cache_index_sha256=hashes,source=str(source),source_sha256={str(p.relative_to(TREE)):sha256(p) for folder in ['tools','visuotactile','scripts/tight_layout_20261116','scripts/tight_balance_20261117'] for p in (TREE/folder).rglob('*.py')},utc=now))
 registration=dict(root=str(ROOT),source76=str(source),parents72=str(parent),protected=protected,development_seeds=list(range(28824000,28824008)),heldout_seeds=list(range(28823000,28823020)),deadline=s['deadline'],training_deadline=s['training_deadline'],max_dp_updates_per_role=1520,backend='unchanged phase_dagger24 cap, refreshed current DP inputs; no sampler or loss change',selection='unchanged independent validation; offline phase review before8 same-wave paired development families',single_learning_factor='correction-only planning starts: half first24 legal expert ticks; original grasp anchors and geometry RNG preserved',historical76_control='same parents, LR, architecture, dataset and phase stopping rules; actual update counts and random draws differ, not strictly budget-matched',space_budget=dict(new_output_cap_gib=6,reserve_gib=12,starting_free=shutil.disk_usage(ROOT).free),created_utc=now)
 atomic_json(ROOT/'registration.json',registration);atomic_json(TREE/'research_docs/evidence/tight_long_20261117/balanced76/registration.json',registration)
 print(ROOT)
if __name__=='__main__':main()
