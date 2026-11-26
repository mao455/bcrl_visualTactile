from pathlib import Path
import json,hashlib,collections
import torch
from visuotactile.compact import sha256
R=Path.cwd();P=R/'outputs/visuotactile/run/tight_paired_dagger_20261122_01';O=R/'outputs/diagnostics/tight_machine_review_20261126_01'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False))
def fingerprint(x):
 if torch.is_tensor(x):return dict(shape=list(x.shape),dtype=str(x.dtype),sha256=hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
 if isinstance(x,dict):return {str(k):fingerprint(v) for k,v in x.items()}
 if isinstance(x,(tuple,list)):return [fingerprint(v) for v in x]
 return x
hist=read(R/'outputs/visuotactile/run/tight_entry_final24_20261122_01/registration.json')['pairs'];current=read(P/'evaluation/test/registration.json')['pairs'];pairs=dict(hist,**current);rows={}
for role,pair in pairs.items():
 info=dict(pair=pair,components={})
 for component in ('frontend','backend'):
  path=Path(pair[component]);p=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
  info['components'][component]=dict(file_sha256=sha256(path),registered_sha256=pair[component+'_sha256'],epoch=p.get('epoch'),updates=p.get('updates'),
   ema_steps=p.get('ema_optimization_step'),model_keys=list(p),contract=p.get('contract'),statistics=fingerprint(p.get('statistics')),normalization=fingerprint(p.get('normalization')),
   base_checkpoint=p.get('configuration',{}).get('base_checkpoint'),base_sha256=p.get('configuration',{}).get('base_sha256'),execution_adapter=p.get('configuration',{}).get('execution_adapter'))
  del p
 rows[role]=info
write(O/'model_identity_normalization.json',rows)
# Reconcile the only same-optimizer interruption against actual committed visits.
root=P/'common180_parents/A/training';receipt=read(next(root.glob('recovery.*.json')));rows=[];boundary=[];bad=[];deck_cache={};counts=collections.Counter()
with (root/'learning.jsonl').open() as f:
 for expected,line in enumerate(f,1):
  x=json.loads(line);u=x['updates'];e=x['epoch'];idx=x['batch_index'];visits=[v['episode'] for v in x['visits']]
  if e not in deck_cache:deck_cache[e]=read(root/f'deck_{e:03d}.json')['order']
  if u!=expected or deck_cache[e][2*(idx-1):2*idx]!=visits:bad.append(dict(update=u,epoch=e,batch_index=idx))
  counts.update(visits)
  if 6839<=u<=6845:boundary.append(dict(update=u,epoch=e,batch_index=idx,visits=visits,deck_sha256=x['deck_sha256'],utc=x['utc'],loss=x['loss']))
  rows.append(u)
abandoned=[json.loads(s) for s in Path(receipt['abandoned_path']).read_text().splitlines()]
# Redundant terminal last alias retired after this audit; selected epoch60 best is preserved.
final=torch.load(root/'best.pt',map_location='cpu',weights_only=False,mmap=True)
resume=dict(native_recovery=receipt,committed_updates=len(rows),ids_unique_contiguous=rows==list(range(1,7201)),all_committed_visits_match_registered_decks=not bad,
 deck_mismatches=bad,boundary=boundary,abandoned_update_ids=[x['updates'] for x in abandoned],abandoned_replayed_same_episode_order=all(
  [v['episode'] for v in x['visits']]==next(z['visits'] for z in boundary if z['update']==x['updates']) for x in abandoned),
 final=dict(epoch=final['epoch'],updates=final['updates'],ema_optimization_step=final['ema_optimization_step'],
 optimizer_state_entries=len(final['optimizer']['state']),optimizer_step_values=sorted(set(int(v['step']) for v in final['optimizer']['state'].values())),
 rng={k:fingerprint(final[k]) for k in ('schedule_rng','noise_rng','torch_rng','cuda_rng')},sampling_exposure_matches_log=dict(counts)==final['sampling_exposure']),
 contemporaneous_restore_metadata=read(R/'outputs/diagnostics/tight_paired_dagger_20261122_01/parent_recovery_readonly_metadata.json'),
 caveat='Actual resume receipt and boundary/EMA/optimizer/RNG fields verified. The overwritten epoch57 payload is no longer available for independent rerunning of its next stochastic update. No exact resumed/uninterrupted optimizer parity claim. Correct resume excludes only a resume explanation, not data/closed-loop failure.')
write(O/'resume_boundary_audit.json',resume)
print('model pairs',len(pairs),'committed',len(rows),'deck errors',len(bad),'EMA',final['ema_optimization_step'],flush=True)
