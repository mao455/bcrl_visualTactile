"""CPU replay of deterministic planned anchor allocation; not training updates."""
import json,collections
from pathlib import Path
import torch
from scripts.tight_balance_20261117.diagnose import BASE,OUT
from scripts.tight_layout_20261116.publication import write,table
from visuotactile.native_data import visual_records
from visuotactile.anchor_sampling import sample_training_anchors,grasp_sampling_profile,onset_sampling_profile,labels
from visuotactile.fresh_visit_sampling import visit
from visuotactile.target_windows import with_target_window
from visuotactile.latest_takeover_sampling import episode_deck

def main():
 root=BASE/'round06/learning';d=json.loads((root/'A/state_data/data.json').read_text());ids=d['splits']['train'];records=with_target_window(visual_records(root/'A/state_data',ids,'G00'),'recovery_onset_commit12_v1')
 p=json.loads((root/'training_sampling.json').read_text());new=dict(p,anchors=onset_sampling_profile())
 counters={v:collections.Counter() for v in ['historical_v2','candidate_v3']};rows=[];g=torch.Generator().manual_seed(2026111402)
 for epoch in range(1,41):
  deck=episode_deck(ids,dict.fromkeys(ids,1),g)
  for idx,i in enumerate(deck,1):
   r=records[i];legal=r.get('supervision_mask',torch.ones(len(r['time']),dtype=torch.bool));ticks=legal.nonzero().flatten();original=int(ticks[0])==0;onset=set(ticks[:24].tolist()) if not original else set()
   selections=[]
   for version,pr in [('historical_v2',p),('candidate_v3',new)]:
    a,receipt,geo=visit(r,pr,identity=i,epoch=epoch,visit_index=idx);_,mask=labels(r,a,24);slots=(a[:,None]+torch.arange(24)[None])[mask].tolist();selections.append((a,geo.get_state()))
    counters[version].update(visits=1,anchors=len(a),target_slots=len(slots),correction_onset_slots=sum(t in onset for t in slots),original_early_slots=sum(t<150 for t in slots) if original else 0)
   if original:
    assert torch.equal(selections[0][0],selections[1][0]);assert torch.equal(selections[0][1],selections[1][1])
   assert torch.equal(selections[0][1],selections[1][1])
  rows.append(dict(epoch=epoch,**{k:dict(v) for k,v in counters.items()}))
 table(OUT/'planned_exposure.csv',rows);write(OUT/'planned_exposure.json',dict(planned_only=True,original_anchors_and_geometry_rng_exact=True,all_geometry_rng_exact=True,epochs=40,updates_if_completed=1520,counts={k:dict(v) for k,v in counters.items()},note='Deterministic CPU sampler dry-run; real exposure comes from executed learning logs; overlapping target slots not independent ticks'))
 print(json.dumps({k:dict(v) for k,v in counters.items()}))
if __name__=='__main__':main()
