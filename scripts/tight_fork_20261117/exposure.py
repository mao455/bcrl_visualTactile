"""Actual-deck counterfactual sampler counts, not a trained control model."""
import collections,json,torch
from .common import *
from visuotactile.fresh_visit_sampling import visit
from visuotactile.anchor_sampling import grasp_sampling_profile,supervision_rows
from visuotactile.target_windows import window_mask
from scripts.tight_layout_20261116.publication import table,write

def main():
 results=[]
 for role,best in [('A',494),('B',342)]:
  records=torch.load(OLD/role/'state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
  data=read(OLD/role/'state_data/data.json');profile=read(OLD/'training_sampling.json');old=dict(profile,anchors=grasp_sampling_profile())
  counts=collections.defaultdict(collections.Counter);matches=0
  for line in (OLD/role/'training/learning.jsonl').open():
   row=json.loads(line)
   for v in row['visits']:
    i=v['episode'];r=records[i];mask,seg=supervision_rows(r);start=int(mask.nonzero()[0]);original=start==0
    receipt=v['anchor_sampling'];record=dict(r,target_window_contract='recovery_onset_commit12_v1')
    aa={}
    for version,p in [('v2_counterfactual',old),('v3_actual',profile)]:
     anchors,check,geo=visit(record,p,identity=i,epoch=row['epoch'],visit_index=receipt['visit_index']);aa[version]=anchors
     if version=='v3_actual':assert anchors.tolist()==receipt['selected_anchors']
     valid=window_mask(mask,seg,anchors,24,'recovery_onset_commit12_v1');ticks=(anchors[:,None]+torch.arange(24)[None])[valid]
     n=len(ticks);early=int((ticks<150).sum()) if original else 0;onset=int((ticks<start+24).sum()) if not original else 0
     for scope in ['actual']+(['selected'] if row['updates']<=best else []):
      c=counts[scope,version];c.update(visits=1,slots=n,original_early_slots=early,onset24_slots=onset,original_visits=int(original),original_objective_mass=v['loss_weight']*original,original_early_base_mass=v['loss_weight']*early/n,onset24_base_mass=v['loss_weight']*onset/n,total_objective_mass=v['loss_weight'])
    if original:assert torch.equal(aa['v2_counterfactual'],aa['v3_actual']);matches+=1
  for (scope,version),c in counts.items():
   results.append(dict(role=role,scope=scope,version=version,**c,original_slot_fraction=c['original_early_slots']/c['slots'],original_objective_fraction=c['original_objective_mass']/c['total_objective_mass'],original_early_base_fraction=c['original_early_base_mass']/c['total_objective_mass'],onset24_base_fraction=c['onset24_base_mass']/c['total_objective_mass']))
 table(OUT/'exposure_objective.csv',results);write(OUT/'exposure_objective.json',dict(rows=results,original_anchors_exact=True,meaning='Actual visit deck/epochs. v2 is sampler-only counterfactual, zero optimizer updates. Base mass uses normalized valid-slot mean inside equal trajectory objective, not gradient norm or geometry loss share.'))
 print(json.dumps(results,indent=2))
if __name__=='__main__':main()
