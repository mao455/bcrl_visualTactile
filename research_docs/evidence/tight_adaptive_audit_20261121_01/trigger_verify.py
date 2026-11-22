from pathlib import Path
import json,sys
import torch,h5py,numpy as np
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from tools.tight_fresh_report import audit_trigger_case

def main():
 torch.set_num_threads(2);rows=[];sums=[]
 for round in (1,2):
  p=RUN/f'round{round:02d}/collection_combined';plan=json.load(open(p/'plan.json'));entries=json.load(open(p/'correction_entries.json'))
  for case in plan['cases']:
   directory=p/case['name'];r=json.load(open(directory/'result.json'));trigger=audit_trigger_case(directory,r)
   with h5py.File(directory/'numeric.h5') as f:
    g=f['episode'];d=g['transition/dagger'];mask=d['expert_label_valid'][:];source=d['control_source'][:];legal=np.flatnonzero(mask);sw=np.flatnonzero(source);switch=int(sw[0]) if len(sw) else None
    checks=dict(accepted_matches_entry=bool(r.get('accepted_for_training'))==(case['name'] in entries),prefix_mask_zero=not mask[:switch].any() if switch is not None else not mask.any(),source_expert_for_all_labels=bool((source[mask]==1).all()),first_legal_equals_switch_plus2=int(legal[0])==switch+2 if len(legal) else None)
    rows.append(dict(round=round,identity=case['name'],path=str(directory),source_role=case['role'],accepted=case['name'] in entries,switch=switch,first_legal=int(legal[0]) if len(legal) else None,legal_ticks=int(mask.sum()),checks=checks,trigger=trigger,load_pair=plan['pairs'][case['role']]))
  subset=[r for r in rows if r['round']==round];sums.append(dict(round=round,opportunities=len(subset),takeovers=sum(r['switch'] is not None for r in subset),accepted=sum(r['accepted'] for r in subset),accepted_ticks=sum(r['legal_ticks'] for r in subset if r['accepted'])))
 (OUT/'trigger_verification.json').write_text(json.dumps(dict(summary=sums,rows=rows),indent=2)+'\n');print(sums);print('trigger passed',all(r['trigger']['passed'] for r in rows));print('bad checks',[(r['identity'],r['checks']) for r in rows if any(v is False for v in r['checks'].values())])
if __name__=='__main__':main()
