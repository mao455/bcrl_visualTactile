"""Real cache/label audit and causal replay; no expert recovery inference."""
import json,h5py,numpy as np,torch
from pathlib import Path
from visuotactile.takeover_trigger import observe_transition,make_trigger
from visuotactile.phase_takeover import profile,STAGES
from visuotactile.native_data import residual_supervision_mask
from visuotactile.target_windows import window_mask
import argparse
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
root=args.root.resolve()
data=json.loads((root/'A/state_data/data.json').read_text());report={'cache':{},'replay':[]}
for role in 'AB':
 raw=torch.load(root/role/'state_data/numeric.pt',weights_only=False,map_location='cpu',mmap=True)['records'];n=0
 for identity in data['aggregation']['new_training_ids']:
  p=root/role/'backend'/role/'nominal'/(identity+'.pt');r=torch.load(p,weights_only=False,map_location='cpu',mmap=True)
  a=raw[identity];mask=a['supervision_mask'];first=int(mask.nonzero()[0]);n+=int(mask.sum())
  for key in ['time','robot','supervision_mask','control_segment']:assert torch.equal(r[key],a[key]),key
  assert torch.equal(r['target'][mask],a['command'][mask])
  assert len(r['time'])==len(a['time']) and not r['supervision_mask'][:first].any()
  assert residual_supervision_mask([r],0,len(mask),'cpu')[0].equal(mask)
  valid=window_mask(mask,a['control_segment'],torch.arange(len(mask)),24)
  for start in range(len(mask)):
   ix=torch.arange(start,min(start+24,len(mask)));assert not (valid[start,:len(ix)]&~mask[ix]).any()
 report['cache'][role]={'corrections':len(data['aggregation']['new_training_ids']),'valid_expert_ticks':n,'factual_full_history_mask_and_rear_targets_equal':True}
for folder in sorted((root/'development').glob('*development*')):
 if not (folder/'numeric.h5').exists():continue
 row={'case':folder.name,'stages':{}}
 with h5py.File(folder/'numeric.h5') as f:
  t=f['episode/transition'];length=len(t['target_after_rad'])
  for stage in STAGES:
   trigger=make_trigger(profile(stage),10**9)
   for tick in range(length):
    if observe_transition(trigger,tick,t,index=tick) is not None:break
   row['stages'][stage]=trigger.receipt()
 report['replay'].append(row)
print(json.dumps({r['case']:{k:v['trigger_completed_tick'] for k,v in r['stages'].items()} for r in report['replay']}))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
