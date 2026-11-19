"""Read-only whole-pool SHA and actual supervision alignment check."""
import json,time
from pathlib import Path
import numpy as np
import h5py,torch
from tools.tight_frozen_frontend import read,write,PROJECT
from visuotactile.compact import sha256
from visuotactile.native_data import visual_records
from visuotactile.effective_exposure import Ledger
root=PROJECT/'outputs/visuotactile/run/tight_exposure50_20261119_01';public=Path(__file__).resolve().parent
began=time.time();data=read(root/'B/state_data/data.json');ids=data['splits']['train']+data['splits']['validation']
a=visual_records(root/'A/state_data',ids,'G00');b=visual_records(root/'B/state_data',ids,'G00');rows=[]
for j,i in enumerate(ids):
 e=data['entries'][i];actual=sha256(e['path']);assert actual==e['source_sha256'],i
 with h5py.File(e['path']) as f:
  g=f['episodes'][e['episode']];t=g['transition'];mask=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if 'dagger' in t else np.ones(len(t['target_after_rad']),bool)
  assert np.array_equal(mask,a[i]['supervision_mask'].numpy()) and np.array_equal(mask,b[i]['supervision_mask'].numpy())
  reached=t['diagnostics_after/joint_position_rad'][:];command=t['target_after_rad'][:]
  assert np.array_equal(a[i]['clean_sample'][mask].numpy(),reached[mask])
  assert np.array_equal(b[i]['clean_sample'][mask].numpy(),command[mask])
  assert np.array_equal(a[i]['robot'].numpy(),b[i]['robot'].numpy())
 rows.append(dict(identity=i,path=e['path'],source_sha256=actual,legal_ticks=int(mask.sum()),masked_ticks=int((~mask).sum()),label_alignment=True))
 if j%20==0:print('verified raw',j,flush=True)
write(public/'raw_integrity.json',dict(passed=True,rows=rows,wall_seconds=time.time()-began,full_raw_sha=True,label_values_exact=True))
# Audit a durable early checkpoint against its complete successful-step log.
p=root/'A/training/last.pt'
if p.exists():
 snapshot=torch.load(p,map_location='cpu',weights_only=False);state=snapshot['effective_exposure_state'];ledger=Ledger(state['legal'])
 for line in (p.parent/'learning.jsonl').open():
  row=json.loads(line)
  if row['updates']>snapshot['updates']:break
  ledger.step(row['updates'],list(row['effective_tick_contributions'].items()))
 assert ledger.updates==snapshot['updates']
 for i in ledger.legal:assert torch.equal(ledger.counts[i],state['counts'][i])
 write(public/'early_checkpoint_exposure_gate.json',dict(passed=True,epoch=snapshot['epoch'],updates=snapshot['updates'],summary=ledger.summary(snapshot['sampling_exposure']),
  restored_counter_values_exact=True,source='Actual durable checkpoint versus successful optimizer transaction log, not expected epochs'))
