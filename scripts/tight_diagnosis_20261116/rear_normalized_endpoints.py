"""Per-part normalization of saved selected-checkpoint predictions; no forward pass."""
import csv,json
from pathlib import Path
import numpy as np,torch
B=Path('outputs/visuotactile/run');O=Path('research_docs/evidence/tight_diagnosis_20261116');rows=[]
for n,parent in [(16,'tight_fresh_20261116_01'),(17,'tight_fresh_20261116_02'),(18,'tight_fresh_20261116_02'),(19,'tight_fresh_20261116_04')]:
 for role in 'AB':
  rear=B/parent/f'round_{n}'/role/'backend'/role
  ckpt=torch.load(rear/'training/best.pt',map_location='cpu',mmap=True,weights_only=False);scale=ckpt['model']['action_scale'].numpy()
  for identity,split in [('formal_050_slow_3','train'),('formal_014_slow_1','validation')]:
   values=list(csv.DictReader((rear/'offline'/(identity+'_targets.csv')).open()));error=np.array([[float(r[f'prediction_{j}_rad'])-float(r[f'truth_{j}_rad']) for j in range(29)] for r in values]);normalized=error*scale
   for phase,mask in [('all',np.ones(len(values),bool)),('before150',np.array([int(r['tick'])<150 for r in values]))]:
    for group,sl in [('arm',slice(0,7)),('hand',slice(7,None))]:
     rows.append(dict(round=n,role=role,identity=identity,split=split,phase=phase,part=group,ticks=int(mask.sum()),rmse_rad=float(np.sqrt((error[mask,sl]**2).mean())),normalized_rmse=float(np.sqrt((normalized[mask,sl]**2).mean())),scope='One fixed saved train/validation trajectory; selected checkpoint only, NOT full6 validation or full348 train.'))
  del ckpt
with (O/'rear_endpoint_normalized.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
