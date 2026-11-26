from pathlib import Path
import gzip,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('outputs/diagnostics/tight_correction_20261127_01')
with gzip.open(D/'grasp_chain_ticks.csv.gz','rt') as f:rows=list(csv.DictReader(f))
fig,axes=plt.subplots(2,3,figsize=(12,6),sharex=True)
for i,seed in enumerate((28911001,28911002)):
 for role in ['A_new','B_new','A_parent','B_parent','expert']:
  selected=[r for r in rows if int(r['seed'])==seed and r['role']==role and int(r['action_tick'])<240]
  t=np.array([int(r['action_tick']) for r in selected])/60
  for j,key in enumerate(['contact_fingers','elevation_mm','root_error_mm']):
   axes[i,j].plot(t,[float(r[key]) for r in selected],label=role,linewidth=1.1)
 for j,title in enumerate(['Object-contact fingers (both physics samples)','Elevation over settled24:48 (mm)','Active goal root error (mm)']):
  axes[i,j].set_title(str(seed)+' / '+title,fontsize=9);axes[i,j].grid(alpha=.25)
  axes[i,j].set_xlabel('Completed trajectory time (s;0-based action /60)')
axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(D/'grasp_comparison.png',dpi=150);plt.close(fig)
