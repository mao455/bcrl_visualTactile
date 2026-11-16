"""Render the completed CPU diagnostic; no model, simulation or updates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=Path('research_docs/evidence/tight_diagnosis_20261116/budget_extension')
report=json.loads((O/'closed_command_gap.json').read_text())
assert report['complete']
cases=sorted({r['case'] for r in report['rows']});fig,axes=plt.subplots(3,len(cases),figsize=(16,9),squeeze=False)
for col,case in enumerate(cases):
 rows=[r for r in report['rows'] if r['case']==case]
 for row,color in zip(rows,['tab:orange','tab:green']):
  with np.load(Path(row['path']).with_name('trace.npz')) as t:
   z=t['object'][:300,2];force=t['object_contact'][:300]
  rise=(z-np.median(z[24:48]))*100;fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
  axes[0,col].plot(np.arange(row['ticks'])/60,row['hand_command_error_rad'],color=color,label=row['group'])
  axes[1,col].plot(np.arange(1,len(z)+1)/60,rise,color=color)
  axes[2,col].step(np.arange(1,len(z)+1)/60,fingers,where='post',color=color)
  if row['first_contact'] is not None:
   for ax in axes[:,col]:ax.axvline(row['first_contact']/60,color=color,ls=':',lw=1)
 axes[0,col].set_title(case);axes[0,col].set_ylabel('Command gap to same-state expert (rad)');axes[0,col].legend(fontsize=6)
 axes[1,col].set_ylabel('Height above settled median (cm)');axes[1,col].axhline(2,color='gray',ls='--',lw=.8)
 axes[2,col].set_ylabel('Contact fingers, both physics samples');axes[2,col].set_ylim(-.2,5.2)
 for ax in axes[:,col]:ax.set_xlabel('Simulation seconds');ax.grid(alpha=.2)
fig.suptitle('B: same DP weights, rear6 versus rear22; dotted lines = first contact. Expert suggestions were NOT executed.')
fig.tight_layout();fig.savefig(O/'closed_command_gap.png',dpi=150,bbox_inches='tight');plt.close(fig)
