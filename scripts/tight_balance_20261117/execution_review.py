"""Posthoc command-to-motion and paired history drift, never training labels."""
import json
import h5py,numpy as np
from scripts.tight_balance_20261117.offline_phases import ROOT,OUT
from scripts.tight_balance_20261117.diagnose import phases
from scripts.tight_balance_20261117.pair_diagnosis import first_run
from scripts.tight_layout_20261116.publication import write,table

def main():
 series={};rows=[];drifts=[]
 for part in range(4):
  folder=ROOT/f'comparison{part:02d}'/'development';plan=json.loads((folder/'plan.json').read_text())
  for c in plan['cases']:
   with h5py.File(folder/c['name']/'numeric.h5') as f:
    t=f['episode/transition'];q=t['diagnostics_after/joint_position_rad'][:];u=t['target_after_rad'][:];phase=phases(t)
   series[c['spec']['seed'],c['role']]=(q,u)
   for name,mask in [('early0_149',np.arange(len(q))<150),('all',np.ones(len(q),bool)),('near',phase==4),('post_subgoal',phase==5)]:
    if not mask.any():continue
    e=q[mask]-u[mask];du=np.diff(u,axis=0)[mask[1:]]
    rows.append(dict(role=c['role'],seed=c['spec']['seed'],phase=name,ticks=int(mask.sum()),arm_tracking_rmse_rad=float(np.sqrt(np.mean(e[:,:7]**2))),hand_tracking_rmse_rad=float(np.sqrt(np.mean(e[:,7:]**2))),arm_target_delta_rms_rad=float(np.sqrt(np.mean(du[:,:7]**2))) if len(du) else None,hand_target_delta_rms_rad=float(np.sqrt(np.mean(du[:,7:]**2))) if len(du) else None))
 for seed in sorted({s for s,r in series}):
  for role in 'AB':
   q,u=series[seed,role];qb,ub=series[seed,role+'_baseline'];n=min(len(q),len(qb));command=np.sqrt(np.mean((u[:n]-ub[:n])**2,axis=1));motion=np.sqrt(np.mean((q[:n]-qb[:n])**2,axis=1))
   drifts.append(dict(seed=seed,role=role,first_target_rms_gt_0p02_for12=first_run(command>.02),first_joint_rms_gt_0p02_for12=first_run(motion>.02),target_rms_first60=float(np.sqrt(np.mean((u[:60]-ub[:60])**2))),joint_rms_first60=float(np.sqrt(np.mean((q[:60]-qb[:60])**2)))))
 table(OUT/'command_motion.csv',rows);table(OUT/'paired_history_drift.csv',drifts)
 write(OUT/'execution_review.json',dict(rows=rows,paired_drifts=drifts,scope='True post-action joint position minus submitted absolute target, with native PD/delay unchanged. This is physical tracking lag, NOT A q_reached label error. Paired differences arise from different evolving observations/history and policies; not a same-input causal comparison.0.02rad/12tick threshold is posthoc description, not failure gate. No rollout commands used as expert labels.'))
 print(json.dumps(drifts,indent=2))
if __name__=='__main__':main()
