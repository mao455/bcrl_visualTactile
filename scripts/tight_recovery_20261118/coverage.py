"""Actual first lawful labels and recovery-state change, without counterfactual claims."""
import csv,json
import h5py,numpy as np
from scripts.tight_recovery_20261118.run import ROOT,OUT,read,write
from scripts.tight_balance_20261117.diagnose import phases
from visuotactile.geometry_sampling import STAGES
from tools.tight_fresh_report import audit_trigger_case

def main(name):
 d=ROOT/name;rows=[]
 for c in read(d/'plan.json')['cases']:
  r=read(d/c['name']/'result.json');audit=audit_trigger_case(d/c['name'],r)
  with h5py.File(d/c['name']/'numeric.h5') as f:
   t=f['episode/transition'];p=phases(t);m=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool);valid=np.flatnonzero(m);receipt=r['dagger']['trigger_receipt'];switch=receipt['trigger_completed_tick'];first=int(valid[0]) if len(valid) else None
   row=dict(case=c['name'],role=c['role'],family=c['spec']['family'],seed=c['spec']['seed'],success=r['success'],accepted=r.get('accepted_for_training',False),switch_tick=switch,screen_onset_tick=receipt['first_regression_transition_tick'],screen=receipt['reason'],screen_state=receipt['at_trigger'],first_legal_tick=first,transition_gap=None if first is None else first-switch,screen_to_label=None if first is None else first-receipt['first_regression_transition_tick'],legal_ticks=len(valid),phase_ticks={s:int(((p==j)&m).sum()) for j,s in enumerate(STAGES)},trigger_replay_passed=audit['passed'])
   if first is not None:
    row.update(first24_phase_ticks={s:int((p[valid[:24]]==j).sum()) for j,s in enumerate(STAGES)},first_legal_state={k:t[k][first].tolist() for k in ['diagnostics_before/object_pose_w','diagnostics_before/object_velocity_w_m_s_rad_s','diagnostics_before/palm_body_pose_w','object_contact_force_w']},first_legal_keypoint_m=float(np.asarray(t['native_keypoints_max_dist'][first]).reshape(-1)[0]),first24_legal_count=min(len(valid),24),suffix_legal_count=max(len(valid)-24,0))
    a=first;b=int(valid[min(23,len(valid)-1)]);row['expert24_reached_change_rad']=(t['diagnostics_after/joint_position_rad'][b]-t['diagnostics_after/joint_position_rad'][a]).tolist();row['expert24_keypoint_change_m']=float(np.asarray(t['native_keypoints_max_dist'][b]-t['native_keypoints_max_dist'][a]).reshape(-1)[0])
   rows.append(row)
 write(OUT/(name+'_coverage.json'),dict(rows=rows,scope='Actual completed-state screen versus first legal expert action, zero-based control indices. Screen onset is not uniquely identified first causal task error. Only valid suffix supervised; all prefixes preserved.'))
 with (OUT/(name+'_coverage.csv')).open('w') as f:
  simple=[{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in rows];keys=list(dict.fromkeys(k for r in simple for k in r));w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(simple)
 print([(r['role'],r['seed'],r['switch_tick'],r['success'],r['first_legal_tick'],r['screen']) for r in rows])
if __name__=='__main__':main(__import__('sys').argv[1])
