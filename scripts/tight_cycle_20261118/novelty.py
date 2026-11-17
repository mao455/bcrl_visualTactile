"""Separate entry and suffix physical support; no label manufacture or admission."""
import numpy as np,h5py
from scripts.tight_cycle_20261118.cycle import ROOT,DATA,OUT,read,write
from scripts.tight_cycle_20261118.analyze import features,buildtree,distance
from visuotactile.geometry_sampling import STAGES

def main():
 data=read(DATA/'A/state_data/data.json');records=[]
 keys=['palm_local_object_m','palm_local_rotation_rad','object_palm_m','object_goal_m','object_rotation_rad','joint_rms_rad','object_speed_m_s','object_spin_rad_s','finger_contact_hamming','issued_history12_rms_rad']
 for i in data['splits']['train']:
  e=data['entries'][i]
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];x,p=features(t);m=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if e.get('trajectory_kind')=='expert_takeover_v1' else np.ones(len(p),bool)
   records.append(({k:x[k][m] for k in keys},p[m]))
 ph=np.concatenate([p for x,p in records]);xx={k:np.concatenate([x[k] for x,p in records]) for k in keys};trees={(int(p),k):buildtree(xx[k][ph==p],k) for p in set(ph) for k in keys}
 thresholds={(r['phase'],r['metric']):r['reference_p95'] for r in read(OUT/'capability/support.json')};rows=[]
 for i,e in read(ROOT/'collection/correction_entries.json').items():
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];x,p=features(t);m=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool);valid=np.flatnonzero(m);onset=np.zeros(len(m),bool);onset[valid[:24]]=True
   switch=int(e['takeover_tick']);last_student=np.zeros(len(m),bool);last_student[max(0,switch-1)]=True
   for region,mask in [('last_student_transition_diagnostic_only',last_student),('first24_legal',onset),('remaining_legal',m&~onset)]:
    for phase in set(p[mask]):
     q=mask&(p==phase)
     for k in keys:
      dd=distance(trees[int(phase),k],x[k][q],k);thr=thresholds[STAGES[int(phase)],k]
      rows.append(dict(identity=i,source=e['source_group'],family=e['family'],region=region,phase=STAGES[int(phase)],metric=k,ticks=int(q.sum()),median=float(np.median(dd)),reference_p95=thr,far_ticks=int((dd>thr+1e-8).sum())))
 write(OUT/'collection/entry_support.json',dict(rows=rows,scope='Last actual student transition is diagnosis only and NEVER supervised. First24 actually legal expert observations versus remaining valid suffix; phase-matched old76 support. Reference threshold old76 leave-family-out p95. Distances per physical unit, no single arbitrary score or recovery guarantee. First legal tick follows real authority/queue transition, not exact switch state.'))
 totals={}
 for region in ['last_student_transition_diagnostic_only','first24_legal','remaining_legal']:
  totals[region]={}
  for k in keys:
   rr=[r for r in rows if r['region']==region and r['metric']==k];totals[region][k]=sum(r['far_ticks'] for r in rr)/sum(r['ticks'] for r in rr)
 print(totals)
if __name__=='__main__':main()
