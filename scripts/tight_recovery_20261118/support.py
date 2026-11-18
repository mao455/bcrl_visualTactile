"""Per-unit old105 support, comparing last student/first legal labels/suffix."""
import numpy as np,h5py
from scripts.tight_recovery_20261118.run import ROOT,DATA,OUT,read,write
from scripts.tight_cycle_20261118.analyze import features,buildtree,distance
from visuotactile.geometry_sampling import STAGES

def main(name):
 keys=['palm_local_object_m','palm_local_rotation_rad','object_speed_m_s','issued_history12_rms_rad'];data=read(DATA/'A/state_data/data.json');rows=[];ref=OUT/'support105_reference.json';records=[]
 for i in data['splits']['train']:
  e=data['entries'][i]
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];x,p=features(t);mask=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool) if e.get('trajectory_kind')=='expert_takeover_v1' else np.ones(len(p),bool)
   records.append((e['family'],{k:x[k][mask] for k in keys},p[mask]))
 ph=np.concatenate([p for _,_,p in records]);fam=np.concatenate([np.repeat(f,len(p)) for f,_,p in records]);xx={k:np.concatenate([x[k] for _,x,_ in records]) for k in keys};trees={};thresholds={};refs=[]
 for phase in set(ph):
  mask=ph==phase;ff=fam[mask]
  for k in keys:
   vals=xx[k][mask];trees[int(phase),k]=buildtree(vals,k);dd=np.full(len(vals),np.nan)
   for family in set(ff):
    q=ff==family
    if (~q).any():dd[q]=distance(buildtree(vals[~q],k),vals[q],k)
   finite=dd[np.isfinite(dd)];thr=float(np.percentile(finite,95)) if len(finite) else None;thresholds[int(phase),k]=thr;refs.append(dict(phase=STAGES[int(phase)],metric=k,p95=thr,ticks=len(vals)))
 write(ref,dict(reference='old105 legal expert states, phase-specific leave-family-out p95, separate units, no causal/recovery guarantee',rows=refs))
 for i,e in read(ROOT/name/'correction_entries.json').items():
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];x,p=features(t);m=t['dagger/expert_label_valid'][:].reshape(-1).astype(bool);v=np.flatnonzero(m);onset=np.zeros(len(m),bool);onset[v[:24]]=True;last=np.zeros(len(m),bool);last[e['takeover_tick']-1]=True
   for region,mask in [('last_student_diagnostic',last),('first24_legal',onset),('remaining_legal',m&~onset)]:
    for phase in set(p[mask]):
     q=mask&(p==phase)
     for k in keys:
      if (int(phase),k) not in trees or thresholds[int(phase),k] is None:continue
      dd=distance(trees[int(phase),k],x[k][q],k);thr=thresholds[int(phase),k]
      rows.append(dict(identity=i,region=region,phase=STAGES[int(phase)],metric=k,ticks=int(q.sum()),median=float(np.median(dd)),reference_p95=thr,far_ticks=int((dd>thr+1e-8).sum())))
 write(OUT/(name+'_support.json'),dict(rows=rows,scope='Only admitted legal experts in supervision. Last student diagnostic uses no expert label. Old105 includes same-family histories; reference leave-family thresholds are conservative descriptive support, not effective sample independence.'))
 print({region:{k:sum(r['far_ticks'] for r in rows if r['region']==region and r['metric']==k)/max(1,sum(r['ticks'] for r in rows if r['region']==region and r['metric']==k)) for k in keys} for region in ['first24_legal','remaining_legal']})
if __name__=='__main__':main(__import__('sys').argv[1])
