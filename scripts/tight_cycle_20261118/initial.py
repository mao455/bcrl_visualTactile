"""Actual pre-action collider clearance and pre-contact settling, no simulation."""
import h5py,numpy as np
from pathlib import Path
from scripts.tight_cycle_20261118.cycle import ROOT,DATA,OUT,read,write
from scripts.tight_layout_20261116.pilot import box_distance
from visuotactile.initial_geometry import boxes

def main():
 data=read(DATA/'A/state_data/data.json');originals=[data['entries'][i] for i in data['normalization_train_ids']];assets=read(Path(originals[0]['path']).parents[1]/'layout_geometry.json')['assets'];rows=[]
 sources=[]
 for e in originals:sources.append((e['family'],'original_training',e['family'],Path(e['path']),e['episode']))
 for c in read(ROOT/'capability/plan.json')['cases']:sources.append((c['name'],c['diagnostic_cohort'],c['spec']['family'],ROOT/'capability'/c['name']/'numeric.h5',None))
 for name,cohort,family,path,episode in sources:
  with h5py.File(path) as f:
   t=f['episodes'][episode]['transition'] if episode else f['episode/transition'];b=t['diagnostics_before'];pose={k:b[k][0] for k in ['object_pose_w','hole_pose_w','table_pose_w']}
   obj=boxes(assets['object_urdf'],pose['object_pose_w']);gaps={}
   for label,asset,pk in [('receptacle','receptive_urdf','hole_pose_w'),('table','table_urdf','table_pose_w')]:
    other=boxes(assets[asset],pose[pk]);gaps[label+'_clearance_m']=min(box_distance(a,b)['distance_m'] for a in obj for b in other)
   a=t['diagnostics_after'];objpose=a['object_pose_w'][24:48];tab=a['table_pose_w'][24:48];vel=a['object_velocity_w_m_s_rad_s'][24:48]
   rows.append(dict(case=name,cohort=cohort,family=family,**gaps,settled24_48_z_above_table_m=float(np.median(objpose[:,2]-tab[:,2])),settled24_48_median_speed_m_s=float(np.median(np.linalg.norm(vel[:,:3],axis=-1))),settled24_48_max_speed_m_s=float(np.linalg.norm(vel[:,:3],axis=-1).max())))
 write(OUT/'capability/clearance_settling.json',dict(rows=rows,scope='Native pre-action original URDF OBB solid distances, no contact/rest margin;24–48 completed control ticks precontact reference, actual dynamics included. No world-origin translation used.',assets=assets,visibility='Standard three cameras448x252 retained; everyframe receipt and initial legal gate. Four recorded full videos visually reviewed for representative visibility; no new all-family visibility estimator claimed.'))
 print({c:{k:[float(np.min([r[k] for r in rows if r['cohort']==c])),float(np.max([r[k] for r in rows if r['cohort']==c]))] for k in ['receptacle_clearance_m','settled24_48_z_above_table_m']} for c in ['original_training','training','new_development']})
if __name__=='__main__':main()
