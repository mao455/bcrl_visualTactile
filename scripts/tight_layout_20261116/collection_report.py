"""Summarize all closed expert attempts, actual reset diversity and surface gaps."""
import json,time
from pathlib import Path
import h5py,numpy as np
from scripts.tight_layout_20261116.pilot import TREE,read,box_distance
from visuotactile.initial_geometry import boxes
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
ROOT=TREE/'outputs/visuotactile/run/tight_layout10_20261116_01_recovery02'

def main():
 assert read(ROOT/'collection_complete.json')['complete']
 source=read(ROOT/'collection_results.json');rows=[]
 entries=read(ROOT/'A/state_data/data.json')['entries']
 for r in source:
  row={k:r.get(k) for k in ['identity','seed','split','legal','success','steps','software_complete']}
  if r['legal']:
   folder=Path(r['dataset_capture']['path']).parents[1]
   initial=read(folder/'initial_observable.json')
   row['initial_observable_sha256']=sha256(folder/'initial_observable.json')
   row['source_hdf5_sha256']=entries[r['identity']]['source_sha256']
   assert Path(entries[r['identity']]['path']).resolve()==Path(r['dataset_capture']['path']).resolve()
   row['initial_surface_gap_m']=read(folder/'layout_geometry.json')['base']['minimum']['distance_m']
   with h5py.File(folder/'numeric.h5') as f:
    d=f['episode/transition/diagnostics_after'];assets=json.loads(f.attrs['metadata_json'])['config']['assets']
    z=d['object_pose_w'][:,2];row['settled_z_range_24_48_m']=float(np.ptp(z[24:48]));row['surface_samples']=[]
    for tick in [24,36,48]:
     obj=boxes(assets['object_urdf'],d['object_pose_w'][tick]);hole=boxes(assets['receptive_urdf'],d['hole_pose_w'][tick]);table=boxes(assets['table_urdf'],d['table_pose_w'][tick])
     row['surface_samples'].append(dict(tick=tick,base_gap_m=min(box_distance(a,b)['distance_m'] for a in obj for b in hole),table_gap_m=min(box_distance(a,b)['distance_m'] for a in obj for b in table)))
  rows.append(row)
 report=dict(complete=True,utc=time.time(),source_results_sha256=sha256(ROOT/'collection_results.json'),
  counts={split:dict(attempted=sum(r['split']==split for r in rows),legal=sum(r['split']==split and r['legal'] for r in rows),native_success=sum(r['split']==split and r['success'] for r in rows)) for split in ['train','validation']},
  rows=rows,interpretation='All registered attempts retained; only native successful captures enter demonstration data. Real URDF box surface distances exclude contact margins and do not identify hand-base contact forces. No student results are inferred.')
 atomic_json(TREE/'research_docs/evidence/tight_layout_20261116/collection_complete.json',report)
 print(json.dumps(report['counts']))
if __name__=='__main__':main()
