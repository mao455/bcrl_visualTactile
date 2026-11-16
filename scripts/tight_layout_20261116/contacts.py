"""Read-only surface-clearance and actual contact diagnostics from closed pilots."""
import json
from pathlib import Path
import h5py,numpy as np
from scripts.tight_layout_20261116.pilot import ROOT,box_distance
from visuotactile.initial_geometry import boxes
from visuotactile.training_resources import atomic_json
rows=[]
for p in sorted((ROOT/'pilot').glob('*/numeric.h5')):
 with h5py.File(p) as f:
  t=f['episode/transition'];after=t['diagnostics_after'];meta=json.loads(f.attrs['metadata_json']);assets=meta['config']['assets']
  objforce=t['object_contact_force_w'][:];total=t['contact_substeps_after_w_N'][:]
  touch=(np.linalg.norm(objforce,axis=-1)>.1).any(axis=(1,2));onset=int(np.flatnonzero(touch)[0]) if touch.any() else len(touch)
  z=after['object_pose_w'][:,2];base=float(np.median(z[24:48]));late=np.flatnonzero((z-base>.02)&(np.arange(len(z))>=48))
  records=[]
  for tick in sorted({12,24,36,48,max(0,onset-1)}):
   objects=boxes(assets['object_urdf'],after['object_pose_w'][tick]);holes=boxes(assets['receptive_urdf'],after['hole_pose_w'][tick]);tables=boxes(assets['table_urdf'],after['table_pose_w'][tick])
   records.append(dict(tick=tick,object_z_m=float(z[tick]),object_pose=after['object_pose_w'][tick].tolist(),
    base_surface_gap_m=min(box_distance(a,b)['distance_m'] for a in objects for b in holes),
    table_surface_gap_m=min(box_distance(a,b)['distance_m'] for a in objects for b in tables),
    object_velocity=after['object_velocity_w_m_s_rad_s'][tick].tolist()))
  rows.append(dict(case=p.parent.name,first_object_contact_tick=onset,settled_z_m=base,settled_z_range_24_48_m=float(np.ptp(z[24:48])),
   precontact_nonobject_contact_ticks=int((np.linalg.norm((total-objforce)[:onset],axis=-1)>.1).any(axis=(1,2)).sum()),
   first_lift2cm_after48_tick=int(late[0]) if len(late) else None,clearance_samples=records))
report=dict(rows=rows,limitation='Surface gaps use actual body poses and original box colliders, excluding contact/rest margins. Fingertip total minus object force cannot identify base rather than table/robot. Lift diagnostic excludes initial fall; it is not a success gate.')
atomic_json(Path('research_docs/evidence/tight_layout_20261116/contact_diagnostics.json'),report)
for r in rows:print(r['case'],r['settled_z_m'],[(x['tick'],round(x['base_surface_gap_m']*1000,3),round(x['table_surface_gap_m']*1000,3)) for x in r['clearance_samples']])
