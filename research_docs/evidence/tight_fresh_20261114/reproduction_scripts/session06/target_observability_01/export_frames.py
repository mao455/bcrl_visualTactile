"""Export unchanged RGB from one accepted, closed training correction only."""
from pathlib import Path
import json,hashlib,time
import h5py,numpy as np
from PIL import Image,ImageDraw
D=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/new_family_09/r09_A_expanded_session06_new64_s28600013_held60')
OUT=Path(__file__).parent
assert not (OUT/'frames.json').exists()
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
result=read(D/'result.json');receipt=read(D/'capture_complete.json');metadata=read(D/'metadata.json')
assert result['accepted_for_training'] and receipt['status']=='source_verified' and receipt['dataset_capture']['verified_source_fields']==185
assert result['dataset_capture']==receipt['dataset_capture']
path=Path(result['dataset_capture']['path']);st=path.stat();rows=[];canvases=[]
with h5py.File(path,'r') as f:
 g=f['episodes'][D.name]
 assert bool(g.attrs['complete']) and g['frames/rgb_u8'].shape==(449,3,3,252,448)
 for tick in (0,183,305,405):
  assert int(g['obs/step_id'][tick])==tick
  rgb=g['frames/rgb_u8'][tick]; poses=g['transition/diagnostics_before/object_pose_w'][tick]
  goal=g['transition/diagnostics_before/goal_pose_w'][tick];hole=g['transition/diagnostics_before/hole_pose_w'][tick]
  records=[];canvas=Image.new('RGB',(448*3,284),'white');draw=ImageDraw.Draw(canvas)
  for view,identity in enumerate(metadata['camera_ids']):
   name=f'obs_{tick:03d}_{identity}.png';image=Image.fromarray(np.transpose(rgb[view],(1,2,0)))
   image.save(OUT/name)
   assert np.array_equal(np.asarray(Image.open(OUT/name)),np.transpose(rgb[view],(1,2,0)))
   canvas.paste(image,(448*view,32));draw.text((448*view+8,8),f'obs {tick} | {identity}',fill='black')
   records.append(dict(path=str(OUT/name),sha256=sha(OUT/name),camera_id=identity,original_rgb_bytes_preserved=True))
  canvas.save(OUT/f'obs_{tick:03d}_three_views.png')
  row=dict(observation_index=tick,completed_transitions_before_observation=tick,time_s=float(g['obs/time_s'][tick]),
   next_action_authority='student' if int(g['transition/dagger/control_source'][tick])==0 else 'expert',
   diagnostic_only=dict(object_pose_w=poses.tolist(),goal_pose_w=goal.tolist(),hole_pose_w=hole.tolist(),
    native_subgoals_before=int(g['transition/diagnostics_before/successes'][tick]),
    retract_phase_before=bool(g['transition/diagnostics_before/retract_phase'][tick])),frames=records,
   contact_sheet=str(OUT/f'obs_{tick:03d}_three_views.png'))
  rows.append(row)
 flags=dict(goal_valid_true=int(g['obs/goal_valid'][:].sum()),task_ids=np.unique(g['obs/task_id'][:]).tolist(),
  goal_commands_all_zero=bool((g['obs/goal_command'][:]==0).all()))
value=dict(version='training_rgb_observability_export_v1',case=D.name,created_utc=time.time(),
 source=dict(path=str(path),verified_capture_sha256=result['dataset_capture']['sha256'],
  large_hdf_sha_recomputed=False,stat=dict(dev=st.st_dev,ino=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns),
  capture_receipt_sha256=sha(D/'capture_complete.json'),metadata_sha256=sha(D/'metadata.json'),result_sha256=sha(D/'result.json')),
 split='train',source_authority='True student prefix then actual expert; no validation/test reads.',
 interpretation='Raw RGB PNGs preserve original pixels. Labelled sheets add headers only. Pose values are diagnostic and not student inputs. Observation N is before action N, after N completed controls; no terminal/reset frame substitution.',
 task_command=metadata['task_command'],command_fields=flags,frames=rows,script_sha256=sha(Path(__file__)))
(OUT/'frames.json').write_text(json.dumps(value,indent=2)+'\n')
print(OUT/'frames.json')
