from pathlib import Path
import sys,json,gc
import torch,h5py,numpy as np
from scipy.spatial.transform import Rotation as R
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.models.native_vision import NativeDinoSpatial
from visuotactile.native_data import Episode,depth_grid
from visuotactile.precision import state_fp32

def main():
 torch.set_num_threads(2);state_fp32();dino=NativeDinoSpatial().cuda().eval();results=[]
 for round in (1,2):
  root=RUN/f'round{round:02d}/learning';data=json.load(open(root/'B/state_data/data.json'));plan=json.load(open(RUN/f'round{round:02d}'/('collection00_recovery02' if round==1 else 'collection00')/'plan.json'))
  for role in ('A','B'):
   i=next(i for i,e in data['entries'].items() if e.get('round_index')==round and e.get('source_group')==role);e=data['entries'][i];ep=Episode(root/'B/state_data',i,'I-HistoryDP');rows=[]
   with h5py.File(e['path']) as f:
    g=f['episodes'][i];o=g['obs'];meta=json.loads(f.attrs['metadata_json']);legal=np.flatnonzero(g['transition/dagger/expert_label_valid'][:]);samples=sorted({0,1,11,12,max(0,e['takeover_tick']-1),e['takeover_tick'],int(legal[0]),len(o['time_s'])-1})
    # All-frame numeric extrinsic and tactile checks, independent of cached indices.
    parent=o['camera_parent_pose_w'][:];cam=o['camera_pose_w'][:,2];rp=R.from_quat(parent[:,[4,5,6,3]]);rc=R.from_quat(cam[:,[4,5,6,3]]);relative_xyz=rp.inv().apply(cam[:,:3]-parent[:,:3]);relative_rot=rp.inv()*rc;rig=meta['camera_rig']['wrist'];rr=R.from_quat(np.array(rig['quat_parent_ros_wxyz'])[[1,2,3,0]])
    target=o['finger_joint_position_rad'][1:];physical=g['transition/physics_samples/joint_position_rad'][:-1];forceerr=np.max(np.abs(o['contact_substeps_local_N'][1:]-g['transition/kinematics_after/contact_substeps_local_N'][:-1]))
    for tick in samples:
     rgb=torch.from_numpy(g['frames/rgb_u8'][tick]).cuda();depth=torch.from_numpy(g['frames/depth_m'][tick:tick+1]).cuda();valid=torch.from_numpy(g['frames/depth_valid'][tick:tick+1]).cuda()
     p=dino.forward_lanes(rgb[None],vision_microbatch=2)[0].cpu();stored=ep.perception['patches'][tick];dg=depth_grid(depth,valid)[0].cpu();previous=ep.perception['patches'][max(0,tick-1)];following=ep.perception['patches'][min(len(o['time_s'])-1,tick+1)]
     distances=[float((p.float()-x.float()).square().mean().sqrt()) for x in (previous,stored,following)]
     rows.append(dict(tick=tick,phase=tick%12,patch_max=float((p.float()-stored.float()).abs().max()),patch_rms=distances[1],patch_rms_to_tminus1_current_tplus1=distances,depth_max=float((dg.float()-ep.perception['depth'][tick].float()).abs().max()),raw_depth_median_m=float(depth[valid].median()),rgb_sha256=__import__('hashlib').sha256(rgb.cpu().numpy().tobytes()).hexdigest()))
    item=dict(round=round,identity=i,camera_ids=meta['camera_ids'],rows=rows,wrist_relative_translation_max_m=float(np.abs(relative_xyz-np.array(rig['pos_parent_m'])).max()),wrist_relative_rotation_max_rad=float((rr.inv()*relative_rot).magnitude().max()),finger_physics_chain_max_rad=float(np.abs(target-physical).max()),local_force_chain_max_N=float(forceerr),frames=len(o['time_s']))
   results.append(item);(OUT/'vision_verification.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(item),flush=True);del ep;gc.collect()
if __name__=='__main__':main()
