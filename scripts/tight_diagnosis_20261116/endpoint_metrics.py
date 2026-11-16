"""Selected/last/source inference on fixed original2 + latest4, one feature read per record.
No optimization, held-out tuning, simulator, synthetic history or cached learned state.
"""
import sys,json,time,csv
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np,h5py,torch
from tools.tight_fresh_train import new_model,supervised_conditions,predict_on_native_clock
from visuotactile.native_data import Episode,visual_records
from visuotactile.anchor_sampling import labels
from visuotactile.target_windows import with_target_window
from visuotactile.offline_planning import predict_contiguous,profile as offline_profile
from visuotactile.position_control import bounded_targets
from visuotactile.frozen_visual_io import mapped_profile
from visuotactile.precision import state_fp32
from visuotactile.episode_io import available_memory
R=Path('outputs/visuotactile/run/tight_fresh_20261116_04/round_19').resolve();O=Path('research_docs/evidence/tight_diagnosis_20261116')
assert not (O/'dp_endpoints.csv').exists();torch.set_num_threads(2);state_fp32();began=time.time();rows=[];bindings=[]
def read(p):return json.loads(p.read_text())
sources=read(R.parent/'night/state.json')['sources'];data=read(R/'A/state_data/data.json');ids=data['normalization_train_ids'][:2]+data['aggregation']['new_training_ids']
assert set(ids)<=set(data['splits']['train'])
with torch.no_grad():
 for role in 'AB':
  dr=R/role/'state_data';records=with_target_window(visual_records(dr,ids,'G00',device='cuda:0'),read(R/role/'continuation.json').get('target_window_contract'));stats=torch.load(dr/'statistics.pt',weights_only=False)
  models={}
  for label,p in [('source',Path(sources[role])/role/'training/best.pt'),('selected',R/role/'training/best.pt'),('last',R/role/'training/last.pt')]:
   saved=torch.load(p,map_location='cpu',weights_only=False,mmap=True);m=new_model(role,stats).eval();m.load_state_dict(saved['ema']);models[label]=m
   bindings.append(dict(role=role,checkpoint=label,path=str(p),epoch=saved.get('epoch'),updates=saved.get('updates')));del saved
  for identity in ids:
   assert time.time()-began<240 and available_memory()>10*2**30 and torch.cuda.mem_get_info()[0]>4*2**30
   rec=records[identity];ep=Episode(dr,identity,'I-HistoryDP',frozen_visual_io=mapped_profile());entry=data['entries'][identity]
   with h5py.File(entry['path']) as f:
    g=f['episodes'][entry['episode']];t=g['transition'];force=t['contact_substeps_after_w_N'][:];pose=t['diagnostics_after/object_pose_w'][:];native=t['diagnostics_after/lifted_object'][:].reshape(-1).astype(bool)
   n=len(pose);touch=(np.linalg.norm(force,axis=-1)>.1).any(axis=(1,2));two=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)>=2
   height=pose[:,2]>=np.median(pose[24:48,2])+.02;first=int(np.flatnonzero(touch)[0]) if touch.any() else n
   held=np.zeros(n,bool)
   for j in range(59,n):held[j]=bool(height[j-59:j+1].all() and two[j-59:j+1].sum()>=54 and np.convolve((~two[j-59:j+1]).astype(int),np.ones(4,dtype=int),'valid').max()<4)
   bins={'all_legal':np.ones(n,bool),'before150':np.arange(n)<150,'pre_first_contact':np.arange(n)<first,'first_contact12':(np.arange(n)>=first)&(np.arange(n)<first+12),'two_finger_contact':two,'native_lift':native,'contact_height':two&height,'hold60_window_end':held,'later_ge150':np.arange(n)>=150}
   for label,m in models.items():
    c,a=supervised_conditions(m,rec,episode=ep);pred=predict_contiguous(m,c,a,configuration=offline_profile(32));pred=bounded_targets(pred.flatten(0,1),rec['joint_limits']).reshape_as(pred);truth,valid=labels(rec,a,24)
    valid=valid[:,:12];err=(pred[:,:12]-truth[:,:12])[valid];ticks=(a[:,None]+torch.arange(12,device=a.device)[None])[valid].cpu().numpy();e=err.cpu().numpy();normalized=(err*m.action_scale).cpu().numpy()
    for phase,b in bins.items():
     mask=b[ticks];v=e[mask];z=normalized[mask]
     if not len(v):continue
     rows.append(dict(role=role,checkpoint=label,identity=identity,cohort='original2' if identity in ids[:2] else 'latest4',phase=phase,ticks=len(v),arm_rmse_rad=float(np.sqrt((v[:,:7]**2).mean())),hand_rmse_rad=float(np.sqrt((v[:,7:]**2).mean())),arm_normalized_mse=float((z[:,:7]**2).mean()),hand_normalized_mse=float((z[:,7:]**2).mean())))
   del ep
   print(role,identity,round(time.time()-began,2),flush=True)
  del models,records,m;torch.cuda.empty_cache()
with (O/'dp_endpoints.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(O/'dp_endpoints.json').write_text(json.dumps(dict(complete=True,seconds=time.time()-began,bindings=bindings,ids=ids,feature_episode_reads=12,model_episode_inferences=36,optimizer_updates=0,simulator_starts=0,scope='Fixed original2 plus latest4 train-only endpoints; native5Hz/commit12 DDIM teacher forcing. Overlapping physical bins from real action-after states, not extra losses or full-pool epoch curves.'),indent=2)+'\n')
