"""Native final32 statistics and recorded-output/execution-chain reconstruction."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,torch,h5py
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.position_control import project_processed_position_targets,REACHABLE_POSITION_EXECUTION

def read(p):return json.loads(Path(p).read_text())
def err(a,b):return float(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float)).max())
def first(x):
 a=np.flatnonzero(x);return int(a[0]) if len(a) else None
def main():
 torch.set_num_threads(2);rows=[];pairs=[];cache={};sources={};hashes={}
 def sha(p):
  p=str(p)
  if p not in hashes:
   h=hashlib.sha256()
   with open(p,'rb') as f:
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
   hashes[p]=h.hexdigest()
  return hashes[p]
 for folder in sorted((RUN/'final_test').glob('paired2_wave*')):
  plan=read(folder/'plan.json');modelcheck={}
  for case in plan['cases']:
   role=case['role'];binding=plan['pairs'][role];p=folder/case['name'];result=read(p/'result.json');t=np.load(p/'trace.npz');plans=np.load(p/'plans.npz')['plans'];n=len(t['command'])
   for k in ('frontend','backend'):modelcheck[role+'_'+k]=sha(binding[k])==binding[k+'_sha256']==result['complete_student_policy'][k+'_sha256']
   saved=torch.load(binding['backend'],weights_only=False,map_location='cpu',mmap=True);scale=saved['model']['action_scale'].numpy();del saved
   with h5py.File(p/'numeric.h5') as f:
    g=f['episode'];o=g['obs'];tr=g['transition'];before=tr['diagnostics_before'];after=tr['diagnostics_after'];native=tr['native_success_diagnostic'];meta=read(p/'metadata.json');limits=torch.tensor(meta['joint_limits_rad'][0]);previous=torch.from_numpy(tr['target_before_rad'][:]);proposed=t['nominal']+t['residual_normalized']/scale
    joint=np.clip(proposed,limits[:,0].numpy(),limits[:,1].numpy());execution=project_processed_position_targets(torch.from_numpy(joint),previous,limits,REACHABLE_POSITION_EXECUTION).numpy()
    currentnom=np.clip(np.stack([plans[tick//12,tick%12] for tick in range(n)]),limits[:,0].numpy(),limits[:,1].numpy())
    checks=dict(joint_clipped_preprojection_rad=err(joint,t['execution_preprojection_target_rad']),nominal_plan_cursor=err(currentnom,t['nominal']),command_from_proposal=err(execution,t['command']),request_raw=err(t['command'],tr['position_requested_rad'][:]),actuator_both_substeps=err(t['actuator_targets'],np.repeat(t['target'][:,None],2,axis=1)),command_to_target=err(t['command'],t['target']),ack=err(tr['position_applied_rad'][:],t['target']),previous_target=err(t['execution_previous_target_rad'],tr['target_before_rad'][:]),q_chain=err(after['joint_position_rad'][:-1],before['joint_position_rad'][1:]),target_chain=err(tr['target_after_rad'][:-1],tr['target_before_rad'][1:]),contact_clock=err(o['contact_time_s'][1:],tr['contact_time_after_s'][:-1]),contact_chain=err(o['contact_substeps_w_N'][1:],tr['contact_substeps_after_w_N'][:-1]))
    reasons={k:bool(v[-1]) for k,v in tr['termination_reasons'].items()};success=bool(native['complete'][:].any());assert success==reasons['max_successes']==result['success']
    objbefore=before['object_pose_w'][:];objafter=after['object_pose_w'][:];objbefore[:,:3]-=before['base_pose_w'][:,:3];key=tr['native_keypoints_max_dist'][:];contact=np.linalg.norm(tr['object_contact_force_w'][:],axis=-1);finger_count=(contact.max(1)>.1).sum(-1)
    event=dict(first_any_contact=first(finger_count>0),first_two_fingers=first(finger_count>=2),last_two_fingers=int(np.flatnonzero(finger_count>=2)[-1]) if (finger_count>=2).any() else None,first_lift=first(after['lifted_object'][:]),first_subgoal=first(after['successes'][:]>0),first_native_alignment=first(tr['native_alignment_condition'][:]),first_near_goal=first(native['near_goal'][:]),first_native_complete=first(native['complete'][:]),minimum_keypoints_mm=1000*float(key.min()),last_keypoints_mm=1000*float(key[-1]),last_object_z_m=float(objafter[-1,2]),minimum_object_z_m=float(objafter[:,2].min()),maximum_tracking_rad=float(np.abs(tr['target_after_rad'][:]-after['joint_position_rad'][:]).max()))
    item=dict(role=role,seed=result['seed'],frames=n,success=success,reasons=reasons,checks=checks,events=event,path=str(p),model_checks=modelcheck.copy(),camera_frames_monotone=bool((np.diff(o['camera_frame'][:],axis=0)>0).all()),subgoal_transitions=np.flatnonzero(np.diff(before['successes'][:])!=0).tolist())
    snap=sorted(set([0,12,60,120,180,240,min(n-1,300),n-1]+[x for x in event.values() if isinstance(x,int) and 0<=x<n]));item['snapshots']=[dict(tick=k,command=t['command'][k].tolist(),joint=t['joint'][k].tolist(),object_pose=objafter[k].tolist(),finger_contacts=int(finger_count[k]),keypoints_mm=float(1000*key[k])) for k in snap]
    cache[role]=(t['command'].copy(),before['joint_position_rad'][:],objbefore,finger_count,key)
   rows.append(item)
  parent,new=[cache[c['role']] for c in plan['cases']];n=min(len(parent[0]),len(new[0]));pc,nc=parent[0][:n],new[0][:n];pq,nq=parent[1][:n],new[1][:n];po,no=parent[2][:n],new[2][:n]
  dif=dict(command_first_gt1e_6_rad=first(np.max(np.abs(pc-nc),axis=-1)>1e-6),joint_pre_first_gt1e_4_rad=first(np.max(np.abs(pq-nq),axis=-1)>1e-4),object_pre_first_gt1mm=first(np.linalg.norm(po[:,:3]-no[:,:3],axis=-1)>.001),finger_contact_count_first_difference=first(parent[3][:n]!=new[3][:n]),initial_q_max_rad=err(pq[0],nq[0]),initial_object_position_max_m=err(po[0,:3],no[0,:3]))
  pairs.append(dict(wave=folder.name,roles=[c['role'] for c in plan['cases']],seed=case['spec']['seed'],differences=dif,paired_initial_check=read(folder/'paired_initial_check.json')))
 summary={role:dict(success=sum(r['success'] for r in rows if r['role']==role),total=sum(r['role']==role for r in rows)) for role in ('A_parent','A_new','B_parent','B_new')};out=dict(summary=summary,records=rows,pairs=pairs,sha256=hashes,priority_pairs=[p for p in pairs if (p['seed'],p['roles'][0][0]) in [(28839004,'A'),(28839005,'B'),(28839007,'B'),(28839002,'B')]])
 (OUT/'final_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(summary));print('max errors',{k:max(r['checks'][k] for r in rows) for k in rows[0]['checks']});print('priority',json.dumps(out['priority_pairs'])[:2500]);print('events',json.dumps([{k:r[k] for k in ('role','seed','success','events')} for r in rows if (r['seed'],r['role'][0]) in [(28839004,'A'),(28839005,'B'),(28839007,'B'),(28839002,'B')]]))
if __name__=='__main__':main()
