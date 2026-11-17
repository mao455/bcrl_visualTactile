"""Read-only coverage and trajectory diagnosis from frozen numeric evidence."""
import collections,json,csv,hashlib
from pathlib import Path
import numpy as np
import torch,h5py
from visuotactile.geometry_sampling import diagnostic_phases,STAGES
from visuotactile.anchor_sampling import supervision_rows,labels
from visuotactile.target_windows import with_target_window
from scripts.tight_layout_20261116.publication import table,write
BASE=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_long_20261117_01')
OUT=Path('research_docs/evidence/tight_long_20261117/balanced76')
def phases(t):
 return diagnostic_phases(t['diagnostics_after/object_pose_w'][:],t['diagnostics_after/palm_body_pose_w'][:],t['object_contact_force_w'][:],t['diagnostics_before/goal_pose_w'][:],t['diagnostics_after/goal_pose_w'][:],t['diagnostics_before/successes'][:].reshape(-1),t['diagnostics_after/successes'][:].reshape(-1),t['native_keypoints_max_dist'][:].reshape(-1)).numpy()
def main():
 root=BASE/'round06/learning'; data=json.loads((root/'A/state_data/data.json').read_text()); ids=data['splits']['train']
 numeric=torch.load(root/'A/state_data/numeric.pt',weights_only=False,map_location='cpu')['records']
 records=with_target_window(numeric,'recovery_onset_commit12_v1'); arrays={};rows=[]
 for i in ids:
  e=data['entries'][i];rec=records[i];mask,_=supervision_rows(rec);mask=mask.numpy();valid=np.flatnonzero(mask)
  with h5py.File(e['path']) as f:
   t=f['episodes'][e['episode']]['transition'];phase=phases(t)
   pose=t['diagnostics_after/object_pose_w'][:];goal=t['diagnostics_before/goal_pose_w'][:]
   angle=2*np.arccos(np.clip(abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
   onset=np.zeros(len(mask),bool)
   if e.get('trajectory_kind')=='expert_takeover_v1':onset[valid[:24]]=True
   arrays[i]=dict(phase=phase,mask=mask,onset=onset)
   source=e.get('source_group','original');row=dict(identity=i,source=source,family=e['family'],round=e.get('round_index',0),valid_ticks=int(mask.sum()),history_ticks=len(mask),onset24_ticks=int(onset.sum()),takeover=e.get('takeover_tick'),takeover_phase=e.get('trigger_receipt',{}).get('configuration',{}).get('stage','early' if source!='original' else None))
   row.update({s:int(((phase==k)&mask).sum()) for k,s in enumerate(STAGES)})
   row['onset_state_sha256']=hashlib.sha256(np.concatenate([t['diagnostics_before/joint_position_rad'][valid[0]],t['diagnostics_before/object_pose_w'][valid[0]],t['target_before_rad'][valid[0]]]).tobytes()).hexdigest()
   row['onset_orientation_rad']=float(angle[valid[0]])
   rows.append(row)
 table(OUT/'coverage.csv',rows)
 exposure=[]
 for role in 'AB':
  chosen=torch.load(root/role/'training/best.pt',weights_only=False,map_location='cpu',mmap=True)['updates']
  stats={i:collections.Counter() for i in ids};selected={i:collections.Counter() for i in ids};unique={i:set() for i in ids}
  for line in (root/role/'training/learning.jsonl').open():
   r=json.loads(line)
   for v in r['visits']:
    i=v['episode'];a=torch.tensor(v['anchor_sampling']['selected_anchors']);_,vm=labels(records[i],a,24)
    ticks=(a[:,None]+torch.arange(24)[None]).expand(vm.shape)[vm].numpy();ticks=ticks[ticks<len(arrays[i]['phase'])]
    c=collections.Counter(visits=1,anchors=len(a),target_slots=len(ticks),onset_slots=int(arrays[i]['onset'][ticks].sum()),original_early_slots=int((ticks<150).sum()) if rows[ids.index(i)]['source']=='original' else 0)
    c.update({s:int((arrays[i]['phase'][ticks]==k).sum()) for k,s in enumerate(STAGES)})
    stats[i].update(c);unique[i].update(ticks.tolist())
    if r['updates']<=chosen:selected[i].update(c)
  for i in ids:
   exposure.append(dict(role=role,identity=i,selected_updates=chosen,actual=dict(stats[i]),selected=dict(selected[i]),unique_target_ticks=len(unique[i])))
 table(OUT/'historical_exposure.csv',exposure)
 groups={}
 for source in ['original','A','B']:
  rr=[r for r in rows if r['source']==source]
  groups[source]=dict(trajectories=len(rr),families=len({r['family'] for r in rr}),states=len({r['onset_state_sha256'] for r in rr}),valid_ticks=sum(r['valid_ticks'] for r in rr),onset24=sum(r['onset24_ticks'] for r in rr),phases={s:sum(r[s] for r in rr) for s in STAGES})
 write(OUT/'coverage.json',dict(groups=groups,distinct_families=len({r['family'] for r in rows}),last4=[r for r in rows if r['round']==6],definitions='Exclusive existing geometry phases are diagnostic proxies, not a strict task funnel. Onset24 overlaps physical phases. Support transfer cannot be inferred from fingertip force alone. Slots overlap; unique ticks count once.',original_manifest_sha256=hashlib.sha256((root/'A/state_data/data.json').read_bytes()).hexdigest()))
 # Compare actual final held-out numeric trajectories only; never make training inputs.
 tests=[];curves=[]
 for part in range(5):
  folder=BASE/'final_test'/f'part{part}'/'development'
  if not (folder/'plan.json').exists():folder=BASE/'final_test'/f'part{part}'
  plan=json.loads((folder/'plan.json').read_text())
  for case in plan['cases']:
   p=folder/case['name'];r=json.loads((p/'result.json').read_text())
   with h5py.File(p/'numeric.h5') as f:
    t=f['episode/transition'];ph=phases(t);pose=t['diagnostics_after/object_pose_w'][:];goal=t['diagnostics_before/goal_pose_w'][:]
    rootdist=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1);angle=2*np.arccos(np.clip(abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1));contact=(np.linalg.norm(t['object_contact_force_w'][:],axis=-1)>.1).sum(-1).min(-1)
    row=dict(seed=case['spec']['seed'],role=case['role'],success=r['success'],steps=r['steps'],phases={s:int((ph==k).sum()) for k,s in enumerate(STAGES)},first_contact=next(iter(np.flatnonzero(contact>=2).tolist()),None),first_held=next(iter(np.flatnonzero(np.isin(ph,[2,3,4])).tolist()),None),first_near=next(iter(np.flatnonzero(ph==4).tolist()),None),final_angle=float(angle[-1]),min_keypoint=float(t['native_keypoints_max_dist'][:].min()))
    tests.append(row)
    for tick in range(0,len(ph),12):curves.append(dict(seed=row['seed'],role=row['role'],tick=tick,phase=STAGES[ph[tick]],root_m=float(rootdist[tick]),angle_rad=float(angle[tick]),contact_fingers=int(contact[tick]),object_z=float(pose[tick,2])))
 table(OUT/'final40_diagnosis.csv',tests);table(OUT/'final40_curves.csv',curves)
 write(OUT/'diagnosis.json',dict(coverage=groups,final40=tests,video_availability='Only seed28823000 A/B recorded video; remaining38 numeric only. Opened test states excluded from all future training/development selection.'))
 print(json.dumps(groups,indent=2))
if __name__=='__main__':main()
