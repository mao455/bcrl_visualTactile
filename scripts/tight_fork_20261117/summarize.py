"""CPU analyses of recorded policy replay and original paired evidence."""
import csv,json,collections
import torch,numpy as np
from .common import *
from .original import load,rms,first
from scripts.tight_layout_20261116.publication import table,write

def main():
 root=ROOT/'replay_fp32';out=torch.load(root/'outputs.pt',weights_only=False);pairs=read(ROOT/'capture/plan.json')['pairs'];scales={}
 for role,pair in pairs.items():
  saved=torch.load(pair['backend'],map_location='cpu',weights_only=False,mmap=True);scales[role]=saved['model']['action_scale'].double()
 rows=[];sensory=[];original=[]
 for role,seed,part in CASES:
  indexed={(s,m):{r['tick']:r for r in out[f'{role}_{s}_{m}'] if r['seed']==seed} for s in ['baseline','candidate'] for m in ['baseline','candidate']}
  for t in range(121):
   values={}
   for (stream,model),rr in indexed.items():
    r=rr[t];values[stream,model]={k:r[k].double().reshape(-1) for k in ['nominal','command']}
    values[stream,model]['proposal_rad']=r['delta'].double().reshape(-1)/scales[role+('_baseline' if model=='baseline' else '')]
    values[stream,model]['unbounded']=values[stream,model]['nominal']+values[stream,model]['proposal_rad']
   row=dict(role=role,seed=seed,tick=t,plan_cursor=t%12)
   for k in ['nominal','proposal_rad','unbounded','command']:
    oo=values['baseline','baseline'][k];no=values['baseline','candidate'][k];nn=values['candidate','candidate'][k];on=values['candidate','baseline'][k]
    for label,v in dict(model_on_baseline=no-oo,feedback_on_candidate=nn-no,total=nn-oo,model_on_candidate=nn-on,feedback_on_baseline=on-oo).items():
     row[k+'_'+label+'_rms']=float(v.square().mean().sqrt())
     for segment,sl in [('arm',slice(0,7)),('hand',slice(7,29))]:row[k+'_'+label+'_'+segment+'_rms']=float(v[sl].square().mean().sqrt())
    row[k+'_decomposition_max']=float(((no-oo)+(nn-no)-(nn-oo)).abs().max())
   rows.append(row)
  a,b=load(path(role+'_baseline',seed,part)),load(path(role,seed,part));names=read(path(role,seed,part)/'metadata.json').get('joint_names',[])
  for t in [0,36,50,60,63,64,68,70,71,72,73,80,84,96,110]:
   row=dict(role=role,seed=seed,tick=t)
   for variant,z in [('baseline',a),('candidate',b)]:
    for key in ['nominal','residual_proposed_rad','command','joint','q_before','qd_before']:
     if variant=='candidate':
      delta=z[key][t]-a[key][t];idx=int(abs(delta).argmax());row[key+'_largest_joint_index']=idx;row[key+'_largest_signed_delta']=float(delta[idx])
    for k in ['object','goal']:
     for j,axis in enumerate('xyz'):row[variant+'_'+k+'_'+axis]=float(z[k][t,j])
    row[variant+'_contact_min']=int((np.linalg.norm(z['object_contact'][t],axis=-1)>.1).sum(-1).min())
    row[variant+'_contact_max']=int((np.linalg.norm(z['object_contact'][t],axis=-1)>.1).sum(-1).max())
    q=z['object'][t,3:].astype(float);g=z['goal'][t,3:].astype(float);row[variant+'_object_goal_angle_rad']=float(2*np.arccos(np.clip(abs(q@g)/(np.linalg.norm(q)*np.linalg.norm(g)),0,1)))
    row[variant+'_projection_dofs']=int((abs(z['execution_projection_rad'][t])>1e-6).sum())
    row[variant+'_command_jump']=rms(z['command'][t]-z['command'][t-1]) if t else 0
   original.append(row)
 for t in range(121):
  x=torch.load(ROOT/'inputs'/f'{t:03d}.pt',weights_only=False)
  for role,ia,ib,seed in [('A',1,0,28824002),('A',5,4,28824006),('B',3,2,28824002)]:
   row=dict(role=role,seed=seed,tick=t)
   for k in ['features']:
    row[k+'_pair_rms']=rms(x[k][ia].float().numpy()-x[k][ib].float().numpy())
   for k in ['depth_m','actuator_target_rad','contact_w_N']:
    row[k+'_pair_rms']=rms(x['packet'][k][ia].numpy()-x['packet'][k][ib].numpy())
   for k in ['robot','native','visual_memory','rear_state']:
    if k=='rear_state':continue
    row[k+'_pair_rms']=rms(x['controllers'][ia][k].numpy()-x['controllers'][ib][k].numpy())
   sensory.append(row)
 table(OUT/'same_input_radians.csv',rows);table(OUT/'original_key_ticks.csv',original);table(OUT/'capture_sensory_difference.csv',sensory)
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,3,figsize=(14,9),sharex=True)
 for i,(role,seed,_) in enumerate(CASES):
  rr=[r for r in rows if r['role']==role and r['seed']==seed]
  for j,k in enumerate(['nominal','proposal_rad','command']):
   for label in ['model_on_baseline','feedback_on_candidate','total']:
    axs[i,j].plot([r['tick'] for r in rr],[r[k+'_'+label+'_rms'] for r in rr],label=label)
   axs[i,j].set_title(f'{role} {seed} {k}');axs[i,j].set_yscale('log');axs[i,j].set_ylabel('RMS rad');axs[i,j].grid(alpha=.2);axs[i,j].legend(fontsize=7)
   for tick in [60,72,84,96]:axs[i,j].axvline(tick,color='gray',alpha=.2)
 for ax in axs[-1]:ax.set_xlabel('Control tick, new capture (not old exact replay)')
 fig.tight_layout();fig.savefig(OUT/'same_input_decomposition.png',dpi=140)
 print('summaries complete')
if __name__=='__main__':main()
