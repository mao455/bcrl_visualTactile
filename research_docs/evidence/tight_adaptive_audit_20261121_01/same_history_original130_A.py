from pathlib import Path
import sys,json,gc
from contextlib import nullcontext
import torch,numpy as np,h5py
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.gpu_batch_controller import GPUBatchedResidualController
from visuotactile.native_data import Episode,visual_records
from visuotactile.continuous_vision import encode_dense
from visuotactile.anchor_continue import residual_prediction
from visuotactile.precision import state_fp32
class Lease:
 def block(self):return nullcontext()
 def check(self):pass
def err(x,y):return float((torch.as_tensor(x).cpu()-torch.as_tensor(y).cpu()).abs().max())
@torch.no_grad()
def main():
 torch.set_num_threads(2);state_fp32();index=2;role='A';learning=RUN/'round02/learning';plan=json.load(open(RUN/'round02/collection01/plan.json'));identity=plan['cases'][0]['name'];data=json.load(open(learning/role/'state_data/data.json'));entry=data['entries'][identity];ep=Episode(learning/role/'state_data',identity,'I-HistoryDP');rec=visual_records(learning/role/'state_data',[identity],'G00')[identity];backend=learning/role/'backend'/role;cached=torch.load(backend/'nominal'/(identity+'.pt'),map_location='cpu',weights_only=False,mmap=True);results=[];allcommands={};allnominal={}
 for tag,path,seed in [('selected_cache_noise',backend/'training/best.pt',7091),('original130_contract_noise',Path(json.load(open(RUN/'round01/collection00_recovery02/plan.json'))['pairs'][role]['backend']),7091)]:
  saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
  with h5py.File(entry['path']) as f:
   g=f['episodes'][identity];o=g['obs'];tr=g['transition'];d=tr['dagger'];metadata=json.loads(f.attrs['metadata_json']);metadata['joint_limits_rad']=metadata['joint_limits_rad'][:1]
   controller=GPUBatchedResidualController(saved,metadata,None,1,device='cuda:0',neural_batch=True,scheduler_index_mode='host',vision_microbatch=1);c=controller.lanes[0];c.base.generator.manual_seed(seed)
   offline_command=residual_prediction(c.residual,cached,Lease()) if seed==7091 else None
   cuda_rec={k:v.cuda() if torch.is_tensor(v) else v for k,v in rec.items()};condition,anchors=encode_dense(c.model,cuda_rec,episode=ep)
   inputs={k:torch.from_numpy(v[:]) for k,v in o.items()};diag={k:torch.from_numpy(tr['diagnostics_before/'+k][:]) for k in ('joint_position_rad','joint_velocity_rad_s')};rows=[];commandsall=[];nominalall=[]
   for tick in range(len(rec['time'])):
    packet={k:v[tick:tick+1] for k,v in inputs.items()};packet.update(depth_m=torch.from_numpy(g['frames/depth_m'][tick:tick+1]),depth_valid=torch.from_numpy(g['frames/depth_valid'][tick:tick+1]));c.base.dino.set(ep.perception['patches'][tick].cuda());commands=packet['actuator_target_rad'].clone();controller._act_neural_batch(packet,{k:v[tick:tick+1] for k,v in diag.items()},[0],commands)
    row=dict(tick=tick,phase=tick%12,robot_max=err(c.base.last_robot,cached['robot'][tick]))
    if tick%12==0:row['condition_vs_full_serial_max']=err(c.base.last_condition,condition[tick//12:tick//12+1])
    if seed==7091:row.update(visual_max=err(c.visual,cached['visual'][tick:tick+1]),nominal_max=err(c.nominal,cached['base'][tick:tick+1]),previous_nominal_max=err(cached['previous_nominal'][tick],torch.zeros(29) if tick==0 else nominalall[-1]),command_vs_offline_backend_max=err(commands,offline_command[tick:tick+1]))
    if tag.startswith('actual'):row.update(self_command_max=err(commands,d['student_proposed_target_rad'][tick:tick+1]),self_nominal_max=err(c.nominal,d['nominal'][tick:tick+1]))
    commandsall.append(commands[0].cpu());nominalall.append(c.nominal[0].cpu());rows.append(row);c.base.command=torch.from_numpy(d['actual_issued'][tick:tick+1]).cuda();c.observe_execution({'position_applied_rad':torch.from_numpy(tr['position_applied_rad'][tick:tick+1])})
   allcommands[tag]=torch.stack(commandsall);allnominal[tag]=torch.stack(nominalall);metrics={k:max(r[k] for r in rows if k in r) for k in set().union(*(r.keys() for r in rows))-{'tick','phase'}};item=dict(tag=tag,path=str(path),seed=seed,identity=identity,metrics=metrics,rows=rows);results.append(item);print(json.dumps({k:v for k,v in item.items() if k!='rows'}),flush=True)
  del controller,c,saved,condition,cuda_rec;gc.collect();torch.cuda.empty_cache()
 contrasts=[]
 for a,b in [('original130_contract_noise','selected_cache_noise')]:
  q=allcommands[a]-allcommands[b];p=allnominal[a]-allnominal[b];contrasts.append(dict(a=a,b=b,command_max_rad=float(q.abs().max()),command_arm_rms_rad=float(q[:,:7].square().mean().sqrt()),command_hand_rms_rad=float(q[:,7:].square().mean().sqrt()),nominal_max_rad=float(p.abs().max()),nominal_arm_rms_rad=float(p[:,:7].square().mean().sqrt()),nominal_hand_rms_rad=float(p[:,7:].square().mean().sqrt()),scope='Same real complete teacher-forced mixed history; no counterfactual physical outcome'))
 out=dict(results=results,contrasts=contrasts);(OUT/'same_history_original130_A.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(contrasts),flush=True)
if __name__=='__main__':main()
