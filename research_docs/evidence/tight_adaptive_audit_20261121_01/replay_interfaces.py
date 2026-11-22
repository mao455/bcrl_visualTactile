"""Fixed-weight replay of preserved real histories; no simulator/optimizer."""
from contextlib import nullcontext
from pathlib import Path
import json,sys,time
import h5py,numpy as np,torch
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS')
SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS'
RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01'
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.gpu_batch_controller import GPUBatchedResidualController
from visuotactile.native_data import Episode,visual_records,cache_residual_episode,residual_batch
from visuotactile.models.native_dp import load_policy,load_residual
from visuotactile.continuous_vision import encode_dense
from visuotactile.trajectory_batch import encode_trajectories
from visuotactile.precision import state_fp32

def read(p):return json.loads(Path(p).read_text())
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def error(x,y):return float((torch.as_tensor(x).cpu()-torch.as_tensor(y).cpu()).abs().max())
class Lease:
    def block(self):return nullcontext()
    def check(self):pass

@torch.no_grad()
def run():
    torch.set_num_threads(2);state_fp32();results=[]
    for index in (1,2):
        learning=RUN/f'round{index:02d}/learning'
        folder=RUN/f'round{index:02d}'/('collection00_recovery02' if index==1 else 'collection00')
        plan=read(folder/'plan.json');case=plan['cases'][1];identity=case['name'];role='B'
        data=read(learning/role/'state_data/data.json');entry=data['entries'][identity]
        ep=Episode(learning/role/'state_data',identity,'I-HistoryDP')
        rec=visual_records(learning/role/'state_data',[identity],'G00')[identity]
        saved=torch.load(plan['pairs'][role]['backend'],map_location='cpu',weights_only=False)
        with h5py.File(entry['path']) as f:metadata=json.loads(f.attrs['metadata_json'])
        metadata['joint_limits_rad']=metadata['joint_limits_rad'][:1]
        controller=GPUBatchedResidualController(saved,metadata,None,1,device='cuda:0',neural_batch=True,scheduler_index_mode='host',vision_microbatch=1)
        c=controller.lanes[0];rows=[]
        with h5py.File(entry['path']) as f:
            g=f['episodes'][identity];o=g['obs'];tr=g['transition'];d=tr['dagger']
            inputs={key:torch.from_numpy(value[:]) for key,value in o.items()}
            diag={key:torch.from_numpy(tr['diagnostics_before/'+key][:]) for key in ('joint_position_rad','joint_velocity_rad_s')}
            for tick in range(len(rec['time'])):
                packet={key:value[tick:tick+1] for key,value in inputs.items()}
                packet.update(depth_m=torch.from_numpy(g['frames/depth_m'][tick:tick+1]),depth_valid=torch.from_numpy(g['frames/depth_valid'][tick:tick+1]))
                c.base.dino.set(ep.perception['patches'][tick].cuda())
                commands=packet['actuator_target_rad'].clone()
                controller._act_neural_batch(packet,{k:v[tick:tick+1] for k,v in diag.items()},[0],commands)
                expected=torch.from_numpy(d['student_proposed_target_rad'][tick:tick+1])
                rows.append(dict(tick=tick,phase=tick%12,source=int(d['control_source'][tick]),command_max_rad=error(commands,expected),nominal_max_rad=error(c.nominal,torch.from_numpy(d['nominal'][tick:tick+1])),robot178_max=error(c.base.last_robot,rec['robot'][tick])))
                # Actual expert takeover may execute a different command. Feed the real acknowledgement.
                c.base.command=torch.from_numpy(d['actual_issued'][tick:tick+1]).cuda()
                c.observe_execution({'position_applied_rad':torch.from_numpy(tr['position_applied_rad'][tick:tick+1])})
        item=dict(round=index,identity=identity,source_pair=plan['pairs'][role],frames=len(rows),rows=rows,
            maximum_command_error_rad=max(r['command_max_rad'] for r in rows),maximum_nominal_error_rad=max(r['nominal_max_rad'] for r in rows),maximum_robot_error=max(r['robot178_max'] for r in rows))
        results.append(item);write('online_self_replay.json',results)
        print(json.dumps({k:v for k,v in item.items() if k not in ('rows','source_pair')}),flush=True)
        del controller,c,saved;torch.cuda.empty_cache()
        # Rebuild selected DP -> current nominal on the exact real observations and fixed seed7091.
        backend=learning/role/'backend'/role
        cached=torch.load(backend/'nominal'/(identity+'.pt'),map_location='cpu',weights_only=False)
        base_saved=torch.load(learning/role/'training/best.pt',map_location='cpu',weights_only=False)
        base=load_policy(base_saved,'cuda:0').eval().requires_grad_(False)
        cfg=read(backend/'configuration.json')['residual_pair']
        regenerated=cache_residual_episode(base,rec,entry,Lease(),cache_identity=cached['cache_identity'],transfer_mode='episode',scheduler_index_mode='host',offline_planning_profile=cfg['cache_offline_planning_profile'])
        checks={key:error(regenerated[key],cached[key]) for key in ('robot','time','visual','plans','base','previous_nominal','target')}
        checks.update(plan_valid_equal=torch.equal(regenerated['plan_valid'],cached['plan_valid']),mask_equal=torch.equal(regenerated['supervision_mask'],cached['supervision_mask']))
        selected=torch.tensor(sorted({0,1,11,12,int(entry['takeover_tick']),int(entry['takeover_tick'])+2}),dtype=torch.long,device='cuda')
        # Same actual trained encoder through B2 training and ordinary complete-history paths.
        cuda_rec={k:v.cuda() if torch.is_tensor(v) else v for k,v in rec.items()}
        a,_=encode_dense(base,cuda_rec,episode=ep,stop=len(rec['time']),selected_anchors=selected)
        b=encode_trajectories(base,[cuda_rec],[ep],selected_anchors=[selected])
        checks['training_B2_vs_serial_condition_max']=error(a,b[0][0])
        out=dict(round=index,identity=identity,checks=checks,source_DP=str(learning/role/'training/best.pt'),optimizer_updates=0,physics_updates=0)
        write(f'nominal_round{index:02d}.json',out);print(json.dumps(out),flush=True)
        del base,base_saved,cached,regenerated;torch.cuda.empty_cache()
if __name__=='__main__':run()
