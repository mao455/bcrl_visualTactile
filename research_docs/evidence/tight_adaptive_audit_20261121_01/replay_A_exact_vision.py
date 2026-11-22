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
from visuotactile.models.native_vision import NativeDinoSpatial

def read(p):return json.loads(Path(p).read_text())
def write(name,x):(OUT/name).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
def error(x,y):return float((torch.as_tensor(x).cpu()-torch.as_tensor(y).cpu()).abs().max())
class Lease:
    def block(self):return nullcontext()
    def check(self):pass

@torch.no_grad()
def run():
    torch.set_num_threads(2);state_fp32();results=[];dino=NativeDinoSpatial().cuda().eval()
    for index in (1,2):
        learning=RUN/f'round{index:02d}/learning'
        folder=RUN/f'round{index:02d}'/('collection02' if index==1 else 'collection01')
        plan=read(folder/'plan.json');case=plan['cases'][0];identity=case['name'];role='A'
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
                patch=dino.forward_lanes(torch.from_numpy(g['frames/rgb_u8'][tick:tick+1]).cuda(),vision_microbatch=1)[0]
                patch_error=error(patch,ep.perception['patches'][tick])
                c.base.dino.set(patch)
                commands=packet['actuator_target_rad'].clone()
                controller._act_neural_batch(packet,{k:v[tick:tick+1] for k,v in diag.items()},[0],commands)
                expected=torch.from_numpy(d['student_proposed_target_rad'][tick:tick+1])
                rows.append(dict(patch_max=patch_error,tick=tick,phase=tick%12,source=int(d['control_source'][tick]),command_max_rad=error(commands,expected),nominal_max_rad=error(c.nominal,torch.from_numpy(d['nominal'][tick:tick+1])),robot178_max=error(c.base.last_robot,rec['robot'][tick])))
                # Actual expert takeover may execute a different command. Feed the real acknowledgement.
                c.base.command=torch.from_numpy(d['actual_issued'][tick:tick+1]).cuda()
                c.observe_execution({'position_applied_rad':torch.from_numpy(tr['position_applied_rad'][tick:tick+1])})
        item=dict(round=index,identity=identity,source_pair=plan['pairs'][role],frames=len(rows),rows=rows,
            maximum_command_error_rad=max(r['command_max_rad'] for r in rows),maximum_nominal_error_rad=max(r['nominal_max_rad'] for r in rows),maximum_robot_error=max(r['robot178_max'] for r in rows))
        results.append(item);write('online_self_replay_A_exact_vision.json',results)
        print(json.dumps({k:v for k,v in item.items() if k not in ('rows','source_pair')}),flush=True)
        del controller,c,saved;torch.cuda.empty_cache()

if __name__=='__main__':run()
