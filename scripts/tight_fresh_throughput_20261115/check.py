"""One focused contract check: real checkpoint/record, frozen patch fixture.

This tests batching numerics; live RGB-D/control acceptance remains necessary.
"""
import json,time,sys
from pathlib import Path
import h5py,torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from visuotactile.batched_controller import BatchedResidualController
from visuotactile.gpu_batch_controller import GPUBatchedResidualController
from visuotactile.compact import EpisodeFile
from visuotactile.batch_recording import snapshot,lane_arrays
from visuotactile.training_resources import atomic_json

def run(output,device='cpu'):
 from visuotactile.precision import state_fp32
 state_fp32();torch.set_num_threads(2);started=time.perf_counter()
 root=Path('outputs/visuotactile/run/tight_fresh_20261115_04/new_family_14')
 plan=json.loads((root/'plan.json').read_text());case=plan['cases'][0];d=root/case['name']
 meta=json.loads((d/'metadata.json').read_text());meta['joint_limits_rad']*=2
 saved=torch.load(plan['pairs']['A']['backend'],map_location='cpu',weights_only=False)
 class Patches:
  def forward_lanes(self,rgb,**kw):
   g=torch.Generator().manual_seed(17)
   return torch.randn(len(rgb),3,384,18,32,generator=g).to(rgb.device)*.02
 old=BatchedResidualController(saved,meta,Patches(),2,device=device,neural_batch=True,scheduler_index_mode='host')
 new=GPUBatchedResidualController(saved,meta,Patches(),2,device=device,neural_batch=True,scheduler_index_mode='host')
 result=json.loads((d/'result.json').read_text());errors=[];visual_errors=[];times={'old':0.,'new':0.}
 def load(group,tick):
  return {k:load(v,tick) if isinstance(v,h5py.Group) else torch.as_tensor(v[tick]).unsqueeze(0).repeat((2,)+(1,)*(v.ndim-1)) for k,v in group.items()}
 with h5py.File(d/'numeric.h5') as f,EpisodeFile(result['dataset_capture']['path']) as ef,torch.no_grad():
  full=next(iter(ef.episodes.items()))[1]
  for tick in range(15):
   p=load(f['episode/obs'],tick);diag=load(f['episode/transition/diagnostics_before'],tick)
   p['image']=torch.from_numpy(full['obs/image'][tick]).unsqueeze(0).repeat(2,1,1,1,1)
   p['depth_valid']=torch.from_numpy(full['obs/depth_valid'][tick]).unsqueeze(0).repeat(2,1,1,1,1)
   active=[True,tick<14]
   a=time.perf_counter();expected=old.act(p,diag,active);times['old']+=time.perf_counter()-a
   a=time.perf_counter();actual=new.act(p,diag,active);times['new']+=time.perf_counter()-a
   error=float((actual.cpu()-expected.cpu()).abs().max());errors.append(error)
   assert torch.isfinite(actual).all()
   for lane,running in enumerate(active):
    if not running:continue
    torch.testing.assert_close(new.lanes[lane].base.last_robot.cpu(),old.lanes[lane].base.last_robot.cpu(),atol=1e-6,rtol=1e-6)
    visual_errors.append(float((new.lanes[lane].visual.cpu()-old.lanes[lane].visual.cpu()).abs().max()))
   tr=load(f['episode/transition'],tick)
   # Teacher-forced same factual execution feedback, regardless of predictions.
   for controller in (old,new):
    for lane,running in enumerate(active):
     if running:controller.lanes[lane].base.command=tr['position_requested_rad'][lane:lane+1]
    controller.observe_execution(tr,active)
  s=snapshot(dict(obs=p,duplicate=p));assert s['obs']['image'] is s['duplicate']['image']
  retained=s['obs']['image'].copy();p['image'].zero_();assert (s['obs']['image']==retained).all()
  assert lane_arrays(s,1,2)['obs']['image'].shape[0]==1
 assert max(errors)<2e-5, ('Material absolute command change',max(errors))
 report=dict(passed=True,device=device,command_tolerance_rad=2e-5,max_visual_difference=max(visual_errors),seconds=time.perf_counter()-started,max_command_error_rad=max(errors),ticks=15,
  native_observations=29,plan_ticks=[0,12],times=times,real_checkpoint=plan['pairs']['A'],
  limitation='Tensor-equivalence check using real sensor/state rows and a deterministic frozen-patch fixture; not a simulation throughput claim')
 atomic_json(output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':run(Path(sys.argv[1]),sys.argv[2] if len(sys.argv)>2 else 'cpu')
