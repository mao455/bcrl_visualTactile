"""Selected-vs-budget8 endpoint fits with their own bound DP inputs; no updates."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from scripts.tight_budget_20261116.run import ROOT,OLD,read
from tools.tight_fresh_train import inputs,new_model,fresh_objective
from visuotactile.bounded_extension import dp_diagnostics,rear_diagnostics,cohorts
from visuotactile.frozen_visual_io import mapped_profile
from visuotactile.offline_planning import profile
from visuotactile.precision import state_fp32
from visuotactile.fresh_visit_sampling import load_profile
from visuotactile.training_resources import atomic_json
from visuotactile.state_resources import lease_for
from visuotactile.models.native_dp import TargetResidual
from visuotactile.compact import sha256
assert read(ROOT/'training_complete.json')['complete']
torch.set_num_threads(2);state_fp32();lease=lease_for(ROOT)
output=ROOT/'endpoint_comparison.json'
assert not output.exists(),'Preserve prior completed endpoints'
rows=[];began=time.time()
def exposure(saved,data,network):
 counts=saved.get('sampling_exposure' if network=='dp' else 'episode_counts',{})
 return dict(optimizer_updates=saved['updates'],ema_updates=saved.get('ema_optimization_step'),
  latest_trajectory_visits={i:counts.get(i,0 if i not in saved.get('train_ids',[]) else None) for i in data['aggregation']['new_training_ids']},
  counts_available=bool(counts),total_trajectory_visits=sum(counts.values()) if counts and all(isinstance(v,int) for v in counts.values()) else None,
  supervised_ticks=saved.get('supervised_ticks'),target_ticks=saved.get('target_ticks'),
  note='Selected checkpoint exposure, not final-run exposure; source checkpoints predate latest4.')
with torch.no_grad():
 for role in 'AB':
  dr,data,records,stats=inputs(ROOT,role)
  objective,_,_=fresh_objective(ROOT,role,data,records,geometry_sampling_profile=load_profile(ROOT)['geometry'] if role=='A' else None)
  for label,p in [('source',Path(read(ROOT/role/'continuation.json')['path'])),('baseline8_selected',OLD/role/'training/best.pt'),('extended_selected',ROOT/role/'training/best.pt')]:
   saved=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
   with lease.block():
    model=new_model(role,stats).eval();model.load_state_dict(saved['ema']);result=dp_diagnostics(model,records,data,objective,mapped_profile(),profile(32),lease)
   rows.append(dict(role=role,network='dp',label=label,checkpoint=str(p),sha256=sha256(p),epoch=saved['epoch'],updates=saved['updates'],exposure=exposure(saved,data,'dp'),diagnostics=result))
   del model,saved;torch.cuda.empty_cache()
  del records,objective
  ids=[i for values in cohorts(data).values() for i in values]
  for label,root,p in [('baseline8_selected',OLD,OLD/role/'backend'/role/'training/best.pt'),('extended_dp_rear8',ROOT,ROOT/role/'backend'/role/'training/best_through_epoch8.pt'),('extended_selected',ROOT,ROOT/role/'backend'/role/'training/best.pt')]:
   saved=torch.load(p,map_location='cpu',mmap=True,weights_only=False);cfg=saved['configuration'];cache=root/role/'backend'/role/'nominal';index=read(cache/'index.json')
   assert index['base_sha256']==cfg['base_sha256']
   records={i:torch.load(cache/(i+'.pt'),map_location='cpu',mmap=True,weights_only=False) for i in ids}
   command_stats=torch.load(cfg['residual_statistics_path'],map_location='cpu',weights_only=False)
   with lease.block():
    model=TargetResidual(command_stats,cfg['fixed_residual_scales'],execution_adapter=cfg['execution_adapter'],loss_contract=cfg['loss_contract'],history_contract=cfg['history_contract'],output_contract=cfg['output_contract']).cuda().eval()
    model.load_state_dict(saved['model']);result=rear_diagnostics(model,records,data,lease)
   rows.append(dict(role=role,network='rear',label=label,checkpoint=str(p),sha256=sha256(p),epoch=saved['epoch'],updates=saved['updates'],base_sha256=cfg['base_sha256'],exposure=exposure(saved,data,'rear'),diagnostics=result))
   del records,model,saved;torch.cuda.empty_cache()
atomic_json(output,dict(complete=True,seconds=time.time()-began,rows=rows,optimizer_updates=0,simulation_starts=0,scope='Fixed train4/historical4/latest4/validation6; each rear gets its OWN selected DP input, current-weight history; executed-command errors, no autonomous claim.'))
print(output)
