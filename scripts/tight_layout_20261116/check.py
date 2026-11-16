"""One focused reset/geometry/budget acceptance entry; no model or GPU updates."""
import copy
import numpy as np
from scripts.tight_layout_20261116.pilot import read,reset_configuration,box_distance,TREE
from visuotactile.narrow_data import direct_configuration
from visuotactile.bounded_extension import Plateau,profile,fresh_layout_profile
from visuotactile.training_resources import atomic_json

def main():
 d=read(TREE/'configs/tight_state.json')['domain'];checks=[]
 for seed in [28816001,28816002,28816003,28818000,28817000]:
  original=direct_configuration(d,seed)
  assert reset_configuration(d,seed,0)==original
  for distance in [.05,.10]:
   a=reset_configuration(d,seed,distance);b=copy.deepcopy(original)
   x=a.pop('fixed_start_pose');y=b.pop('fixed_start_pose')
   assert a==b and x[2:]==y[2:] and np.isclose(np.linalg.norm(np.array(x[:2])-y[:2]),distance)
 checks.append('Only registered peg XY changes; zero displacement byte-value identical')
 a=(np.zeros(3),np.eye(3),np.ones(3));b=(np.array([3.,0,0]),np.eye(3),np.ones(3))
 assert abs(box_distance(a,b)['distance_m']-1)<1e-9
 b=(np.array([3.,3.,0]),np.eye(3),np.ones(3))
 assert abs(box_distance(a,b)['distance_m']-np.sqrt(2))<1e-9
 checks.append('Collision solid Euclidean separation includes diagonal distance')
 class Optimizer:
  def __init__(self):self.param_groups=[{'lr':1e-4}]
 for p in [profile(),fresh_layout_profile(),fresh_layout_profile('backend')]:
  a=Plateau(p);opt=Optimizer();resume=None
  for epoch in range(1,p['maximum_epochs']+1):
   event=a.observe(epoch,1.,opt)
   if epoch==10:
    resume=Plateau(p,copy.deepcopy(a.state));other=copy.deepcopy(opt)
   elif epoch>10:
    assert resume.observe(epoch,1.,other)==event and resume.state==a.state
   if a.state['stop']:break
  assert p['minimum_epochs']<=epoch<=p['maximum_epochs'] and a.state['reductions']==1 and opt.param_groups[0]['lr']==5e-5
  growing=Plateau(p);opt=Optimizer()
  for epoch in range(1,p['maximum_epochs']+1):growing.observe(epoch,.98**epoch,opt)
  assert not growing.state['stop'] and growing.state['reductions']==0
 checks.append('Historical and fresh schedules stop constant validation after minimum exposure/LR reduction; continuing improvement reaches finite max; state resume exact')
 out=dict(passed=True,checks=checks,optimizer_updates=0,gpu_work=False)
 atomic_json(TREE/'research_docs/evidence/tight_layout_20261116/check.json',out);print(out)
if __name__=='__main__':main()
