"""Additional fixed-weight gradients and all nominal-cache provenance; zero updates."""
from training_verify import *
from visuotactile.compact import sha256

def main():
 torch.set_num_threads(2);state_fp32();provenance=[]
 for rnd in (1,2):
  for role in ('A','B'):
   root=RUN/f'round{rnd:02d}/learning';backend=root/role/'backend'/role;cfg=read(backend/'configuration.json')['residual_pair'];expected=sha256(root/role/'training/best.pt');identities=[];bad=[]
   for path in sorted((backend/'nominal').glob('*.pt')):
    rec=torch.load(path,map_location='cpu',weights_only=False,mmap=True);identity=rec['cache_identity'];identities.append(rec['identity']);
    if identity['base_checkpoint_sha256']!=expected or rec['nominal_supervision']!=('q_reached' if role=='A' else 'u_expert'):bad.append(path.name)
    del rec
   assert cfg['bases'][role]['sha256']==expected and not bad
   provenance.append(dict(round=rnd,role=role,entries=len(identities),base_sha256=expected,bad=bad,nominal_supervision='q_reached' if role=='A' else 'u_expert',cache_profile=cfg['cache_offline_planning_profile']))
 write('nominal_provenance.json',provenance);print('provenance',provenance,flush=True)
 root=RUN/'round02/learning';result=[]
 for role,identity in [('A','r21_A_layout10_train_28817024_post_subgoal'),('B','r21_B_layout10_train_28817012_post_subgoal'),('B','r20_B_layout10_train_28817000_instability')]:
  dr,data,records,stats=inputs(root,role);saved=torch.load(root/role/'training/best.pt',map_location='cpu',weights_only=False,mmap=True);g=gradient(root,role,data,records,saved,identity);g['role']=role;result.append(g);write('extra_gradients.json',result);print('gradient',role,identity,g['loss'],g['gradient_norm_before_clip'],flush=True);del saved,records;gc.collect();torch.cuda.empty_cache()
 import replay_A_exact_vision,rebuild_A_nominal
 replay_A_exact_vision.run();rebuild_A_nominal.run()
if __name__=='__main__':main()
