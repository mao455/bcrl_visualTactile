"""Registered complete-pair candidate comparison on the same capability families."""
import copy,time
from scripts.tight_cycle_20261118.cycle import ROOT,PARENT,OUT,read,write,hashes

def prepare(part, *, parts=2):
 from tools.tight_fresh_rollouts import completed_pair
 part=int(part);assert parts in (2,4) and 0<=part<parts
 width=8//parts
 source=ROOT/'learning';assert read(source/'owner_exit.json')['children_reaped'] and read(source/'complete.json')['complete']
 old=read(ROOT/'capability/plan.json');reg=read(ROOT/'registration.json');session=read(ROOT/'session.json')
 pairs={r+'_baseline':completed_pair(PARENT,r) for r in 'AB'};pairs.update({r+'_candidate':completed_pair(source,r) for r in 'AB'})
 selected=set(reg['training_seeds'][part*width:part*width+width]+reg['new_development_seeds'][part*width:part*width+width])
 d=ROOT/f'comparison{part:02d}';d.mkdir();session.update(resource_root=str(d),owned_worker_roots=[str(d)])
 plan=copy.deepcopy(old);plan.update(pairs=pairs,cases=[],resource_root=str(d),source_sha256=hashes())
 for c in old['cases']:
  if c['role']!='A' or c['spec']['seed'] not in selected:continue
  for role in pairs:
   x=copy.deepcopy(c);x.update(role=role,name=role+'_'+c['spec']['identity'],video=c['spec']['seed'] in [reg['training_seeds'][part*width],reg['new_development_seeds'][part*width]])
   plan['cases'].append(x)
 plan['batch']['lanes']=len(plan['cases'])
 assert len(plan['cases'])==64//parts
 plan['evaluation_scope']=dict(protocol='Candidate versus72 full A/B same-wave; actual four-way pre-action check. Registered waves cover original16families. Open development, no independent final test.',selection='One candidate per role, independent validation-selected matched frontend/rear. No new hyperparameter grid.',test_used=False)
 write(d/'session.json',session);write(d/'plan.json',plan);print(d)
if __name__=='__main__':prepare(__import__('sys').argv[1])
