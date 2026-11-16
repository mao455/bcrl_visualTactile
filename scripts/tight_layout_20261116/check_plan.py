"""Check actual preparation routes without loading student weights or simulation."""
import tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.tight_layout_20261116 import learn
from visuotactile.training_resources import atomic_json

def main():
 original=learn.ROOT
 try:
  with tempfile.TemporaryDirectory(prefix='layout-plan-') as name:
   learn.ROOT=Path(name)
   atomic_json(learn.ROOT/'session.json',learn.read(learn.SOURCE/'session.json'))
   with patch('tools.tight_fresh_rollouts.completed_pair',side_effect=lambda root,role:{'role':role}):learn.evaluation_prepare()
   p=learn.read(learn.ROOT/'development/plan.json')
   assert len(p['cases'])==p['batch']['lanes']==8
   assert p['batch']['queue_profile']=='independent_lane_queue_v1'
   assert p['execution_backend']==learn.read(learn.RUNS/'tight_reliable_eval_20261116_01/28500100/plan.json')['execution_backend']
   for seed in learn.DEV_SEEDS:
    c=[x for x in p['cases'] if x['spec']['seed']==seed]
    assert len(c)==2 and {x['role'] for x in c}=={'A','B'} and c[0]['spec']['narrow_reset']==c[1]['spec']['narrow_reset']
   assert sum(x['video'] for x in p['cases'])==2
   atomic_json(learn.TREE/'research_docs/evidence/tight_layout_20261116/evaluation_plan_check.json',dict(passed=True,cases=8,distinct_development_families=4,matched_pair_reset=True,accepted_gpu_queue=True,video_cases=2,optimizer_updates=0,simulation_starts=0,required_owner_affinity='16 CPUs for the unchanged accepted Kit thread setting; torch/BLAS threads2'))
 finally:learn.ROOT=original
 print('Mixed-policy paired queue registration passes; no GPU/simulation')
if __name__=='__main__':main()
