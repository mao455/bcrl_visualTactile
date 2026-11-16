"""Preparation regression: real new-layout resets and negative admission checks."""
import copy,tempfile
from pathlib import Path
from unittest.mock import patch
from scripts.tight_layout_20261116 import dagger,learn
from tools.tight_fresh_batch_run import validate_takeover_plan
from visuotactile.training_resources import atomic_json


def main():
 source=learn.ROOT
 with tempfile.TemporaryDirectory(prefix='layout-dagger-check-') as directory:
  temp=Path(directory);fixture=temp/'source';fixture.mkdir();(fixture/'development').mkdir()
  for role in 'AB':
   (fixture/role).mkdir();(fixture/role/'state_data').symlink_to(source/role/'state_data',target_is_directory=True)
  atomic_json(fixture/'session.json',learn.read(source/'session.json'))
  atomic_json(fixture/'complete.json',dict(complete=True));atomic_json(fixture/'owner_exit.json',dict(children_reaped=True))
  atomic_json(fixture/'development/result.json',dict(software_complete=True))
  atomic_json(fixture/'development/paired_initial_check.json',dict(passed=True))
  atomic_json(fixture/'development/plan.json',learn.read(learn.RUNS/'tight_reliable_eval_20261116_01/28500100/plan.json'))
  with patch('tools.tight_fresh_rollouts.completed_pair',side_effect=lambda root,role:dict(role=role)):
   dagger.prepare(fixture,temp/'round',1,0,'approach')
  plan=learn.read(temp/'round/collection/plan.json');validate_takeover_plan(plan)
  assert len(plan['cases'])==plan['batch']['lanes']==8
  for i in range(4):
   a,b=plan['cases'][2*i:2*i+2]
   assert a['spec']['narrow_reset']==b['spec']['narrow_reset']
   assert a['spec']['layout']['distance_m']==b['spec']['layout']['distance_m']==.1
   assert a['spec']['split']==b['spec']['split']=='train'
   assert a['spec']['source_identity'].startswith('layout10_train_')
   assert 'grasp_evaluation_ticks' not in a
  for mutate in ['fallback','trigger']:
   bad=copy.deepcopy(plan)
   if mutate=='fallback':bad['cases'][0]['spec']['takeover_tick']=80
   else:bad['cases'][0]['dagger']['trigger']['approach_distance_m']=.2
   try:validate_takeover_plan(bad)
   except ValueError:pass
   else:raise AssertionError('Invalid collection plan admitted: '+mutate)
  assert learn.read(source/'session.json')['deadline']==learn.read(temp/'round/session.json')['deadline']
  from scripts.tight_layout_20261116 import final_test
  with patch('tools.tight_fresh_rollouts.completed_pair',side_effect=lambda root,role:dict(role=role)):
   final_test.prepare(fixture,temp/'final')
  test_seeds=[]
  for index in range(2):
   plan=learn.read(temp/'final'/f'part{index}'/'plan.json');final_test.validate(plan)
   test_seeds.extend(c['spec']['seed'] for c in plan['cases'])
   bad=copy.deepcopy(plan);bad['cases'][0]['spec']['seed']=28820000
   try:final_test.validate(bad)
   except AssertionError:pass
   else:raise AssertionError('Development seed admitted as final test')
  assert len(test_seeds)==16 and len(set(test_seeds))==8
 atomic_json(learn.TREE/'research_docs/evidence/tight_layout_20261116/dagger_plan_check.json',dict(passed=True,
  simulated_policy_identities=True,real_source_reset_metadata=True,cases=8,families=4,shared_AB_layout=True,
  rejects_changed_trigger_and_fixed_fallback=True,original_deadline_preserved=True,final_test_registration_8_paired_families=True,rejects_development_seed_in_test=True,simulation_starts=0,optimizer_updates=0))
 print('Actual new-layout train reset registration and negative takeover admission pass; no model or GPU work')
if __name__=='__main__':main()
