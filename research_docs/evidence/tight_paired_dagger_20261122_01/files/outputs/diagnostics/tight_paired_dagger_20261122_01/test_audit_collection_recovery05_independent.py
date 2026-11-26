import copy,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE));import audit_collection_recovery05 as m
class AssignmentTests(unittest.TestCase):
 def setUp(self):
  self.plan=m.read(m.PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01/collection/plan.json')
  self.assign=[dict(case=c['name'],role=c['role'],ordinal=i,lane=i%8,sampler_seed=7091,attempt='original' if i<15 else 'continued',attempt_ordinal=i if i<15 else i-15) for i,c in enumerate(self.plan['cases'])]
  rows=[]
  for i in range(0,96,2):
   a,b=self.plan['cases'][i:i+2];rows.append(dict(family=a['spec']['family'],opportunity=a['opportunity'],reference=a['name'],case=b['name'],passed=True,reference_legal=True,legal=True,differences={'position':0.},attempt='original' if i<14 else 'continued'))
  missing=rows.pop(7);self.initial=dict(passed=False,full48_verified=False,all_recorded_checks_passed=True,rows=rows,unverified_rows=[dict(family=missing['family'],opportunity=missing['opportunity'],passed=None,reason='Actually crosses attempts')])
 def test_truthful47plus1(self):
  pairs=m.recovered_assignments(self.plan,self.assign,self.initial);self.assertEqual(len(pairs),48)
 def test_fake_allpassed_rejected(self):
  self.initial['passed']=True
  with self.assertRaises(ValueError):m.recovered_assignments(self.plan,self.assign,self.initial)
 def test_wrong_assigned_role_rejected(self):
  self.assign[0]['role']='B'
  with self.assertRaises(ValueError):m.recovered_assignments(self.plan,self.assign,self.initial)
 def test_tolerance_rejected(self):
  self.initial['rows'][0]['differences']['position']=.001
  with self.assertRaises(ValueError):m.recovered_assignments(self.plan,self.assign,self.initial)
 def test_pair_names_rejected(self):
  self.initial['rows'][0]['case']='unknown'
  with self.assertRaises(ValueError):m.recovered_assignments(self.plan,self.assign,self.initial)
 def test_missing_not_invented(self):
  self.initial['unverified_rows'][0]['passed']=True
  with self.assertRaises(ValueError):m.recovered_assignments(self.plan,self.assign,self.initial)
class NoiseAvailabilityTests(AssignmentTests):
 def context(self,complete=0):
  pairs=m.recovered_assignments(self.plan,self.assign,self.initial)
  byname={}
  for i,c in enumerate(self.plan['cases']):byname[c['name']]=dict(case=c['name'],role=c['role'],evidence_directory='/fixture/'+c['name'],result=dict(software_complete=i<complete,legal=True,batch_execution=dict(lane=i%8,queue_ordinal=i if i<15 else i-15)))
  return (self.plan,self.assign,self.initial,pairs,list(byname.values()),{},{'observable_pair_checks':dict(passed=47,unverified=1)})
 def test_all_incomplete_never_get_endpoint_certificates(self):
  with patch.object(m,'inputs',return_value=self.context()):
   result=m.noise('unused')
  self.assertEqual(result['cpu_receipt_checks_passed'],0);self.assertEqual(result['uncertified_execution_endpoints'],96);self.assertFalse(result['all96_execution_endpoints_certified'])
  self.assertTrue(all(p['paired_start_equal'] is None for p in result['pairs']))
 def test_only_available_endpoints_counted(self):
  scope=dict(steps=13,actual_native_plans=2,student_executed_plan_calls=2,switch_tick=None)
  def endpoint(*args,**kwargs):return dict(receipt_hashes_verified=True,planning_calls=2,native_draw_reconstruction_verified=False,start_state=[1],end_state=[2])
  with patch.object(m,'inputs',return_value=self.context(2)),patch.object(m,'read',return_value={}),patch.object(m,'planning_scope',return_value=scope),patch.object(m,'verify_receipts',return_value=0),patch.object(m,'numeric_schedule',return_value={}):
   result=m.noise('unused',rng_auditor=endpoint)
  self.assertEqual(result['cpu_receipt_checks_passed'],2);self.assertEqual(result['exact_cuda_full_endpoint_checks'],0);self.assertEqual(result['uncertified_execution_endpoints'],94)
  self.assertEqual(sum(p['paired_start_equal'] is True for p in result['pairs']),1)
if __name__=='__main__':unittest.main()
