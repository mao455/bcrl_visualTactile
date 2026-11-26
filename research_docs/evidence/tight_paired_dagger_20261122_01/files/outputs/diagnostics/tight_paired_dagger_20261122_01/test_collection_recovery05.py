import importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('recovery',Path(__file__).with_name('collection_recovery05.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class RecoveryTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.original=self.root/'old';self.original.mkdir()
  self.plan={'cases':[{'name':str(i)} for i in range(6)]}
  for f,v in [('plan.json',self.plan),('paired_queue_assignments.json',[{'case':'0'},{'case':'1'},{'case':'2'}]),('session.json',{'deadline':1}),('registration.json',{'wrapper_sha256':'x'})]:(self.original/f).write_text(json.dumps(v))
  for n in range(3):(self.original/str(n)).mkdir()
 def test_only_unassigned(self):self.assertEqual(m.census(self.original)['remaining_cases'],['3','4','5'])
 def test_assignment_is_not_completion(self):self.assertTrue(all(r['status']=='software_incomplete' for r in m.census(self.original)['rows']))
 def test_closed_and_pending(self):
  (self.original/'0/result.json').write_text('{}');(self.original/'1/execution_complete.json').write_text('{}')
  self.assertEqual([r['status'] for r in m.census(self.original)['rows']],['closed','execution_pending_verification','software_incomplete'])
 def test_nonprefix_rejected(self):
  (self.original/'paired_queue_assignments.json').write_text('[{"case":"2"}]')
  with self.assertRaises(ValueError):m.census(self.original)
 def test_unrecorded_directory_rejected(self):
  (self.original/'3').mkdir()
  with self.assertRaises(ValueError):m.census(self.original)
 def test_prepare_preserves_full_registration(self):
  new=self.root/'new';m.prepare(self.original,new)
  self.assertEqual((new/'plan.json').read_bytes(),(self.original/'plan.json').read_bytes())
  self.assertEqual(m.read(new/'recovery05_binding.json')['assigned'],3)
 def test_no_overwrite(self):
  new=self.root/'new';m.prepare(self.original,new)
  with self.assertRaises(FileExistsError):m.prepare(self.original,new)


 def test_runtime_dispatch_exact_suffix_and_restoration(self):
  import contextlib,sys,types
  from unittest.mock import patch
  new=self.root/'new';m.prepare(self.original,new)
  runner=types.ModuleType('tools.tight_fresh_batch_run');runner.read=m.read
  tc=types.ModuleType('tools.tight_paired_collection');seen=[]
  def valid(plan):
   if plan!=self.plan:raise ValueError('not full original')
   seen.append('full_validation')
  @contextlib.contextmanager
  def pairing(plan,output):
   seen.append([c['name'] for c in plan['cases']]);yield
  def fake_worker(output):
   tc.validate_plan(m.read(output/'plan.json'))
   selected=runner.read(output/'plan.json');tc.validate_plan(selected)
   self.assertEqual([c['name'] for c in selected['cases']],['3','4','5'])
   self.assertEqual(runner.read(output/'session.json'),{'deadline':1})
   with tc.pairing_scope(self.plan,output):pass
  tc.validate_plan=valid;tc.pairing_scope=pairing;tc.worker=fake_worker
  pkg=types.ModuleType('tools');pkg.tight_paired_collection=tc;pkg.tight_fresh_batch_run=runner
  with patch.dict(sys.modules,{'tools':pkg,'tools.tight_paired_collection':tc,'tools.tight_fresh_batch_run':runner}):m.worker(new)
  self.assertIn(['3','4','5'],seen)
  self.assertIs(tc.validate_plan,valid);self.assertIs(tc.pairing_scope,pairing);self.assertIs(runner.read,m.read)
 def test_runtime_refuses_changed_assignment(self):
  new=self.root/'new';m.prepare(self.original,new)
  (self.original/'paired_queue_assignments.json').write_text('[]')
  with self.assertRaises(ValueError):m.worker(new)
if __name__=='__main__':unittest.main()
