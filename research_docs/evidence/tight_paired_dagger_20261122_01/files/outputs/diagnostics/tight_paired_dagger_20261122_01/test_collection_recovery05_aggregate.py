import json,sys,tempfile,types,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parent))
import collection_recovery05_aggregate as m
import collection_recovery05_scope as scope
class AggregateTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.original=self.root/'original';self.recovered=self.root/'recovered';self.continued=self.root/'continued';self.output=self.root/'aggregate'
  for p in (self.original,self.recovered,self.continued):p.mkdir()
  self.plan={'cases':[dict(name=str(i),role='AB'[i%2],opportunity='support',spec={'identity':str(i),'family':'f'+str(i//2)}) for i in range(96)]}
  self.put(self.original/'plan.json',self.plan);self.put(self.original/'session.json',{});self.put(self.original/'paired_queue_assignments.json',[dict(case=str(i),ordinal=i) for i in range(15)])
  old=[];new=[]
  for i,c in enumerate(self.plan['cases']):
   root=self.recovered if i<15 else self.continued;(root/str(i)).mkdir()
   if i<15:(self.original/str(i)).mkdir()
   result=dict(identity=str(i),legal=True if i<7 or i>=15 else None,software_complete=i<7 or i>=15,success=False,accepted_for_training=False)
   (old if i<15 else new).append(dict(case=str(i),role=c['role'],result=result))
  self.put(self.recovered/'recovered_results.json',old);self.put(self.continued/'results.json',new)
  for p in (self.recovered,self.continued):self.put(p/'correction_entries.json',{})
  self.put(self.continued/'paired_queue_assignments.json',[dict(case=str(i),ordinal=i-15) for i in range(15,96)])
  self.put(self.continued/'recovery05_binding.json',{'assignment_sha256':m.digest(self.original/'paired_queue_assignments.json')})
  checks=[dict(family='f'+str(i),opportunity='support',passed=True) for i in range(48) if i!=7]
  self.put(self.original/'paired_initial_check.json',dict(rows=checks[:7],passed=True));self.put(self.continued/'paired_initial_check.json',dict(rows=checks[7:],passed=True))
 def put(self,p,x):p.write_text(json.dumps(x))
 def run_aggregate(self):return m.aggregate(self.original,self.recovered,self.continued,self.output)
 def validation_modules(self):
  tools=types.ModuleType('tools');tc=types.ModuleType('tools.tight_paired_collection');tc.validate_plan=lambda p:True;tools.tight_paired_collection=tc
  return patch.dict(sys.modules,{'tools':tools,'tools.tight_paired_collection':tc})
 def test_full96_retained_incomplete_excluded_no_trigger(self):
  x=self.run_aggregate();self.assertEqual(x['opportunities'],96)
  self.assertEqual(sum(r['software_incomplete'] for r in x['roles'].values()),8)
  self.assertEqual(sum(r['no_trigger'] for r in x['roles'].values()),88)
  self.assertEqual(x['observable_pair_checks'],dict(passed=47,unverified=1,failed=0,all_verified=False))
  self.assertFalse(m.read(self.output/'result.json')['software_complete'])
 def test_duplicates_rejected(self):
  rows=m.read(self.continued/'results.json');rows.append(rows[0]);self.put(self.continued/'results.json',rows)
  with self.assertRaises(ValueError):self.run_aggregate()
 def test_missing_case_rejected(self):
  rows=m.read(self.continued/'results.json');self.put(self.continued/'results.json',rows[1:])
  with self.assertRaises(ValueError):self.run_aggregate()
 def test_index_publication_does_not_overwrite_original_checks(self):
  before=(self.original/'paired_initial_check.json').read_bytes();self.run_aggregate();m.publish_index(self.original,self.output)
  self.assertEqual(before,(self.original/'paired_initial_check.json').read_bytes())
  with self.assertRaises(FileExistsError):m.publish_index(self.original,self.output)
 def test_scope_validates_truthful_aggregate(self):
  self.run_aggregate();m.publish_index(self.original,self.output)
  with self.validation_modules():self.assertEqual(scope.validated_summary(self.original)['opportunities'],96)
 def test_scope_rejects_tampered_summary(self):
  self.run_aggregate();self.put(self.output/'collection_summary.json',{})
  with self.validation_modules():
   with self.assertRaises(ValueError):scope.validated_summary(self.output)
 def test_scope_rejects_tampered_published_index(self):
  self.run_aggregate();m.publish_index(self.original,self.output);self.put(self.original/'results.json',[])
  with self.validation_modules():
   with self.assertRaises(ValueError):scope.validated_summary(self.original)
if __name__=='__main__':unittest.main()
