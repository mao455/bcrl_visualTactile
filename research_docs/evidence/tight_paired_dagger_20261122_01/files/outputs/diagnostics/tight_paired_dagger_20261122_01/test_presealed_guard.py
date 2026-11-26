import copy,importlib.util,tempfile,unittest
from pathlib import Path
p=Path(__file__).with_name('presealed_guard.py');s=importlib.util.spec_from_file_location('guard',p);g=importlib.util.module_from_spec(s);s.loader.exec_module(g)
class Guard(unittest.TestCase):
 def pool(self,role):
  original=['old'+str(i) for i in range(180)];full=['full'+str(i) for i in range(24)];own=[role+'new']
  entries={i:dict(family=i) for i in original+full};entries[own[0]]=dict(family='newfamily',source_group=role)
  return dict(pool_partitions=dict(old_full=original[:30],old_correction=original[30:],new_full=full,own_new_correction=own),paired_pool=dict(original_ids=original),splits=dict(train=original+full+own),entries=entries,aggregation=dict(student_prefix_supervised=False,new_training_ids=own))
 def test_own_pool_valid_and_cross_owner_rejected(self):
  d=self.pool('A');c=dict(roles=dict(A=dict(valid_trajectories=1)));g.validate_pool(d,'A',c);d['entries']['Anew']['source_group']='B'
  with self.assertRaises(ValueError):g.validate_pool(d,'A',c)
 def test_pool_missing_old_or_prefix_label_rejected(self):
  c=dict(roles=dict(A=dict(valid_trajectories=1)))
  for kind in ('drop','prefix','duplicate'):
   d=self.pool('A')
   if kind=='drop':d['splits']['train'].pop(0)
   elif kind=='prefix':d['aggregation']['student_prefix_supervised']=True
   else:d['pool_partitions']['own_new_correction'].append('old0')
   with self.assertRaises(ValueError):g.validate_pool(d,'A',c)
 def test_pending_no_checkpoint_reads(self):
  with tempfile.TemporaryDirectory() as t:
   r=g.audit(Path(t),'main');self.assertEqual(r['status'],'pending');self.assertEqual(r['checkpoint_reads'],0)
if __name__=='__main__':unittest.main()
