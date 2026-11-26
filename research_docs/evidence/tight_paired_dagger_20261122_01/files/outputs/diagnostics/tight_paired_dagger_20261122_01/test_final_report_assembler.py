"""CPU-only fixtures for final evidence gates; never reads experiment artifacts."""
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
path=Path(__file__).with_name('final_report_assembler.py');spec=importlib.util.spec_from_file_location('assembler',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class Gates(unittest.TestCase):
 def fixture(self):
  pairs=[];roll=[]
  for seed in range(64):
   row=dict(seed=str(seed),family='f'+str(seed))
   for r in m.ROLES:
    success=seed<(40 if r=='A_new' else 32)
    row[r+'_success']=str(success);roll.append(dict(seed=str(seed),role=r,success=str(success)))
   pairs.append(row)
  rates=[dict(role=r,success=40 if r=='A_new' else 32,n=64,success_rate=(40 if r=='A_new' else 32)/64) for r in m.ROLES]
  contrasts=[]
  for name,left,right in [('A_new_minus_B_new','A_new','B_new'),('A_new_minus_parent','A_new','A_parent'),('B_new_minus_parent','B_new','B_parent')]:
   a=next(x['success'] for x in rates if x['role']==left);b=next(x['success'] for x in rates if x['role']==right)
   contrasts.append(dict(name=name,n=64,left_success=a,right_success=b,paired_difference_pp=(a-b)*100/64,both_success=min(a,b),gained=max(a-b,0),lost=max(b-a,0),both_failure=64-max(a,b)))
  return dict(cohort='test',n=64,full_denominator_preserved=True,episodes=256,absolute_success=rates,contrasts=contrasts,both_new_reach_50_percent=True),dict(n=64,seeds=list(range(64))),pairs,roll
 def test_valid_paired_denominator_and_transitions(self):
  rates,contrasts=m.validate_primary(*self.fixture());self.assertEqual(rates['A_new']['success'],40);self.assertEqual(contrasts['A_new_minus_parent']['gained'],8)
 def test_denominator_or_gains_fabrication_rejected(self):
  for change in ['missing','gain','absolute','seed','duplicate']:
   s,r,p,o=self.fixture()
   if change=='missing':p.pop()
   elif change=='gain':s['contrasts'][1]['gained']+=1
   elif change=='absolute':s['absolute_success'][0]['success']+=1
   elif change=='seed':r['seeds'][-1]=100
   else:o[-1]=copy.deepcopy(o[0])
   with self.assertRaises(ValueError):m.validate_primary(s,r,p,o)
 def test_pending_never_reads_active_checkpoints_or_writes_draft(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'root';root.mkdir();out=Path(tmp)/'out';out.mkdir();audit=Path(tmp)/'audit';audit.mkdir()
   result=m.assemble(root,out,audit)
   self.assertEqual(result['status'],'pending');self.assertEqual(result['actual_checkpoint_reads'],0);self.assertFalse((out/'DRAFT_REPORT.md').exists());self.assertGreater(len(result['checklist']),30)
 def test_report_rendering_uses_actual_absolute_counts(self):
  s,r,p,o=self.fixture();rates,contrasts=m.validate_primary(s,r,p,o)
  for x in rates.values():x.update(wilson95=[.3,.8],software_incomplete=0,protocol_unverified=0)
  for x in contrasts.values():x['paired_bootstrap95_pp']=[-5.,25.]
  s.update(observed_utc=10000.,funnels=[dict(role=r,stable_hold=50,first_15mm_gate=45,native_first_subgoal=40,final_insertion=rates[r]['success']) for r in m.ROLES])
  collection=dict(roles={r:dict(opportunities=48,no_trigger=10,takeovers=38,expert_recovered=30,valid_trajectories=30,legal_expert_ticks=1000,admitted_legal_expert_ticks=800,stages={}) for r in 'AB'})
  actual={scope:{r:{c:dict(updates=60,exposure=dict(minimum_tick_steps=50),entry=dict(minimum_steps=50) if c=='frontend' else None,wall_seconds=1) for c in ('frontend','backend')} for r in 'AB'} for scope in ('learning','common180_parents')}
  pools={r:{**dict.fromkeys(('old_full','old_correction','new_full','own_new_correction'),10),'dp_visits_per_epoch_by_partition':dict.fromkeys(('old_full','old_correction','new_full','own_new_correction'),20)} for r in 'AB'}
  text=m.report(Path('/fixture'),s,rates,contrasts,collection,dict(accepted=24,families=24,attempts=25),{},actual,pools,{}, {},dict(started_utc=100.))
  self.assertIn('A 新版 40/64',text);self.assertIn('B 新版 32/64',text);self.assertIn('不能单独估计新数据贡献',text)
 def test_owner_attempt_union_never_double_counts_and_open_excluded(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   for name,start,end in [('parent_train',10.,30.),('parent_train_recovery01',25.,50.),('other_nested',28.,29.),('still_running',45.,None)]:
    d=root/'owners'/name;d.mkdir(parents=True);(d/'project_job_started.json').write_text(json.dumps(dict(utc=start)))
    if end is not None:(d/'project_exit.json').write_text(json.dumps(dict(utc=end,children_reaped=True)))
   d=root/'common180_parents/A/training';d.mkdir(parents=True)
   (d/'recovery.fixture.json').write_text(json.dumps(dict(durable_updates=6840,completed_epochs=57,known_discarded_update_ids=[6841,6842,6843,6844],abandoned_rows=4)))
   result=m.attempt_costs(root)
   self.assertEqual(result['closed_owner_wall_union_seconds'],40.);self.assertEqual(result['closed_owner_sum_before_overlap_removal_seconds'],46.)
   self.assertEqual(result['open_or_unreaped_owner_count'],1);self.assertEqual(result['known_discarded_actual_update_count'],4);self.assertEqual(result['selected_age_or_exposure_contribution_from_discarded'],0)
   self.assertEqual([r['owner'] for r in result['owner_attempts'] if 'parent_train' in r['owner']],['parent_train','parent_train_recovery01'])
   with self.assertRaises(ValueError):m.interval_union_seconds([(20.,10.)])
 def test_bad_attempt_receipt_conservative_warning_no_inferred_cost(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);d=root/'owners/bad';d.mkdir(parents=True)
   (d/'project_job_started.json').write_text(json.dumps(dict(utc=20.)));(d/'project_exit.json').write_text(json.dumps(dict(utc=10.,children_reaped=True)))
   result=m.attempt_costs(root);self.assertEqual(result['closed_owner_wall_union_seconds'],0.);self.assertEqual(len(result['warnings']),1)
 def test_actual_selected_ledger_not_finished_summary(self):
  with tempfile.TemporaryDirectory() as tmp:
   d=Path(tmp);(d/'best.pt').write_bytes(b'fixture checkpoint, no model')
   dep=dict(epoch=60,selection='preregistered_epoch60_only',validation_used_for_selection=False,checkpoint_sha256=m.digest(d/'best.pt'),update=60)
   (d/'deployment.json').write_text(json.dumps(dep));(d/'finished.json').write_text('{}');(d/'family_phase_curves.csv').write_text('epoch,split,family,phase\n')
   state=dict(epoch=60,updates=60,effective_exposure_state=dict(updates=60,legal={'i':[True]}),sampling_exposure={'i':60},entry_exposure_state=dict(updates=60,entries={'i':{}},counts={'i':{'0':50}}))
   class Loader:
    @staticmethod
    def load(*a,**k):return state
   class Ledger:
    def __init__(self,*a):pass
    def summary(self,*a):return dict(eligible=True,minimum_tick_steps=50,ancestors_counted=False)
   row=m.actual_component(d,'frontend',Loader,Ledger);self.assertEqual(row['entry']['minimum_steps'],50)
   state['entry_exposure_state']['counts']['i']['0']=49
   with self.assertRaises(ValueError):m.actual_component(d,'frontend',Loader,Ledger)
   state['entry_exposure_state']['counts']['i']['0']=50;(d/'best.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):m.actual_component(d,'frontend',Loader,Ledger)
if __name__=='__main__':unittest.main()
