"""CPU regression checks for immutable capture reuse and paired scheduling."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock,patch

PUBLIC=Path(__file__).resolve().parent
PROJECT=PUBLIC.parents[2]
if str(PROJECT) not in sys.path:sys.path.insert(0,str(PROJECT))

def load(name):
    spec=importlib.util.spec_from_file_location(name,PUBLIC/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

cycle=load('round2_cycle')
closed=load('recover_round2_closed')
cleanup=load('cleanup_stale_assets')

class RecoveryTests(unittest.TestCase):
    def test_existing_144_cases_and_test_flags(self):
        r=cycle.ROOT/'round01/evaluation'
        registered=cycle.read(r/'registration_4lane_unexecuted.json')
        total=0
        for folder in registered['waves']:
            original=cycle.read(Path(folder)/'plan.json')
            pairs=cycle.paired_plans(original)
            self.assertEqual(len(pairs),2)
            expected={c['name']:dict(c,video=True) for c in original['cases']}
            actual={c['name']:c for _,p in pairs for c in p['cases']}
            self.assertEqual(actual,expected)
            for _,p in pairs:
                self.assertFalse(p['test_used'])
                self.assertEqual(p['source_sha256'],original['source_sha256'])
                self.assertEqual(p['batch']['seed'],original['batch']['seed'])
                self.assertEqual(p['batch']['lanes'],2)
            total+=len(actual)
        self.assertEqual(total,144)

    def test_heldout_pair_dictionary_and_contract(self):
        r=cycle.ROOT/'round01/evaluation'
        original=cycle.read(Path(cycle.read(r/'registration_4lane_unexecuted.json')['waves'][0])/'plan.json')
        original['test_used']=True;original['holdout_contract']='layout_final_test_v1'
        for _,p in cycle.paired_plans(original):
            self.assertTrue(p['test_used'])
            self.assertTrue(p['evaluation_scope']['test_used'])
            self.assertEqual(p['pairs'],original['pairs'])
            self.assertEqual(p['holdout_contract'],'layout_final_test_v1')

    def test_census_excludes_partial_and_unstarted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);plan={'cases':[{'name':x} for x in ['closed','partial','unstarted']]}
            for name in ['closed','partial']:(root/name).mkdir();(root/name/'started.json').write_text('{}')
            (root/'closed/capture_complete.json').write_text('{}')
            (root/'closed/result.json').write_text(json.dumps(dict(software_complete=True,legal=True,residual_ticks=20,steps=20)))
            with patch.object(closed,'OLD',root):a,b,c=closed.census(plan)
            self.assertEqual([x['name'] for x in a],['closed'])
            self.assertEqual(b,['partial']);self.assertEqual(c,['unstarted'])

    def test_completed_batches_can_finish_without_new_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'round02').mkdir();batch=root/'old';batch.mkdir()
            cases=[dict(name=str(i),role='A' if i<8 else 'B',adaptive_stage='contact') for i in range(20)]
            for c in cases:(batch/c['name']).mkdir()
            for name,value in {'plan':{'cases':cases},'result':{'software_complete':True},
                    'results':[dict(role=c['role'],case=c['name'],result={}) for c in cases],
                    'correction_entries':{c['name']:{'source_group':c['role']} for c in cases}}.items():
                (batch/(name+'.json')).write_text(json.dumps(value))
            (root/'session.json').write_text('{}')
            strategy=dict(schedule=[dict(families=[28817003],stages=[{'A':'contact','B':'contact'}])],target_accepted=24)
            with patch.object(cycle,'ROOT',root),patch.object(cycle.experiment,'collection_report'),\
                    patch.object(cycle.shutil,'disk_usage',return_value=Mock(free=0)),\
                    patch.object(cycle,'load',return_value=Mock()),patch.object(cycle.experiment,'sources',return_value={}),\
                    patch.object(cycle.experiment,'prepare_collection') as prepare,patch.object(cycle.experiment,'execute') as execute:
                combined=cycle.collect(strategy,resume_batches=[str(batch)],output_prefix='micro',round_offset=40)
                prepare.assert_not_called();execute.assert_not_called()
            self.assertEqual(len(cycle.read(combined/'results.json')),20)
            self.assertTrue(cycle.read(root/'round02/collection_budget_stop.json')['population']=={'A':8,'B':12})

    def test_scratch_age_lower_bound(self):
        with self.assertRaises(ValueError):cleanup.plan(.49)

if __name__=='__main__':unittest.main()
