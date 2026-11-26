"""Tiny CPU HDF5 fixtures and summary write interception."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import h5py
import numpy as np

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('integrity',HERE/'audit_collection_integrity.py')
m=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(m)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.numeric=self.root/'numeric.h5';self.compact=self.root/'compact.h5'
        self.result=dict(steps=6,identity='case',dagger=dict(expert_valid_ticks=2))
        fields={'obs/step_id':np.arange(6),'obs/time_s':np.arange(6)/60.,'obs/proprio':np.zeros((6,61)),
                'obs/actuator_target_rad':np.zeros((6,29)),'obs/contact_substeps_w_N':np.zeros((6,2,5,3)),
                'obs/contact_substeps_local_N':np.zeros((6,2,5,3)),'obs/contact_time_s':np.zeros((6,2)),
                'obs/contact_valid':np.ones(6,dtype=bool),
                'transition/dagger/control_source':np.array([0,0,1,1,1,1]),
                'transition/dagger/expert_label_valid':np.array([0,0,0,0,1,1],dtype=bool),
                'transition/dagger/switch_clean':np.array([0,0,0,0,1,1],dtype=bool),
                'transition/dagger/segment_id':np.array([0,0,1,1,2,2]),
                'terminated':np.array([0,0,0,0,0,1],dtype=bool),'truncated':np.zeros(6,dtype=bool)}
        for key in ('actual_issued','actual_applied','u_expert','q_reached','previous_issued'):
            fields['transition/dagger/'+key]=np.zeros((6,29))
        for key in ('diagnostics_before/joint_position_rad','diagnostics_before/joint_velocity_rad_s',
                    'diagnostics_after/joint_position_rad','target_before_rad','target_after_rad'):
            fields['transition/'+key]=np.zeros((6,29))
        for path,prefix in ((self.numeric,'episode'),(self.compact,'episodes/case')):
            with h5py.File(path,'w') as handle:
                group=handle.create_group(prefix)
                for key,value in fields.items():group.create_dataset(key,data=value)

    def test_complete_prefix_history_and_legal_suffix(self):
        result=m.check_numeric(self.root,self.result,compact_path=self.compact)
        self.assertEqual(result['legal_expert_ticks'],2)
        self.assertEqual(result['student_prefix_ticks'],2)
        self.assertEqual(result['queue_or_invalid_expert_ticks'],2)
        self.assertTrue(result['compact_complete_numeric_history_checked'])

    def test_student_label_is_rejected(self):
        with h5py.File(self.numeric,'a') as handle:handle['episode/transition/dagger/expert_label_valid'][0]=True
        with self.assertRaisesRegex(ValueError,'student/queue'):m.check_numeric(self.root,self.result)

    def test_feedback_prediction_substitution_rejected(self):
        with h5py.File(self.numeric,'a') as handle:handle['episode/transition/dagger/previous_issued'][1,0]=1
        with self.assertRaisesRegex(ValueError,'Historical feedback'):m.check_numeric(self.root,self.result)

    def test_compact_prefix_mutation_rejected(self):
        with h5py.File(self.compact,'a') as handle:handle['episodes/case/obs/proprio'][0,0]=1
        with self.assertRaisesRegex(ValueError,'Compact historical'):m.check_numeric(self.root,self.result,compact_path=self.compact)

    def test_mask_change_without_segment_rejected(self):
        with h5py.File(self.numeric,'a') as handle:handle['episode/transition/dagger/segment_id'][4]=1
        with self.assertRaisesRegex(ValueError,'lacks segment'):m.check_numeric(self.root,self.result)

    def test_production_summary_interception_does_not_write_inputs(self):
        sentinel=self.root/'collection_summary.json';sentinel.write_bytes(b'original')
        fake=SimpleNamespace(atomic_json=lambda *_:self.fail('writer not intercepted'))
        def summary(output):
            value=dict(opportunities=96)
            fake.atomic_json(output/'collection_summary.json',value)
            return value
        fake.summarize=summary
        self.assertEqual(m.production_summary(fake,self.root),dict(opportunities=96))
        self.assertEqual(sentinel.read_bytes(),b'original')


if __name__=='__main__':unittest.main(verbosity=2)
