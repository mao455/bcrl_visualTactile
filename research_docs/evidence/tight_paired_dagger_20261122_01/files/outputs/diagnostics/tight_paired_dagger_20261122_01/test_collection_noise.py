"""Pure CPU fixtures; no torch import, CUDA, Kit or process launches."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('collection_noise', HERE/'audit_collection_noise.py')
n = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(n)


class Fixture:
    def __init__(self, root):
        self.root = Path(root)/'collection'
        self.root.mkdir()
        self.numeric = {}
        self.calls = []
        self.cases, self.assignments, self.initial, self.results = [], [], [], []
        for family in range(24):
            for op in ('support', 'alignment'):
                names = []
                for role in 'AB':
                    name = f'{role}_f{family}_{op}'
                    ordinal = len(self.cases)
                    spec = dict(identity=name, family=f'family{family}', seed=1000+family,
                                event_seed=2000+family, narrow_reset=dict(pose=family),
                                initial_configuration_sha256='reset'+str(family))
                    case = dict(name=name, role=role, opportunity=op, spec=spec)
                    self.cases.append(case)
                    names.append(name)
                    self.assignments.append(dict(case=name, role=role, ordinal=ordinal, lane=ordinal % 8, sampler_seed=7091))
                    steps = 31 if role == 'A' else 25
                    switch = 13 if op == 'support' else None
                    counts = dict(plans=len(range(0, steps, 12)), history_updates=steps, robot_observations=steps, backend_updates=steps)
                    result = dict(identity=name, family=spec['family'], steps=steps, software_complete=True, legal=True,
                                  batch_execution=dict(lane=ordinal % 8, queue_ordinal=ordinal),
                                  native_inference=dict(batch_observed_counts=counts),
                                  dagger=dict(student_ticks=steps if switch is None else switch,
                                              expert_ticks=0 if switch is None else steps-switch,
                                              recovery_coverage=dict(switch_tick=switch)))
                    self.results.append(dict(case=name, role=role, result=result))
                    directory = self.root/name
                    directory.mkdir()
                    self.put(directory/'result.json', result)
                    self.put(directory/'native_inference_timing.json', dict(batch_observed_counts=counts))
                    self.numeric[name] = dict(step_id=list(range(steps)), control_source=[0 if switch is None or tick < switch else 1 for tick in range(steps)])
                    for stage in ('start', 'end'):
                        path = directory/f'sampler_rng_{stage}.pt'
                        path.write_bytes(b'CPU fixture, not a serialized torch state '+stage.encode())
                        self.put(directory/f'sampler_rng_{stage}.json', dict(identity=name, stage=stage,
                                 next_step=(0 if ordinal < 8 else 601) if stage == 'start' else steps,
                                 state_path=path.name, state_sha256=n.sha(path), initial_seed=7091,
                                 checkpoint_sampler_seed=7091, extra_random_draws=0, vision_arithmetic=dict(version='fixture')))
                self.initial.append(dict(family=f'family{family}', opportunity=op, reference=names[0], case=names[1], passed=True))
        self.plan = dict(stage_takeover_contract='paired_own_student_recovery_collection_v1',kind='takeover',test_used=False,
                         sampler_seed=7091,record_sampler_rng=True,inference_scheduler_index_mode='host',
                         cases=self.cases,batch=dict(lanes=8),visual_arithmetic=dict(version='fixture'))
        self.save()

    def put(self, path, value):
        path.write_text(json.dumps(value))

    def save(self):
        self.put(self.root/'plan.json', self.plan)
        self.put(self.root/'registration.json', dict(plan_sha256=n.sha(self.root/'plan.json')))
        self.put(self.root/'paired_queue_assignments.json', self.assignments)
        self.put(self.root/'paired_initial_check.json', dict(passed=True, rows=self.initial))
        self.put(self.root/'results.json', self.results)

    def rng(self, directory, plans, *, reconstruct_cuda):
        if reconstruct_cuda:
            raise AssertionError('CPU fixture must not request CUDA')
        self.calls.append((directory.name, plans))
        return dict(receipt_hashes_verified=True, planning_calls=plans, native_draw_reconstruction_verified=False,
                    start_state=[7091,0], end_state=[7091,plans])

    def audit(self, **kwargs):
        return n.audit(self.root, rng_auditor=self.rng, numeric_loader=lambda directory:self.numeric[directory.name],
                       verify_sources=False, **kwargs)


class CollectionNoiseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def test_full96_cpu_receipts_and48_paired_assignments(self):
        result = self.f.audit()
        self.assertEqual(result['assignments'], 96)
        self.assertEqual(result['paired_assignments'], 48)
        self.assertEqual(result['cpu_receipt_checks_passed'], 96)
        self.assertEqual(len(self.f.calls), 96)
        self.assertEqual(result['exact_cuda_full_endpoint_checks'], 0)
        self.assertFalse(result['cuda_used'])
        self.assertNotIn('torch', sys.modules)

    def test_reconstruction_uses_full_endpoint_counts_and_separate_prefix(self):
        recorded = []
        def simulated_rng(directory, plans, *, reconstruct_cuda):
            self.assertTrue(reconstruct_cuda)
            recorded.append(plans)
            return dict(receipt_hashes_verified=True, planning_calls=plans, native_draw_reconstruction_verified=True,
                        start_state=[7091,0], end_state=[7091,plans])
        result = n.audit(self.f.root, reconstruct_cuda=True, rng_auditor=simulated_rng,
                         numeric_loader=lambda directory:self.f.numeric[directory.name],
                         prefix_reconstructor=lambda draws:[7091,draws],verify_sources=False)
        self.assertEqual(recorded, [3]*96)
        self.assertEqual(result['rows'][0]['reconstructed_student_boundary_state'], [7091,2])
        self.assertEqual(result['rows'][0]['endpoint_rng']['end_state'], [7091,3])
        self.assertEqual(result['observed_takeover_rng_states'], 0)
        self.assertNotIn('torch', sys.modules)

    def test_takeover_prefix_excludes_first_expert_tick_and_shadow_calls(self):
        row = self.f.audit()['rows'][0]
        self.assertEqual(row['switch_tick'], 13)
        self.assertEqual(row['student_plan_ticks'], [0,12])
        self.assertEqual(row['actual_native_plans'], 3)
        self.assertEqual(row['student_executed_plan_calls'], 2)
        self.assertEqual(row['shadow_after_switch_plan_calls'], 1)
        self.assertFalse(row['observed_takeover_boundary_state_available'])

    def test_autonomous_scope_covers_all_native_calls(self):
        row = self.f.audit()['rows'][2]
        self.assertIsNone(row['switch_tick'])
        self.assertEqual(row['student_executed_plan_calls'], row['actual_native_plans'])
        self.assertEqual(row['shadow_after_switch_plan_calls'], 0)

    def test_queued_start_prior_controller_clock_is_not_rng_age(self):
        row = self.f.audit()['rows'][10]
        self.assertEqual(row['start_receipt_prior_controller_next_step'], 601)

    def test_changed_assignment_seed_or_order_rejected(self):
        self.f.assignments[0]['sampler_seed'] = 99
        self.f.save()
        with self.assertRaisesRegex(ValueError, 'assignment differs'):
            self.f.audit()

    def test_missing_pair_role_rejected(self):
        self.f.cases[1]['role'] = 'A'
        self.f.assignments[1]['role'] = 'A'
        self.f.save()
        with self.assertRaisesRegex(ValueError, 'Duplicate pair role'):
            self.f.audit()

    def test_wrong_observed_pair_names_rejected(self):
        self.f.initial[0]['case'] = self.f.initial[1]['case']
        self.f.save()
        with self.assertRaisesRegex(ValueError, 'names do not match'):
            self.f.audit()

    def test_missing_rng_end_or_changed_sha_rejected(self):
        path = self.f.root/self.f.cases[0]['name']/'sampler_rng_end.pt'
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'state SHA differs'):
            self.f.audit()

    def test_repeated_native_tick_rejected(self):
        self.f.numeric[self.f.cases[0]['name']]['step_id'][1] = 0
        with self.assertRaisesRegex(ValueError, 'ticks skipped/duplicated'):
            self.f.audit()

    def test_actual_counter_disagrees_with_native_schedule_rejected(self):
        name = self.f.cases[0]['name']
        path = self.f.root/name/'native_inference_timing.json'
        value = n.read(path)
        value['batch_observed_counts']['plans'] = 2
        self.f.put(path, value)
        with self.assertRaisesRegex(ValueError, 'counts disagree'):
            self.f.audit()

    def test_numeric_switch_differs_from_result_rejected(self):
        self.f.numeric[self.f.cases[0]['name']]['control_source'][12] = 1
        with self.assertRaisesRegex(ValueError, 'switch disagrees'):
            self.f.audit()

    def test_takeover_exactly_on_planning_tick_excludes_that_tick(self):
        result = copy.deepcopy(self.f.results[0]['result'])
        result['dagger'].update(student_ticks=12, expert_ticks=19, recovery_coverage=dict(switch_tick=12))
        numeric = dict(step_id=list(range(31)),control_source=[0]*12+[1]*19)
        scope = n.planning_scope(result, dict(batch_observed_counts=result['native_inference']['batch_observed_counts']), numeric)
        self.assertEqual(scope['student_plan_ticks'], [0])
        self.assertEqual(scope['shadow_after_switch_plan_calls'], 2)

    def test_source_changing_back_to_student_rejected(self):
        self.f.numeric[self.f.cases[0]['name']]['control_source'][20] = 0
        with self.assertRaisesRegex(ValueError, 'contiguous switch'):
            self.f.audit()

    def test_result_plan_binding_changed_rejected(self):
        (self.f.root/'plan.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen plan SHA differs'):
            self.f.audit()

    def test_unpaired_noise_start_rejected(self):
        original = self.f.rng
        def rng(directory, plans, **kwargs):
            value = original(directory,plans,**kwargs)
            if directory.name.startswith('B'):
                value['start_state'] = [0,0]
            return value
        self.f.rng = rng
        with self.assertRaisesRegex(ValueError, 'not reseeded'):
            self.f.audit()


if __name__ == '__main__':
    unittest.main(verbosity=2)
