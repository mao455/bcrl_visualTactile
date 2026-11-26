"""CPU-only tests: temporary roots, synthetic process registry, no signals/GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('paired_resource_watch_v4', HERE/'resource_watch_v4.py')
w = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(w)
# Pure synthetic empty external scan for inherited unit fixtures.
_real_inspect = w.inspect
w.inspect = lambda *args, **kwargs: _real_inspect(*args, external_scan=lambda: dict(processes=[], errors=[]), **kwargs)
sys.path.insert(0, str(w.PROJECT))
sys.path.insert(0, str(HERE))


class Fixture:
    def __init__(self, folder):
        self.root = Path(folder)/'run'
        self.config = Path(folder)/'config'
        self.root.mkdir()
        self.config.mkdir()
        self.procs = {}
        self.count = 0

    def put(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def add(self, name):
        self.count += 1
        spec = next(s for s in w.specs(self.root, self.config) if s['name'] == name)
        owner = spec['owner']
        opid = 100*self.count
        mpid, wpid = opid+1, opid+2
        command = ['python', '-m', spec['module']]
        if spec['module'] != 'tools.tight_project':
            command.append('project')
        command.extend([spec['operation'], '--job', str(spec['job'])])
        for pid, ppid, born, cmd in (
            (opid, 1, 10., command),
            (mpid, opid, 10.3, ['python', '-m', 'visuotactile.state_resources', '--root', str(owner)]),
        ):
            self.procs[pid] = dict(pid=pid, ppid=ppid, created=born, command=cmd, cwd=str(self.root))
        output = self.root/'collection' if name == 'collection' else owner/'wave00'
        if name in ('collection', 'development', 'test'):
            module = 'tools.tight_paired_collection' if name == 'collection' else 'tools.tight_paired_evaluate'
            entry = 'paired_collection' if name == 'collection' else 'paired_evaluate'
            worker = ['python', '-m', module, 'worker', '--output', str(output)]
            args = ['--output', str(output)] if name == 'collection' else ['worker', '--output', str(output)]
        else:
            entry = 'paired_frontend'
            args = ['--root', str(self.root/'common180_parents'), '--role', 'A']
            worker = ['python', '-m', 'tools.tight_paired_learning', 'frontend', *args]
        self.procs[wpid] = dict(pid=wpid, ppid=opid, created=12., command=worker, cwd=str(self.root))
        job = dict(version='tight_project_job_v1', operation=spec['operation'], resource_root=str(owner),
                   stages=[dict(entry=entry, args=args)])
        self.put(spec['job'], job)
        self.put(owner/'project_job_started.json', dict(pid=opid, utc=10.1, job=job, commands=[worker]))
        self.put(owner/'monitor.json', dict(pid=mpid, started=10.4))
        self.put(owner/'project_status.json', dict(pid=wpid, utc=12.1, stage=0, command=worker))
        raw = dict(pid=mpid, utc_epoch=100., gpu_processes=[], external_gpu_processes=[], process_telemetry_errors=[])
        self.put(owner/'shared_resources.json', raw)
        if name in ('collection', 'development', 'test'):
            self.put(output/'plan.json', dict(resource_root=str(self.root if name == 'collection' else owner)))
            plan_sha = w.sha((output/'plan.json').read_bytes())
            if name == 'collection':
                binding = dict(root=str(self.root), owner=str(owner), output=str(output), job=str(spec['job']), plan_sha256=plan_sha)
                self.put(output/'registration.json', binding)
            else:
                binding = dict(job=str(spec['job']), job_sha256=w.sha(spec['job'].read_bytes()),
                               wave_plan_sha256={str(output): plan_sha})
                self.put(owner/'registration.json', binding)
        return dict(spec=spec, owner=owner, opid=opid, mpid=mpid, wpid=wpid, output=output, raw=raw)

    def gpu(self, row, pid=None, owned=False):
        proc = dict(pid=row['wpid'] if pid is None else pid, owned=owned, name='python', vram_mib=8000)
        row['raw']['gpu_processes'] = [proc]
        row['raw']['external_gpu_processes'] = [] if owned else [proc]
        self.put(row['owner']/'shared_resources.json', row['raw'])

    def inspect(self):
        return w.inspect(self.root, self.config, 101., self.procs.get)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.f = Fixture(self.tmp.name)

    def test_registered_parent_monitor_keeps_existing_dp_running(self):
        row = self.f.add('parent_train')
        self.f.gpu(row, owned=True)
        state = self.f.inspect()
        self.assertFalse(state['paused'])
        self.assertEqual(state['active_resource_roots'], [str(row['owner'])])

    def test_recovery_owner_exact_spec_and_monitor(self):
        row = self.f.add('parent_train_recovery01')
        self.f.gpu(row, owned=True)
        state = self.f.inspect()
        self.assertFalse(state['paused'], state)
        self.assertEqual(state['active_resource_roots'], [str(row['owner'])])
        self.assertEqual(row['spec']['job'].name, 'parent_train_recovery01.json')

    def test_closed_root_reused_pid_wrong_command_or_timestamp_ignored(self):
        closed = self.f.add('parent_train')
        self.f.add('parent_train_recovery01')
        original = copy.deepcopy(self.f.procs[closed['opid']])
        # Historical files may be unavailable; unrelated PID must not create an error.
        closed['spec']['job'].unlink()
        for mutation in ('timestamp', 'command'):
            with self.subTest(mutation=mutation):
                proc = copy.deepcopy(original)
                if mutation == 'timestamp':
                    proc['created'] = 90.
                else:
                    proc['command'] = ['python', '-m', 'unrelated_test']
                self.f.procs[closed['opid']] = proc
                state = self.f.inspect()
                self.assertFalse(state['paused'], state)
                self.assertFalse(state['errors'])

    def test_cpu_only_external_test_keeps_pause_when_gpu_view_clears(self):
        self.f.add('parent_train_recovery01')
        record = dict(pid=9001, command=['/venv/python', '-m', 'run_test'],
                      cwd=str(w.EXTERNAL_TEST_ROOTS[0]), created=80.)
        scan = lambda: w.external_test_processes([record], self_pid=8000)
        for _ in range(3):
            state = _real_inspect(self.f.root, self.f.config, 101., self.f.procs.get, external_scan=scan)
            self.assertTrue(state['paused'])
            self.assertEqual(state['reason'], 'external_test_python_present')
            self.assertFalse(state['external_gpu_processes'])
            self.assertEqual(state['external_test_python_processes'][0]['pid'], 9001)

    def test_external_scan_self_exclusion_and_cmdline_match(self):
        rows = [dict(pid=1, command=['python', 'watch.py', str(w.EXTERNAL_TEST_ROOTS[0])], cwd='/tmp'),
                dict(pid=2, command=['python3', str(w.EXTERNAL_TEST_ROOTS[0]/'test.py')], cwd='/tmp'),
                dict(pid=3, command=['bash', str(w.EXTERNAL_TEST_ROOTS[0]/'test.sh')], cwd='/tmp'),
                dict(pid=4, command=['python', 'normal.py'], cwd='/tmp')]
        scan = w.external_test_processes(rows, self_pid=1)
        self.assertEqual([p['pid'] for p in scan['processes']], [2])
        self.assertFalse(scan['errors'])

    def test_independent_external_scan_failure_fails_closed(self):
        self.f.add('parent_train_recovery01')
        def broken():
            raise OSError('process enumeration failed')
        state = _real_inspect(self.f.root, self.f.config, 101., self.f.procs.get, external_scan=broken)
        self.assertTrue(state['paused'])
        self.assertTrue(state['external_test_python_scan_errors'])

    def test_new_course_demo_relative_python_and_exact_cwd_match(self):
        self.f.add('parent_train_recovery01')
        record = dict(pid=9002, command=['.venv/bin/python', '-m', 'course.demo'],
                      cwd=str(w.EXTERNAL_TEST_ROOTS[1]), created=80.)
        scan = lambda: w.external_test_processes([record], self_pid=8000)
        state = _real_inspect(self.f.root, self.f.config, 101., self.f.procs.get, external_scan=scan)
        self.assertTrue(state['paused'])
        self.assertEqual(state['external_test_python_processes'][0]['pid'], 9002)

    def test_exact_root_boundaries_and_nested_cwd(self):
        records = [dict(pid=i+100, command=['python', '-m', 'course.demo'], cwd=cwd)
                   for i, cwd in enumerate((str(w.EXTERNAL_TEST_ROOTS[1]/'nested'),
                                             str(w.EXTERNAL_TEST_ROOTS[1])+'-other',
                                             '/tmp/FingerEyeLab-copy',
                                             str(w.EXTERNAL_TEST_ROOTS[0])+'-backup'))]
        result = w.external_test_processes(records, self_pid=8000)
        self.assertEqual([row['pid'] for row in result['processes']], [100])

    def test_external_absolute_script_path_matches_outside_cwd(self):
        record = dict(pid=9100, command=['python', str(w.EXTERNAL_TEST_ROOTS[1]/'course/demo.py')], cwd='/tmp')
        result = w.external_test_processes([record], self_pid=8000)
        self.assertEqual(result['processes'][0]['matched_project_roots'], [str(w.EXTERNAL_TEST_ROOTS[1])])

    def test_proposed_gate_waits_two_clean_scans_and_resets_on_external(self):
        import driver_support_v4 as gate
        seen = []
        records = iter([dict(processes=[], errors=[]), dict(processes=[dict(pid=9002)], errors=[]),
                        dict(processes=[], errors=[]), dict(processes=[], errors=[])])
        with patch.object(gate.time, 'time', return_value=100.), patch.object(gate.time, 'sleep'):
            gate.wait_external(200., self.f.root/'STOP', seen.append, scan=lambda: next(records))
        self.assertEqual(len(seen), 4)

    def test_proposed_gate_scan_error_never_counts_as_quiet(self):
        import driver_support_v4 as gate
        seen = []
        with patch.object(gate.time, 'time', side_effect=[100., 100., 201.]), patch.object(gate.time, 'sleep'):
            with self.assertRaises(TimeoutError):
                gate.wait_external(200., self.f.root/'STOP', seen.append,
                                   scan=lambda: dict(processes=[], errors=['unavailable']))
        self.assertEqual(len(seen), 1)

    def test_proposed_gate_own_stop_preserved(self):
        import driver_support_v4 as gate
        stop = self.f.root/'STOP'
        stop.touch()
        with self.assertRaises(TimeoutError):
            gate.wait_external(10**12, stop, lambda state: None, scan=lambda: self.fail('scan after STOP'))

    def test_early_age_guard_does_not_wait_for_hard_8second_failure(self):
        row = self.f.add('parent_train_recovery01')
        row['raw']['utc_epoch'] = 97.5
        self.f.put(row['owner']/'shared_resources.json', row['raw'])
        state = self.f.inspect()
        self.assertTrue(state['paused'])
        self.assertEqual(state['errors'][0]['error'], 'early_monitor_age_guard')

    def test_exact_eval_monitor_roots_are_used_without_owners_glob(self):
        for name in ('development', 'test'):
            with self.subTest(name=name):
                row = self.f.add(name)
                self.f.gpu(row)
                state = self.f.inspect()
                self.assertFalse(state['paused'], state)
                self.assertIn(str(row['owner']), state['active_resource_roots'])

    def test_custom_workers_excluded_only_after_registration_and_ancestry(self):
        for name in ('collection', 'development', 'test'):
            with self.subTest(name=name):
                row = self.f.add(name)
                self.f.gpu(row)
                state = self.f.inspect()
                self.assertFalse(state['paused'], state)
                self.assertIn(row['wpid'], [p['pid'] for p in state['registered_workers_excluded']])

    def test_raw_telemetry_preserved_byte_for_byte(self):
        row = self.f.add('collection')
        self.f.gpu(row)
        path = row['owner']/'shared_resources.json'
        original = path.read_bytes()
        state = self.f.inspect()
        self.assertEqual(path.read_bytes(), original)
        self.assertFalse(state['paused'])
        self.assertEqual(state['raw_monitor_rows'][0]['raw']['external_gpu_processes'][0]['pid'], row['wpid'])

    def test_external_gpu_pid_is_never_excluded(self):
        row = self.f.add('collection')
        self.f.gpu(row, pid=9999)
        self.assertEqual(self.f.inspect()['external_gpu_processes'][0]['pid'], 9999)
        self.assertTrue(self.f.inspect()['paused'])

    def test_wrong_module_output_owner_ancestry_or_pid_reuse_not_owned(self):
        row = self.f.add('collection')
        self.f.gpu(row)
        original = copy.deepcopy(self.f.procs[row['wpid']])
        for mutation in ('module', 'output', 'parent', 'create_time'):
            with self.subTest(mutation=mutation):
                proc = copy.deepcopy(original)
                if mutation == 'module':
                    proc['command'][2] += '_lookalike'
                elif mutation == 'output':
                    proc['command'][-1] += '_other'
                elif mutation == 'parent':
                    proc['ppid'] = 9999
                else:
                    proc['created'] = 90.
                self.f.procs[row['wpid']] = proc
                state = self.f.inspect()
                self.assertTrue(state['paused'], state)
                self.assertFalse(state['registered_workers_excluded'])
        self.f.procs[row['wpid']] = original

    def test_changed_plan_is_not_an_owned_worker(self):
        row = self.f.add('development')
        self.f.gpu(row)
        self.f.put(row['output']/'plan.json', dict(resource_root=str(row['owner']), changed=True))
        state = self.f.inspect()
        self.assertTrue(state['paused'])
        self.assertFalse(state['registered_workers_excluded'])

    def test_dead_owner_does_not_grant_worker_ownership(self):
        row = self.f.add('collection')
        self.f.gpu(row)
        self.f.procs.pop(row['opid'])
        self.assertTrue(self.f.inspect()['paused'])
        self.assertFalse(self.f.inspect()['registered_workers_excluded'])

    def test_monitor_failure_cannot_be_masked_by_clean_second_monitor(self):
        row = self.f.add('parent_train')
        self.f.add('main_cache')
        row['raw']['process_telemetry_errors'] = ['nvidia-smi timeout']
        self.f.put(row['owner']/'shared_resources.json', row['raw'])
        state = self.f.inspect()
        self.assertTrue(state['paused'])
        self.assertEqual(state['reason'], 'telemetry_or_registration_error')

    def test_malformed_process_telemetry_fails_closed(self):
        row = self.f.add('parent_train')
        for malformed in (None, {}, 'not a list', [None], ['bad row']):
            with self.subTest(malformed=malformed):
                raw = copy.deepcopy(row['raw'])
                raw['gpu_processes'] = malformed
                self.f.put(row['owner']/'shared_resources.json', raw)
                self.assertTrue(self.f.inspect()['paused'])

    def test_missing_stale_or_false_monitor_fails_closed(self):
        row = self.f.add('development')
        for mutation in ('stale', 'future', 'missing', 'different_pid'):
            with self.subTest(mutation=mutation):
                raw = copy.deepcopy(row['raw'])
                if mutation == 'stale':
                    raw['utc_epoch'] = 90.
                elif mutation == 'future':
                    raw['utc_epoch'] = 110.
                elif mutation == 'different_pid':
                    raw['pid'] = 9999
                self.f.put(row['owner']/'shared_resources.json', raw)
                if mutation == 'missing':
                    (row['owner']/'shared_resources.json').unlink()
                self.assertTrue(self.f.inspect()['paused'])

    def test_unregistered_telemetry_root_ignored(self):
        self.f.put(self.f.root/'owners/rogue/shared_resources.json', dict(utc_epoch=100., external_gpu_processes=[]))
        state = self.f.inspect()
        self.assertTrue(state['paused'])
        self.assertFalse(state['raw_monitor_rows'])

    def test_fulllease_actual_A_B_and_DP_groups_block_without_gpu(self):
        from visuotactile.state_resources import FullLease
        for group in (None, 'A', 'B'):
            with self.subTest(group=group):
                folder = self.f.root/('dp' if group is None else group)
                session = dict(resource_policy='full_gpu_v1', training_deadline=10000., reserve_bytes=0,
                               residual_pair=dict(groups=['A', 'B']), resources=dict(min_ram_gib=0, min_gpu_free_mib=0))
                self.f.put(folder/'session.json', session)
                self.f.put(folder/'concurrency.json', dict(pause_groups=list(w.PAUSES)))
                lease = FullLease(folder)
                with patch('sys.argv', ['python', '--root', str(folder if group else self.f.root/'frontend')]), \
                     patch.object(lease, 'check'), \
                     patch('visuotactile.state_resources.time.sleep', side_effect=TimeoutError('paused')):
                    with self.assertRaisesRegex(TimeoutError, 'paused'):
                        with lease.block(cuda=False):
                            self.fail('Paused group entered a block')

    def test_pause_and_restore_exact_original_bytes_and_prior_groups(self):
        self.f.put(self.f.root/'session.json', dict(resource_policy='full_gpu_v1'))
        path = self.f.root/'concurrency.json'
        original = b'{ "pause_groups": ["prior"], "note": "keep spaces" }\n'
        path.write_bytes(original)
        schedule = w.Schedules(self.f.root, Path(self.tmp.name)/'backups')
        schedule.pause()
        value = w.read(path)
        self.assertEqual(value['pause_groups'], ['prior', *w.PAUSES])
        schedule.restore()
        self.assertEqual(path.read_bytes(), original)

    def test_created_schedule_is_deleted_on_restore(self):
        self.f.put(self.f.root/'session.json', dict(resource_policy='full_gpu_v1'))
        schedule = w.Schedules(self.f.root, Path(self.tmp.name)/'backups')
        schedule.pause()
        self.assertTrue((self.f.root/'concurrency.json').exists())
        schedule.restore()
        self.assertFalse((self.f.root/'concurrency.json').exists())

    def test_third_party_edit_and_symlink_never_overwritten(self):
        self.f.put(self.f.root/'session.json', dict(resource_policy='full_gpu_v1'))
        path = self.f.root/'concurrency.json'
        schedule = w.Schedules(self.f.root, Path(self.tmp.name)/'backups')
        schedule.pause()
        external = b'{"external_edit": true}'
        path.write_bytes(external)
        schedule.pause()
        schedule.restore()
        self.assertEqual(path.read_bytes(), external)
        self.assertIn(str(path), schedule.conflicts)
        path.unlink()
        outside = Path(self.tmp.name)/'outside.json'
        outside.write_bytes(external)
        path.symlink_to(outside)
        schedule2 = w.Schedules(self.f.root, Path(self.tmp.name)/'backups2')
        schedule2.pause()
        self.assertEqual(outside.read_bytes(), external)


if __name__ == '__main__':
    unittest.main(verbosity=2)
