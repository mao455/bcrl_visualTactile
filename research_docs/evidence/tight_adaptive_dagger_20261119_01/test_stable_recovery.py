"""CPU checks for stable admission and refusal before loading model code."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('stable_recovery', Path(__file__).with_name('round2_stable_recovery.py'))
stable = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stable)


class StableAdmissionTests(unittest.TestCase):
    def test_short_pulse_and_interruption_reset(self):
        window = stable.StableWindow(120, 15)
        for stamp in range(0, 120, 15):
            self.assertFalse(window.observe(stamp, True))
        self.assertFalse(window.observe(120, False))
        for stamp in range(135, 255, 15):
            self.assertFalse(window.observe(stamp, True))
        self.assertTrue(window.observe(255, True))

    def test_missing_samples_reset_instead_of_crediting_gap(self):
        window = stable.StableWindow(120, 15)
        for stamp in range(0, 90, 15):
            self.assertFalse(window.observe(stamp, True))
        self.assertFalse(window.observe(200, True))
        for stamp in range(215, 320, 15):
            self.assertFalse(window.observe(stamp, True))
        self.assertTrue(window.observe(320, True))

    def test_full_window_and_new_low_sample(self):
        window = stable.StableWindow(120, 15)
        for stamp in range(0, 120, 15):
            self.assertFalse(window.observe(stamp, True))
        self.assertTrue(window.observe(120, True))
        self.assertFalse(window.observe(135, False))
        self.assertFalse(window.observe(150, True))

    def fixture(self, root):
        (root / 'round02').mkdir()
        (root / 'batch').mkdir()
        (root / 'round02/strategy.json').write_text('{}')
        (root / 'failure.json').write_text('{}')
        (root / 'batch/result.json').write_text('{}')
        def sha(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()
        session = dict(deadline=500, resources=dict(min_disk_gib=12))
        registration = dict(original_deadline=500, previous_failure=str(root / 'failure.json'),
            previous_failure_sha256=sha(root / 'failure.json'), verified_batches=[str(root / 'batch')],
            verified_batch_result_sha256={str(root / 'batch'): sha(root / 'batch/result.json')},
            source_strategy_sha256=sha(root / 'round02/strategy.json'),
            stable_resource_admission=dict(stable_seconds=120))
        (root / 'session.json').write_text(json.dumps(session))
        (root / 'round02/ram_recovery02.json').write_text(json.dumps(registration))

    def test_tampered_predecessor_refused_before_wait_or_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            (root / 'failure.json').write_text('{"changed":true}')
            with patch.object(stable, 'ROOT', root), patch.object(stable, 'PUBLIC', root), \
                    patch.object(stable.time, 'time', return_value=100), \
                    patch.object(stable, 'stable_admission') as wait:
                with self.assertRaises(AssertionError):
                    stable.run()
                wait.assert_not_called()
                self.assertFalse((root / 'round02/ram_recovery02_bootstrap_started.json').exists())

    def test_expired_session_and_open_final_freeze_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            with patch.object(stable, 'ROOT', root), patch.object(stable, 'PUBLIC', root):
                with patch.object(stable.time, 'time', return_value=501):
                    with self.assertRaises(TimeoutError):
                        stable.validate_registration()
                (root / 'final_freeze.json').write_text('{}')
                with patch.object(stable.time, 'time', return_value=100):
                    with self.assertRaises(AssertionError):
                        stable.validate_registration()

    def test_expiry_cannot_query_gpu_or_sleep(self):
        with patch.object(stable, 'read', side_effect=[dict(deadline=1),
                dict(stable_resource_admission=dict(stable_seconds=120, poll_seconds=15,
                    simulation_min_ram_gib=34, simulation_min_vram_mib=16000))]), \
                patch.object(stable.time, 'time', return_value=2), \
                patch.object(stable.subprocess, 'check_output') as query, \
                patch.object(stable.time, 'sleep') as sleep:
            with self.assertRaises(TimeoutError):
                stable.stable_admission(Path('/unused'), simulation=True)
            query.assert_not_called()
            sleep.assert_not_called()


if __name__ == '__main__':
    unittest.main()
