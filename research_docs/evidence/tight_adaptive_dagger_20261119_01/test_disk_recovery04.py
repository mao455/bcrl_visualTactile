"""Recovery gates prevent duplicated A work and learning after the test boundary."""
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

PUBLIC = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('disk_recovery_under_test', PUBLIC / 'round2_disk_recovery04.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


class DiskRecoveryTests(unittest.TestCase):
    def test_remaining_work_never_retrains_completed_A_or_recaches_visuals(self):
        stages = recovery.remaining_stages(Path('/test/learning'))
        fronts = [s for s in stages if s['entry'] == 'frontend']
        self.assertEqual(len(fronts), 1)
        self.assertEqual(fronts[0]['args'][fronts[0]['args'].index('--role') + 1], 'B')
        self.assertNotIn('visual_cache', [s['entry'] for s in stages])
        self.assertEqual([s['entry'] for s in stages].count('backend_prepare'), 2)

    def test_original_deadline_and_freeze_both_block_learning(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with patch.object(recovery, 'ROOT', root):
                (root / 'session.json').write_text(json.dumps(dict(deadline=time.time()-1)))
                with self.assertRaises(AssertionError):
                    recovery.session_gate()
                (root / 'session.json').write_text(json.dumps(dict(deadline=time.time()+60)))
                (root / 'final_freeze.json').write_text('{}')
                with self.assertRaises(AssertionError):
                    recovery.session_gate()
                (root / 'final_freeze.json').unlink()
                (root / 'final_test').mkdir()
                with self.assertRaises(AssertionError):
                    recovery.session_gate()

    def test_registration_rejects_changed_checkpoint_before_start(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / 'learning'
            checkpoint = target / 'B/training/last.pt'
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b'durable checkpoint')
            deadline = time.time()+60
            (root / 'session.json').write_text(json.dumps(dict(deadline=deadline)))
            reg = root / 'registration.json'
            sha = recovery.digest(checkpoint)
            reg.write_text(json.dumps(dict(original_deadline=deadline,
                identities={str(checkpoint): sha}, B_durable=dict(checkpoint_sha256=sha))))
            with patch.object(recovery, 'ROOT', root), patch.object(recovery, 'TARGET', target), patch.object(recovery, 'REG', reg):
                recovery.validate_registration()
                checkpoint.write_bytes(b'changed by another writer')
                with self.assertRaises(AssertionError):
                    recovery.validate_registration()


if __name__ == '__main__':
    unittest.main()
