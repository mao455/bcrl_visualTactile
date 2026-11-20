import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

PUBLIC = Path(__file__).resolve().parent

def load(name):
    spec = importlib.util.spec_from_file_location(name, PUBLIC / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

supervised = load('round2_supervised_recovery')
failed = load('archive_failed_captures')

class SupervisionTests(unittest.TestCase):
    def test_storage_is_rechecked_after_ram_wait(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            public = project / 'research_docs/evidence/test'
            public.mkdir(parents=True)
            root = project / 'run'
            root.mkdir()
            owner = root / 'owners/capture'
            output = root / 'batch'
            output.mkdir()
            (output / 'plan.json').write_text(json.dumps(dict(minimum_start_disk_gib=32.5)))
            config = project / 'configs/experiments/adaptive_dagger'
            config.mkdir(parents=True)
            (config / 'capture.json').write_text(json.dumps(dict(stages=[dict(args=['--output',str(output)])])))
            (root / 'session.json').write_text(json.dumps(dict(deadline=500,resources=dict(min_disk_gib=12))))
            (root / 'round02').mkdir()
            policy = dict(stable_seconds=60,poll_seconds=5,collection_min_ram_gib=28,
                simulation_min_ram_gib=30,training_min_ram_gib=24,simulation_min_vram_mib=16000,training_min_vram_mib=11000)
            (root / 'round02/ram_recovery03.json').write_text(json.dumps(dict(stable_resource_admission=policy)))
            now = [100]
            with patch.object(supervised,'ROOT',root),patch.object(supervised,'PUBLIC',public), \
                    patch.object(supervised.time,'time',side_effect=lambda:now[0]), \
                    patch.object(supervised.time,'sleep',side_effect=lambda _:now.__setitem__(0,501)), \
                    patch.object(supervised.shutil,'disk_usage',return_value=Mock(free=31*2**30)), \
                    patch('psutil.virtual_memory',return_value=Mock(available=40*2**30)), \
                    patch.object(supervised.subprocess,'check_output',return_value='23000\n'):
                with self.assertRaises(TimeoutError):
                    supervised.stable_admission(owner,simulation=True,lanes=1)
            receipt=json.loads((owner/'resource_admission.json').read_text())
            self.assertFalse(receipt['instantaneous_ready'])
            self.assertEqual(receipt['required_disk_gib'],32.5)
            self.assertEqual(receipt['required_ram_gib'],28)

    def test_failed_capture_scope_preserves_valid_raw(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            path=base/failed.SCOPES[0]/'case/demonstration/data.h5.partial'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'failed evidence')
            with patch.object(failed.a,'BASE',base):
                self.assertEqual(failed.safe(path),path)
                path.with_name('data.h5').write_bytes(b'valid raw')
                with self.assertRaises(AssertionError):failed.safe(path)
            other=base/'current/data.h5.partial';other.parent.mkdir();other.write_bytes(b'active')
            with patch.object(failed.a,'BASE',base):
                with self.assertRaises(AssertionError):failed.safe(other)

    def test_actual_registration_keeps_cases_and_runtime_reserves(self):
        root=supervised.ROOT
        previous=json.loads((root/'round02/ram_recovery02.json').read_text())
        current=json.loads((root/'round02/ram_recovery03.json').read_text())
        self.assertEqual(current['remaining_schedule'],previous['remaining_schedule'])
        self.assertEqual(current['verified_batch_result_sha256'],previous['verified_batch_result_sha256'])
        self.assertEqual(current['original_deadline'],previous['original_deadline'])
        self.assertEqual(current['collection_lanes'],1)
        self.assertEqual(json.loads((root/'session.json').read_text())['resources'],
            dict(min_gpu_free_mib=4096,min_ram_gib=10,min_disk_gib=12))

if __name__=='__main__':unittest.main()
