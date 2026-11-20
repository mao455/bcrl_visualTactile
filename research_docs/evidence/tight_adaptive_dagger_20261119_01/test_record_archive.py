import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zstandard as zstd

spec = importlib.util.spec_from_file_location('closed_record_archive', Path(__file__).with_name('archive_closed_records.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class ArchiveTests(unittest.TestCase):
    def test_reject_raw_checkpoint_and_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            for name in ['data.h5', 'data.h5.partial', 'best.pt', 'data.json']:
                path = base / name
                path.write_bytes(b'evidence')
                with patch.object(a, 'BASE', base), patch.object(a, 'ROOT', base / 'active'):
                    with self.assertRaises(AssertionError):
                        a.safe(path)

    def test_reject_external_symlink_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            original = base / 'learning.jsonl'
            original.write_bytes(b'evidence')
            alias = base / 'shared.jsonl'
            alias.symlink_to(original)
            with patch.object(a, 'BASE', base), patch.object(a, 'ROOT', base / 'active'):
                with self.assertRaises(AssertionError):
                    a.safe(alias)
                alias.unlink()
                alias.hardlink_to(original)
                with self.assertRaises(AssertionError):
                    a.safe(original)
            with patch.object(a, 'BASE', base / 'other'), patch.object(a, 'ROOT', base / 'active'):
                with self.assertRaises(AssertionError):
                    a.safe(original)

    def test_refuse_current_round2_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / 'active/round02/learning/A/training/learning.jsonl'
            path.parent.mkdir(parents=True)
            path.write_bytes(b'evidence')
            with patch.object(a, 'BASE', base), patch.object(a, 'ROOT', base / 'active'):
                with self.assertRaises(AssertionError):
                    a.safe(path)

    def test_byte_roundtrip_and_original_removed_only_after_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / 'historical/training/learning.jsonl'
            path.parent.mkdir(parents=True)
            content = b'{"tick":17,"exposure":50}\n' * 20000
            path.write_bytes(content)
            with patch.object(a, 'BASE', base), patch.object(a, 'ROOT', base), \
                    patch.object(a, 'safe', side_effect=lambda x: Path(x)), \
                    patch.object(a.helper, 'process_references', return_value=(set(), set(), {})):
                row = a.archive(path, a.helper.digest(path), opened=set())
            self.assertFalse(path.exists())
            with Path(row['archive']).open('rb') as source, zstd.ZstdDecompressor().stream_reader(source) as decoded:
                self.assertEqual(decoded.read(), content)
            self.assertTrue(row['original_removed'])
            self.assertTrue(row['byte_roundtrip_verified'])
            self.assertEqual(json.loads(path.with_name(path.name + '.archive.json').read_text()), row)

    def test_open_inode_refused_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            path = base / 'learning.jsonl'
            path.write_bytes(b'unique evidence')
            info = path.stat()
            with patch.object(a, 'safe', side_effect=lambda x: Path(x)):
                with self.assertRaises(AssertionError):
                    a.archive(path, a.helper.digest(path), opened={(info.st_dev, info.st_ino)})
            self.assertEqual(path.read_bytes(), b'unique evidence')
            self.assertFalse(path.with_name(path.name + '.zst').exists())


if __name__ == '__main__':
    unittest.main()
