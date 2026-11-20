"""Deletion boundary checks use only disposable synthetic files."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PUBLIC=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('nominal_cleanup',PUBLIC/'cleanup_retired_nominals.py')
cleanup=importlib.util.module_from_spec(spec);spec.loader.exec_module(cleanup)

class Boundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.allowed=self.base/cleanup.ALLOWED[0]/'A/backend/A/nominal';self.allowed.mkdir(parents=True)
        self.patch=patch.object(cleanup,'BASE',self.base);self.patch.start()
    def tearDown(self):self.patch.stop();self.temp.cleanup()
    def test_only_pinned_derivative_paths(self):
        good=self.allowed/'episode.pt';good.write_bytes(b'cache');self.assertEqual(cleanup.safe(good),good)
        for name in ['best.pt','last.pt','statistics.pt','source.h5']:
            p=self.allowed/name;p.write_bytes(b'protected')
            with self.assertRaises(AssertionError):cleanup.safe(p)
        other=self.base/'other_project/nominal/episode.pt';other.parent.mkdir(parents=True);other.write_bytes(b'other')
        with self.assertRaises(AssertionError):cleanup.safe(other)
        self.assertEqual(other.read_bytes(),b'other')
    def test_links_rejected(self):
        p=self.allowed/'episode.pt';p.write_bytes(b'cache')
        linked=self.allowed/'alias.pt';linked.symlink_to(p)
        with self.assertRaises(AssertionError):cleanup.safe(linked)
        hard=self.allowed/'hard.pt';hard.hardlink_to(p)
        with self.assertRaises(AssertionError):cleanup.safe(p)
    def test_changed_manifest_cannot_start_deletion(self):
        ledger=self.base/'ledger';ledger.mkdir();(ledger/'manifest.json.gz').write_bytes(b'changed')
        with patch.object(cleanup,'LEDGER',ledger):
            with self.assertRaises(AssertionError):cleanup.apply('0'*64)
        self.assertFalse((ledger/'deleted.jsonl').exists())

if __name__=='__main__':unittest.main()
