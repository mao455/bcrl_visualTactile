"""Whole-root protection checks for the extra historical derivative scope."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('more_cleanup',Path(__file__).with_name('cleanup_more_retired_nominals.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class ProtectedRoots(unittest.TestCase):
    def test_protected_subtree_and_future_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            protected=base/m.retired.ALLOWED[0]/'round05/learning'
            path=protected/'A/backend/A/nominal/future_episode.pt'
            path.parent.mkdir(parents=True);path.write_bytes(b'protected future cache')
            with patch.object(m.retired,'BASE',base),patch.object(m,'PROTECTED_ROOTS',(protected,)):
                with self.assertRaises(AssertionError):m.safe(path)
            self.assertEqual(path.read_bytes(),b'protected future cache')

    def test_sibling_derivative_allowed_but_external_and_models_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary)
            protected=base/m.retired.ALLOWED[0]/'round05/learning'
            folder=base/m.retired.ALLOWED[0]/'round06/learning/A/backend/A/nominal'
            folder.mkdir(parents=True)
            good=folder/'episode.pt';good.write_bytes(b'derivative')
            model=folder/'best.pt';model.write_bytes(b'model')
            external=base/'another_project/nominal/episode.pt';external.parent.mkdir(parents=True);external.write_bytes(b'other')
            with patch.object(m.retired,'BASE',base),patch.object(m,'PROTECTED_ROOTS',(protected,)):
                self.assertEqual(m.safe(good),good)
                with self.assertRaises(AssertionError):m.safe(model)
                with self.assertRaises(AssertionError):m.safe(external)
            self.assertEqual(external.read_bytes(),b'other')

if __name__=='__main__':unittest.main()
