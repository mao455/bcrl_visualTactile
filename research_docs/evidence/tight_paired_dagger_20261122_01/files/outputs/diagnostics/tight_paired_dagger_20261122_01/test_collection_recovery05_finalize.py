import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
HERE=Path(__file__).parent;sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('finalize',HERE/'collection_recovery05_finalize.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class OverlayTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
  self.source=self.root/'source';self.source.mkdir();(self.source/'result.json').write_text('{"v":1}');(self.source/'numeric.h5').write_bytes(b'unchanged payload')
 def test_overlay_json_independent_payload_linked(self):
  out=self.root/'out';m.overlay(self.source,out);(out/'result.json').write_text('{}')
  self.assertEqual((self.source/'result.json').read_text(),'{"v":1}')
  self.assertTrue((out/'numeric.h5').is_symlink());self.assertEqual((out/'numeric.h5').read_bytes(),b'unchanged payload')
 def test_overlay_no_overwrite(self):
  out=self.root/'out';out.mkdir()
  with self.assertRaises(FileExistsError):m.overlay(self.source,out)
 def test_interrupted_unknown_not_called_illegal(self):
  result=m.interrupted_result({'spec':{'identity':'x'}},self.source)
  self.assertIsNone(result['legal']);self.assertFalse(result['software_complete']);self.assertFalse(result['accepted_for_training']);self.assertTrue(result['assignment_consumed'])
 def test_interrupted_known_legality(self):
  (self.source/'initial_collision.json').write_text('{"legal":true}')
  self.assertIs(m.interrupted_result({'spec':{}},self.source)['legal'],True)
if __name__=='__main__':unittest.main()
