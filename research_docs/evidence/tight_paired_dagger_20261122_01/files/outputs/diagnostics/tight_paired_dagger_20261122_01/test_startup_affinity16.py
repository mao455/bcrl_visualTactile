import importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('affinity16',Path(__file__).with_name('startup_affinity16.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
class P:
 pid=123
 def create_time(self):return 100.
 def is_running(self):return True
 def cmdline(self):return [m.PYTHON,'-m',m.MODULE,'worker','--output',str(m.R/'evaluation/test/wave33')]
 def cwd(self):return str(m.B)
 def nice(self):return 10
class Tests(unittest.TestCase):
 def state(self):return dict(command=P().cmdline(),pid=123,stage=16,utc=1.)
 def test_identity(self):m.identity(self.state(),P(),100.,m.R/'evaluation/test/wave33')
 def test_wrong_command(self):
  p=P();p.cmdline=lambda:['external']
  with self.assertRaises(ValueError):m.identity(self.state(),p,100.,m.R/'evaluation/test/wave33')
 def test_reused_pid(self):
  p=P();p.create_time=lambda:101.
  self.assertFalse(m.same(p,100.))
 def test_outside_registered_remaining(self):
  s=self.state();s['stage']=15
  with self.assertRaises(ValueError):m.identity(s,P(),100.,m.R/'evaluation/test/wave33')
 def run_probe(self,ready,error=False):
  calls=[];writes=[]
  def ch(p,b,cpus):calls.append(set(cpus));return[123]
  clock=iter([1.,2.,100.])
  with patch.object(m.psutil,'Process',return_value=P()),patch.object(m,'identity'),patch.object(m,'read',return_value={'current_stage':'environment_constructor_started'}),patch.object(m,'ready',side_effect=ready),patch.object(m,'masks',return_value=[(123,m.OLD)]),patch.object(m,'change',side_effect=ch),patch.object(m,'write',side_effect=lambda p,x:writes.append(p.name)),patch.object(m.time,'time',return_value=200.),patch.object(m.time,'monotonic',side_effect=lambda:next(clock)):
   if error:
    with self.assertRaises(RuntimeError):m.probe(self.state())
   else:m.probe(self.state())
  return calls,writes
 def test_ready_does_not_change(self):
  calls,writes=self.run_probe([True]);self.assertEqual(calls,[]);self.assertEqual(writes,[])
 def test_cap_restores(self):
  calls,writes=self.run_probe([False,False,False]);self.assertEqual(calls,[m.WIDE,m.OLD]);self.assertEqual(writes,['wave33_intent.json','wave33_exit.json'])
 def test_exception_restores(self):
  calls,writes=self.run_probe([False,False,RuntimeError('injected')],True);self.assertEqual(calls,[m.WIDE,m.OLD]);self.assertEqual(writes,['wave33_intent.json','wave33_exit.json'])
if __name__=='__main__':unittest.main()
