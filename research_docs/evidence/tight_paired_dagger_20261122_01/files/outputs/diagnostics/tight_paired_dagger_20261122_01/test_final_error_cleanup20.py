"""CPU-only safety fixtures. Every signal syscall and process mutation is mocked."""
import contextlib
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).with_name('final_error_cleanup20.py')
spec = importlib.util.spec_from_file_location('cleanup20_tested', SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

TRACE = '''Traceback (most recent call last):
  File "/repo/visuotactile/queued_rollout.py", line 291, in run_queue
    with lease.block():
  File "/repo/visuotactile/state_resources.py", line 219, in block
    if low:raise RuntimeError('RAM reserve')
RuntimeError: RAM reserve
'''


class FakeProc:
    def __init__(self, pid=10, birth=100., parent=9, ancestors=(), status='running'):
        self.pid, self.birth, self.parent, self.ancestors, self.state = pid, birth, parent, ancestors, status
    def create_time(self): return self.birth
    def ppid(self): return self.parent
    def parents(self): return list(self.ancestors)
    def status(self): return self.state
    def cmdline(self): return ['fake-own-worker']


class Clock:
    def __init__(self): self.value = 0.
    def monotonic(self): return self.value
    def time(self): return 1000.+self.value
    def sleep(self, duration): self.value += duration


class ExceptionPolicy(unittest.TestCase):
    def test_complete_known_error(self): self.assertTrue(m.complete_known_exception(TRACE))
    def test_no_traceback_is_no_trigger(self): self.assertFalse(m.complete_known_exception('RuntimeError: RAM reserve'))
    def test_incomplete_stack_is_no_trigger(self): self.assertFalse(m.complete_known_exception(TRACE.replace('visuotactile/queued_rollout.py','other.py')))
    def test_other_error_is_no_trigger(self): self.assertFalse(m.complete_known_exception(TRACE.replace('RuntimeError: RAM reserve','RuntimeError: something else')))
    def test_post_exception_append_is_no_trigger(self): self.assertFalse(m.complete_known_exception(TRACE+'running again\n'))
    def test_only_last_traceback_is_used(self): self.assertFalse(m.complete_known_exception(TRACE+'Traceback (most recent call last):\nOtherError: no\n'))


class StabilityPolicy(unittest.TestCase):
    def setUp(self):
        self.obs = dict(fingerprint=(59,100.,'log','progress'), log_mtime=100., progress_mtime=99., progress_utc=99.)
        self.prev = dict(fingerprint=self.obs['fingerprint'], first_seen=100., first_seen_monotonic=5.)
    def test_full120_seconds_required(self):
        self.assertFalse(m.stable_ready(self.obs,self.prev,300.,124.999))
        self.assertTrue(m.stable_ready(self.obs,self.prev,300.,125.))
    def test_wall_clock_jump_cannot_authorize(self): self.assertFalse(m.stable_ready(self.obs,self.prev,100000.,6.))
    def test_no_previous_observation_cannot_authorize(self): self.assertFalse(m.stable_ready(self.obs,None,300.,125.))
    def test_changed_progress_cannot_authorize(self):
        self.obs['fingerprint']=(59,100.,'log','newprogress'); self.assertFalse(m.stable_ready(self.obs,self.prev,300.,125.))
    def test_changed_recent_log_cannot_authorize(self):
        self.obs['log_mtime']=250.; self.assertFalse(m.stable_ready(self.obs,self.prev,300.,125.))


class GroupIdentity(unittest.TestCase):
    def setUp(self): self.root=FakeProc();self.child=FakeProc(11,101.,10,(self.root,))
    def census(self, rows, sid=10):
        with patch.object(m.os,'getpgid',return_value=10),patch.object(m.os,'getsid',return_value=sid):
            return m.group_census(self.root,rows)
    def test_own_descendant_accepted(self): self.assertEqual([r['pid'] for r in self.census([self.root,self.child])],[10,11])
    def test_foreign_member_rejected(self):
        with self.assertRaises(ValueError):self.census([self.root,FakeProc(12,101.,77,())])
    def test_recycled_root_pid_rejected(self):
        with self.assertRaises(ValueError):self.census([FakeProc(10,200.)])
    def test_wrong_session_rejected(self):
        with self.assertRaises(ValueError):self.census([self.root,self.child],sid=99)
    def test_older_descendant_rejected(self):
        with self.assertRaises(ValueError):self.census([self.root,FakeProc(11,99.,10,(self.root,))])


class MockedSignalBoundary(unittest.TestCase):
    def setUp(self):
        self.expected=dict(index=59,pid=10,birth=100.,fingerprint=(59,100.,'log','progress'))
        self.clock=Clock()
    def run_cleanup(self, out, fresh_side_effect=None):
        import psutil
        patches=[patch.object(m,'schedule_lock',lambda:contextlib.nullcontext()),
            patch.object(m,'fresh_match',side_effect=fresh_side_effect,return_value=self.expected),
            patch.object(m.os,'killpg'),patch.object(m.time,'time',self.clock.time),
            patch.object(m.time,'monotonic',self.clock.monotonic),patch.object(m.time,'sleep',self.clock.sleep),
            patch.object(psutil,'Process',return_value=FakeProc()),patch.object(m,'cleanup_guard')]
        with contextlib.ExitStack() as stack:
            mocks=[stack.enter_context(p) for p in patches]
            try:m.cleanup(self.expected,{},Path(out),'fake-root-registration-sha',Path(out)/'registration.json')
            except ValueError:return mocks[2].call_args_list,mocks[1].call_count,True
            return mocks[2].call_args_list,mocks[1].call_count,False
    def test_term_then_fresh_verified_kill_and15s_grace(self):
        with tempfile.TemporaryDirectory() as out:
            calls,fresh,failed=self.run_cleanup(out)
            self.assertFalse(failed);self.assertEqual([c.args[1] for c in calls],[m.signal.SIGTERM,m.signal.SIGKILL])
            self.assertGreaterEqual(self.clock.value,15.);self.assertEqual(fresh,5)
            self.assertTrue((Path(out)/'wave59_exit.json').exists())
            self.assertTrue((Path(out)/'wave59_term_intent.json').exists())
            self.assertTrue((Path(out)/'wave59_kill_intent.json').exists())
    def test_noop_identity_race_produces_zero_signals_and_zero_intent(self):
        with tempfile.TemporaryDirectory() as out:
            calls,_,failed=self.run_cleanup(out,[ValueError('identity changed')])
            self.assertTrue(failed);self.assertEqual(calls,[]);self.assertEqual(list(Path(out).iterdir()),[])
    def test_kill_reverification_failure_never_signals_replacement(self):
        with tempfile.TemporaryDirectory() as out:
            calls,_,failed=self.run_cleanup(out,[self.expected,self.expected,self.expected,ValueError('new worker')])
            self.assertTrue(failed);self.assertEqual([c.args[1] for c in calls],[m.signal.SIGTERM])
            self.assertTrue((Path(out)/'wave59_failure.json').exists())
            self.assertFalse((Path(out)/'wave59_kill.json').exists())
    def test_real_helper_stop_guard_precedes_any_signal(self):
        with tempfile.TemporaryDirectory() as out, patch.object(m.os,'killpg') as signals:
            (Path(out)/'STOP').touch()
            with self.assertRaisesRegex(ValueError,'Helper-local stop'):
                m.cleanup(self.expected,{},Path(out),'fake-sha',Path(out)/'registration.json')
            signals.assert_not_called()
            self.assertEqual([p.name for p in Path(out).iterdir()],['STOP'])
    def test_registration_change_guard_precedes_signal(self):
        with tempfile.TemporaryDirectory() as out, patch.object(m.os,'killpg') as signals, patch.object(m,'digest',return_value='new-sha'):
            with self.assertRaisesRegex(ValueError,'registration changed'):
                m.cleanup(self.expected,{},Path(out),'old-sha',Path(out)/'registration.json')
            signals.assert_not_called()


if __name__=='__main__':unittest.main()
