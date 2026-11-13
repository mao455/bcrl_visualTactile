import copy
import json
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from tools import tight_live_publish as p


def command(*args):
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def repository(self):
        remote = self.root/'remote.git'
        repo = self.root/'publisher'
        command('git', 'init', '--bare', str(remote))
        command('git', 'clone', str(remote), str(repo))
        p.git(repo, 'config', 'user.name', 'publisher-test')
        p.git(repo, 'config', 'user.email', 'publisher-test@example.invalid')
        p.git(repo, 'checkout', '-b', p.BRANCH)
        (repo/'README').write_text('base\n')
        p.git(repo, 'add', 'README')
        p.git(repo, 'commit', '-m', 'base')
        p.git(repo, 'push', '-u', 'origin', p.BRANCH)
        return remote, repo

    def data(self):
        return dict(published_utc=1794560000, watch='test', state={'state':'RUNNING'},
                    completion_record_present=False, failure_record_present=False,
                    fairness={'matched_increment_passed':True}, evaluations=[], training=[], evidence_errors=[])

    def test_actual_push_idempotence_and_unrelated_dirt(self):
        remote, repo = self.repository()
        data = self.data()
        first = p.publish(repo, data)
        self.assertEqual(p.publish(repo, data), first)
        self.assertEqual(command('git','--git-dir',str(remote),'rev-parse',p.BRANCH),first)
        self.assertEqual(set(p.git(repo,'diff','--name-only','HEAD^','HEAD').splitlines()),set(p.PATHS))
        (repo/'private.json').write_text('do not publish')
        with self.assertRaisesRegex(RuntimeError,'unrelated'):
            p.publish(repo,data)
        self.assertEqual(command('git','--git-dir',str(remote),'rev-parse',p.BRANCH),first)

    def test_remote_advances_during_push_preserves_other_change(self):
        remote, repo = self.repository()
        peer=self.root/'peer'
        command('git','clone','--branch',p.BRANCH,str(remote),str(peer))
        p.git(peer,'config','user.name','peer');p.git(peer,'config','user.email','peer@example.invalid')
        original=p.git
        advanced=[]
        def racing(path,*args,**kw):
            if args[0]=='push' and not advanced:
                advanced.append(True)
                (peer/'reviewed.md').write_text('other reviewed work\n')
                original(peer,'add','reviewed.md');original(peer,'commit','-m','parallel review')
                original(peer,'push','origin',p.BRANCH)
            return original(path,*args,**kw)
        with patch.object(p,'git',side_effect=racing):
            head=p.publish(repo,self.data())
        self.assertTrue((repo/'reviewed.md').exists())
        self.assertEqual(command('git','--git-dir',str(remote),'rev-parse',p.BRANCH),head)

    def test_interrupted_bot_owned_write_is_recovered(self):
        _,repo=self.repository()
        p.publish(repo,self.data())
        (repo/p.PATHS[0]).write_text('interrupted write')
        data=self.data();data['state']['state']='COMPLETE'
        p.publish(repo,data)
        self.assertFalse(p.git(repo,'status','--porcelain'))
        self.assertIn('COMPLETE',(repo/p.PATHS[0]).read_text())

    def test_failed_push_retains_commit_and_recovers(self):
        _,repo=self.repository();original=p.git
        def offline(path,*args,**kw):
            if args[0]=='push':raise RuntimeError('network unavailable')
            return original(path,*args,**kw)
        with patch.object(p,'git',side_effect=offline):
            with self.assertRaisesRegex(RuntimeError,'network'):
                p.publish(repo,self.data())
        pending=p.git(repo,'rev-parse','HEAD')
        self.assertEqual(p.publish(repo,self.data()),pending)

    def test_receipt_hash_and_actual_plan_must_match(self):
        root=self.root/'eval';group=root/'G01';case=group/'evaluation'/'case0';case.mkdir(parents=True)
        spec=dict(identity='case0',seed=17)
        p.write(root/'evaluation_plan.json',[dict(identity='case0',spec=spec)])
        result=dict(spec,software_complete=True,legal=True,success=False,checkpoint_sha256='frozen')
        p.write(case/'result.json',result)
        receipt=dict(result_sha256=p.sha(case/'result.json'),binding=dict(spec=spec,checkpoint_sha256='frozen',plan_sha256=p.sha(root/'evaluation_plan.json')))
        p.write(case/'evaluation_receipt.json',receipt)
        summary=dict(results=[result],completed=1,valid=1,successes=0,planned=1,complete=True,software_failures=0,native_lifts=0,expert_takeover=False,formal_test22_used=False)
        path=group/'evaluation_summary.json';p.write(path,summary)
        self.assertEqual(p.checked_summary(path)['completed'],1)
        summary['successes']=1;p.write(path,summary)
        with self.assertRaisesRegex(ValueError,'count mismatch'):p.checked_summary(path)
        summary['successes']=0;p.write(path,summary)
        result['seed']=18;p.write(case/'result.json',result)
        with self.assertRaisesRegex(ValueError,'binding mismatch'):p.checked_summary(path)

    def test_unchanged_snapshot_never_republishes_timestamp(self):
        config=dict(runtime=str(self.root),deadline=time.time()+30,worktree='unused')
        data=self.data();data.pop('published_utc')
        with patch.object(p,'snapshot',return_value=data),patch.object(p,'publish',return_value='head') as pub:
            p.run(config,once=True)
            p.run(config,once=True)
            self.assertEqual(pub.call_count,1)

    def test_deadline_does_not_call_git(self):
        with patch.object(p.subprocess,'run') as run:
            with self.assertRaises(TimeoutError):p.git(self.root,'status',deadline=time.time()-1)
            run.assert_not_called()


if __name__=='__main__':unittest.main()
