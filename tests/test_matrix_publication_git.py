import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest
import tools.tight_diagnostics as d

SOURCE='a'*40
REPORT='b'*40
TARGET='refs/heads/feature/tight-state-foundation'

class Git:
 def __init__(self, extra='', staged='', head=SOURCE, remote=SOURCE, push_fail=False, commit_fail=False):
  self.extra=extra;self.initial_staged=staged;self.head=head;self.remote=remote;self.push_fail=push_fail;self.commit_fail=commit_fail;self.calls=[];self.paths=[];self.staged=[];self.committed=False
 def __call__(self,cmd,**kwargs):
  assert cmd[0]=='git';a=cmd[1:];self.calls.append(a)
  if a==['rev-parse','HEAD']:out=self.head+'\n'
  elif a==['rev-parse','HEAD^']:out=SOURCE+'\n'
  elif a[:2]==['remote','get-url']:out='https://github.com/mao455/bcrl_visualTactile.git\n'
  elif a[0]=='ls-remote':out=self.remote+'\t'+TARGET+'\n'
  elif a[0]=='show':out='measured failure, not success'
  elif a[:3]==['diff','--cached','--name-only']:out='\0'.join(([self.initial_staged] if self.initial_staged else [])+self.staged)
  elif a[:2]==['diff','--name-only']:out=self.extra or ('' if self.committed else 'research_docs/STATE_DP_RESULTS.md\0')
  elif a[0]=='diff-tree':out='\0'.join(self.paths)
  elif a[0]=='add':self.paths=a[2:];self.staged=self.paths.copy();out=''
  elif a[0]=='commit':
   if self.commit_fail:raise subprocess.CalledProcessError(1,cmd,stderr='commit failure')
   self.head=REPORT;self.staged=[];self.committed=True;out='report committed'
  elif a[0]=='restore':self.staged=[];out=''
  elif a[0]=='push':
   assert a==['push','origin','HEAD:'+TARGET]
   if self.push_fail:raise subprocess.CalledProcessError(1,cmd,stderr='network failure')
   out='pushed'
  else:raise AssertionError(a)
  return SimpleNamespace(stdout=out,stderr='',returncode=0)

def fixture(tmp_path, monkeypatch):
 repo=tmp_path/'repo';repo.mkdir();root=tmp_path/'run';root.mkdir();out=repo/'research_docs/evidence/matrix/results';out.mkdir(parents=True)
 monkeypatch.setattr(d,'CODE',repo)
 (repo/'runtime.py').write_text('source')
 (repo/'research_docs/STATE_DP_RESULTS.md').write_text('measured failure, not success')
 status=dict(state='FAILED',owned_workers_exited=True)
 d.write(root/'status.json',status);d.write(root/'session.json',dict(source_head=SOURCE));d.write(root/'source_freeze.json',{'runtime.py':d.sha(repo/'runtime.py')})
 d.write(out/'summary.json',dict(schema='method_matrix_cpu_publication_v1',status=status,session=dict(source_head=SOURCE),source_run=str(root),stages={}))
 manifest={p.name:dict(bytes=p.stat().st_size,sha256=d.sha(p)) for p in out.iterdir()};d.write(out/'artifacts.json',manifest)
 return repo,root,out

def invoke(root,out,git):
 before=(root/'status.json').read_bytes()
 with patch.object(d.subprocess,'run',git):result=d.matrix_commit_publication(root,out)
 assert (root/'status.json').read_bytes()==before
 return result

def test_success(tmp_path,monkeypatch):
 repo,root,out=fixture(tmp_path,monkeypatch);git=Git();r=invoke(root,out,git)
 assert r['complete'] and r['pushed'] and r['report_commit']==REPORT
 assert set(r['files'])=={'research_docs/STATE_DP_RESULTS.md','research_docs/evidence/matrix/results/summary.json','research_docs/evidence/matrix/results/artifacts.json'}
 assert (root/'public_git.json').is_file() and not (root/'public_git_error.json').exists()

@pytest.mark.parametrize('kwargs,reason',[
 ({'staged':'user.txt'},'Pre-existing staged'),({'head':'c'*40},'HEAD differs'),
 ({'remote':'c'*40},'Remote target differs'),({'extra':'user.txt\0'},'Other tracked')])
def test_git_guard(tmp_path,monkeypatch,kwargs,reason):
 repo,root,out=fixture(tmp_path,monkeypatch);git=Git(**kwargs);r=invoke(root,out,git)
 assert not r['complete'] and reason in r['error'];assert not any(a[0] in ('add','commit','push','restore') for a in git.calls)
 assert (out/'summary.json').is_file() and (root/'public_git_error.json').is_file()

def test_runtime_guard(tmp_path,monkeypatch):
 repo,root,out=fixture(tmp_path,monkeypatch);(repo/'runtime.py').write_text('changed');git=Git();r=invoke(root,out,git)
 assert 'Frozen runtime changed' in r['error'];assert not any(a[0]=='add' for a in git.calls)

def test_user_result_prose_is_not_committed(tmp_path,monkeypatch):
 repo,root,out=fixture(tmp_path,monkeypatch)
 (repo/'research_docs/STATE_DP_RESULTS.md').write_text('User edited independent result prose')
 git=Git();r=invoke(root,out,git)
 assert 'outside the owner-managed' in r['error'] and not any(a[0]=='add' for a in git.calls)

@pytest.mark.parametrize('kind',['symlink','raw','oversized','stale'])
def test_output_guard(tmp_path,monkeypatch,kind):
 repo,root,out=fixture(tmp_path,monkeypatch)
 if kind=='symlink':(out/'groups.csv').symlink_to(repo/'runtime.py')
 elif kind=='raw':(out/'weights.pt').write_bytes(b'weights')
 elif kind=='oversized':
  with (out/'groups.csv').open('wb') as stream:stream.truncate(20*1024**2+1)
 else:
  value=d.read(out/'summary.json');value['session']['source_head']='d'*40;d.write(out/'summary.json',value)
 d.write(out/'artifacts.json',{p.name:dict(bytes=p.stat().st_size,sha256=d.sha(p)) for p in out.iterdir() if p.name!='artifacts.json'})
 git=Git();r=invoke(root,out,git);assert not r['complete'] and not any(a[0]=='add' for a in git.calls)

def test_push_failure_keeps_commit_and_receipt_without_rewind(tmp_path,monkeypatch):
 repo,root,out=fixture(tmp_path,monkeypatch);git=Git(push_fail=True);r=invoke(root,out,git)
 assert not r['complete'] and r['report_commit']==REPORT and not r['pushed']
 assert r['git_stderr']=='network failure' and not any(a[0] in ('reset','restore','update-ref') for a in git.calls)

def test_commit_failure_unstages_only_our_report(tmp_path,monkeypatch):
 repo,root,out=fixture(tmp_path,monkeypatch);git=Git(commit_fail=True);r=invoke(root,out,git)
 assert not r['complete'] and r['report_commit'] is None and git.staged==[]
 assert any(a==['restore','--staged','--',*r['files']] for a in git.calls)
 assert not any(a[0]=='push' for a in git.calls)
