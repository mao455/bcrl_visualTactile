"""Finite post-training chain; no retry or parameter tuning after evaluation."""
import json,os,subprocess,sys,time,traceback
from pathlib import Path
from visuotactile.training_resources import atomic_json
PROJECT=Path.cwd();PUBLIC=Path(__file__).resolve().parent
ROOT=PROJECT/'outputs/visuotactile/run/tight_exposure50_20261119_01'
SESSION=json.loads((ROOT/'session.json').read_text())
POST_ROOT=Path(os.environ.get('TIGHT_POST_ATTEMPT_ROOT', str(ROOT))).resolve()
assert POST_ROOT==ROOT or ROOT in POST_ROOT.parents
assert POST_ROOT.is_dir()
try:
 while not (ROOT/'project_complete.json').exists():
  if (ROOT/'project_failure.json').exists():raise RuntimeError('Training owner failed; preserve results and inspect recovery')
  if time.time()>SESSION['deadline']:raise TimeoutError('Registered total deadline')
  time.sleep(30)
 for mode in ['evaluation','offline']:
  with (POST_ROOT/f'post_{mode}.log').open('x') as log:
   subprocess.run([sys.executable,str(PUBLIC/'analyze.py'),mode],stdout=log,stderr=subprocess.STDOUT,check=True)
 with (POST_ROOT/'post_evaluation_run.log').open('x') as log:
  subprocess.run([sys.executable,'-m','tools.tight_project','evaluate','--job','configs/experiments/exposure50/evaluate.json'],stdout=log,stderr=subprocess.STDOUT,check=True)
 with (POST_ROOT/'post_report.log').open('x') as log:
  subprocess.run([sys.executable,str(PUBLIC/'analyze.py'),'report'],stdout=log,stderr=subprocess.STDOUT,check=True)
 atomic_json(POST_ROOT/'post_complete.json',dict(utc=time.time(),scope='Training exposure/export +64 closed-loop cases; scientific review and publication remain'))
except BaseException as e:
 atomic_json(POST_ROOT/'post_failure.json',dict(utc=time.time(),error=repr(e)));raise
