"""One explicit resource recovery: unchanged frozen trainers, sequential DP jobs."""
import importlib.util,json,sys,time,subprocess,traceback
from pathlib import Path
F=Path('/home/CNS2025331827/docs/bc+rl_RSS_budget_20261116');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(F))
spec=importlib.util.spec_from_file_location('frozen_run',F/'scripts/tight_budget_20261116/run.py');run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
R=run.ROOT
assert run.read(R/'failure.json')['stage']=='train'
assert not (R/'serial_recovery_started.json').exists(),'One recovery only'
import torch
for role in 'AB':
 assert run.read(R/'logs'/('dp_'+role)/'state.json')['exit_code']!=0
 s=torch.load(R/role/'training/last.pt',map_location='cpu',mmap=True,weights_only=False)
 assert s['epoch']==11 and s['updates']==2156 and s['ema_optimization_step']==2156
 del s
run.atomic_json(R/'serial_recovery_started.json',dict(utc=time.time(),frozen_source='76352bd',
 source_failure=str(R/'failure.json'),reason='Synchronized long-trajectory B2 graphs exceeded VRAM; only one DP GPU worker at a time',
 durable_epochs=11,durable_updates=2156,partial12_preserved=True,maximum_additional_attempts_per_role=1,
 maximum_effective_total_updates_per_role=4704,maximum_discarded_attempt_updates_per_role=140))
prefetch=subprocess.Popen([sys.executable,str(HERE/'read_ahead_serial.py')],cwd=F)
try:
 for role in 'AB':
  run.execute('dp_'+role,['tools.tight_fresh_train','train','--root',R,'--role',role,'--epochs','24','--offline-anchor-batch','32','--full-trajectory-batch'],R/role/'training/finished.json')
 run.atomic_json(R/'dp_complete.json',dict(complete=True,utc=time.time(),recovery='serialized GPUs, unchanged sampler/model/update semantics'))
 prefetch.wait(timeout=30)
 run.rear()
 run.atomic_json(R/'serial_recovery_complete.json',dict(complete=True,utc=time.time(),source_failure_preserved=True))
except BaseException as e:
 run.atomic_json(R/'serial_recovery_failed.json',dict(utc=time.time(),error=repr(e),traceback=traceback.format_exc()));raise
finally:
 if prefetch.poll() is None:
  prefetch.terminate()
  try:prefetch.wait(timeout=30)
  except subprocess.TimeoutExpired:prefetch.kill();prefetch.wait()
