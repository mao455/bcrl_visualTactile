"""Cooperative handover only after both real DP trainers finish; no model polling."""
import argparse,json,time,os,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root
out=root/'round_15';ledger=out.with_name(out.name+'_stages');receipt=root/'night_boundary_request.json'
assert not receipt.exists();deadline=json.loads((root/'session.json').read_text())['deadline']-180
started=time.time()
def write(value):
 tmp=receipt.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(receipt)
while time.time()<deadline:
 if (root/'continuation/owner_finished.json').exists():raise RuntimeError('Owner ended before requested frontend boundary')
 if all((out/r/'training/finished.json').exists() for r in 'AB'):
  done={r:json.loads((out/r/'training/finished.json').read_text()) for r in 'AB'}
  assert all(v['complete'] and v['epochs']==12 and v['updates']==2460 for v in done.values())
  if list(ledger.glob('backend_prepare_*_worker.json')):raise RuntimeError('Backend already started; do not interrupt a later stage')
  write(dict(started=started,utc=time.time(),pid=os.getpid(),stage='requesting_after_both_DP_complete',
   reason='Latest user authorizes new bounded overnight resource window; preserve completed DP and old deadline',
   dp_finished=done,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
  (ledger/'STOP').touch(exist_ok=False)
  break
 time.sleep(.25)
else:raise TimeoutError('Original deadline reached before complete DP boundary')
