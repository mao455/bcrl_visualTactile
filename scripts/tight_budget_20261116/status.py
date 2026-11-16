"""One read-only handoff/status entry; never starts or retries work."""
import json,time,sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.tight_budget_20261116.run import ROOT,read

def last_update(path):
 if not path.exists():return None
 with path.open('rb') as f:
  f.seek(0,2);f.seek(max(0,f.tell()-4*2**20));raw=f.read()
 lines=raw.splitlines()
 if not raw.endswith(b'\n'):lines=lines[:-1]
 for line in reversed(lines):
  try:
   row=json.loads(line)
   return {k:row[k] for k in ['epoch','updates','update','loss'] if k in row}
  except json.JSONDecodeError:pass
 return None

def main():
 out=dict(root=str(ROOT),deadline=datetime.fromtimestamp(read(ROOT/'session.json')['deadline']).astimezone().isoformat(),networks={},caches={},active_workers=[])
 for role in 'AB':
  for kind,path in [('dp',ROOT/role/'training'),('rear',ROOT/role/'backend'/role/'training')]:
   p=path/('validation.jsonl' if kind=='dp' else 'validation.json')
   if not p.exists():
    latest=last_update(path/('learning.jsonl' if kind=='dp' else 'loss.jsonl'))
    if latest is not None:out['networks'][role+'_'+kind]=dict(epoch=0,updates=0,score=None,best_epoch=None,best_score=None,stopping=None,complete=False,
     latest_logged_update=latest,latest_logged_scope='First epoch in progress; no completed independent validation yet.')
    continue
   if kind=='dp':
    raw=p.read_text();lines=raw.splitlines();lines=lines if raw.endswith('\n') else lines[:-1]
    rows=[json.loads(x) for x in lines]
   else:rows=read(p)
   if not rows:continue
   score=lambda x:x['validation']['selection'] if kind=='dp' else x['validation']['all']['normalized_equal_mse']
   best=min(rows,key=score);last=rows[-1]
   out['networks'][role+'_'+kind]=dict(epoch=last['epoch'],updates=last.get('updates',last.get('update')),score=score(last),best_epoch=best['epoch'],best_score=score(best),stopping=last.get('stopping'),complete=(path/'finished.json').exists(),
    latest_logged_update=last_update(path/('learning.jsonl' if kind=='dp' else 'loss.jsonl')),latest_logged_scope='May be an uncommitted in-flight epoch; durable epoch/selection are reported separately.')
 for p in (ROOT/'logs').glob('*/state.json'):
  s=read(p)
  try:cmd=Path(f"/proc/{s['pid']}/cmdline").read_bytes()
  except FileNotFoundError:continue
  if str(ROOT).encode() not in cmd and not any(x in cmd for x in [b'scripts.tight_budget_20261116.endpoints',b'scripts.tight_budget_20261116.evaluate']):continue
  out['active_workers'].append(dict(name=s['name'],pid=s['pid'],heartbeat_age_s=round(time.time()-s['heartbeat'],1),elapsed_s=round(time.time()-s['started'],1)))
 for role in 'AB':
  rear=ROOT/role/'backend';cache=rear/role/'nominal'
  if cache.exists():out['caches'][role]=dict(written_shards=sum(1 for p in cache.glob('*.pt') if not p.name.startswith('.')),complete=(rear/'cache_complete.json').exists())
 out['current_failure']={p.name:read(p) for p in [ROOT/'serial_recovery_failed.json',ROOT/'finish_failed.json'] if p.exists()}
 out['historical_parallel_failure_preserved']=(ROOT/'failure.json').exists()
 out['finish']=read(ROOT/'finish_state.json') if (ROOT/'finish_state.json').exists() else None
 out['completed']={k:(ROOT/(k+'.json')).exists() for k in ['dp_complete','training_complete','evaluation_complete','finish_complete']}
 print(json.dumps(out,indent=2))

if __name__=='__main__':main()
