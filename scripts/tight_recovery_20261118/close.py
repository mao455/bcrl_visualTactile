"""Compact final receipts, resource/identity closure; no simulation or training."""
import csv,json,shutil,time,os,gzip
from pathlib import Path
from scripts.tight_recovery_20261118.run import ROOT,OUT,DATA,PARENT,TREE,read,write,COLLECTION
from visuotactile.compact import sha256

def main():
 assert read(ROOT/'tail_recovery02_complete.json')['complete']
 assert read(ROOT/'tail_recovery02_exit.json')['children_reaped']
 source_checks={p:sha256(TREE/p)==h for p,h in read(ROOT/'learning/prepared.json')['source_sha256'].items()}
 assert all(source_checks.values()),'Training source identity drift'
 for part in range(4):assert read(ROOT/f'comparison{part:02d}/paired_initial_check.json')['passed']
 data_a=read(ROOT/'learning/A/state_data/data.json');data_b=read(ROOT/'learning/B/state_data/data.json')
 assert data_a['splits']==data_b['splits'] and len(data_a['splits']['train'])==130
 stages=[]
 for p in sorted(ROOT.glob('*_complete.json')):
  v=read(p);stages.append(dict(stage=p.stem,seconds=v.get('seconds'),utc=v.get('utc'),receipt=str(p)))
 for p in sorted((ROOT/'learning/logs').glob('*_complete.json')):
  v=read(p);stages.append(dict(stage='learning/'+p.stem,seconds=v.get('seconds'),utc=v.get('utc'),receipt=str(p)))
 for name in ['pilot','collection','collection_recovery01','collection16_0','collection16_1']+[p.name for p in sorted(ROOT.glob('comparison*')) if p.is_dir()]:
  p=ROOT/name/'owner_exit.json'
  if p.exists():
   v=read(p);stages.append(dict(stage='simulation/'+name,seconds=v.get('seconds'),utc=v.get('utc'),receipt=str(p)))
 for name in ['pilot_admission_recovery02']:
  p=ROOT/name/'owner_exit.json'
  if p.exists():
   v=read(p);stages.append(dict(stage='cpu_admission/'+name,seconds=v.get('seconds'),utc=v.get('utc'),receipt=str(p)))
 write(OUT/'timing_scope.json',dict(note='Parent learning/offline/evaluation owner times contain child stage timings. Do not sum nested rows; absolute registered-window duration is authoritative. Failed32 and pre-action rejected recovery included separately. Kit initialization and execution profiles are separate receipts.'))
 with (OUT/'stage_times.csv').open('w') as f:w=csv.DictWriter(f,['stage','seconds','utc','receipt'],lineterminator='\n');w.writeheader();w.writerows(stages)
 resources=[]
 names=['pilot','collection','collection_recovery01','collection16_0','collection16_1','learning','learning/phase_review']+[p.name for p in sorted(ROOT.glob('comparison*')) if p.is_dir()]
 for name in names:
  d=ROOT/name;dest=OUT/'closure_receipts'/name;dest.mkdir(parents=True,exist_ok=True)
  for filename in ['owner_exit.json','owner_failure.json','owner_complete.json','initialization.json','monitor_exit.json','result.json','wave_profiles.json','session.json']:
   if (d/filename).exists():shutil.copyfile(d/filename,dest/filename)
  p=d/'resources/resources.jsonl'
  if p.exists():
   rr=[json.loads(l) for l in p.read_text().splitlines()];write(dest/'resource_rows.json',rr)
   resources.append(dict(stage=name,samples=len(rr),first=rr[0] if rr else None,last=rr[-1] if rr else None))
 protected=read(OUT/'protected.json')['files'];checked={p:sha256(p)==h for p,h in protected.items()};assert all(checked.values())
 raw={}
 for role in 'AB':
  data=read(ROOT/'learning'/role/'state_data/data.json')
  for e in data['entries'].values():raw[e['path']]=e['source_sha256']
 assert all(Path(p).is_file() for p in raw)
 running=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit() or int(p.name)==os.getpid():continue
  try:
   cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
   argv=(p/'cmdline').read_bytes().split(b'\0')
   if argv and b'python' in Path(argv[0].decode()).name.encode() and (str(ROOT) in cmd or 'scripts.tight_recovery_20261118.tail' in cmd or 'scripts.tight_recovery_20261118.eval_recovery' in cmd):running.append(dict(pid=int(p.name),matches_registered_root=True))
  except OSError:pass
 # This review's launcher may contain ROOT in its command, so report actual inventory
 # and require direct known GPU worker PIDs are no longer alive independently.
 for name in names:
  p=ROOT/name/'worker_started.json'
  if p.exists():assert not Path('/proc/'+str(read(p)['pid'])).exists(),name
 assert not running,running
 write(OUT/'identity_closure.json',dict(source_hashes=source_checks,common_train_trajectories=130,actual_initial_checks_passed=4,owned_python_workers_remaining=running,scope='Training/collection source and observable initial fields, not a hidden simulator-state replay guarantee.'))
 write(OUT/'closure.json',dict(utc=time.time(),protected_hashes=checked,raw_dependencies_existing=len(raw),resources=resources,known_simulator_workers_exited=True,remaining_root_command_matches=running,source_head=read(ROOT/'session.json')['source_commit'],elapsed_from_registered_start_s=read(ROOT/'tail_recovery02_complete.json')['utc']-read(ROOT/'session.json')['started_utc'],free_disk_bytes=shutil.disk_usage(ROOT).free,scope='Includes failed attempts and CPU admission recovery; full_gpu policy, no50percent quota claim. Raw/model artifacts stay local; compact reports/configs/CSV/curves/videos published.'))
 for p in OUT.rglob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
 print('closed',len(raw),'raw dependencies and',len(protected),'protected identities')
if __name__=='__main__':main()
