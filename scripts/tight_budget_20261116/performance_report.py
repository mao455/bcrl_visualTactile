"""Measured whole-stage wall times; preserve failed compute in the accounting."""
import json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.tight_budget_20261116.run import ROOT,OLD,read
O=Path('research_docs/evidence/tight_diagnosis_20261116/budget_extension')
O.mkdir(parents=True,exist_ok=True)
assert read(ROOT/'training_complete.json')['complete']
assert read(ROOT/'evaluation_complete.json')['complete']

def interval(names):
 states=[read(ROOT/'logs'/n/'state.json') for n in names]
 assert all(x.get('exit_code')==0 for x in states)
 return dict(seconds=max(x['ended'] for x in states)-min(x['started'] for x in states),
  started=min(x['started'] for x in states),ended=max(x['ended'] for x in states),
  workers=[dict(name=x['name'],pid=x['pid'],seconds=x['ended']-x['started'],attempt=x['attempt']) for x in states])

first_dp=min(read(ROOT/'logs'/('dp_'+r)/'attempt1.json')['started'] for r in 'AB')
dp_end=read(ROOT/'dp_complete.json')['utc']
stages=dict(dp=dict(seconds=dp_end-first_dp,started=first_dp,ended=dp_end,
 scope='Original parallel attempts, failed partial12, explicit serial recovery, validation and waits all included; baseline epochs1-8 predate this run.'))
for stage in ['cache','train','offline']:stages['rear_'+stage]=interval(['rear_'+stage+'_'+r for r in 'AB'])
stages['endpoints']=interval(['finish_endpoints'])
groups=read(ROOT/'evaluation_complete.json')['groups']
stages['autonomy']=interval(['evaluation_'+g for g in groups])
epochs=[]
for r in 'AB':
 for line in (ROOT/r/'training/validation.jsonl').read_text().splitlines():
  x=json.loads(line);io=x.get('training_epoch_io',{})
  epochs.append(dict(role=r,epoch=x['epoch'],optimizer_updates=x['updates'],new_experiment=x['epoch']>8,
   seconds=io.get('wall_seconds'),started=io.get('started_utc'),ended=io.get('ended_utc'),
   main_process_io=io.get('process_io'),scope='Training epoch only, excludes validation; CPU pre-reader I/O is separate, not counted as main-process physical bytes.'))
old_stages=read(OLD/'cycle_readout.json')['stages']
old_dp=[old_stages['frontend_'+r] for r in 'AB']
summary=dict(stages=stages,dp_epoch_times=epochs,
 whole_run_seconds=read(ROOT/'evaluation_complete.json')['utc']-read(ROOT/'prepared.json')['utc'],
 original_round19_dp_seconds=max(x['ended'] for x in old_dp)-min(x['started'] for x in old_dp),
 original_round19_stages=old_stages,
 discarded_optimizer_updates=dict(A=140,B=140),discarded_scope='Actual partial epoch12 work, excluded from restored-model effective exposure; retained logs and checkpoints.',
 new_collected_trajectories=0,new_disk_visual_feature_copies=0,
 failure_receipt=read(ROOT/'failure.json'),recovery_receipt=read(ROOT/'serial_recovery_started.json'),
 io_observations={p:read(ROOT/p) for p in ['io_observation_1136.json','sequential_warm_probe.json'] if (ROOT/p).exists()},
 resource_limit_violation='Initial paired DP peak OOM; free GPU memory128.69MiB, below required4GiB. Do not claim whole-session compliance.',
 timing_limitations='Different epoch budgets; total stage speedup is not inferred from parallel worker counts, a few fast batches, or timing excluding the failed attempt.')
(O/'performance.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:round(v['seconds'],3) for k,v in stages.items()}))
