"""Finite CPU-only completion readout for the one registered recovery cycle.

Never starts simulation, changes a candidate, retries training or releases review.
Produces the existing cycle_readout.json only when the actual cycle completed.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from visuotactile.training_resources import atomic_json
read=lambda p:json.loads(p.read_text())

def main(root,index):
 ledger=root/'night/completion_report';ledger.mkdir(exist_ok=False)
 deadline=read(root/'session.json')['deadline'];candidate=root/f'round_{index:02d}'
 output=candidate/'cycle_readout.json';started=time.time()
 def state(stage,**extra):atomic_json(ledger/'state.json',dict(stage=stage,utc=time.time(),pid=os.getpid(),started=started,deadline=deadline,**extra))
 try:
  while time.time()<deadline:
   owner=read(root/'night/state.json')
   state('waiting_for_real_cycle',owner_stage=owner.get('stage'),owner_error=owner.get('error'))
   if any(r['index']==index and r['complete'] for r in owner.get('rounds',[])):
    assert read(candidate.with_name(candidate.name+'_stages')/'finished.json')['software_complete']
    if not output.exists():
     state('building_completed_readout')
     with (ledger/'readout.log').open('x') as stream:
      subprocess.run([sys.executable,'-m','tools.tight_fresh_cycle_readout','--candidate',str(candidate),
       '--collection',str(root/f'new_family_{index:02d}'),'--output',str(output)],
       stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=max(1,min(120,deadline-time.time())))
    proof=read(output);assert proof['coordinator_complete']
    state('complete',report=str(output),full_task_legal=proof['full_task_legal'],full_task_insertions=proof['full_task_insertions'])
    return
   if owner.get('safe_stop'):
    state('owner_safe_stop_before_cycle_complete',owner_state=owner,
      human_entry='Inspect night/state.json and its stage_ledger; preserve every completed stage. No automatic bad-task restart.')
    return
   time.sleep(min(30,max(.1,deadline-time.time())))
  state('report_deadline',report_exists=output.exists())
 except BaseException as error:
  state('report_failure',error=repr(error));raise

if __name__=='__main__':
 p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--index',type=int,required=True);a=p.parse_args();main(a.root.resolve(),a.index)
