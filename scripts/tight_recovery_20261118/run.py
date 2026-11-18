"""Bounded state-regression recovery cycle reusing existing batch/training owners."""
import copy,sys,time,subprocess
from pathlib import Path
from scripts.tight_cycle_20261118 import cycle as old
from scripts.tight_cycle_20261118.cycle import read,write,hashes,TREE,PARENT
COLLECTION='collection_completed'
ROOT=old.RUNS/'tight_recovery_20261118_01';DATA=old.ROOT/'learning';OUT=TREE/'research_docs/evidence/tight_recovery_20261118'
def prepare():
 ROOT.mkdir(exist_ok=False);s=read(DATA/'session.json');now=time.time()
 s.update(started_utc=now,deadline=now+6*3600,training_deadline=now+5*3600,cache_deadline=now+5*3600,offline_deadline=now+6*3600,collection_deadline=now+2*3600,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),authorization='Latest attachment: diagnose29+3, pilot8 then32 opportunities, protected105 plus new legal suffixes, own72 complete A/B parents, bounded learning and paired full evaluation')
 write(ROOT/'session.json',s);write(OUT/'registration.json',dict(root=str(ROOT),session=s,parent=read(old.ROOT/'registration.json')['parent_pairs'],protected105=str(DATA),pilot_families=[28817000,28817008,28817012,28817024],pilot_opportunities=8,batch_opportunities=32,pilot_gate='At least4 actual triggers and3 legal rescues, at least2 near-stage recovered states; verified first legal labels2ticks after takeover; inspect ambiguity/coverage before main batch. Preserve all failures, no quota retries.',scope='No final tests, no network/loss/control changes; 6h fixed window'))
 prepare_wave('pilot',[0,8,12,24],8,8)
def prepare_wave(name,indices,lanes,round_index):
 from tools.tight_fresh_batch_run import prepare as batch_prepare
 from tools.tight_fresh_stage_takeover import PHASE_CONTRACT,validate_plan
 from visuotactile.deviation_takeover import profile
 d=ROOT/name;staging=ROOT/(name+'_registration');staging.mkdir();s=read(ROOT/'session.json');s.update(resource_root=str(d),owned_worker_roots=[str(d)]);write(staging/'session.json',s)
 p=batch_prepare(PARENT,d,lanes=lanes,identities=[f'layout10_train_{28817000+i}' for i in indices],delays=[0]*len(indices),round_index=round_index,seed=28841000+round_index,session_root=staging,execution_backend=read(old.ROOT/'collection/plan.json')['execution_backend'],queued_gpu=True,vision_microbatch=4,writer_bytes=512<<20,verification_workers=2,verification_max_pending=4,capture_audit_memo=True)
 p.update(resource_root=str(d),stage_takeover_contract=PHASE_CONTRACT,source_sha256=hashes())
 for c in p['cases']:
  c['dagger']['trigger']=profile();c['name']=c['name'].replace('_contact_height_only_d0','_regression');c['spec'].update(identity=c['name'],layout=s['layout']);c['video']=c['spec']['seed']==28817000;c['target_stage']='first_observable_pre_subgoal_regression'
 p['recovery_registration']=dict(pilot=name=='pilot',trigger='task_regression_takeover_v2',selection='causal regression, no tick deadline/fallback; excludes achieved subgoals and aligned support release',limitations='screens also occur in4/30 expert successes; not proof of irreversible failure',parent='protected72 full A/B')
 validate_plan(p);write(d/'plan.json',p);write(OUT/(name+'_manifest.json'),p)
def wave(name):old.run_wave(ROOT/name)
def summarize(name):
 from scripts.tight_layout_20261116.phase_report import summarize as summary
 summary(ROOT/name,OUT/(name+'_summary.json'))
def collections():
 for name in ['collection16_0','collection16_1']:old.run_wave(ROOT/name)
 dest=ROOT/COLLECTION;dest.mkdir(exist_ok=False);plan=copy.deepcopy(read(ROOT/'collection16_0/plan.json'));plan['cases']=[];results=[];entries={};profiles=[];checks=[];elapsed=0.;init=0.
 for name in ['collection16_0','collection16_1']:
  d=ROOT/name;p=read(d/'plan.json');plan['cases'].extend(p['cases']);results.extend(read(d/'results.json'));entries.update(read(d/'correction_entries.json'));profiles.extend(read(d/'wave_profiles.json'));checks.extend(read(d/'paired_initial_check.json')['rows']);elapsed+=read(d/'owner_complete.json')['seconds'];init+=read(d/'initialization.json')['seconds']
  for c in p['cases']:(dest/c['name']).symlink_to(d/c['name'],target_is_directory=True)
 plan.update(intake_only=True,physical_waves=['collection16_0','collection16_1']);write(dest/'plan.json',plan);write(dest/'results.json',results);write(dest/'correction_entries.json',entries);write(dest/'wave_profiles.json',profiles);write(dest/'paired_initial_check.json',dict(passed=all(read(ROOT/n/'paired_initial_check.json')['passed'] for n in ['collection16_0','collection16_1']),rows=checks,scope='Actual within-wave A/B initial pairs; separate16-lane worlds'))
 write(dest/'initialization.json',dict(seconds=init,lanes=16,waves=2));write(dest/'result.json',dict(software_complete=True,worker_error=None,cases=32,lanes=16,waves=2,elapsed_seconds=elapsed,accepted_training_trajectories=len(entries)));write(dest/'owner_complete.json',dict(seconds=elapsed,utc=time.time()));write(dest/'owner_exit.json',dict(children_reaped=True,utc=time.time(),joined_closed_waves=True));write(dest/'session.json',read(ROOT/'session.json'))

if __name__=='__main__':globals()[sys.argv[1]](*sys.argv[2:])
