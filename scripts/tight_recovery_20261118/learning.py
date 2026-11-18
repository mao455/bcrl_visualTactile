"""Preserve original30 exposure with finite expert replay, matched DP/rear pairs."""
import copy,shutil
from scripts.tight_recovery_20261118.run import ROOT,DATA,OUT,COLLECTION,read,write,hashes,PARENT

def prepare():
 from tools.tight_fresh_dagger import prepare_aggregate
 from visuotactile.trajectory_training import profile,budget
 from visuotactile.bounded_extension import profile as bounded
 from visuotactile.latest_takeover_sampling import sampling_plan
 from visuotactile.compact import sha256
 assert read(OUT/'pilot_decision.json')['passed']
 intake=ROOT/'intake';intake.mkdir();entries={};cases=[];results=[]
 p=copy.deepcopy(read(ROOT/COLLECTION/'plan.json'))
 for name in ['pilot_admission_recovery02',COLLECTION]:
  d=ROOT/name;assert read(d/'owner_exit.json')['children_reaped'] and read(d/'result.json')['software_complete']
  e=read(d/'correction_entries.json');assert not(set(e)&set(entries));entries.update(e)
  for c in read(d/'plan.json')['cases']:
   cases.append(c);(intake/c['name']).symlink_to(d/c['name'],target_is_directory=True)
  results.extend(read(d/'results.json'))
 p.update(cases=cases,intake_only=True,physical_waves=['pilot8','collection16_0','collection16_1'])
 write(intake/'plan.json',p);write(intake/'correction_entries.json',entries);write(intake/'results.json',results);write(intake/'result.json',dict(software_complete=True,intake_only=True,waves=3,opportunities=40))
 target=ROOT/'learning';prepare_aggregate(DATA,intake,target);(target/'logs').mkdir()
 s=copy.deepcopy(read(ROOT/'session.json'));s.update(resource_root=str(target),owned_worker_roots=[str(target)],phase_dagger=True,fixed56_long=False,matched_update_diagnostic=False)
 write(target/'session.json',s);(target/'shared_resources.json').unlink()
 # Three original-demonstration visits, one for each legal correction. No
 # pseudoexpert old-policy targets and no uniform latest-data amplification.
 spec=dict(version='original_demonstration_replay_v1',original_repeat=3)
 write(target/'backend_episode_sampling.json',spec);reg=dict(profile=profile(),roles={})
 for role in 'AB':
  child=target/role;(child/'shared_resources.json').unlink();(child/'shared_resources.json').symlink_to(target/'shared_resources.json');write(child/'session.json',s)
  data=read(child/'state_data/data.json');cont=read(child/'continuation.json');cont['sampling_contract']=spec;write(child/'continuation.json',cont)
  _,mult=sampling_plan(data,spec);reg['roles'][role]=dict(**budget(sum(mult.values()),24),data_sha256=sha256(child/'state_data/data.json'),episode_multiplicities=mult)
  assert set(read(DATA/role/'state_data/data.json')['splits']['train'])<=set(data['splits']['train'])
 for name in ['training_sampling.json','cuda_allocator_policy.json','frozen_visual_io.json']:shutil.copyfile(DATA/name,target/name)
 write(target/'frozen_visual_hotset.json',dict(version='formal_immutable_hotset_v1',max_bytes=4*1024**3,minimum_free_bytes=8*1024**3))
 write(target/'full_trajectory_batch.json',reg);write(target/'bounded_extension.json',dict(policy=bounded(),roles={}))
 p=read(target/'prepared.json');p.update(continue_student_pair=True,source_sha256=hashes(),defer_development=True,original_replay=spec,newly_collected=len(entries),parent='protected72 full A/B',budget_reason='DP16–24 complete replay decks, max about2400 updates; rear existing12–24 traversals with identical original replay. Validation plateau then complete composed/offline and autonomous review; no extension/grid.')
 write(target/'prepared.json',p);write(OUT/'learning_registration.json',dict(prepared=p,training=reg,session=s,selection='Validation-selected complete pair is a proposal only; old/new phase and composed outputs then actual path retention govern promotion',original_base_mass_expected=90/sum(mult.values()),new_to_parent=len(set(data['splits']['train'])-set(read(PARENT/'A/state_data/data.json')['splits']['train']))))
if __name__=='__main__':prepare()
