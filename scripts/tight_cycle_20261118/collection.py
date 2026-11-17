"""Register one whole recovery batch after capability evidence, then legal aggregation."""
import copy,shutil,time
from pathlib import Path
from scripts.tight_cycle_20261118.cycle import ROOT,DATA,PARENT,OUT,TREE,read,write,hashes

def prepare():
 from tools.tight_fresh_batch_run import prepare as batch_prepare
 from tools.tight_fresh_stage_takeover import PHASE_CONTRACT,validate_plan
 from visuotactile.grasp_curriculum import APPROACH,CONTACT_ONSET
 from visuotactile.phase_takeover import profile
 decision=read(ROOT/'decision.json');assert decision['action']=='registered_recovery32'
 assert read(ROOT/'capability/owner_exit.json')['children_reaped']
 d=ROOT/'collection';ids=decision['family_stage_assignments'];assert len(ids)==16
 session=read(ROOT/'session.json');session.update(resource_root=str(d),owned_worker_roots=[str(d)])
 # Preparation requires a separate pre-existing resource root; exact session copied below.
 staging=ROOT/'collection_registration';staging.mkdir();write(staging/'session.json',session)
 p=batch_prepare(PARENT,d,lanes=32,identities=[r['identity'] for r in ids],delays=[0]*16,round_index=7,seed=28831000,session_root=staging,
  execution_backend=read(ROOT/'capability/plan.json')['execution_backend'],queued_gpu=True,vision_microbatch=4,writer_bytes=512<<20,verification_workers=2,verification_max_pending=4,capture_audit_memo=True)
 p.update(resource_root=str(d),stage_takeover_contract=PHASE_CONTRACT,source_sha256=hashes())
 for c in p['cases']:
  a=next(a for a in ids if a['identity']==c['spec']['source_identity']);stage=a['stage']
  c['dagger']['trigger']=copy.deepcopy(APPROACH if stage=='approach' else CONTACT_ONSET if stage=='contact' else profile(stage))
  c['name']=c['name'].replace('_contact_height_only_d0','_cycle_'+stage);c['spec'].update(identity=c['name'],layout=session['layout']);c['video']=False
  c['collection_type']='existing_family_actual_student_recovery';c['target_stage']=stage
 p['cycle_registration']=dict(decision=str(ROOT/'decision.json'),opportunities=32,independent_families=16,source='72 selected full A/B; aggregate into retained76 after closure',capacity='First actual32 collection load; same standard cameras, full student/expert and bounded writer; no artificial performance pilot',fallback='Existing causal approach/contact recovery screens; later phase never-fallback, untriggered kept in denominator')
 validate_plan(p);write(d/'plan.json',p);write(OUT/'collection_manifest.json',p);print(d)

def learning():
 from tools.tight_fresh_dagger import prepare_aggregate
 from visuotactile.trajectory_training import profile,budget
 from visuotactile.bounded_extension import fresh_layout_profile
 collection=ROOT/'collection';target=ROOT/'learning'
 assert read(collection/'owner_exit.json')['children_reaped'];assert read(collection/'paired_initial_check.json')['passed']
 entries=read(collection/'correction_entries.json');assert entries
 # Retain all76, while plan explicitly supplies selected72 frontend/rear parents.
 prepare_aggregate(DATA,collection,target);(target/'logs').mkdir()
 session=copy.deepcopy(read(ROOT/'session.json'));session.update(resource_root=str(target),owned_worker_roots=[str(target)],phase_dagger=False,fixed56_long=False,matched_update_diagnostic=False)
 write(target/'session.json',session);(target/'shared_resources.json').unlink()
 reg=dict(profile=profile(),roles={});parentdata=read(PARENT/'A/state_data/data.json')
 for role in 'AB':
  child=target/role;(child/'shared_resources.json').unlink();(child/'shared_resources.json').symlink_to(target/'shared_resources.json');write(child/'session.json',session)
  data=read(child/'state_data/data.json');ids=data['splits']['train'];
  cont=read(child/'continuation.json');pair=read(ROOT/'registration.json')['parent_pairs'][role];assert cont['path']==pair['frontend'] and cont['previous_backend']==pair['backend'];
  assert set(read(DATA/role/'state_data/data.json')['splits']['train'])<=set(ids)
  reg['roles'][role]=dict(**budget(len(ids),80),data_sha256=__import__('visuotactile.compact',fromlist=['sha256']).sha256(child/'state_data/data.json'),episode_multiplicities=dict.fromkeys(ids,1))
 for name in ['training_sampling.json','cuda_allocator_policy.json','frozen_visual_io.json']:
  shutil.copyfile(DATA/name,target/name)
 write(target/'frozen_visual_hotset.json',dict(version='formal_immutable_hotset_v1',max_bytes=4*1024**3,minimum_free_bytes=8*1024**3))
 write(target/'full_trajectory_batch.json',reg);write(target/'bounded_extension.json',dict(policy=fresh_layout_profile(),roles={}))
 p=read(target/'prepared.json');p.update(continue_student_pair=True,source_sha256=hashes(),defer_development=True,
  newly_collected=len(entries),new_to_parent=sorted(set(ids)-set(parentdata['splits']['train'])),parent='selected72 full A/B per collection plan',budget_reason='40–80 full traversals frontend,20–40 rear; existing validation plateau/LR rule, full visits per trajectory. More exposure than prior13-traversal A control; compare curves and autonomous outcomes, no tiny-update conclusion.')
 write(target/'prepared.json',p);write(OUT/'learning_registration.json',dict(prepared=p,training=reg,session=session));print(target)
if __name__=='__main__':globals()[__import__('sys').argv[1]]()
