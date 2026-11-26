from pathlib import Path
import copy,json,time,subprocess
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from scripts.tight_layout_20261116.pilot import reset_configuration,displacement
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_correction_20261127_01';C=P/'configs/experiments/correction_20261127';OLD=P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
read=lambda p:json.loads(p.read_text())
owner=R/'development_before';s=session_at(owner,duration=12*3600)
s.update(version='tight_correction_20261127_v1',authorization='Correct actual100mm reset,32 original-development autonomous observe-only opportunities; terminal64 excluded; targeted correction and independent existingB72 optimization pilot authorized',cache_deadline=s['deadline'])
atomic_json(owner/'session.json',s)
freeze=read(OLD/'evaluation/development/registration.json');pairs=freeze['pairs'];stages=[];plans={}
for i in range(8):
 p=read(OLD/f'evaluation/development/wave{i:02d}/plan.json');out=owner/f'wave{i:02d}';out.mkdir()
 p.update(resource_root=str(owner),pairs=pairs,source_sha256=sources(),source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),observe_only=True,video_full_decode=False,first_plan_forensics=False)
 p.pop('evaluation_contract',None)
 p['evaluation_scope']=dict(protocol='Corrected registered100mm layout; four frozen full students; observe_only never transfers authority',software_failure='Preserve first attempt; technical-only same-family retry allowed; no physical outcome replacement',terminal64_reused=False)
 p['batch'].update(verification_workers=0)
 for case in p['cases']:
  spec=case['spec'];seed=spec['seed'];old=copy.deepcopy(spec['narrow_reset']);corrected=reset_configuration(s['domain'],seed,.1)
  assert old['hole_x_range']==corrected['hole_x_range'] and old['hole_y_range']==corrected['hole_y_range']
  spec.update(narrow_reset=corrected,reset_admission=dict(object_xy_m=corrected['fixed_start_pose'][:2],hole_xy_m=[old['hole_x_range'][0],old['hole_y_range'][0]],unshifted_object_xy_m=old['fixed_start_pose'][:2],displacement_xy_m=displacement(s['domain'],.1).tolist(),tolerance_m=1e-5,baseline_source=str(OLD/f'evaluation/development/wave{i:02d}/plan.json')))
  case['video']=False
 atomic_json(out/'session.json',s);atomic_json(out/'plan.json',p)
 plans[str(out/'plan.json')]=sha256(out/'plan.json')
 stages.append(dict(entry='machine_review',args=['worker','--output',str(out),'--mode','autonomous']))
atomic_json(C/'development_before.json',dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(owner),stages=stages))
atomic_json(R/'registration.json',dict(utc=time.time(),families=list(range(28911000,28911008)),opportunities=32,pairs=pairs,plan_sha256=plans,terminal64_access=False,noise_seed=7091,reset_subtraction='corrected actual XY minus original unshifted family XY',second_repository='frozen: no files/refs/tags/PR/settings writes',publication_remote='https://github.com/mao455/bcrl_visualTactile.git',publication_branch='feature/tight-state-foundation',initial_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
print(C/'development_before.json')
