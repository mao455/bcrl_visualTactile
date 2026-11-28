from pathlib import Path
import copy,json,time
from tools.tight_machine_review import session_at,sources
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.narrow_data import direct_configuration
from scripts.tight_layout_20261116.pilot import reset_configuration
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';D=P/'outputs/diagnostics/tight_full_curve_20261127_01';C=P/'configs/experiments/full_curve_20261127';OLD=P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01';PREV=P/'outputs/visuotactile/run/tight_correction_20261127_01';read=lambda p:json.loads(p.read_text())
s=session_at(R,duration=96*3600);s.update(version='corrected_full_demo_learning_curve_v1',cache_deadline=s['deadline'],authorization='One fixed-architecture full collection/training learning curve; same-start old-pool control versus added actual100mm full experts; 0/30/60complete matched checkpoints. No pose auxiliary or sealed terminal64.')
s['resources']['min_ram_gib']=6;atomic_json(R/'session.json',s)
pairs={r:read(PREV/'registration.json')['pairs'][r+'_new'] for r in 'AB'}
for r,pair in pairs.items():
 for k in ['frontend','backend']:assert sha256(pair[k])==pair[k+'_sha256']
reg=dict(version=s['version'],utc=time.time(),deadline=s['deadline'],starting_pairs=pairs,starting_pool={r:str(PREV/'short_with_corrections'/r/'state_data/data.json') for r in 'AB'},conditions=['old_only','plus_corrected24'],milestones=[0,30,60],checkpoint_selection='Fixed milestones; no validation/development winner selection',architecture='unchanged full DP->Transformer, frozen DINO and affine, native120/60/5Hz, commit12',midpoint_rear='Fresh own original backend continuation for30epochs bound to DP30; final separately from same original backend for60epochs bound to DP60; extra midpoint cost reported',sampling='One real history visit per base epoch, explicit old-history repeats align actual DP and rear update counts across A/B and both conditions; no artificial padding counted',training_families=list(range(28910000,28910008)),development_families=list(range(28911000,28911008)),corrected_demo_families=list(range(28910000,28910024)),sealed_historical64=list(range(28912000,28912064)),noise_seed=7091,software_failure='Retain first attempt and denominator; retry technical failures only with explicit new attempt identity. Physical failures not replaced.',expert_retries_per_family=3,pose_auxiliary_enabled=False,wrong_layout_data='Retained in both pools, explicitly annotated historical unshifted',new_own_corrections='Previous correction A3/B1 included in both arms; no new correction collection this round',publication_target='Only mao455/bcrl_visualTactile; second repository remains frozen, zero writes')
atomic_json(R/'registration.json',reg)
for shard in range(2):
 owner=R/f'experts_{shard:02d}';osess=copy.deepcopy(s);osess['resource_root']=str(owner);owner.mkdir();atomic_json(owner/'session.json',osess)
 cases=[]
 for seed in range(28910000+shard*12,28910012+shard*12):
  family=f'paired_train_{seed}';reset=reset_configuration(s['domain'],seed,.1);base=direct_configuration(s['domain'],seed)
  admission=dict(object_xy_m=reset['fixed_start_pose'][:2],hole_xy_m=[reset['hole_x_range'][0],reset['hole_y_range'][0]],unshifted_object_xy_m=base['fixed_start_pose'][:2],displacement_xy_m=[reset['fixed_start_pose'][i]-base['fixed_start_pose'][i] for i in range(2)],tolerance_m=1e-5,baseline_source='Historical family direct_configuration; actual-minus-unshifted signed components')
  cases.append(dict(seed=seed,family=family,narrow_reset=reset,reset_admission=admission))
 atomic_json(owner/'plan.json',dict(cases=cases,source_sha256=sources(),attempts=3))
 job=dict(version='tight_project_job_v1',operation='prepare',resource_root=str(owner),stages=[dict(entry='curve_experts',args=['experts','--root',str(owner)])]);atomic_json(C/f'experts_{shard:02d}.json',job)
print(R)
