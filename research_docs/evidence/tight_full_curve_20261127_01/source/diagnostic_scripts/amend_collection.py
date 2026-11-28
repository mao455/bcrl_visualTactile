from pathlib import Path
import copy,json,time
from tools.tight_machine_review import sources
from visuotactile.training_resources import atomic_json
from visuotactile.compact import sha256
from visuotactile.narrow_data import direct_configuration
from scripts.tight_layout_20261116.pilot import reset_configuration
P=Path.cwd();R=P/'outputs/visuotactile/run/tight_full_curve_20261127_01';C=P/'configs/experiments/full_curve_20261127';read=lambda p:json.loads(p.read_text())
old=read(R/'registration.json');atomic_json(R/'registration_before_legality_amendment.json',old)
s=read(R/'session.json');owner=R/'experts_02';owner.mkdir();atomic_json(owner/'session.json',dict(s,resource_root=str(owner)))
cases=[]
for seed in range(28910024,28910031):
 reset=reset_configuration(s['domain'],seed,.1);base=direct_configuration(s['domain'],seed)
 cases.append(dict(seed=seed,family=f'paired_train_{seed}',narrow_reset=reset,reset_admission=dict(object_xy_m=reset['fixed_start_pose'][:2],hole_xy_m=[reset['hole_x_range'][0],reset['hole_y_range'][0]],unshifted_object_xy_m=base['fixed_start_pose'][:2],displacement_xy_m=[reset['fixed_start_pose'][i]-base['fixed_start_pose'][i] for i in range(2)],tolerance_m=1e-5,baseline_source='Historical direct family reset')))
atomic_json(owner/'plan.json',dict(cases=cases,source_sha256=sources(),attempts=3,complete_previous_shards=[str(R/f'experts_{n:02d}') for n in range(2)],target_total_accepted=24))
atomic_json(C/'experts_02.json',dict(version='tight_project_job_v1',operation='prepare',resource_root=str(owner),stages=[dict(entry='curve_experts',args=['experts','--root',str(owner)])]))
history=read(P/'outputs/visuotactile/run/tight_paired_dagger_20261122_01/learning/A/state_data/data.json');historical=[history['entries'][i]['seed'] for i in history['paired_pool']['shared_full_ids']]
old.update(corrected_demo_families=historical,initial_dispatch_families=list(range(28910000,28910024)),fallback_families=list(range(28910025,28910031)),target_accepted_full=24,training_families=[28910000,28910002,28910003,28910004,28910006,28910013,28910014,28910015],legality_amendment=dict(utc=time.time(),reason='Initial dispatch missed historical replacement24 and included rejected-old12;1/5 collision-rejected at corrected layout. Keep all attempts. Collect historical24 then sequential bounded fallback25..30 until24full. Train curve picks8 already initial-collision-legal families before student outcomes.',all_original24_attempts_preserved=True,old12='Captured extra legal corrected demonstration retained, not relabeled as one of original24'))
atomic_json(R/'registration.json',old)
# Preserve the unexecuted pre-amendment evaluation manifest, then update train cases.
o=R/'evaluation_e00_start/rollouts';p=read(o/'plan.json');atomic_json(o/'plan_before_legality_amendment.json',p)
from outputs.diagnostics.tight_full_curve_20261127_01.prepare_evaluation import prepare
# Rebuild only the unexecuted owner with its archived plan kept outside.
assert not (o.parent/'project_job_started.json').exists()
import shutil
archive=R/'registration_unexecuted_e00_start';o.parent.rename(archive)
print(prepare(0))
