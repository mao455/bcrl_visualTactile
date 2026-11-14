"""Independent CPU post-hoc review. Never edits frozen code or rollout evidence."""
import ast
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np

# Helpers are imported from the explicitly supplied, fully hash-verified frozen tree.
# Run as a standalone file, so a development tools package cannot shadow that tree.
import argparse
import hashlib


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_helpers(tree):
    # Reject an imported development implementation instead of silently mixing it
    # with historical frozen helpers. Direct script invocation starts clean.
    for package in ('tools', 'visuotactile'):
        module = sys.modules.get(package)
        if module is not None and list(getattr(module, '__path__', [])) != [str(tree / package)]:
            raise ValueError('Run this tool as a standalone file with no foreign research packages imported')
    sys.path.insert(0, str(tree))
    global phase_case, contact_motion, causal_contact_height, role_report, completed_pair
    from tools.tight_fresh_phases import case as phase_case
    from tools.tight_fresh_visual_closeout import contact_motion
    from tools.tight_fresh_contact_height_coverage import causal_contact_height
    from tools.tight_fresh_learning_report import role_report
    from tools.tight_fresh_rollouts import completed_pair


def read(path):return json.loads(Path(path).read_text())
def require(value,message):
    if not value:raise ValueError(message)


def check_sources(plan,tree):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip()==plan['source_commit'],'Frozen commit differs')
    for name,digest in plan['source_sha256'].items():
        require(sha256(tree/name)==digest,'Frozen file changed: '+str(tree/name))
    return dict(tree=str(tree),commit=plan['source_commit'],verified_source_files=len(plan['source_sha256']))


def held_orientation(directory):
    with h5py.File(directory/'numeric.h5','r') as f:
        t=f['episode/transition'];p=t['diagnostics_after/object_pose_w'][:]
        goal=t['diagnostics_before/goal_pose_w'][:];forces=t['object_contact_force_w'][:]
        mask,height=causal_contact_height(p,forces)
        indices=np.flatnonzero(mask & (t['diagnostics_before/successes'][:]==0))
        cosine=np.abs((p[:,3:]*goal[:,3:]).sum(-1))/(np.linalg.norm(p[:,3:],axis=-1)*np.linalg.norm(goal[:,3:],axis=-1))
        angle=2*np.arccos(np.clip(cosine,0.,1.));keypoint=t['native_keypoints_max_dist'][:]
        tolerance=t['native_success_diagnostic/insertion_tolerance_m'][:]
        def snapshot(i):
            return dict(completed_tick=int(i+1),root_angle_rad=float(angle[i]),
                root_distance_m=float(np.linalg.norm(p[i,:3]-goal[i,:3])),
                native_keypoint_distance_m=float(keypoint[i]),native_tolerance_m=float(tolerance[i]))
        if not len(indices):return dict(eligible_contact_height_windows=0)
        first=int(indices[0]);best=int(indices[np.argmin(angle[indices])])
        return dict(eligible_contact_height_windows=len(indices),first=snapshot(first),
            smallest_root_angle=snapshot(best),angle_decrease_from_first_window_rad=float(angle[first]-angle[best]),
            interpretation='Both snapshots have full causal 12-tick contact-height windows and the same first goal; quaternion angles ignore object symmetry.')


def audit_group(label,root,tree):
    directory=root/'development';plan=read(directory/'plan.json');finished=read(directory/'result.json')
    source_receipt=check_sources(plan,tree);rows=read(directory/'results.json')
    require(finished['software_complete'] and not finished.get('worker_error') and finished['cases']==8,'Incomplete worker')
    require(read(directory/'run_started.json')['plan_sha256']==sha256(directory/'plan.json'),'Startup plan hash differs')
    require(plan['kind']=='autonomous' and plan['test_used'] is False and len(rows)==len(plan['cases'])==8,'Wrong cohort')
    pairs={role:completed_pair(root,role) for role in ('A','B')}
    require(plan['pairs']==pairs,'Completed pair differs from executed pair')
    for pair in pairs.values():
        for key in ('frontend','backend'):
            require(sha256(pair[key])==pair[key+'_sha256'],'Model changed')
    indexed={r['case']:r for r in rows};require(len(indexed)==8,'Duplicate case')
    cases=[]
    for planned in plan['cases']:
        name=planned['name'];role=planned['role'];path=directory/name;row=indexed[name]
        r=read(path/'result.json');started=read(path/'started.json');pair=pairs[role]
        require(row['result']==r and row['role']==role,'Index/case mismatch')
        require(started['spec']==planned['spec'] and started['checkpoint']==pair['backend'],'Started identity mismatch')
        require(all(r[k]==v for k,v in planned['spec'].items()),'Executed reset differs')
        require(r['software_complete'] and r['legal'] and not r.get('censored') and r.get('dagger') is None,'Not complete autonomous execution')
        require(r['checkpoint_sha256']==pair['backend_sha256'] and r['residual_version']==pair['residual_version'],'Model contract mismatch')
        require(r['residual_ticks']==r['steps'] and r['simulation_step_hz']==60 and r['physics_hz']==120 and r['plan_hz']==5,'Clock mismatch')
        require(r['action_control_mode']=='absolute_joint_position_target_rad' and not r['expert_shadow_computed'],'Control mismatch')
        for stage in ('dino_rgb','learned_visual_and_history','residual_forward_compose'):
            require(r['native_inference']['summary'][stage]['wall_s']['count']==r['steps'],'Dropped real-frame student stage')
        require(r['native_inference']['summary']['ddim_plan']['wall_s']['count']==(r['steps']+11)//12,'Planning clock mismatch')
        phase=phase_case(path);require(phase['authority']['expert_ticks']==0 and phase['steps']==r['steps'],'Expert action or numeric length mismatch')
        require(phase['native_lift']==r['native_lift'] and phase['native_subgoals']==r['native_subgoals'],'Native numeric/result mismatch')
        with h5py.File(path/'numeric.h5','r') as f:
            t=f['episode/transition'];targets=t['target_after_rad'][:];previous=f['episode/obs/actuator_target_rad'][:]
            require(np.array_equal(targets[:-1],previous[1:]),'Actual target/history continuity differs')
            require(np.isfinite(targets).all(),'Nonfinite targets')
        cases.append(dict(group=label,role=role,**phase,contact_motion=contact_motion(path,phase),
            held_orientation=held_orientation(path),wall_seconds=row['timing']['total_seconds']))
    totals={}
    for role in ('A','B'):
        chosen=[c for c in cases if c['role']==role]
        totals[role]=dict(opportunities=4,successes=sum(c['success'] for c in chosen),
            native_lift_cases=sum(c['native_lift'] for c in chosen),
            old_stable_window_cases=sum(c['settled_height_stable_grasp'] for c in chosen),
            contact_height12_cases=sum(c['contact_motion']['first_contact_height_window_end_tick'] is not None for c in chosen),
            native_alignment_cases=sum(c['native_alignment_first_tick'] is not None for c in chosen),
            native_near_goal_cases=sum(c['native_near_goal_first_tick'] is not None for c in chosen),
            first_subgoal_cases=sum(c['native_subgoals']>0 for c in chosen),
            termination_reasons=dict(Counter(k for c in chosen for k,v in c['reasons'].items() if v)),
            total_control_ticks=sum(c['steps'] for c in chosen),rollout_wall_seconds=sum(c['wall_seconds'] for c in chosen))
    return dict(root=str(root),plan_sha256=sha256(directory/'plan.json'),results_sha256=sha256(directory/'results.json'),
        completion_sha256=sha256(directory/'result.json'),source_verification=source_receipt,pairs=pairs,totals=totals,cases=cases,
        initialization=read(directory/'initialization.json'))


def compact_learning(root, roles):
    """Separate completed-run exposure from the validation-selected frontend."""
    from tools.tight_fresh_learning_report import exposure_rows, learning_sources
    result = {}
    for role, report in roles.items():
        directory = root / role
        data = read(directory / 'state_data/data.json')
        corrections, current, sources = learning_sources(data)
        receipts = [json.loads(line) for line in (directory / 'training/learning.jsonl').read_text().splitlines()]
        selected = receipts[:report['selected_updates']]
        counts, exposure = exposure_rows(selected, set(data['splits']['train']), corrections, source_groups=sources)
        epoch = next(row for row in report['history'] if row['epoch'] == report['selected_epoch'])
        require(dict(counts) == epoch['sampling_exposure'], 'Selected exposure differs from epoch receipt')
        compact = lambda value: {key: item for key, item in value.items() if key != 'episodes'}
        result[role] = dict(
            frontend_sha256=report['checkpoint_sha256'],
            backend_sha256=report['backend_selected']['checkpoint_sha256'],
            source_populations={key: len(ids) for key, ids in report['source_populations'].items()},
            completed_frontend_epochs=report['latest_epoch'], completed_frontend_updates=report['latest_updates'],
            completed_frontend_exposure=report['actual_exposure_by_correction_age'],
            selected_frontend_epoch=report['selected_epoch'], selected_frontend_updates=report['selected_updates'],
            selected_frontend_exposure=exposure,
            selected_current_episode_visits={identity: counts[identity] for identity in sorted(current)},
            validation_initial=compact(report['initial_metrics']['validation']),
            validation_selected=compact(report['selected_validation']),
            validation_relative_change=report['selected_validation_relative_change'],
            correction_initial=compact(report['initial_metrics']['correction']),
            correction_last_epoch=compact(report['last_epoch_correction_metrics']),
            correction_last_epoch_is_selected=report['correction_metrics_match_selected'],
            selected_backend_epoch=report['backend_selected_metrics']['epoch'],
            selected_backend_validation=report['backend_selected_metrics']['validation'],
            completed_backend_exposure=report['backend_actual_exposure_by_correction_age'],
            backend_cache=report['backend_cache'],
            test_used=False,
            limits='Exposure reports actual supervised slots/draws, not independent states. '
                'B last-epoch correction fitting is not the deployed epoch3 checkpoint result.')
    return result


def run(source_root, candidate_root, source_tree, candidate_tree, output):
    OUT = output.resolve()
    FROZEN, OLD_FROZEN = candidate_tree.resolve(), source_tree.resolve()
    ROOTS = dict(source_round05=source_root.resolve(), candidate_round06=candidate_root.resolve())
    TREES = dict(source_round05=OLD_FROZEN, candidate_round06=FROZEN)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Preserve previous review evidence")
    for label, root in ROOTS.items():
        check_sources(read(root / "development/plan.json"), TREES[label])
    load_helpers(FROZEN)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'autonomy.json').exists():raise FileExistsError('Preserve prior CPU analysis')
    # Old source files differ by additions, but original stability arithmetic is exact.
    stable=[]
    for tree in (OLD_FROZEN,FROZEN):
        source=(tree/'visuotactile/takeover_trigger.py').read_text()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='stable_windows')
        stable.append(ast.get_source_segment(source,node))
    require(stable[0]==stable[1],'Old stable arithmetic differs')
    groups={label:audit_group(label,root,TREES[label]) for label,root in ROOTS.items()}
    old,new=(read(root/'development/plan.json') for root in ROOTS.values())
    require([(c['name'],c['role'],c['spec']) for c in old['cases']]==[(c['name'],c['role'],c['spec']) for c in new['cases']],'Development opportunities changed')
    helpers=['tools/tight_fresh_phases.py','tools/tight_fresh_visual_closeout.py',
        'tools/tight_fresh_contact_height_coverage.py','tools/tight_fresh_learning_report.py','visuotactile/takeover_trigger.py']
    atomic_json(OUT/'autonomy.json',dict(version='round06_independent_cpu_review_v1',complete=True,
        source_sha256=sha256(__file__),helper_root=str(FROZEN),helper_sha256={n:sha256(FROZEN/n) for n in helpers},
        old_stable_function_source_identical=True,groups=groups,simulator_starts=0,optimizer_updates=0,test_used=False,
        limits='Four repeatedly reused development starts per role, not new-generalization samples or exact hidden-state pairs. Contact-height permits palm rotation and is not native lift/success. Missing windows do not prove dropping. Angular and position minima are shown at their actual same-tick snapshots.'))
    print(json.dumps({label:g['totals'] for label,g in groups.items()},indent=2),flush=True)
    for label,root in ROOTS.items():
        if label!='candidate_round06':continue
        roles={r:role_report(root,r) for r in ('A','B')}
        atomic_json(OUT/'learning_round06.json',dict(roles=roles,
            helper_root=str(FROZEN),helper_sha256=sha256(FROZEN/'tools/tight_fresh_learning_report.py'),test_used=False))
        atomic_json(OUT/'learning_summary.json',dict(roles=compact_learning(root,roles),
            full_report_sha256=sha256(OUT/'learning_round06.json'),source_sha256=sha256(__file__),test_used=False))
    print('Finished CPU autonomy and learning audits',flush=True)


def parser():
    value = argparse.ArgumentParser(__doc__)
    value.add_argument('--source-root', type=Path, required=True)
    value.add_argument('--candidate-root', type=Path, required=True)
    value.add_argument('--source-tree', type=Path, required=True)
    value.add_argument('--candidate-tree', type=Path, required=True)
    value.add_argument('--output', type=Path, required=True)
    return value


if __name__ == '__main__':
    arguments = parser().parse_args()
    run(**vars(arguments))
