"""One-shot completed round09 behavior review; no weights, old numeric, or payload hashes.

Run only after round09_stages/finished.json and development/result.json complete.
Source behavior is reused from the completed per-role source preflight.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
SESSION=HERE.parent.parent
DEVELOPMENT=SESSION/'round_09/development'
MANIFEST=SESSION/'cpu_audits/round09_source_manifest.json'
PREFLIGHT=SESSION/'cpu_audits/round09_source_preflight_01/source_preflight.json'
COMPARISON=Path('/home/CNS2025331827/docs/bc+rl_RSS_continuous_vision_20261114/tools/tight_fresh_round_compare.py')
BINDINGS={}


def require(value,message):
    if not value:raise ValueError(message)


def small_sha(path):
    path=Path(path)
    require(path.suffix in ('.json','.py') and path.stat().st_size<32*1024*1024,
        'Only small JSON/source hashes are allowed: '+str(path))
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
    BINDINGS[str(path)]=dict(sha256=digest,bytes=len(raw))
    return digest


def read(path):
    path=Path(path);small_sha(path)
    raw=path.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==BINDINGS[str(path)]['sha256'],'Small input changed while read')
    return json.loads(raw)


def numeric_stat(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)


def helper_input_token(path):
    """Local adapter: retain stability checks without claiming a numeric SHA."""
    path=Path(path)
    if path.suffix=='.h5':
        require(path.name=='numeric.h5' and path.parent.parent==DEVELOPMENT,
            'Never open/hash source trajectories or capture payloads')
        return 'stat_only:'+json.dumps(numeric_stat(path),sort_keys=True)
    return small_sha(path)


def main():
    global SESSION,DEVELOPMENT,MANIFEST,PREFLIGHT,COMPARISON
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--comparison-helper',type=Path,default=COMPARISON,
        help='Small per-role comparison helper; its actual source SHA is retained')
    args=parser.parse_args();output=args.output.resolve();SESSION=args.session.resolve(strict=True)
    DEVELOPMENT=SESSION/'round_09/development'
    MANIFEST=SESSION/'cpu_audits/round09_source_manifest.json'
    PREFLIGHT=SESSION/'cpu_audits/round09_source_preflight_01/source_preflight.json'
    COMPARISON=args.comparison_helper.resolve(strict=True)
    require(not output.exists(),'Preserve existing report; no automatic rerun or overwrite')
    began=time.time()
    # No numeric helpers are imported until the entire candidate is closed.
    cycle=read(SESSION/'round_09_stages/finished.json')
    finished=read(DEVELOPMENT/'result.json')
    require(cycle.get('software_complete') is True and finished.get('software_complete') is True
        and finished.get('worker_error') is None and finished.get('cases')==16,
        'Round09 cycle and all16 autonomous opportunities must first complete')
    manifest=read(MANIFEST);preflight=read(PREFLIGHT)
    require(preflight.get('version')=='per_role_source_preflight_v1' and preflight.get('complete') is True
        and preflight.get('candidate_audited') is False and preflight.get('test_used') is False,
        'Completed non-test source-only preflight required')
    require(preflight['source_manifest']['path']==str(MANIFEST)
        and preflight['source_manifest']['sha256']==BINDINGS[str(MANIFEST)]['sha256']
        and preflight['source_manifest']['roles']==manifest['roles'], 'Source manifest/preflight binding differs')
    plan=read(DEVELOPMENT/'plan.json');indexed=read(DEVELOPMENT/'results.json')
    require(plan['source_commit']=='87577e84f5f9183ba739e483b34729029629bb99',
        'This report is for the actually registered frozen875 round09 candidate')
    require(len(indexed)==len(plan['cases'])==16 and {r['case'] for r in indexed}=={c['name'] for c in plan['cases']},
        'All sixteen opportunities, including illegal starts, must remain indexed')
    require(read(DEVELOPMENT/'run_started.json')['plan_sha256']==BINDINGS[str(DEVELOPMENT/'plan.json')]['sha256'],
        'Candidate startup plan differs')

    # Use exactly the previously reviewed phase/contact arithmetic for both cohorts.
    helper_tree=Path(preflight['helper_root'])
    helper_names=['tools/tight_fresh_phases.py','tools/tight_fresh_visual_closeout.py',
        'tools/tight_fresh_contact_height_coverage.py','visuotactile/takeover_trigger.py']
    # Prior helper/source verification is carried from preflight, not rerun.
    helper_binding=dict(root=str(helper_tree),source_sha256={name:preflight['helper_sha256'][name] for name in helper_names},
        verification='Carried from completed source preflight; no repeat scan of old frozen source trees')
    sys.path.insert(0,str(helper_tree))
    from tools import tight_fresh_phases as phases
    from tools import tight_fresh_visual_closeout as motion
    from tools.tight_fresh_contact_height_coverage import causal_contact_height
    spec=importlib.util.spec_from_file_location('round09_readonly_comparison_helpers',COMPARISON)
    compare=importlib.util.module_from_spec(spec);spec.loader.exec_module(compare)
    small_sha(COMPARISON);small_sha(__file__)
    compare.read=read;compare.sha256=small_sha;compare.causal_contact_height=causal_contact_height
    # Only this private process changes helper globals. No source files are edited.
    # contact_motion's initial input check now compares size/inode/mtime tokens;
    # remove the misleading legacy numeric_sha256 key before writing the report.
    phases.sha256=helper_input_token;motion.sha256=helper_input_token
    compare.validate_development_plan(plan)
    require(Path(plan['resource_root']).resolve()==SESSION/'round_09','Candidate resource root differs')
    execution=compare.verify_execution_receipt(DEVELOPMENT,plan)
    require(execution['lanes']==8 and execution['vision_microbatch']==8,'Expected registered M8/eight-lane comparison')
    sources=preflight['source_by_role'];population={};initial={};matching={}
    for role in ('A','B'):
        olddir=Path(manifest['roles'][role]['development']);oldplan=read(olddir/'plan.json')
        source=sources[role]
        require(source['development']==str(olddir) and source['plan_sha256']==BINDINGS[str(olddir/'plan.json')]['sha256'],
            'Source preflight plan binding differs')
        require(small_sha(olddir/'result.json')==source['completion_sha256']
            and small_sha(olddir/'results.json')==source['results_sha256'], 'Source reviewed small results changed')
        require(source['pairs']=={role:oldplan['pairs'][role]} and len(source['cases'])==8
            and all(c['role']==role for c in source['cases']), 'Source preflight role/pair population differs')
        population[role]=compare.compare_role_plans(oldplan,plan,role)
        initial[role]=compare.initial_observation_comparison(olddir,DEVELOPMENT,
            [c for c in oldplan['cases'] if c['role']==role])

    by_name={r['case']:r for r in indexed};assignments=compare.wave_assignments(plan);rows=[];numeric_files=[]
    for case in plan['cases']:
        name=case['name'];role=case['role'];directory=DEVELOPMENT/name;pair=plan['pairs'][role]
        record=by_name[name];r=read(directory/'result.json');started=read(directory/'started.json')
        metadata=read(directory/'metadata.json');batch=r['batch_execution'];wave,lane=assignments[name]
        require(record['result']==r and record['role']==role and started['spec']==case['spec'], 'Candidate case/index identity differs')
        require(all(r.get(k)==v for k,v in case['spec'].items()),'Executed reset differs from plan')
        require(started['batch_execution']==metadata['batch_execution']==batch
            and (batch['wave'],batch['lane'])==(wave,lane) and batch['lanes']==8 and batch['vision_microbatch']==8
            and batch['physics_seed']==plan['batch']['seed']+wave
            and batch['event_seed']==plan['batch']['seed']+100000+wave,
            'Candidate actual wave/lane/seed/microbatch differs')
        require(metadata['policy_checkpoint_sha256']==pair['backend_sha256']
            and metadata['execution_backend']==execution['execution_backend'], 'Executed policy/backend receipt differs')
        if not r['legal']:
            rows.append(dict(group='candidate',role=role,case=name,legal=False,
                initial_rejection=compare.invalid_initialization_evidence(directory,r)))
            continue
        require(r['software_complete'] and not r.get('censored') and r.get('dagger') is None
            and not r['expert_shadow_computed'] and r['complete_student_policy']==pair,
            'Candidate must be complete actual full-student autonomous execution')
        require(r['checkpoint_sha256']==pair['backend_sha256'] and r['residual_version']==pair['residual_version']
            and r['residual_ticks']==r['steps'] and r['simulation_step_hz']==60 and r['physics_hz']==120
            and r['plan_hz']==5 and r['action_control_mode']=='absolute_joint_position_target_rad',
            'Candidate model or control-frequency receipt differs')
        for stage in ('dino_rgb','learned_visual_and_history','residual_forward_compose'):
            require(r['native_inference']['summary'][stage]['wall_s']['count']==r['steps'], 'Missing actual per-control student observation stage')
        path=directory/'numeric.h5';before=numeric_stat(path)
        phase=phases.case(directory)
        require(phase['authority']['expert_ticks']==0 and phase['steps']==r['steps']
            and phase['native_lift']==r['native_lift'] and phase['native_subgoals']==r['native_subgoals'],
            'Candidate numeric authority/length/native outcome differs from result')
        contact=motion.contact_motion(directory,phase);orientation=compare.held_orientation(directory)
        require(numeric_stat(path)==before,'Closed candidate numeric changed during review')
        phase.pop('numeric_sha256')
        numeric_files.append(dict(path=str(path),stat=before,sha256_recomputed=False))
        rows.append(dict(group='candidate',role=role,**phase,contact_motion=contact,held_orientation=orientation,
            numeric_stat=before,numeric_sha256_recomputed=False,wall_seconds=record['timing']['total_seconds']))
    candidate=dict(root=str(SESSION/'round_09'),development=str(DEVELOPMENT),execution=execution,pairs=plan['pairs'],cases=rows,
        totals={role:compare.autonomy_totals([r for r in rows if r['role']==role],batch=True,explicit=True) for role in ('A','B')})
    for role in ('A','B'):
        selected=dict(candidate,cases=[r for r in rows if r['role']==role])
        matching[role]=compare.matched_legal_comparison(sources[role],selected,roles=(role,))['by_role'][role]
    report=dict(version='round09_behavior_source_reuse_v1',complete=True,started_utc=began,finished_utc=time.time(),
        source_by_role=sources,candidate=candidate,population_by_role=population,
        matched_legal_comparison=matching,initial_observation_comparison_by_role=initial,
        source_preflight_reused=True,input_and_code_small_sha256=BINDINGS,numeric_files=numeric_files,
        phase_helper_binding=helper_binding,
        scope=dict(candidate_numeric_files=len(numeric_files),source_numeric_files_opened=0,checkpoint_files_opened=0,
            RGB_or_DINO_payloads_opened=0,numeric_payload_hashes_recomputed=0,simulator_starts=0,optimizer_updates=0,test_used=False),
        limits=[
            'Source A08 and B06 behavior comes from two already audited original runs; no synthetic mixed-source execution was created.',
            'This is a behavior summary, not a repeated model-byte or complete camera/control provenance audit. Model identity is bound by existing small plan/metadata/result receipts; candidate numeric stability uses stat before/after, not SHA.',
            'Native lift, old stable, rotation-permitting contact-height12, turning, original alignment/subgoal and original success remain separate measurements; missing a held window does not prove a drop.',
            'Direction summaries use the same first native goal; minima retain their actual ticks. Quaternion angles ignore object symmetry and improvement does not by itself establish insertion.',
            'All registered opportunities and illegal starts remain visible; behavior comparison additionally uses each role\'s matched-legal intersection.',
            'Equality of saved visible initial states does not prove identical PhysX/contact/renderer/RNG hidden states. Repeated development starts are not an independent held-out generalization estimate.',
            'No learning or selected-versus-last fitting claim is made by this report.'])
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(output),source={k:v['totals'] for k,v in sources.items()},candidate=candidate['totals'],
        matched_legal=matching),indent=2))


if __name__=='__main__':main()
