"""Freeze and audit this round's complete paired students; no historical globals.

Preparation and summary are CPU-only. ``worker`` is an explicit native simulation
stage. It disables expert construction and actions within that worker process.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import csv
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from tools.tight_paired_dagger import PROJECT, ROOT, CONFIG, AUDIT, read, session_at
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

CONTRACT = 'tight_paired_dagger_frozen_autonomy_v1'
ROLES = ('A_new', 'B_new', 'A_parent', 'B_parent')
SAMPLER_SEED = 7091
STABLE_RULE = dict(contact_threshold_N=.1, minimum_fingers_both_substeps=2,
    minimum_consecutive_ticks=12, settled_reference='median(object_world_z[24:48])',
    minimum_height_above_settled_m=.02, earliest_available_transition=47,
    scope='Existing held12 screen; diagnostic, never native success')


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def wilson(successes, n):
    if not n: raise ValueError('Positive full registered denominator required')
    z = 1.959963984540054
    p = successes/n; denominator = 1+z*z/n
    center = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def contrast(left, right, *, repetitions=200000):
    left = np.asarray(left, dtype=np.int8); right = np.asarray(right, dtype=np.int8)
    if left.shape != right.shape or left.ndim != 1 or not len(left):
        raise ValueError('One aligned outcome per registered family required')
    delta = left-right
    rng = np.random.default_rng(20261122)
    # Batch resamples to keep memory bounded even if cohort grows accidentally.
    means = np.concatenate([delta[rng.integers(0,len(delta),(min(10000,repetitions-i),len(delta)))].mean(1)
                            for i in range(0,repetitions,10000)])
    return dict(n=len(delta), left_success=int(left.sum()), right_success=int(right.sum()),
        paired_difference_pp=float(delta.mean()*100),
        paired_bootstrap95_pp=(np.quantile(means,[.025,.975])*100).tolist(),
        both_success=int(((left==1)&(right==1)).sum()),
        gained=int(((left==1)&(right==0)).sum()), lost=int(((left==0)&(right==1)).sum()),
        both_failure=int(((left==0)&(right==0)).sum()),
        method='paired family percentile bootstrap; 200000 resamples; seed20261122')


def geometry_metrics(objects, forces, keypoints, goals):
    """Existing retrospective held12 and native 15mm gate, on actual transitions."""
    objects=np.asarray(objects); forces=np.asarray(forces); keypoints=np.asarray(keypoints)
    goals=np.asarray(goals); n=len(objects)
    count=(np.linalg.norm(forces,axis=-1)>.1).sum(-1).min(-1)
    stable=np.zeros(n,dtype=bool)
    if n>=48:
        good=(count>=2)&(objects[:,2]>=float(np.median(objects[24:48,2]))+.02)
        runs=0
        for i, value in enumerate(good):
            runs=runs+1 if value else 0
            stable[i]=runs>=12 and i>=47
    gate=(keypoints<=.015)&(goals==0)
    def first(mask):
        ticks=np.flatnonzero(mask); return int(ticks[0]) if len(ticks) else None
    return dict(stable_hold=bool(stable.any()), first_stable_hold_tick=first(stable),
        first_15mm_gate=bool(gate.any()), first_15mm_gate_tick=first(gate),
        left_15mm_gate_after_entry=bool(gate.any() and ((~gate)&(np.arange(n)>first(gate))&(goals==0)).any()))


def numeric_metrics(directory):
    import h5py
    path=directory/'numeric.h5'
    if not path.exists():
        return dict(stable_hold=False, first_stable_hold_tick=None, first_15mm_gate=False,
            first_15mm_gate_tick=None,left_15mm_gate_after_entry=False, geometry_observed=False)
    with h5py.File(path,'r') as stream:
        tr=stream['episode/transition']
        metrics=geometry_metrics(tr['diagnostics_after/object_pose_w'][:], tr['object_contact_force_w'][:],
            tr['native_keypoints_max_dist'][:],tr['diagnostics_before/successes'][:])
    return dict(metrics,geometry_observed=True,numeric_path=str(path),numeric_sha256=sha256(path))


def validate_pair(pair, supervision):
    if pair.get('version')!='complete_student_policy_v1' or pair.get('controller')!='ResidualController':
        raise ValueError('Full DP plus Transformer required')
    if pair.get('supervision')!=supervision: raise ValueError('Nominal target contract differs')
    for key in ('frontend','backend'):
        if sha256(pair[key])!=pair[key+'_sha256']: raise ValueError('Policy bytes differ: '+key)


def frozen_pairs(root=ROOT):
    """Read explicit completed deployments; never infer a validation winner."""
    import torch
    from tools.tight_fresh_rollouts import completed_pair
    parents=read(root/'parent_pairs.json')
    for role in 'AB':
        actual=completed_pair(root/'common180_parents',role)
        if actual!=parents[role]:
            raise ValueError('Active parent identity differs from completed reconstructed pair')
    pairs={}
    for role,target in (('A','q_reached'),('B','u_expert')):
        for tag, pair in [('new',completed_pair(root/'learning',role)),('parent',parents[role])]:
            validate_pair(pair,target)
            frontend=torch.load(pair['frontend'],map_location='cpu',weights_only=False,mmap=True)
            if frontend['contract'].get('sampler_seed')!=SAMPLER_SEED:
                raise ValueError('Registered inference noise seed differs')
            del frontend
            # Both reconstructed parents and new students must carry actual epoch60
            # deployment receipts; this rejects still-registered historical parents.
            for component in ('frontend','backend'):
                receipt=read(Path(pair[component]).parent/'deployment.json')
                if (receipt.get('epoch')!=60 or receipt.get('selection')!='preregistered_epoch60_only'
                        or receipt.get('checkpoint_sha256')!=pair[component+'_sha256']):
                    raise ValueError('Missing final-epoch deployment binding')
            pairs[role+'_'+tag]=pair
    return {role:pairs[role] for role in ROLES}


def validate_cohorts(root):
    """Keep original seal; compare identity/reset metadata, never test trajectories."""
    cohorts=read(root/'cohorts.json')
    registration_path=root/'registration.json'
    if registration_path.exists():
        initial=read(registration_path)
        if initial.get('cohorts_sha256')!=sha256(root/'cohorts.json'):
            raise ValueError('Initial sealed cohorts SHA changed')
    families=set(); seeds=set(); resets=set()
    for name,rows in cohorts.items():
        for row in rows:
            reset=json.dumps(row['narrow_reset'],sort_keys=True,separators=(',',':'))
            if row['family'] in families or row['seed'] in seeds or reset in resets:
                raise ValueError('Training/development/test family or reset overlap')
            families.add(row['family']);seeds.add(row['seed']);resets.add(reset)
    historical=AUDIT/'all_frozen_families.json'
    if root.resolve()==ROOT.resolve():
        previous=read(historical)
        for row in previous:
            reset=json.dumps(row['narrow_reset'],sort_keys=True,separators=(',',':')) if 'narrow_reset' in row else None
            if row.get('family') in families or row.get('seed') in seeds or reset in resets:
                raise ValueError('New cohort overlaps historical family/reset')
    return cohorts


def prepare(cohort, *, root=ROOT, pairs=None, template=None, job_path=None):
    """Freeze all opportunities once, after all four policies are complete."""
    if cohort not in ('development','test'): raise ValueError('Explicit cohort required')
    output=root/'evaluation'/cohort
    if output.exists(): raise FileExistsError('Preserve attempted evaluation identity')
    pairs=frozen_pairs(root) if pairs is None else pairs
    if tuple(pairs)!=ROLES: raise ValueError('Exactly four complete registered roles required')
    for role,pair in pairs.items(): validate_pair(pair,'q_reached' if role.startswith('A') else 'u_expert')
    session=read(root/'session.json')
    if session['deadline']<=time.time(): raise ValueError('Finite session expired')
    cohorts=validate_cohorts(root); specifications=cohorts[cohort]
    previous=root/'evaluation/development/registration.json'
    if cohort=='test' and previous.exists() and read(previous)['pairs']!=pairs:
        raise ValueError('Policies changed after descriptive development freeze')
    expected=8 if cohort=='development' else 64
    start=28911000 if cohort=='development' else 28912000
    if len(specifications)!=expected or [s['seed'] for s in specifications]!=list(range(start,start+expected)):
        raise ValueError('Frozen cohort changed')
    if template is None:
        template=read(PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01/wave00/plan.json')
    # The old result/template contributes execution configuration only.
    plan_template={k:copy.deepcopy(template[k]) for k in ('batch','execution_backend')}
    sources={str(p.relative_to(PROJECT)):sha256(p) for folder in ('tools','visuotactile')
             for p in sorted((PROJECT/folder).rglob('*.py'))}
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    output.mkdir(parents=True)
    evaluation_session=session_at(output)
    stages=[]; wave_hashes={}
    for index, registered in enumerate(specifications):
        seed=registered['seed']; wave=output/f'wave{index:02d}'; wave.mkdir()
        spec=dict(identity=f'paired_{cohort}_{seed}',family=registered['family'],split=cohort,
            seed=seed,event_seed=seed+100000,cohort='fresh',collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=copy.deepcopy(registered['narrow_reset']),layout=copy.deepcopy(session['layout']))
        names=list(ROLES[index%4:]+ROLES[:index%4])
        plan=copy.deepcopy(plan_template)
        plan.update(kind='autonomous',pairs=pairs,cases=[dict(name=name+'_'+spec['identity'],role=name,
            spec=copy.deepcopy(spec),video=index==0,diagnostic_cohort=cohort) for name in names],
            resource_root=str(output),source_commit=commit,source_sha256=sources,
            inference_scheduler_index_mode='host',first_plan_forensics=False,test_used=cohort=='test',
            cohort=cohort,utc=time.time(),expert_sha256=session['expert_sha256'],record_sampler_rng=True,
            visual_arithmetic=copy.deepcopy(session['visual_arithmetic']),
            evaluation_contract=CONTRACT,expert_permission=False,
            evaluation_scope=dict(protocol='New A/B versus own matched complete parents; fixed epoch60',
                hidden_state_replay_claim=False,strict_success='unchanged native complete insertion',
                milestone='Both new policies >=50% of full sealed64 denominator',
                software_failure='Primary failure; no outcome-dependent retries',
                native_first_subgoal='Cumulative 10 control ticks; not necessarily consecutive'))
        if cohort=='test': plan['holdout_contract']='layout_final_test_v1'
        plan['batch'].update(lanes=4,vision_microbatch=2,seed=seed,writer_bytes=256<<20,
            ordering='One family x four frozen complete policies; cyclic lane rotation; no lane recycling')
        plan['execution_backend'].update(kit_tasking_threads=8,physics_threads=2)
        atomic_json(wave/'session.json',evaluation_session); atomic_json(wave/'plan.json',plan)
        wave_hashes[str(wave)]=sha256(wave/'plan.json')
        stages.append(dict(entry='paired_evaluate',args=['worker','--output',str(wave)]))
    job_path=job_path or CONFIG/f'evaluate_{cohort}.json'
    job_path.parent.mkdir(parents=True,exist_ok=True)
    job=dict(version='tight_project_job_v1',operation='evaluate',resource_root=str(output),stages=stages)
    atomic_json(job_path,job)
    freeze=dict(version=CONTRACT,registered_utc=time.time(),cohort=cohort,pairs=pairs,
        n=expected,seeds=[s['seed'] for s in specifications],specifications=specifications,
        wave_plan_sha256=wave_hashes,source_commit=commit,source_sha256=sources,
        job=str(job_path),job_sha256=sha256(job_path),deadline=session['deadline'],
        cohorts_sha256=sha256(root/'cohorts.json'),parent_pairs_sha256=sha256(root/'parent_pairs.json'),
        expert_permission=False,optimizer_updates=0,new_dagger_trajectories=0,
        strict_denominator='Every registered initial state for every role; missing software/invalid starts are primary failures',
        noise=dict(sampler_seed=7091,draw_shape=[1,24,29],scheduler='host',record_start_end=True),
        stable_hold=STABLE_RULE,first_gate='native keypoints max distance <=15mm while first subgoal unconfirmed',
        inference='120/60/5Hz; commit12; every-frame DINO microbatch2, duplicate-last-real-lane tail',
        checkpoint_selection='All new components preregistered epoch60; development is descriptive only',
        historical24='Reference existing tight_entry_final24_20261122_01; no re-execution',
        compute_policy='One own GPU worker; CPU8-15; nice10; ionice idle; external jobs untouched')
    atomic_json(output/'registration.json',freeze)
    return dict(output=str(output),job=str(job_path),episodes=expected*4)


@contextmanager
def no_expert_scope(output, lanes):
    """Supply an inert protocol stub; never load an expert network or execute it."""
    import torch
    from visuotactile import expert,queued_rollout
    old_factory=expert.create_expert; old_actor=queued_rollout.expert_actor_input
    class NoExpert:
        is_rnn=False
        def reset(self): pass
        def get_action(self,*args,**kwargs): raise RuntimeError('Expert actions forbidden in autonomous evaluation')
    stub=NoExpert()
    def blocked_factory(*args,**kwargs):
        atomic_json(output/'expert_disabled.json',dict(expert_permission=False,network_loaded=False,
            player='inert reset-only protocol stub',expert_action_calls=0,utc=time.time()))
        return stub,None
    expert.create_expert=blocked_factory
    queued_rollout.expert_actor_input=lambda teacher,player: torch.zeros((lanes,1),device='cpu')
    try: yield
    finally:
        expert.create_expert=old_factory; queued_rollout.expert_actor_input=old_actor


def validate_wave(output):
    plan=read(output/'plan.json'); root=Path(plan['resource_root']); freeze=read(root/'registration.json')
    if (freeze['version']!=CONTRACT or plan.get('evaluation_contract')!=CONTRACT
            or plan['kind']!='autonomous' or plan.get('expert_permission') is not False
            or plan['pairs']!=freeze['pairs'] or len(plan['cases'])!=4
            or {c['role'] for c in plan['cases']}!=set(ROLES)
            or sha256(output/'plan.json')!=freeze['wave_plan_sha256'][str(output)]):
        raise ValueError('Frozen autonomous wave binding failed')
    if (plan['batch'].get('lanes')!=4 or plan['batch'].get('vision_microbatch')!=2
            or plan.get('record_sampler_rng') is not True
            or plan.get('inference_scheduler_index_mode')!='host'):
        raise ValueError('Native inference execution contract differs')
    if any(c.get('dagger') or c.get('source_data') for c in plan['cases']):
        raise ValueError('No expert or training-data privileges in evaluation')
    if len({c['spec']['seed'] for c in plan['cases']})!=1:
        raise ValueError('Exactly one initial state shared by four policies required')
    return plan,freeze


def worker(output):
    output=Path(output).resolve(); plan,freeze=validate_wave(output)
    from visuotactile.paired_execution import install
    from tools.tight_evaluate import run
    install(output)
    with no_expert_scope(output,4): run(output,allow_heldout=freeze['cohort']=='test')


def audit_rng(directory, planning_calls, *, reconstruct_cuda=False):
    """CPU hashes/state parity always; optional exact native CUDA draw reconstruction."""
    import torch
    states={}; receipts={}
    for stage in ('start','end'):
        receipt=read(directory/f'sampler_rng_{stage}.json'); path=directory/receipt['state_path']
        if (sha256(path)!=receipt['state_sha256'] or receipt['initial_seed']!=SAMPLER_SEED
                or receipt['checkpoint_sampler_seed']!=SAMPLER_SEED or receipt['extra_random_draws']!=0):
            raise ValueError('Inference noise receipt mismatch')
        states[stage]=torch.load(path,map_location='cpu',weights_only=False)['generator_state']
        receipts[stage]=dict(receipt_path=str(directory/f'sampler_rng_{stage}.json'),state_path=str(path),state_sha256=sha256(path))
    exact=False
    if reconstruct_cuda:
        generator=torch.Generator(device='cuda:0').manual_seed(SAMPLER_SEED)
        if not torch.equal(generator.get_state().cpu(),states['start']): raise ValueError('Noise start differs')
        for _ in range(planning_calls): torch.randn((1,24,29),device='cuda:0',generator=generator)
        if not torch.equal(generator.get_state().cpu(),states['end']): raise ValueError('Noise consumption differs')
        exact=True
    return dict(receipts=receipts,planning_calls=planning_calls,receipt_hashes_verified=True,
        native_draw_reconstruction_verified=exact,start_state=states['start'].tolist(),end_state=states['end'].tolist())


def summarize(output, *, reconstruct_cuda=False):
    """Intention-to-evaluate summary, preserving unfinished opportunities as failures."""
    output=Path(output).resolve(); freeze=read(output/'registration.json')
    if freeze['version']!=CONTRACT: raise ValueError('Wrong evaluation registration')
    rows=[]; audits=[]; rng_rows=[]; starts=[]; observable_states=[]
    for wave_name,digest in freeze['wave_plan_sha256'].items():
        wave=Path(wave_name); plan,_=validate_wave(wave)
        if sha256(wave/'plan.json')!=digest: raise ValueError('Plan changed')
        check=read(wave/'paired_initial_check.json') if (wave/'paired_initial_check.json').exists() else None
        initial_ok=bool(check and check['passed'] and len(check['rows'])==3)
        if initial_ok:
            for row in check['rows']:
                values=[row['q_max_rad'],row['qd_max_rad_s'],row['target_max_rad'],*row['physical'].values()]
                if any(not math.isfinite(value) or value>1e-5 for value in values) or row['active']!=row['reference_active']:
                    raise ValueError('Observable initial pair receipt contradicts tolerance')
            reference=wave/plan['cases'][0]['name']/'initial_observable.json'
            if reference.exists():
                observed=read(reference)
                obj=np.asarray(observed['object_pose_w'][0]);base=np.asarray(observed['base_pose_w'][0])
                observable_states.append(dict(seed=plan['cases'][0]['spec']['seed'],
                    vector=np.r_[observed['joint_position_rad'][0],obj[:3]-base[:3],obj[3:]].tolist()))
        result_rows=read(wave/'results.json') if (wave/'results.json').exists() else []
        if len({r['case'] for r in result_rows})!=len(result_rows): raise ValueError('Duplicate physical result')
        names={c['name'] for c in plan['cases']}
        if any(r['case'] not in names or r['role']!=next(c['role'] for c in plan['cases'] if c['name']==r['case']) for r in result_rows):
            raise ValueError('Unknown or role-mismatched result')
        known={r['case']:r['result'] for r in result_rows}
        profiles=read(wave/'wave_profiles.json') if (wave/'wave_profiles.json').exists() else []
        for lane,case in enumerate(plan['cases']):
            directory=wave/case['name']; result=known.get(case['name'])
            # Closed per-lane results can survive a later lane/worker failure.
            if result is None and (directory/'result.json').exists(): result=read(directory/'result.json')
            result=result or {}; legal=result.get('legal') is True; software=result.get('software_complete') is True
            complete=legal and software and initial_ok
            protocol_missing=[]
            if software and legal:
                required=[directory/'native_inference_timing.json',wave/'expert_disabled.json',
                    directory/'sampler_rng_start.json',directory/'sampler_rng_end.json']
                protocol_missing.extend(str(path.name) for path in required if not path.exists())
            if software and legal and not protocol_missing:
                if result.get('seed')!=case['spec']['seed'] or result.get('family')!=case['spec']['family']:
                    raise ValueError('Executed reset identity differs')
                if result.get('complete_student_policy')!=freeze['pairs'][case['role']]: raise ValueError('Executed policy differs')
                if result.get('expert_shadow_computed') is not False or result.get('dagger'): raise ValueError('Expert computation in autonomy')
                if result['residual_ticks']!=result['steps']: raise ValueError('Incomplete full controller')
                if (result['physics_hz'],result['simulation_step_hz'],result['plan_hz'],result['residual_hz'])!=(120,60,5,60):
                    raise ValueError('Native rates differ')
                if len(profiles)==0:
                    protocol_missing.append('wave_profiles.json')
                elif len(profiles)!=1:
                    raise ValueError('Multiple profiles for single physical wave')
                else:
                    vision=profiles[0]['vision_execution']['roles'][case['role']]
                    if (vision['vision_microbatch']!=2 or vision['controller_calls']!=result['steps']
                            or vision['real_lanes']!=result['steps']
                            or vision['consumed_observations_per_lane'][lane]!=result['steps']):
                        raise ValueError('Every-frame native RGB-D consumption differs')
                timing=read(directory/'native_inference_timing.json')['batch_observed_counts']
                if any(timing[k]!=result['steps'] for k in ('history_updates','robot_observations','backend_updates')) or timing['plans']!=(result['steps']+11)//12:
                    raise ValueError('Native control/vision history counts differ')
                if not (wave/'expert_disabled.json').exists() or read(wave/'expert_disabled.json')['network_loaded'] is not False:
                    raise ValueError('Missing no-expert receipt')
                noise=audit_rng(directory,timing['plans'],reconstruct_cuda=reconstruct_cuda)
                rng_rows.append(dict(seed=case['spec']['seed'],role=case['role'],**noise))
                if not starts:
                    starts=noise['start_state']
                if noise['start_state']!=starts: raise ValueError('Unpaired inference noise start')
                metrics=numeric_metrics(directory)
            else: metrics=dict(stable_hold=False,first_stable_hold_tick=None,first_15mm_gate=False,
                first_15mm_gate_tick=None,left_15mm_gate_after_entry=False,geometry_observed=False)
            complete=complete and not protocol_missing
            rows.append(dict(seed=case['spec']['seed'],family=case['spec']['family'],role=case['role'],lane=lane,
                success=bool(complete and result.get('success')),legal=legal,software_complete=software,
                observed_initial_pair_passed=initial_ok,missing_result=not bool(result),
                strict_primary_admissible=complete,protocol_missing=';'.join(protocol_missing),steps=result.get('steps'),
                native_first_subgoal=bool(complete and result.get('native_subgoals',0)>=1),
                stable_hold=bool(complete and metrics['stable_hold']),first_15mm_gate=bool(complete and metrics['first_15mm_gate']),
                geometry_observed=metrics['geometry_observed'],first_stable_hold_tick=metrics['first_stable_hold_tick'],
                first_15mm_gate_tick=metrics['first_15mm_gate_tick'],
                left_15mm_gate_after_entry=metrics['left_15mm_gate_after_entry'],result_path=str(directory/'result.json')))
        audits.append(dict(wave=wave.name,initial_pair_passed=initial_ok,closed_result_count=len(known)))
    separations=[float(np.max(np.abs(np.asarray(a['vector'])-b['vector'])))
        for i,a in enumerate(observable_states) for b in observable_states[i+1:]]
    if separations and min(separations)<=1e-5:
        raise ValueError('Distinct family labels produced duplicate observed initial states')
    n=freeze['n']
    if len(rows)!=4*n or len({(r['seed'],r['role']) for r in rows})!=4*n: raise ValueError('Registered population differs')
    index={(r['seed'],r['role']):r for r in rows}
    def outcomes(role): return [index[seed,role]['success'] for seed in freeze['seeds']]
    absolute=[]; funnels=[]
    for role in ROLES:
        subset=[index[seed,role] for seed in freeze['seeds']]; k=sum(r['success'] for r in subset)
        absolute.append(dict(role=role,success=k,n=n,success_rate=k/n,wilson95=wilson(k,n),
            software_incomplete=sum(not r['software_complete'] for r in subset),illegal_start=sum(not r['legal'] and not r['missing_result'] for r in subset),
            missing=sum(r['missing_result'] for r in subset),protocol_unverified=sum(bool(r['protocol_missing']) for r in subset),observed_initial_pair_failures=sum(not r['observed_initial_pair_passed'] for r in subset)))
        funnels.append(dict(role=role,n=n,stable_hold=sum(r['stable_hold'] for r in subset),
            first_15mm_gate=sum(r['first_15mm_gate'] for r in subset),native_first_subgoal=sum(r['native_first_subgoal'] for r in subset),
            final_insertion=k,geometry_unobserved=sum(not r['geometry_observed'] for r in subset),
            denominator='all registered initial states; unobserved/unfinished primary failure'))
    contrasts=[dict(name='A_new_minus_B_new',**contrast(outcomes('A_new'),outcomes('B_new')))]
    for role in 'AB': contrasts.append(dict(name=role+'_new_minus_parent',**contrast(outcomes(role+'_new'),outcomes(role+'_parent'))))
    pairs=[]
    for seed in freeze['seeds']:
        row=dict(seed=seed,family=index[seed,'A_new']['family'])
        for role in ROLES:
            r=index[seed,role]
            for key in ('success','legal','software_complete','stable_hold','first_15mm_gate','native_first_subgoal','result_path'):row[role+'_'+key]=r[key]
        for role in 'AB':
            new=index[seed,role+'_new']['success']; parent=index[seed,role+'_parent']['success']
            row[role+'_success_retained']=new and parent; row[role+'_success_gained']=new and not parent; row[role+'_success_lost']=parent and not new
        pairs.append(row)
    for role,pair in freeze['pairs'].items(): validate_pair(pair,'q_reached' if role.startswith('A') else 'u_expert')
    session=read(output/'session.json')
    if sha256(session['dino_checkpoint'])!=session['dino_sha256']:
        raise ValueError('Frozen encoder bytes differ')
    for path,digest in freeze['source_sha256'].items():
        if sha256(PROJECT/path)!=digest: raise ValueError('Frozen source changed: '+path)
    evidence=output/'summary'; evidence.mkdir(exist_ok=True)
    summary=dict(absolute_success=absolute,contrasts=contrasts,
        both_new_reach_50_percent=all(x['success_rate']>=.5 for x in absolute[:2]),
        cohort=freeze['cohort'],n=n,episodes=4*n,full_denominator_preserved=True,
        final_execution_complete=all(r['software_complete'] for r in rows),
        primary_protocol_verified=all(not r['protocol_missing'] for r in rows),
        all_software_complete=all(r['software_complete'] for r in rows),
        observed_paired_starts_passed=sum(w['initial_pair_passed'] for w in audits),
        inference_rng_receipts_verified=len(rng_rows),inference_rng_reconstruction_verified=sum(r['native_draw_reconstruction_verified'] for r in rng_rows),
        funnels=funnels,waves=audits,stable_hold_rule=STABLE_RULE,
        interpretation='One overall expanded-initial-state + own-correction + paired-training intervention; no data-only causal effect',
        hidden_physics_or_future_stream_exactness_claim=False,development_selects_checkpoint=False,
        registration_sha256=sha256(output/'registration.json'),observed_utc=time.time())
    atomic_json(evidence/'distinct_observed_initials.json',dict(states=observable_states,tolerance=1e-5,
        minimum_separation=min(separations) if separations else None,
        scope='Observed q29 + base-relative object position + quaternion; not hidden physical state'))
    atomic_json(evidence/'primary_summary.json',summary); atomic_json(evidence/'noise_audit.json',rng_rows)
    write_csv(evidence/'rollouts.csv',rows); write_csv(evidence/'per_initial_state.csv',pairs)
    write_csv(evidence/'funnels.csv',funnels); write_csv(evidence/'paired_contrasts.csv',contrasts)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='stage',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--cohort',choices=('development','test'),required=True)
    work=sub.add_parser('worker');work.add_argument('--output',type=Path,required=True)
    summary=sub.add_parser('summarize');summary.add_argument('--output',type=Path,required=True)
    summary.add_argument('--reconstruct-cuda-noise',action='store_true',help='Explicit tiny CUDA audit; default summary stays CPU-only')
    sub.add_parser('project')
    args,unknown=parser.parse_known_args()
    if args.stage=='project':
        import sys
        from tools import tight_project
        tight_project.ENTRIES['paired_evaluate']=('evaluate','tools.tight_paired_evaluate',[])
        sys.argv=[sys.argv[0],*unknown];tight_project.main();return
    if unknown: parser.error('Unknown arguments: '+' '.join(unknown))
    if args.stage=='prepare': result=prepare(args.cohort)
    elif args.stage=='worker': worker(args.output);return
    else: result=summarize(args.output,reconstruct_cuda=args.reconstruct_cuda_noise)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
