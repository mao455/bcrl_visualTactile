"""Read-only recovery05 audit: all96 opportunities, truthful47/1 init checks.

CPU by default. CUDA endpoint reconstruction requires a separate explicit flag.
No original audit gates are patched and no missing pair check is manufactured.
"""
import argparse
import json
import math
import math
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_collection_noise import (PROJECT, SOURCES, SEED, SHAPE, read, require,
    planning_scope, numeric_schedule, verify_receipts, native_rng_auditor,
    native_cuda_prefix_state)
from audit_collection_integrity import sha, check_numeric
from collection_recovery05_scope import validated_summary


def recovered_assignments(plan, assignments, checks):
    """Validate real checks directly; the missing cross-attempt pair stays absent."""
    cases = plan['cases']
    require(plan.get('stage_takeover_contract') == 'paired_own_student_recovery_collection_v1'
            and plan.get('kind') == 'takeover' and plan.get('test_used') is False,
            'Exact training-only collection contract required')
    require(plan.get('sampler_seed') == SEED and plan.get('record_sampler_rng') is True
            and plan.get('inference_scheduler_index_mode') == 'host', 'Noise contract differs')
    require(len(cases) == len(assignments) == 96 and len({c['name'] for c in cases}) == 96,
            'Exactly96 unique original cases and assignments required')
    pairs = {}; attempt_counts = {}
    for ordinal, (case, assigned) in enumerate(zip(cases, assignments)):
        attempt = assigned.get('attempt')
        require(attempt in ('original', 'continued'), 'Unknown assignment attempt')
        expected_attempt_ordinal = attempt_counts.get(attempt, 0)
        require(assigned.get('attempt_ordinal') == expected_attempt_ordinal,
                'Noncontiguous attempt-local assignment ordinal')
        attempt_counts[attempt] = expected_attempt_ordinal+1
        require(assigned.get('case') == case['name'] and assigned.get('role') == case['role']
                and assigned.get('ordinal') == ordinal and assigned.get('sampler_seed') == SEED,
                'Global assignment order/identity/role/seed changed')
        require(type(assigned.get('lane')) is int and 0 <= assigned['lane'] < plan['batch']['lanes'],
                'Invalid physical lane')
        require(case['role'] in ('A','B') and case['opportunity'] in ('support','alignment')
                and case['spec']['identity'] == case['name'], 'Invalid case contract')
        key = (case['spec']['family'], case['opportunity'])
        require(case['role'] not in pairs.setdefault(key, {}), 'Duplicate pair role')
        pairs[key][case['role']] = case
    require(attempt_counts == {'original':15, 'continued':81}
            and [a['attempt'] for a in assignments] == ['original']*15+['continued']*81,
            'Exact consumed15 plus unassigned81 prefix/suffix required')
    require(len(pairs) == 48 and len({k[0] for k in pairs}) == 24, '24 families/48 pairs required')
    for pair in pairs.values():
        require(set(pair) == {'A','B'}, 'Missing paired role')
        for key in ('seed','event_seed','narrow_reset','initial_configuration_sha256'):
            require(pair['A']['spec'][key] == pair['B']['spec'][key], 'Configured paired reset differs')
    require(checks.get('passed') is False and checks.get('full48_verified') is False
            and checks.get('all_recorded_checks_passed') is True
            and len(checks['rows']) == 47 and len(checks['unverified_rows']) == 1,
            'Truthful47 verified and1 unverified required')
    observed = set()
    by_name = {a['case']:a for a in assignments}
    for row in checks['rows']:
        key = (row['family'], row['opportunity'])
        require(key in pairs and key not in observed and row.get('passed') is True,
                'Unknown/duplicate/failed observed pair')
        names = {c['name'] for c in pairs[key].values()}
        require({row.get('reference'),row.get('case')} == names, 'Observed pair names differ')
        differences=row.get('differences')
        require(isinstance(differences,dict) and bool(differences)
                and all(type(v) in (int,float) and math.isfinite(v) and 0 <= v <= 1e-5
                        for v in differences.values()), 'Observed pair numeric tolerance differs')
        require(row.get('reference_legal') is True and row.get('legal') is True,
                'Observed pair initialization legality not verified')
        require({by_name[n]['attempt'] for n in names} == {row.get('attempt')},
                'Observed pair crosses attempts without saved check')
        observed.add(key)
    missing = checks['unverified_rows'][0]
    key = (missing['family'], missing['opportunity'])
    require(set(pairs)-observed == {key} and missing.get('passed') is None
            and bool(missing.get('reason')), 'Unverified pair identity/reason differs')
    require({by_name[c['name']]['attempt'] for c in pairs[key].values()} == {'original','continued'},
            'Unverified pair must be the actual cross-attempt pair')
    return pairs


def admitted_capture_receipts(directory,result,compact,payload_sha):
    """Bind the original full-field verifier and all-frame camera acceptance."""
    receipt=read(directory/'capture_complete.json');binding=receipt.get('dataset_capture',{})
    capture=result['dataset_capture'];asynchronous=result.get('async_capture',{})
    require(receipt.get('status')=='source_verified' and receipt.get('steps')==result['steps']
            and Path(receipt['numeric_path']).resolve()==(directory/'numeric.h5').resolve(),
            'Admitted capture completion/steps/numeric path differs')
    require(binding==capture and asynchronous.get('dataset_capture')==capture
            and asynchronous.get('status')=='source_verified' and asynchronous.get('steps')==result['steps'],
            'Admitted source verifier/result binding differs')
    require(Path(binding['path']).resolve()==compact.resolve() and binding.get('sha256')==payload_sha
            and type(binding.get('verified_source_fields')) is int and binding['verified_source_fields']>0,
            'Admitted source field verification/hash binding differs')
    camera=read(directory/'camera_acceptance.json')
    require(camera.get('passed') is True and camera.get('frames')==result['steps']
            and camera.get('views')==['fixed_high','fixed_side','wrist_rgbd']
            and camera.get('invalid_frames_per_view')==[0,0,0]
            and all(camera.get(k) is True for k in ('depth_mask_exact','camera_flags_all_true',
                    'synchronized_clock','frames_strictly_increase')), 'Admitted camera acceptance differs')
    return dict(source_verified_fields=binding['verified_source_fields'],all_frame_camera_acceptance_bound=True)


def inputs(collection):
    collection = Path(collection).resolve()
    summary = validated_summary(collection)
    pointer = read(collection/'recovery05_active_aggregate.json') if (collection/'recovery05_active_aggregate.json').exists() else None
    aggregate = Path(pointer['path']).resolve() if pointer else collection
    binding = read(aggregate/'recovery05_aggregate_binding.json')
    original, recovered, continued = map(Path, binding['source_directories'])
    plan = read(original/'plan.json'); registration = read(original/'registration.json')
    require(sha(original/'plan.json') == registration['plan_sha256'], 'Frozen plan registration differs')
    require(sha(PROJECT/'tools/tight_paired_collection.py') == registration['wrapper_sha256'], 'Collector wrapper changed')
    require(sha(plan['parent_pairs_path']) == plan['parent_pairs_sha256'], 'Collection parent ledger changed')
    for source in SOURCES:
        require(sha(PROJECT/source) == plan['source_sha256'][source], 'Frozen source changed: '+source)
    assignments = read(aggregate/'paired_queue_assignments.json')
    checks = read(aggregate/'paired_initial_check.json')
    pairs = recovered_assignments(plan, assignments, checks)
    rows = read(aggregate/'results.json'); entries = read(aggregate/'correction_entries.json')
    for i,(case,row) in enumerate(zip(plan['cases'],rows)):
        expected = (recovered if i < 15 else continued)/case['name']
        directory = Path(row['evidence_directory'])
        require(directory.absolute() == expected.absolute() and directory.is_dir(), 'Wrong physical evidence directory')
        result = row['result']
        require(result.get('identity') == case['name'] and result.get('family') == case['spec']['family']
                and row['role'] == case['role'], 'Result identity/family/role differs')
        filename = 'result.json' if result.get('software_complete') else 'interrupted_outcome.json'
        # Continuation software failures are normally written as result.json.
        if not (directory/filename).exists() and not result.get('software_complete'):
            filename = 'result.json'
        require(read(directory/filename) == result, 'Aggregate/physical outcome differs')
    return plan, assignments, checks, pairs, rows, entries, summary


def integrity(collection):
    from tools import tight_paired_collection as collector
    plan, assignments, checks, pairs, rows, entries, summary = inputs(collection)
    funnels = {r:{op:dict(opportunities=0,invalid_initialization=0,unknown_initialization=0,
        software_incomplete=0,untriggered=0,takeovers=0,expert_recovered=0,valid_trajectories=0,
        legal_expert_ticks=0,never_entered_at_takeover=0,ever_entered_then_left_at_takeover=0,
        ever_entered_still_in_gate_at_takeover=0) for op in ('support','alignment')} for r in 'AB'}
    details=[]; failures=[]; hashes={}
    for case,row in zip(plan['cases'],rows):
        name=case['name']; result=row['result']; directory=Path(row['evidence_directory'])
        f=funnels[case['role']][case['opportunity']]; f['opportunities']+=1
        legal=result.get('legal'); complete=result.get('software_complete') is True
        f['invalid_initialization']+=legal is False; f['unknown_initialization']+=legal is None
        f['software_incomplete']+=not complete
        if legal is not True or not complete:
            require(name not in entries and not result.get('accepted_for_training'), 'Unsafe incomplete/illegal admission')
            failures.append(dict(case=name,legal=legal,software_complete=complete,
                issue='software_incomplete' if not complete else 'invalid_initialization',retained_in_denominator=True))
            continue
        require(result['complete_student_policy'] == plan['pairs'][case['role']], 'Not own complete parent')
        trigger=collector.audit_trigger(directory,result)
        switched=trigger['actual_switch_tick'] is not None
        recovered=bool(result['dagger']['expert_takeover_success']); accepted=name in entries
        f['takeovers']+=switched; f['untriggered']+=not switched
        f['expert_recovered']+=switched and recovered; f['valid_trajectories']+=accepted
        receipt=result['dagger']['trigger_receipt']
        require(not recovered or switched, 'Expert success without takeover')
        require(result['dagger']['student_autonomous_success'] == bool(result['success'] and not switched), 'Wrong autonomous success definition')
        if switched:
            never=bool(receipt['never_entered_15mm']); ever=bool(receipt['ever_entered_15mm']); left=bool(receipt['ever_entered_then_left_15mm'])
            require(never != ever and (not left or ever), 'Inconsistent gate history')
            f['never_entered_at_takeover']+=never;f['ever_entered_then_left_at_takeover']+=left
            f['ever_entered_still_in_gate_at_takeover']+=ever and not left
        compact=Path(result['dataset_capture']['path']) if result.get('dataset_capture') else None
        numeric=check_numeric(directory,result,compact_path=compact)
        scope=planning_scope(result,read(directory/'native_inference_timing.json'),numeric_schedule(directory))
        f['legal_expert_ticks']+=numeric['legal_expert_ticks']
        require(accepted == bool(result.get('accepted_for_training')), 'Admission flag differs')
        if compact is not None:
            hashes[name]=sha(compact)
            require(hashes[name] == result['dataset_capture']['sha256'], 'Complete capture SHA differs')
        if accepted:
            entry=entries[name]
            require(switched and recovered and result['success'] and numeric['legal_expert_ticks']>0, 'Unsafe correction')
            require(entry['source_group']==case['role'] and entry['opportunity']==case['opportunity']
                and entry['episode']==name and entry['family']==case['spec']['family']
                and entry['length']==result['steps'], 'Own correction identity differs')
            require(compact is not None and Path(entry['path']).resolve()==compact.resolve()
                and hashes[name]==entry['source_sha256'], 'Own correction capture binding differs')
            for field in ('never_entered_15mm','ever_entered_15mm','ever_entered_then_left_15mm'):
                require(entry[field+'_at_takeover']==receipt[field], 'Own correction gate history differs')
            admitted_capture_receipts(directory,result,compact,hashes[name])
        details.append(dict(case=name,role=case['role'],opportunity=case['opportunity'],
            evidence_directory=str(directory),takeover=switched,expert_recovered=recovered,
            accepted_for_own_pool=accepted,trigger_reason=receipt['reason'],**numeric,**scope))
    require(all(f['opportunities']==24 for role in funnels.values() for f in role.values()), 'Assigned24 denominator differs')
    import torch
    require(not torch.cuda.is_initialized(), 'CPU integrity audit initialized CUDA')
    return dict(version='paired_collection_integrity_recovery05_v1',passed=True,utc=time.time(),
        opportunities=96,paired_opportunities=48,observable_pair_checks=summary['observable_pair_checks'],
        full48_observed_initial_states_verified=False,production_summary=summary,
        by_assigned_opportunity=funnels,details=details,technical_failures=failures,
        complete_payload_sha256=hashes,admitted_payload_sha256={k:hashes[k] for k in entries},
        cuda_initialized=False,read_only_inputs=True)


def noise(collection, *, reconstruct_cuda=False, rng_auditor=None, prefix_reconstructor=native_cuda_prefix_state):
    plan, assignments, checks, pairs, rows, entries, summary = inputs(collection)
    rng_auditor = native_rng_auditor if rng_auditor is None else rng_auditor
    audited=[]; unavailable=[]; starts={}; scopes={}; available_receipts=[]
    for case,assigned,row in zip(plan['cases'],assignments,rows):
        directory=Path(row['evidence_directory']); result=row['result']
        available_receipts.append(dict(case=case['name'],start_receipt=(directory/'sampler_rng_start.json').exists(),
            end_receipt=(directory/'sampler_rng_end.json').exists(),
            complete_legal_execution=result.get('software_complete') is True and result.get('legal') is True))
        if result.get('software_complete') is not True or result.get('legal') is not True:
            unavailable.append(dict(case=case['name'],reason='No verified complete legal execution; endpoint consumption not certified',retained_in_denominator=True))
            continue
        batch=result['batch_execution']
        require(batch.get('lane')==assigned['lane'] and batch.get('queue_ordinal')==assigned['attempt_ordinal'], 'Actual attempt-local physical lane/order differs')
        scope=planning_scope(result,read(directory/'native_inference_timing.json'),numeric_schedule(directory))
        prior=verify_receipts(directory,case,scope['steps'],plan['visual_arithmetic'])
        value=rng_auditor(directory,scope['actual_native_plans'],reconstruct_cuda=reconstruct_cuda)
        require(value['receipt_hashes_verified'] is True and value['planning_calls']==scope['actual_native_plans'], 'Endpoint contract differs')
        if reconstruct_cuda:require(value['native_draw_reconstruction_verified'] is True, 'CUDA full endpoint differs')
        if starts:require(value['start_state']==next(iter(starts.values())), 'Assignment reseeding differs')
        starts[case['name']]=value['start_state'];scopes[case['name']]=scope
        boundary=prefix_reconstructor(scope['student_executed_plan_calls']) if reconstruct_cuda else None
        if reconstruct_cuda and scope['switch_tick'] is None:require(boundary==value['end_state'], 'Autonomous prefix/end differs')
        audited.append(dict(case=case['name'],role=case['role'],family=case['spec']['family'],opportunity=case['opportunity'],
            lane=assigned['lane'],ordinal=assigned['ordinal'],attempt=assigned['attempt'],attempt_ordinal=assigned['attempt_ordinal'],
            **scope,endpoint_rng=value,start_receipt_prior_controller_next_step=prior,
            reconstructed_student_boundary_state=boundary,observed_takeover_boundary_state_available=False,
            saved_end_scope='entire real episode including post-takeover student shadow calls'))
    paired=[]
    for (family,op),roles in pairs.items():
        a,b=roles['A']['name'],roles['B']['name'];both=a in starts and b in starts
        if both:require(starts[a]==starts[b], 'A/B verified RNG starts differ')
        paired.append(dict(family=family,opportunity=op,A=a,B=b,paired_start_equal=True if both else None,
            common_student_noise_draw_prefix=min(scopes[a]['student_executed_plan_calls'],scopes[b]['student_executed_plan_calls']) if both else None,
            reason=None if both else 'At least one consumed opportunity has no verified complete endpoint',later_draw_counts_may_differ=True))
    return dict(version='paired_collection_native_noise_recovery05_v1',utc=time.time(),assignments=96,
        paired_assignments=48,observable_pair_checks=summary['observable_pair_checks'],
        cpu_receipt_checks_passed=len(audited),uncertified_execution_endpoints=len(unavailable),
        exact_cuda_full_endpoint_checks=sum(bool(r['endpoint_rng']['native_draw_reconstruction_verified']) for r in audited),
        student_prefix_cuda_reconstructed=sum(r['reconstructed_student_boundary_state'] is not None for r in audited),
        observed_takeover_rng_states=0,independent_observed_takeover_rng_claim=False,
        sampler_seed=SEED,per_native_plan_draw_shape=list(SHAPE),commit_control_ticks=12,
        cuda_used=reconstruct_cuda,rows=audited,pairs=paired,unavailable=unavailable,available_receipts=available_receipts,
        all96_execution_endpoints_certified=not unavailable,
        scope='Only verified complete trajectories certify full endpoint consumption; incomplete opportunities stay in96. No fabricated boundary or observable-state match.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('integrity','noise','both'))
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reconstruct-cuda-noise',action='store_true')
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False);began=time.time()
    try:
        if a.mode in ('integrity','both'):
            (a.output/'integrity.json').write_text(json.dumps(integrity(a.collection),indent=2)+'\n')
        if a.mode in ('noise','both'):
            (a.output/'noise_audit.json').write_text(json.dumps(noise(a.collection,reconstruct_cuda=a.reconstruct_cuda_noise),indent=2)+'\n')
        print(json.dumps(dict(audited=True,wall_s=time.time()-began,output=str(a.output))))
    except BaseException as exc:
        (a.output/'failure.json').write_text(json.dumps(dict(error=repr(exc),utc=time.time(),wall_s=time.time()-began),indent=2)+'\n')
        raise


if __name__=='__main__':main()
