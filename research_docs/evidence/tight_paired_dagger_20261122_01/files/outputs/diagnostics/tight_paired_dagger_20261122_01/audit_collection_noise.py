"""Audit real collection RNG endpoints; CUDA reconstruction is explicit opt-in.

The frozen queue keeps computing the student in shadow AFTER expert takeover.
Saved RNG end is at episode end, NOT takeover. Full endpoint consumption is
verified separately from the student-executed prefix. No hidden-physics claim.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
DEFAULT_ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
SOURCES = ('tools/tight_paired_collection.py', 'tools/tight_paired_evaluate.py',
           'visuotactile/queued_rollout.py', 'visuotactile/gpu_batch_controller.py',
           'visuotactile/models/native_dp.py', 'visuotactile/sampler_audit.py')
SEED = 7091
SHAPE = (1, 24, 29)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def count(value, name):
    require(type(value) is int and value >= 0, name+' must be a nonnegative integer')
    return value


def assignment_checks(plan, assignments, initial):
    cases = plan['cases']
    require(plan.get('stage_takeover_contract') == 'paired_own_student_recovery_collection_v1'
            and plan.get('kind') == 'takeover' and plan.get('test_used') is False,
            'Exact training-only collection contract required')
    require(plan.get('sampler_seed') == SEED and plan.get('record_sampler_rng') is True
            and plan.get('inference_scheduler_index_mode') == 'host', 'Frozen noise contract changed')
    require(len(cases) == 96 and len({c['name'] for c in cases}) == 96, 'Exactly96 unique cases required')
    require(len(assignments) == 96, 'Exactly96 actual assignments required')
    keys, names = {}, set()
    for ordinal, (case, assigned) in enumerate(zip(cases, assignments)):
        require(assigned.get('case') == case['name'] and assigned.get('role') == case['role']
                and assigned.get('ordinal') == ordinal and assigned.get('sampler_seed') == SEED,
                'Actual queue assignment differs from frozen case/ordinal/role/seed')
        require(type(assigned.get('lane')) is int and 0 <= assigned['lane'] < plan['batch']['lanes'],
                'Invalid actual lane assignment')
        spec = case['spec']
        require(spec.get('identity') == case['name'] and case['role'] in ('A', 'B')
                and case['opportunity'] in ('support', 'alignment'), 'Invalid case identity/role/opportunity')
        key = (spec['family'], case['opportunity'])
        require(case['role'] not in keys.setdefault(key, {}), 'Duplicate pair role')
        keys[key][case['role']] = case
        names.add(case['name'])
    require(len(keys) == 48 and len({key[0] for key in keys}) == 24, 'Exactly24 families/48 pairs required')
    for key, pair in keys.items():
        require(set(pair) == {'A', 'B'}, 'Missing A/B partner')
        for field in ('seed', 'event_seed', 'narrow_reset', 'initial_configuration_sha256'):
            require(pair['A']['spec'][field] == pair['B']['spec'][field], 'Paired reset/event differs')
    require(initial.get('passed') is True and len(initial['rows']) == 48, 'Exactly48 passed observed-state checks required')
    checked = set()
    for row in initial['rows']:
        key = (row['family'], row['opportunity'])
        require(key in keys and key not in checked and row.get('passed') is True, 'Duplicate/failed observed-state pair')
        require({row.get('reference'), row.get('case')} == {c['name'] for c in keys[key].values()},
                'Observed-state check names do not match assigned pair')
        checked.add(key)
    return keys


def numeric_schedule(directory):
    import h5py
    with h5py.File(Path(directory)/'numeric.h5', 'r') as handle:
        return dict(step_id=handle['episode/obs/step_id'][:].tolist(),
                    control_source=handle['episode/transition/dagger/control_source'][:].tolist())


def planning_scope(result, timing, numeric):
    steps = count(result.get('steps'), 'steps')
    require(steps > 0 and result.get('software_complete') is True and result.get('legal') is True,
            'Full real legal software-complete trajectory required; never replace missing attempts')
    ticks = numeric['step_id']
    sources = numeric['control_source']
    require(ticks == list(range(steps)) and len(sources) == steps, 'Native ticks skipped/duplicated or source length changed')
    require(all(type(source) in (int, bool) and source in (0, 1) for source in sources), 'Unknown source code')
    switch = next((tick for tick, source in enumerate(sources) if source), None)
    prefix = steps if switch is None else switch
    require(sources == [0]*prefix+[1]*(steps-prefix), 'Student/expert source is not one contiguous switch')
    dagger = result['dagger']
    require(dagger.get('recovery_coverage', {}).get('switch_tick') == switch, 'Recorded switch disagrees with actual numeric source')
    require(dagger.get('student_ticks') == prefix and dagger.get('expert_ticks') == steps-prefix,
            'Recorded source counts disagree with numeric trace')
    total_plan_ticks = [tick for tick in ticks if tick % 12 == 0]
    student_plan_ticks = [tick for tick in total_plan_ticks if tick < prefix]
    observed = timing['batch_observed_counts']
    require(result['native_inference']['batch_observed_counts'] == observed, 'Timing/result native counts disagree')
    for key in ('history_updates', 'robot_observations', 'backend_updates'):
        require(observed[key] == steps, 'Observed native complete history count differs')
    require(observed['plans'] == len(total_plan_ticks), 'Actual native plan counter differs from recorded tick schedule')
    return dict(steps=steps, actual_native_plans=count(observed['plans'], 'native plans'), switch_tick=switch,
                student_executed_ticks=prefix, student_executed_plan_calls=len(student_plan_ticks),
                shadow_after_switch_plan_calls=len(total_plan_ticks)-len(student_plan_ticks),
                student_plan_ticks=student_plan_ticks, all_plan_ticks=total_plan_ticks,
                student_scope_end='expert_switch_before_first_expert_action' if switch is not None else 'autonomous_native_outcome')


def verify_receipts(directory, case, steps, arithmetic):
    prior_next_step = None
    for stage in ('start', 'end'):
        receipt = read(directory/f'sampler_rng_{stage}.json')
        require(receipt.get('identity') == case['name'] and receipt.get('stage') == stage,
                'RNG receipt identity/stage differs')
        if stage == 'end':
            require(receipt.get('next_step') == steps, 'RNG end/controller clock differs')
        else:
            # Reassigned lane generator is reseeded before recording, while the
            # history clock resets only on first act(). This diagnostic field
            # can still name the previous episode length; it is not RNG age.
            prior_next_step = count(receipt.get('next_step'), 'start prior-controller next_step')
        require(receipt.get('state_path') == f'sampler_rng_{stage}.pt', 'Unexpected RNG state path')
        require(receipt.get('initial_seed') == SEED and receipt.get('checkpoint_sampler_seed') == SEED
                and receipt.get('extra_random_draws') == 0, 'RNG receipt seed/draw contract differs')
        require(receipt.get('vision_arithmetic') == arithmetic, 'Recorded visual arithmetic differs')
        require(sha(directory/receipt['state_path']) == receipt.get('state_sha256'), 'Saved RNG state SHA differs')
    return prior_next_step


def native_cuda_prefix_state(draws):
    """Only reached after explicit CUDA flag; NOT an observed boundary state."""
    import torch
    generator = torch.Generator(device='cuda:0').manual_seed(SEED)
    for _ in range(draws):
        torch.randn(SHAPE, device='cuda:0', generator=generator)
    return generator.get_state().cpu().tolist()


def native_rng_auditor(directory, plans, *, reconstruct_cuda):
    """Reuse eval helper, adding collection state-payload metadata checks."""
    import torch
    from tools.tight_paired_evaluate import audit_rng
    for stage in ('start', 'end'):
        receipt = read(directory/f'sampler_rng_{stage}.json')
        payload = torch.load(directory/receipt['state_path'], map_location='cpu', weights_only=False)
        state = payload.get('generator_state')
        require(torch.is_tensor(state) and state.dtype == torch.uint8 and state.ndim == 1
                and state.numel() == receipt.get('state_bytes'), 'Saved generator state layout differs')
        require(payload.get('initial_seed') == SEED and payload.get('device') == 'cuda:0',
                'Saved generator payload seed/device differs')
    return audit_rng(directory, plans, reconstruct_cuda=reconstruct_cuda)


def audit(output, *, reconstruct_cuda=False, rng_auditor=None, numeric_loader=numeric_schedule,
          prefix_reconstructor=native_cuda_prefix_state, project=PROJECT, verify_sources=True):
    output = Path(output).resolve()
    plan = read(output/'plan.json')
    registration = read(output/'registration.json')
    require(registration.get('plan_sha256') == sha(output/'plan.json'), 'Frozen plan SHA differs')
    bindings = {}
    if verify_sources:
        for source in SOURCES:
            expected = plan['source_sha256'][source]
            require(sha(Path(project)/source) == expected, 'Frozen source differs: '+source)
            bindings[source] = expected
    assignments = read(output/'paired_queue_assignments.json')
    pairs = assignment_checks(plan, assignments, read(output/'paired_initial_check.json'))
    raw_results = read(output/'results.json')
    require(len(raw_results) == 96 and len({r['case'] for r in raw_results}) == 96, 'Full96 distinct results required')
    by_name = {row['case']: row for row in raw_results}
    require(set(by_name) == {c['name'] for c in plan['cases']}, 'Unknown/missing result cases')
    if rng_auditor is None:
        # Existing helper loads state tensors on CPU unless explicitly requested.
        rng_auditor = native_rng_auditor
    starts, scopes, rows = {}, {}, []
    for case, assigned in zip(plan['cases'], assignments):
        directory = output/case['name']
        result = read(directory/'result.json')
        require(by_name[case['name']]['role'] == case['role'] and by_name[case['name']]['result'] == result,
                'Aggregate/individual result identity differs')
        require(result.get('identity') == case['name'] and result.get('family') == case['spec']['family'], 'Result case differs')
        batch = result['batch_execution']
        require(batch.get('lane') == assigned['lane'] and batch.get('queue_ordinal') == assigned['ordinal'], 'Actual lane/ordinal differs')
        scope = planning_scope(result, read(directory/'native_inference_timing.json'), numeric_loader(directory))
        prior_next_step = verify_receipts(directory, case, scope['steps'], plan['visual_arithmetic'])
        noise = rng_auditor(directory, scope['actual_native_plans'], reconstruct_cuda=reconstruct_cuda)
        require(noise['receipt_hashes_verified'] is True and noise['planning_calls'] == scope['actual_native_plans'], 'Endpoint helper contract differs')
        if reconstruct_cuda:
            require(noise['native_draw_reconstruction_verified'] is True, 'Exact CUDA endpoint reconstruction failed')
        # Every assignment is explicitly reseeded, not merely each lane's first case.
        if starts:
            require(noise['start_state'] == next(iter(starts.values())), 'Assignment was not reseeded to same native noise start')
        starts[case['name']] = noise['start_state']
        scopes[case['name']] = scope
        boundary = None
        if reconstruct_cuda:
            boundary = prefix_reconstructor(scope['student_executed_plan_calls'])
            if scope['switch_tick'] is None:
                require(boundary == noise['end_state'], 'Autonomous boundary/end mismatch')
        rows.append(dict(case=case['name'], role=case['role'], family=case['spec']['family'],
                         opportunity=case['opportunity'], lane=assigned['lane'], ordinal=assigned['ordinal'],
                         **scope, endpoint_rng=noise,
                         start_receipt_prior_controller_next_step=prior_next_step,
                         reconstructed_student_boundary_state=boundary,
                         observed_takeover_boundary_state_available=False,
                         saved_end_scope='entire real episode including post-takeover student shadow calls'))
    paired = []
    for (family, opportunity), roles in pairs.items():
        a, b = roles['A']['name'], roles['B']['name']
        require(starts[a] == starts[b], 'A/B pair RNG starts differ')
        shared = min(scopes[a]['student_executed_plan_calls'], scopes[b]['student_executed_plan_calls'])
        paired.append(dict(family=family, opportunity=opportunity, A=a, B=b,
                           paired_start_equal=True, common_student_noise_draw_prefix=shared,
                           later_draw_counts_may_differ=True))
    return dict(version='paired_collection_native_noise_audit_v1', utc=time.time(), output=str(output),
                plan_sha256=sha(output/'plan.json'), source_sha256=bindings, assignments=96, paired_assignments=48,
                sampler_seed=SEED, per_native_plan_draw_shape=list(SHAPE), commit_control_ticks=12,
                cpu_receipt_checks_passed=96, exact_cuda_full_endpoint_checks=sum(r['endpoint_rng']['native_draw_reconstruction_verified'] for r in rows),
                student_prefix_cuda_reconstructed=sum(r['reconstructed_student_boundary_state'] is not None for r in rows),
                observed_takeover_rng_states=0, independent_observed_takeover_rng_claim=False,
                cuda_used=reconstruct_cuda, rows=rows, pairs=paired,
                scope='Student execution ends at actual source switch; observed RNG end remains full episode. Prefix boundary is inferred from verified native draw sequence, not independently saved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, default=DEFAULT_ROOT/'collection')
    parser.add_argument('--output', type=Path, required=True, help='New audit evidence directory; existing attempts never overwritten')
    parser.add_argument('--reconstruct-cuda-noise', action='store_true', help='Explicit tiny CUDA use; run only after external tests are quiet')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.time()
    try:
        result = audit(args.collection, reconstruct_cuda=args.reconstruct_cuda_noise)
        result['wall_s'] = time.time()-began
        (args.output/'noise_audit.json').write_text(json.dumps(result, indent=2)+'\n')
        with (args.output/'per_assignment.csv').open('w', newline='') as stream:
            fields = ['case', 'role', 'family', 'opportunity', 'lane', 'ordinal', 'steps', 'actual_native_plans',
                      'switch_tick', 'student_executed_ticks', 'student_executed_plan_calls', 'shadow_after_switch_plan_calls']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({key: row[key] for key in fields} for row in result['rows'])
        print(json.dumps({key: result[key] for key in ('assignments', 'paired_assignments', 'cpu_receipt_checks_passed', 'exact_cuda_full_endpoint_checks', 'observed_takeover_rng_states', 'wall_s')}))
    except BaseException as exc:
        (args.output/'failure.json').write_text(json.dumps(dict(error=repr(exc), utc=time.time(), wall_s=time.time()-began), indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
