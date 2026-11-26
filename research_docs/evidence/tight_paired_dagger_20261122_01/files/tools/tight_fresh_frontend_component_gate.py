"""Finite B-component diagnostic: same-input admission, then two fresh Kit arms.

This tool never trains, collects expert actions, or registers a frontend-only
controller as a complete student. CPU preparation does not start CUDA or Kit.
The durable owner launches each registered stage once, with one original
session deadline and a hard 1800 second upper bound including cleanup.
"""
import argparse
import copy
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from tools.tight_fresh_selected_fit import read, sha256, checked_hash, exclusive_json

VERSION = 'B_frontend_projection_component_gate_v1'
STAGES = ('admission', 'full_B', 'projected_front_B')
LANES, TICKS, SEED = 8, 25, 28500000
TARGET_ATOL = 2e-5
BACKEND = dict(version='tight_execution_backend_v2', physics_device='cpu', camera_backend='standard',
    kit_tasking_threads=16, physics_threads=2, renderer_descriptor_sets=360000,
    renderer_reserved_descriptors=900000)


def require(condition, message):
    if not condition: raise ValueError(message)


def sources(tree):
    return {str(p.relative_to(tree)): sha256(p) for folder in ('tools', 'visuotactile')
        for p in sorted((tree/folder).rglob('*.py'))}


def registered_cases(development, preregistration, domain):
    from visuotactile.narrow_data import direct_configuration
    require(development.get('kind') == 'autonomous' and development.get('test_used') is False,
        'Completed non-test development registration required')
    cases = sorted((copy.deepcopy(c['spec']) for c in development['cases'] if c['role'] == 'B'),
        key=lambda c:c['seed'])
    expected = sorted(preregistration['development'], key=lambda c:c['seed'])
    require(len(cases) == len(expected) == LANES, 'Exactly the original eight development opportunities required')
    for index, (case, original) in enumerate(zip(cases, expected)):
        require(case['seed'] == original['seed'] == SEED+index and case['event_seed'] == SEED+100000+index,
            'Registered development seed/event population changed')
        require(case['family'] == original['family'] and case['split'] == 'development'
            and original['split'] == 'development_only', 'Development family/split changed')
        require(case['narrow_reset'] == original['narrow_reset'] == direct_configuration(domain, SEED+index),
            'Development initial configuration differs from its preregistration')
    require(development['execution_backend'] == BACKEND and development['batch']['lanes'] == LANES
        and development['batch']['vision_microbatch'] == 8
        and development['inference_scheduler_index_mode'] == 'host', 'Source development execution profile differs')
    return cases


def selected_train(data):
    original = data['normalization_train_ids']
    train = set(data['splits']['train'])
    heldout = set(data['splits'].get('validation', [])) | set(data['splits'].get('test', []))
    require(len(original) == len(set(original)) == 30 and set(original) <= train
        and not train & heldout, 'Original train30 and authoritative disjoint splits required')
    selected = sorted(original)[:LANES]
    heldout_families = {data['entries'][i]['family'] for i in heldout}
    for identity in selected:
        entry = data['entries'][identity]
        require(identity not in heldout and entry.get('split', 'train') == 'train'
            and entry.get('trajectory_kind') is None and entry['family'] not in heldout_families
            and entry['length'] >= TICKS, 'Only original training prefixes may enter admission')
    return selected


def source_batch_receipts(root, cases, pair):
    """Case seeds describe reset configurations; actual native draws use wave seeds."""
    receipts = {}; common = None
    for lane, case in enumerate(cases):
        path = root/'development'/('B_'+case['identity'])/'result.json'
        row = read(path); batch = copy.deepcopy(row['batch_execution'])
        require(row.get('software_complete') is True and row['complete_student_policy'] == pair
            and row['seed'] == case['seed'] and row['narrow_reset'] == case['narrow_reset'],
            'Prior B outcome does not bind the selected policy and registered initial configuration')
        require(batch.pop('lane') == lane and batch['lanes'] == LANES and batch['vision_microbatch'] == 8,
            'Prior actual batch lane/profile differs')
        require(batch['physics_seed'] == SEED+1 and batch['event_seed'] == SEED+100001 and batch['wave'] == 1,
            'Expected prior B wave1 native RNG receipt')
        require(common is None or batch == common, 'Prior B wave RNG receipts disagree')
        common = batch; receipts[str(path)] = sha256(path)
    return common, receipts


def prepare(root, resource_root, output, *, max_seconds=1800):
    import torch
    from tools.tight_fresh_rollouts import completed_pair
    from visuotactile.batched_frontend_diagnostic import selected_frontend, VERSION as FRONT_VERSION
    from visuotactile.position_control import REACHABLE_POSITION_EXECUTION
    from visuotactile.residual_output import REACHABLE_CENTERED_OUTPUT
    require(type(max_seconds) is int and 60 <= max_seconds <= 1800, 'Finite owner bound must be 60..1800 seconds')
    root, resource_root, output = (Path(p).resolve() for p in (root, resource_root, output))
    output.mkdir(parents=True, exist_ok=False)
    try:
        tree = Path(__file__).resolve().parents[1]
        session = read(resource_root/'session.json')
        require(time.time() < session['deadline'] and session['precision_policy'] == 'unified_fp32'
            and session['camera_content_validation'] is True, 'Live original FP32 validated-camera session required')
        pair = completed_pair(root, 'B')
        require(Path(pair['frontend']).resolve() == root/'B/training/best.pt'
            and pair['supervision'] == 'u_expert' and pair['modality'] == 'G00'
            and pair['execution_adapter'] == REACHABLE_POSITION_EXECUTION,
            'The selected round frontend and complete projected B rear must match')
        saved_front, selected = selected_frontend(pair['frontend'], pair['frontend_sha256'])
        rear = torch.load(pair['backend'], map_location='cpu', weights_only=False, mmap=True)
        require(rear['configuration'].get('output_contract') == REACHABLE_CENTERED_OUTPUT,
            'Full B must use the same nominal-centered final projection')
        del rear
        devpath = root/'development/plan.json'
        dev = read(devpath)
        require(dev['pairs']['B'] == pair and read(root/'development/result.json').get('software_complete') is True,
            'Completed source development does not bind this selected B pair')
        prereg = tree/'research_docs/evidence/tight_fresh_20261114/session06_preregistration/initial_families.json'
        cases = registered_cases(dev, read(prereg), session['domain'])
        datapath = root/'B/state_data/data.json'; data = read(datapath)
        require(saved_front['configuration']['data_sha256'] == sha256(datapath), 'Selected frontend training manifest differs')
        ids = selected_train(data)
        prior_batch, prior_receipts = source_batch_receipts(root, cases, pair)
        bindings = {str(p): sha256(p) for p in (resource_root/'session.json', devpath,
            root/'development/result.json', prereg, datapath, Path(pair['frontend']), Path(pair['backend']),
            root/'B/training/finished.json', root/'B/backend/B/training/finished.json',
            root/'B/backend/B/offline/result.json')}
        bindings.update(prior_receipts)
        for identity in ids:
            entry = data['entries'][identity]
            bindings[entry['path']] = checked_hash(entry['path'], entry['source_sha256'])
        for key in ('expert', 'dino'):
            bindings[session[key+'_checkpoint']] = checked_hash(session[key+'_checkpoint'], session[key+'_sha256'])
        plan = dict(version=VERSION, resource_root=str(resource_root), source_root=str(root), pair=pair,
            selected_frontend=selected, original_session_deadline=session['deadline'], max_seconds=max_seconds,
            source_tree=str(tree), source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip(),
            source_sha256=sources(tree), bound_files=bindings, session_sha256=sha256(resource_root/'session.json'),
            cases=cases, train_ids=ids, train_data=str(datapath), ticks_per_train_prefix=TICKS,
            execution_backend=BACKEND, lanes=LANES, vision_microbatch=8, neural_batch=True,
            inference_scheduler_index_mode='host', precision_policy='unified_fp32',
            batch_seed=prior_batch['physics_seed'], batch_event_seed=prior_batch['event_seed'],
            source_B_batch_execution=prior_batch,
            stages=list(STAGES), projection_mode='projection', frontend_controller_version=FRONT_VERSION,
            target_atol_rad=TARGET_ATOL, history_tolerance=dict(atol=2e-5, rtol=2e-5, discrete='exact'),
            native_termination='Unmodified native success/fall/timeout; same constructed bound in both fresh Kit arms',
            identities=dict(full_B=dict(complete_student_pair=True, pair=pair),
                projected_front_B=dict(complete_student_pair=False, rear_network_loaded=False,
                    frontend=pair['frontend'], frontend_sha256=pair['frontend_sha256'],
                    controller=FRONT_VERSION, mode='projection', execution_adapter=pair['execution_adapter'])),
            optimizer_updates=0, accepted_training_trajectories=0, test_used=False, automatic_retry=False,
            interpretation='Repeated development component diagnostic; removing rear also removes tactile feedback. Fixed full-first ordering; no causal success/generalization claim from an eight-case cohort.',
            rng='Both new arms use prior B wave1 actual physics_seed28500001/event_seed28600001, not the per-case reset specification seeds. Each uses a fresh Kit and fresh policy lane generators. Prior B ran after an A wave in one persistent Kit; its hidden state is not reproduced by matching seeds. Old outcomes remain separate.',
            cuda_initialized=torch.cuda.is_initialized(), prepared_utc=time.time())
        require(not plan['cuda_initialized'], 'Preparation must remain CPU only')
        exclusive_json(output/'plan.json', plan)
        return plan
    except BaseException as error:
        exclusive_json(output/'prepare_error.json', dict(error=repr(error), utc=time.time(), automatic_retry=False))
        raise


def verify(plan, tree, *, inputs=True):
    require(plan['version'] == VERSION and plan['stages'] == list(STAGES), 'Unknown component gate identity/stages')
    require(plan['projection_mode'] == 'projection' and plan['execution_backend'] == BACKEND
        and plan['lanes'] == LANES and plan['vision_microbatch'] == 8 and plan['target_atol_rad'] == TARGET_ATOL,
        'Registered component execution or acceptance threshold changed')
    require(type(plan['max_seconds']) is int and 60 <= plan['max_seconds'] <= 1800, 'Finite owner bound changed')
    require(sources(tree) == plan['source_sha256'], 'Frozen candidate source bytes/population changed')
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=tree,text=True).strip() == plan['source_commit'],
        'Frozen candidate commit changed')
    if inputs:
        for path, digest in plan['bound_files'].items(): checked_hash(path, digest)
    checked_hash(Path(plan['resource_root'])/'session.json', plan['session_sha256'])
    require(read(Path(plan['resource_root'])/'session.json')['deadline'] == plan['original_session_deadline'],
        'Original session deadline changed')


def check_bounds(plan, output, owner):
    if time.time() >= owner['work_deadline']: raise TimeoutError('Finite component gate deadline reached')
    if (Path(output)/'STOP').exists() or (Path(plan['resource_root'])/'STOP').exists():
        raise InterruptedError('Component gate STOP requested')


def stage_session(plan, directory):
    """Environment reads its local session; preserve original bytes and deadline."""
    source = Path(plan['resource_root'])/'session.json'
    checked_hash(source, plan['session_sha256'])
    with (directory/'session.json').open('xb') as stream:
        stream.write(source.read_bytes()); stream.flush(); os.fsync(stream.fileno())
    checked_hash(directory/'session.json', plan['session_sha256'])


def frontend_state(base):
    from tools.tight_fresh_forensics_gate import freeze
    # Full mixed command and pure projected command intentionally differ. After
    # the explicit source acknowledgement both histories must nevertheless agree.
    return freeze(dict(memory=base.memory, plan=base.plan, robot=base.last_robot,
        condition=base.last_condition, actual_previous=base.position_previous,
        next_step=base.next_step, episode=base.episode, started=base.started,
        generator=base.generator.get_state()))


def acknowledge_recorded_source(full, front, packet, predicted, target, active):
    """Offline-only teacher forcing after predictions/pending output are audited.

    This deliberately writes the candidate's transaction receipt because the
    compared predictions were never submitted to a simulator. Live simulate()
    never calls this function and requires the unchanged actual receipt path.
    """
    import torch
    require(torch.equal(front._pending_commands, predicted), 'Offline acknowledgement must start from untouched candidate commands')
    for lane, running in enumerate(active):
        if running:
            full.lanes[lane].base.command = target[lane:lane+1].clone()
            front.lanes[lane].command = target[lane:lane+1].clone()
    front._pending_commands = target.clone()
    transition = dict(position_requested_rad=target, position_applied_rad=target,
        episode_id=packet['episode_id'], step_id=packet['step_id'])
    full.observe_execution(transition, active); front.observe_execution(transition, active)


def run_same_input(full, front, samples, *, lease, cuda, progress=None):
    import torch
    from tools.tight_fresh_forensics_gate import freeze, rng_state
    from tools.tight_fresh_batch_gate import numeric_history_comparison, exact_state_equal
    require(len(samples) == LANES and all(len(s) == TICKS for s in samples), 'Eight complete 25-tick train prefixes required')
    rows = []; active = [True]*LANES
    for tick in range(TICKS):
        packet = {k: torch.cat([s[tick][0][k] for s in samples]) for k in samples[0][tick][0]}
        diag = {k: torch.cat([s[tick][1][k] for s in samples]) for k in samples[0][tick][1]}
        target = torch.cat([s[tick][2] for s in samples])
        require(bool((packet['step_id'] == tick).all()), 'Source history skipped a native tick')
        before = freeze(dict(packet=packet, diagnostics=diag, source_target=target))
        with lease.block():
            rng = rng_state(cuda)
            full.act(packet, diag, active)
            projected = front.act(packet, diag, active)
            require(exact_state_equal(before, freeze(dict(packet=packet, diagnostics=diag, source_target=target))),
                'Diagnostic changed shared recorded inputs')
            require(exact_state_equal(rng, rng_state(cuda)), 'Diagnostic consumed shared inference RNG')
            checks = []
            for lane in range(LANES):
                rear, pure = full.lanes[lane], front.lanes[lane]
                history = numeric_history_comparison(frontend_state(rear.base), frontend_state(pure))
                require(torch.equal(rear.base.generator.get_state(), pure.generator.get_state()), 'Independent sampler draw order differs')
                delta = float((rear.nominal-front.last_nominal[lane]).abs().max())
                require(delta <= TARGET_ATOL, 'Frontend nominal target parity failed')
                # Independent full rear compose implementation, zero proposal,
                # using the candidate nominal; no rear forward in the pure arm.
                nominal = front.last_nominal[lane].to(rear.device)
                expected, _ = rear.residual.compose(nominal, torch.zeros_like(nominal),
                    torch.zeros(1, dtype=torch.bool, device=rear.device), pure.limits.to(rear.device),
                    previous=pure.last_robot[None,58:87].to(rear.device))
                require(torch.equal(expected.cpu(), projected[lane:lane+1]), 'Pure target differs from full zero-proposal projection math')
                checks.append(dict(lane=lane, nominal_max_abs_rad=delta, history=history, projection_math_exact=True))
            # OFFLINE ONLY: replace both submitted histories with the same actual
            # recorded source execution. Neither prediction was executed in Kit.
            acknowledge_recorded_source(full, front, packet, projected, target, active)
            for lane in range(LANES): numeric_history_comparison(frontend_state(full.lanes[lane].base), frontend_state(front.lanes[lane]))
            require(full.visual_frames == front.visual_frames == [tick+1]*LANES, 'Every source frame must enter both histories')
        rows.append(dict(tick=tick, checks=checks, teacher_forced_recorded_target=True))
        if progress is not None: progress(rows[-1])
    report = front.execution_report()
    require(report['all_submitted_acknowledged'] and report['rear_forward_ticks'] == 0, 'Pure controller boundary failed')
    return dict(passed=True, rows=rows, source_ticks_per_lane=TICKS, projection_math='Exact full-rear compose with zero proposal; no rear forward in pure arm',
        frontend=report, full_vision=full.vision_execution_report(),
        scope='Train-only same-input current/past sensor history and recorded target acknowledgements; not autonomous execution')


def create_controllers(plan, metadata, dino, *, full=True, front=True):
    import torch
    from visuotactile.batched_controller import BatchedResidualController
    from visuotactile.batched_frontend_diagnostic import BatchedFrontendDiagnostic
    pair = plan['pair']; result = {}
    if full:
        saved = torch.load(pair['backend'], map_location='cpu', weights_only=False)
        result['full'] = BatchedResidualController(saved, metadata, dino, LANES,
            neural_batch=True, scheduler_index_mode='host', vision_microbatch=8)
        result['full'].lanes[0].model.eval().requires_grad_(False)
        result['full'].lanes[0].residual.eval().requires_grad_(False)
    if front:
        result['front'] = BatchedFrontendDiagnostic(pair['frontend'], metadata, dino, LANES,
            mode='projection', checkpoint_sha256=pair['frontend_sha256'], execution_adapter=pair['execution_adapter'])
    return result


def admission(plan, directory, lease):
    from tools.tight_fresh_forensics_gate import load_source
    from visuotactile.models.native_vision import NativeDinoSpatial
    data = read(plan['train_data'])
    require(selected_train(data) == plan['train_ids'], 'Train admission population changed')
    loaded = [load_source(data['entries'][i], i, TICKS) for i in plan['train_ids']]
    metadata = copy.deepcopy(loaded[0][0])
    metadata['joint_limits_rad'] = [copy.deepcopy(item[0]['joint_limits_rad'][0]) for item in loaded]
    require(all(item[0]['config']['action'] == metadata['config']['action'] for item in loaded), 'Original control metadata differs across train lanes')
    with lease.block():
        dino = NativeDinoSpatial().cuda().eval()
        controllers = create_controllers(plan, metadata, dino)
    from visuotactile.training_resources import atomic_json
    result = run_same_input(controllers['full'], controllers['front'], [item[1] for item in loaded], lease=lease, cuda=True,
        progress=lambda row: atomic_json(directory/'progress.json', dict(last_complete_tick=row, utc=time.time())))
    result.update(software_complete=True, version=VERSION, sources=[item[2] for item in loaded], optimizer_updates=0, test_used=False)
    exclusive_json(directory/'result.json', result)


def counts(controller, lane, steps, stage):
    from tools.tight_fresh_batch_eval import inference_counts
    report = controller.inference_metrics(lane)
    result = inference_counts(report, steps, controller.visual_frames[lane], strict=stage == 'full_B')
    require(all(result[k] == steps for k in ('visual_frames','history_updates','robot_observations'))
        and result['plans'] == (steps+11)//12 and result['backend_updates'] == (steps if stage == 'full_B' else 0),
        'Component inference counts differ from each real control tick')
    return report, result


def simulate(plan, directory, lease, stage):
    import h5py
    import numpy as np
    import torch
    from tools.tight_state import environment
    from tools.tight_fresh_batch_eval import check_sensor_clock_with_evidence, lane_outcome
    from tools.tight_fresh_stage_takeover import native_bound_guard
    from visuotactile.batched_controller import lane_tree
    from visuotactile.environment_rng import IsolatedEnvironment
    from visuotactile.live_collision import clearance
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.narrow_data import configure_direct_reset_lanes
    from visuotactile.recording import BufferedNumericWriter
    from visuotactile.training_resources import atomic_json
    began = time.perf_counter()
    app, env, player, lease = environment(directory, evaluation=True, lease=lease, num_envs=LANES,
        execution_backend=plan['execution_backend'], camera_content_validation=True, startup_diagnostics=True)
    atomic_json(directory/'startup_timing.json', dict(seconds=time.perf_counter()-began))
    configure_direct_reset_lanes(env, [c['narrow_reset'] for c in plan['cases']])
    isolated = IsolatedEnvironment(env, plan['batch_seed'])
    with lease.block():
        env.clear_recorded_events(); env.action_control_mode = 'native_command_clipped_pre_delay'
        isolated.reset(plan['batch_seed']); isolated.begin_events(plan['batch_event_seed'])
        env.action_control_mode = 'absolute_joint_position_target_rad'
        initial = env._diagnostics()
        atomic_json(directory/'initial_observable.json', {k:v.detach().cpu().tolist() for k,v in initial.items()})
        legal = [clearance(env, lane=i) for i in range(LANES)]
        atomic_json(directory/'initial_collision.json', legal)
        bound = native_bound_guard(env); atomic_json(directory/'native_bound.json', bound)
        require(abs(float(env.physics_dt)-1/120) < 1e-10 and abs(float(env.step_dt)-1/60) < 1e-10, 'Native clock changed')
        metadata = env.research_metadata(); metadata['config'] = env.cfg.to_dict()
        atomic_json(directory/'metadata.json', metadata)
        dino = NativeDinoSpatial().cuda().eval()
        controller = create_controllers(plan, metadata, dino, full=stage == 'full_B', front=stage != 'full_B')[
            'full' if stage == 'full_B' else 'front']
    active = [bool(row['legal']) for row in legal]
    files, writers, lane_dirs = [], [], []
    for lane, case in enumerate(plan['cases']):
        path = directory/case['identity']; path.mkdir(); lane_dirs.append(path)
        f = h5py.File(path/'numeric.h5', 'x'); files.append(f)
        writers.append(BufferedNumericWriter(f.create_group('episode'), 32))
    steps = [0]*LANES; reasons = [None]*LANES; flags = [None]*LANES; frames = [None]*LANES
    maxima = [dict(native_lift=False, native_alignment=False, native_subgoals=0, position_target_max_error=0.) for _ in range(LANES)]
    profile = dict(controller_s=0., environment_s=0., recording_s=0.); start = time.perf_counter()
    try:
        for tick in range(bound['collector_max_control_ticks']):
            if not any(active): break
            packet = env.research_packet
            check_sensor_clock_with_evidence(packet, active, frames, directory=directory, env=env, control_tick=tick)
            with lease.block():
                before = time.perf_counter(); command = controller.act(packet, env._diagnostics(), active)
                profile['controller_s'] += time.perf_counter()-before
                before = time.perf_counter(); _, _, terminated, truncated, _ = isolated.step(command)
                profile['environment_s'] += time.perf_counter()-before
                transition = env.last_transition
                controller.observe_execution(transition, active)
                before = time.perf_counter()
                numeric = dict(obs={k:v for k,v in packet.items() if torch.is_tensor(v)
                    and k not in ('image','depth_m','depth_valid')}, transition=transition)
                for lane, running in enumerate(active):
                    if not running: continue
                    require(int(packet['step_id'][lane]) == steps[lane]
                        and int(transition['step_id'][lane]) == steps[lane]
                        and int(transition['episode_id'][lane]) == int(packet['episode_id'][lane]), 'Native observation/transition clock changed')
                    require(abs(float(transition['time_before_s'][lane])-float(packet['time_s'][lane])) < 1e-8
                        and abs(float(transition['time_after_s'][lane]-transition['time_before_s'][lane])-1/60) < 1e-8,
                        'A control action did not advance the actual native 60Hz clock')
                    record = lane_tree(numeric, lane, LANES)
                    base = controller.lanes[lane].base if stage == 'full_B' else controller.lanes[lane]
                    nominal = controller.lanes[lane].nominal if stage == 'full_B' else controller.last_nominal[lane]
                    record['component'] = dict(nominal_rad=nominal, submitted_rad=command[lane:lane+1],
                        submitted_minus_nominal_rad=command[lane:lane+1]-nominal,
                        current_plan_rad=base.plan, plan_cursor=torch.tensor([tick%12]))
                    writers[lane].append(record, steps[lane]); steps[lane] += 1
                    after = transition['diagnostics_after']; row = maxima[lane]
                    row['native_lift'] |= bool(after['lifted_object'][lane])
                    row['native_alignment'] |= bool(transition['native_alignment_condition'][lane])
                    row['native_subgoals'] = max(row['native_subgoals'], int(after['successes'][lane]))
                    difference = float((command[lane]-transition['target_after_rad'][lane]).abs().max())
                    require(np.isfinite(difference) and difference <= 2e-5, 'Requested absolute target differs from native execution')
                    row['position_target_max_error'] = max(row['position_target_max_error'], difference)
                    reason = {k:bool(v[lane]) for k,v in transition['termination_reasons'].items()}
                    require(not reason['max_successes'] or bool(terminated[lane]), 'Native success/termination disagreement')
                    if bool((terminated|truncated)[lane]):
                        reasons[lane] = reason; flags[lane] = dict(terminated=bool(terminated[lane]), truncated=bool(truncated[lane])); active[lane] = False
                profile['recording_s'] += time.perf_counter()-before
            atomic_json(directory/'progress.json', dict(control_tick=tick+1, steps=steps, active=active, utc=time.time()))
        require(not any(active), 'Native finite bound exhausted before natural termination')
    except BaseException as error:
        atomic_json(directory/'partial_result.json', dict(software_complete=False, error=repr(error),
            steps=steps, active=active, reasons=reasons, native_termination_flags=flags,
            observed_stage_maxima=maxima, initial_admissions=legal, accepted_training_trajectories=0,
            automatic_retry=False, utc=time.time()))
        raise
    finally:
        for writer, f in zip(writers, files):
            try: writer.flush()
            finally: f.close()
    results = []
    for lane, path in enumerate(lane_dirs):
        result = dict(lane=lane, spec=plan['cases'][lane], legal=legal[lane]['legal'], steps=steps[lane],
            reasons=reasons[lane], native_termination_flags=flags[lane], software_complete=True,
            controller_identity=plan['identities'][stage], **maxima[lane])
        result.update(lane_outcome(legal=result['legal'], steps=steps[lane], reasons=reasons[lane], still_active=False,
            max_control_ticks=None, native_bound=bound['collector_max_control_ticks']))
        # Always label this repeated component cohort; never merge formal scores.
        result.update(component_diagnostic=True, diagnostic_only=True,
            autonomous_evaluation_eligible=False, formal_evaluation_eligible=False, accepted_for_training=False)
        if steps[lane]:
            inference, result['inference_counts'] = counts(controller, lane, steps[lane], stage)
            atomic_json(path/'inference.json', inference)
        result['numeric_sha256'] = sha256(path/'numeric.h5')
        atomic_json(path/'result.json', result); results.append(result)
    if stage == 'projected_front_B':
        execution = controller.execution_report()
        require(execution['all_submitted_acknowledged'] and execution['rear_forward_ticks'] == 0
            and [r['acknowledged_control_ticks'] for r in execution['lanes']] == steps, 'Pure inference receipt incomplete')
        torch.save(controller.plan_records, directory/'frontend_plans.pt')
    else: execution = controller.vision_execution_report()
    profile.update(loop_and_flush_s=time.perf_counter()-start, wall_s=time.perf_counter()-began,
        control_ticks=sum(steps), aggregate_ticks_per_second=sum(steps)/(time.perf_counter()-start),
        resource_wait_s=getattr(lease, 'wait_s', 0.),
        timing_scope='Unsynchronized host sections; aggregate wall includes writers. Not isolated kernel attribution.')
    atomic_json(directory/'result.json', dict(version=VERSION, software_complete=True, results=results,
        profile=profile, execution=execution, stage=stage, test_used=False, accepted_training_trajectories=0,
        diagnostic_only=True, autonomous_evaluation_eligible=False,
        formal_evaluation_eligible=False, complete_student_pair=stage == 'full_B'))


def worker(output, stage):
    import torch
    from visuotactile.training_resources import atomic_json
    output = Path(output).resolve(); plan = read(output/'plan.json'); owner = read(output/'owner_started.json')
    require(stage in STAGES and owner['plan_sha256'] == sha256(output/'plan.json'), 'Worker registration changed')
    verify(plan, Path(__file__).resolve().parents[1]); check_bounds(plan, output, owner)
    directory = output/stage
    checked_hash(directory/'session.json', plan['session_sha256'])
    exclusive_json(directory/'started.json', dict(pid=os.getpid(), utc=time.time(), stage=stage, plan_sha256=owner['plan_sha256']))
    try:
        if stage != 'admission': require(read(output/'admission/result.json').get('passed') is True, 'CUDA admission did not pass')
        atomic_json(directory/'resource_before.json', read(Path(plan['resource_root'])/'shared_resources.json'))
        from visuotactile.state_resources import lease_for
        from visuotactile.precision import state_fp32, snapshot
        require(torch.cuda.is_available(), 'Actual CUDA required; CPU tests never pass live admission')
        state_fp32(); atomic_json(directory/'precision.json', snapshot())
        real = lease_for(Path(plan['resource_root']), deadline=owner['work_deadline'])
        class BoundedLease:
            def __getattr__(self, key): return getattr(real, key)
            def check(self): check_bounds(plan, output, owner); real.check()
            @contextmanager
            def block(self, **kwargs):
                self.check()
                with real.block(**kwargs): yield
                self.check()
        lease = BoundedLease()
        if stage == 'admission':
            admission(plan, directory, lease)
            atomic_json(directory/'worker_memory.json', lease.memory())
            atomic_json(directory/'resource_after.json', read(Path(plan['resource_root'])/'shared_resources.json'))
        else:
            simulate(plan, directory, lease, stage)
            atomic_json(directory/'worker_memory.json', lease.memory())
            atomic_json(directory/'resource_after.json', read(Path(plan['resource_root'])/'shared_resources.json'))
            from visuotactile.native_evaluation import exit_completed_kit_worker
            exit_completed_kit_worker(directory)
    except BaseException as error:
        atomic_json(directory/'worker_error.json', dict(error=repr(error), traceback=traceback.format_exc(), utc=time.time(), automatic_retry=False))
        raise


def initial_comparison(output):
    import numpy as np
    left, right = [read(output/stage/'initial_observable.json') for stage in STAGES[1:]]
    require(left.keys() == right.keys(), 'Initial observable field populations differ')
    fields = {}
    for key in left:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        require(a.shape == b.shape, 'Initial observable lane shape differs')
        fields[key] = dict(exact=bool(np.array_equal(a,b)), maximum_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float)))) if a.size else 0.)
    bounds = [read(output/stage/'native_bound.json') for stage in STAGES[1:]]
    require(bounds[0] == bounds[1], 'Two arms constructed different native termination bounds')
    return dict(fields=fields, all_observables_equal=all(row['exact'] for row in fields.values()),
        native_bound=bounds[0], interpretation='Visible initial-state comparison only; independent Kit hidden physical state equivalence is not established')


def owner(output, worker_tree):
    from tools.tight_batch_owner import bind_worker_group, cleanup_owned_group
    from tools.tight_fresh_eval_recovery import stop_owned
    from visuotactile.training_resources import atomic_json
    start = time.time()
    output = Path(output).resolve(); worker_tree = Path(worker_tree).resolve()
    plan = read(output/'plan.json'); verify(plan, worker_tree, inputs=False)
    deadline = min(plan['original_session_deadline'], start+plan['max_seconds'])
    require(deadline-start >= 60, 'Insufficient original deadline for finite component gate')
    started = dict(pid=os.getpid(), utc=start, deadline=deadline, work_deadline=deadline-30,
        plan_sha256=sha256(output/'plan.json'), worker_tree=str(worker_tree), stages=list(STAGES), automatic_retry=False)
    exclusive_json(output/'owner_started.json', started)
    complete = False; error = None; rows = []; handlers = {}
    def interrupted(signum, frame): raise SystemExit(128+signum)
    try:
        for sig in (signal.SIGTERM, signal.SIGINT): handlers[sig] = signal.signal(sig, interrupted)
        for stage in STAGES:
            check_bounds(plan, output, started)
            directory = output/stage; directory.mkdir(); stage_session(plan, directory)
            command = [sys.executable,'-m','tools.tight_fresh_frontend_component_gate','worker','--output',str(output),'--stage',stage]
            process = binding = None; cleanup = None
            try:
                with (directory/'worker.log').open('x') as stream:
                    process = subprocess.Popen(command, cwd=worker_tree, stdin=subprocess.DEVNULL, stdout=stream,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        env=dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONFAULTHANDLER='1'))
                    binding = bind_worker_group(process.pid)
                    atomic_json(directory/'owner_worker_started.json', dict(binding=binding, command=command, utc=time.time()))
                    while process.poll() is None:
                        check_bounds(plan, output, started); time.sleep(.25)
                result = read(directory/'result.json') if (directory/'result.json').is_file() else {}
                require(process.returncode == 0 and result.get('software_complete') is True
                    and (stage != 'admission' or result.get('passed') is True), 'Component stage failed: '+stage)
            finally:
                if process is not None:
                    try:
                        stop_owned(process)
                        cleanup = cleanup_owned_group(binding) if binding else dict(live_descendants_exited=False, error='Missing process binding')
                    except BaseException as caught: cleanup = dict(live_descendants_exited=False, error=repr(caught))
                    atomic_json(directory/'owner_cleanup.json', cleanup)
            require(cleanup is not None and cleanup['live_descendants_exited'], 'Owned worker cleanup incomplete')
            rows.append(dict(stage=stage, complete=True, result_sha256=sha256(directory/'result.json')))
            atomic_json(output/'owner_progress.json', dict(completed_stages=rows, utc=time.time()))
        require(sha256(output/'plan.json') == started['plan_sha256'], 'Plan changed during execution')
        comparison = initial_comparison(output)
        atomic_json(output/'result.json', dict(version=VERSION, software_complete=True, stages=rows,
            initial_observables=comparison, all_component_outcomes_retained=True,
            scope='Independent B full versus projected frontend development diagnostic; no success threshold for software completion',
            accepted_training_trajectories=0, optimizer_updates=0, test_used=False))
        complete = True
    except BaseException as caught: error = repr(caught)
    finally:
        for sig, handler in handlers.items(): signal.signal(sig, handler)
        atomic_json(output/'owner_finished.json', dict(complete=complete, error=error, completed_stages=rows,
            seconds=time.time()-start, deadline=deadline, original_session_deadline=plan['original_session_deadline'], automatic_retry=False))
    require(complete, 'Bounded component diagnostic failed; preserve evidence, no automatic retry')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('root','resource-root','output'): prep.add_argument('--'+name, type=Path, required=True)
    prep.add_argument('--max-seconds', type=int, default=1800)
    own = sub.add_parser('owner'); own.add_argument('--output', type=Path, required=True); own.add_argument('--worker-tree', type=Path, required=True)
    work = sub.add_parser('worker'); work.add_argument('--output', type=Path, required=True); work.add_argument('--stage', choices=STAGES, required=True)
    args = vars(parser.parse_args()); command = args.pop('command')
    result = globals()[command](**args)
    if command == 'prepare': print(json.dumps(dict(prepared=True, cuda_initialized=result['cuda_initialized'], stages=result['stages'])), flush=True)


if __name__ == '__main__': main()
