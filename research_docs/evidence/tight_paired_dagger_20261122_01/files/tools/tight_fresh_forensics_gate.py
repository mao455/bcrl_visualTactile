"""Strict optional-recorder parity on real RGB-D and complete frozen students.

This is an offline inference gate, not a simulator rollout or a speed benchmark.
Every native observation is consumed. Recorded actual targets are acknowledged
as teacher-forced history identically in both arms; predicted targets are checked
before that acknowledgement. A mismatch fails admission, with no tolerance.
"""
import argparse
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import h5py
import torch

from visuotactile.compact import sha256
from visuotactile.first_plan_forensics import FirstPlanRecorder, tensor_snapshot
from visuotactile.training_resources import atomic_json


def flatten(value, prefix='root'):
    if is_dataclass(value):
        value = {f.name: getattr(value, f.name) for f in fields(value)}
    elif isinstance(value, SimpleNamespace):
        value = vars(value)
    if isinstance(value, dict):
        answer = {prefix: ('dict', sorted(value))}
        for key in sorted(value):
            answer.update(flatten(value[key], prefix+'/'+key))
        return answer
    if isinstance(value, (tuple, list)):
        answer = {prefix: (type(value).__name__, len(value))}
        for index, item in enumerate(value):
            answer.update(flatten(item, prefix+'/'+str(index)))
        return answer
    if torch.is_tensor(value) or value is None or isinstance(value, (str, int, float, bool)):
        return {prefix: value}
    raise TypeError('Unsupported gate state: '+type(value).__name__)


def freeze(value):
    """Read-only, CPU snapshots; never retain mutable controller references."""
    return {key: item.detach().cpu().clone() if torch.is_tensor(item) else item
        for key, item in flatten(value).items()}


def compare(name, left, right):
    mismatches = []; identities = []
    for path in sorted(set(left) | set(right)):
        if path not in left or path not in right:
            mismatches.append(dict(path=path, reason='missing_state_field')); continue
        a, b = left[path], right[path]
        if torch.is_tensor(a) and torch.is_tensor(b):
            x, dx = tensor_snapshot(a); y, dy = tensor_snapshot(b)
            identities.append((path, dx['sha256'], dy['sha256']))
            if dx['sha256'] != dy['sha256']:
                error = None
                if x.shape == y.shape and (x.is_floating_point() or y.is_floating_point()):
                    delta = (x.double()-y.double()).abs()
                    if torch.isfinite(delta).all():
                        error = float(delta.max()) if delta.numel() else 0.
                mismatches.append(dict(path=path, reason='tensor_bytes_shape_or_dtype_differ',
                    left=dx, right=dy, max_abs_difference=error))
        elif torch.is_tensor(a) or torch.is_tensor(b) or type(a) is not type(b) or a != b:
            mismatches.append(dict(path=path, reason='state_value_or_type_differs',
                left=repr(a), right=repr(b)))
        else:
            identities.append((path, repr(a), repr(b)))
    digest = lambda side: hashlib.sha256(json.dumps([(row[0], row[side]) for row in identities],
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(name=name, bitwise_equal=not mismatches, leaves=len(identities),
        left_sha256=digest(1), right_sha256=digest(2), mismatches=mismatches)


def rng_state(cuda):
    value = dict(cpu=torch.random.get_rng_state())
    if cuda:
        value['cuda0'] = torch.cuda.get_rng_state(0)
    return freeze(value)


def controller_state(controller):
    base = controller.base
    return freeze(dict(plan=controller.plan, output=controller.command,
        nominal=controller.nominal, delta=controller.delta,
        previous_nominal=controller.previous_nominal, backend_state=controller.state,
        backend_visual=controller.visual, contact_gate=controller.contact_gate,
        backend_diagnostics=controller.residual_diagnostics,
        frontend_memory=base.memory, robot178=base.last_robot,
        condition=base.last_condition, frontend_previous_execution=base.position_previous,
        next_step=base.next_step, episode=base.episode, started=base.started,
        independent_generator=base.generator.get_state()))


class CountedDino:
    def __init__(self, model):
        self.model, self.calls = model, 0

    def __call__(self, rgb):
        self.calls += 1
        return self.model(rgb)


def run_sequence(baseline, recorded, samples, *, lease, cuda, enabled_first=False):
    """CPU-testable exact gate using unchanged complete-controller act/ack paths."""
    rows = []
    for index, (packet, diagnostics, actual_target) in enumerate(samples):
        tick = int(packet['step_id'][0])
        if tick != index:
            raise ValueError('The gate requires every actual tick from episode start')
        original = freeze(dict(packet=packet, diagnostics=diagnostics, actual_target=actual_target))
        checks = []
        with lease.block():
            checks.append(compare('independent_rng_before',
                freeze(baseline.base.generator.get_state()), freeze(recorded.base.generator.get_state())))
            global_before = rng_state(cuda)
            order = (('enabled', recorded), ('disabled', baseline)) if enabled_first else (
                ('disabled', baseline), ('enabled', recorded))
            predictions = {}
            for label, controller in order:
                predictions[label] = controller.act(packet, diagnostics).detach().cpu().clone()
                checks.append(compare(label+'_global_rng_unchanged', global_before, rng_state(cuda)))
                checks.append(compare(label+'_inputs_unchanged', original,
                    freeze(dict(packet=packet, diagnostics=diagnostics, actual_target=actual_target))))
            checks.append(compare('predicted_absolute_target', freeze(predictions['disabled']), freeze(predictions['enabled'])))
            checks.append(compare('complete_history_plan_and_rng_after_act', controller_state(baseline), controller_state(recorded)))
            # Real source execution history, explicitly teacher forcing. The
            # compared student prediction above is not called an executed action.
            for controller in (baseline, recorded):
                controller.base.command = actual_target.detach().clone()
                controller.observe_execution(dict(position_applied_rad=actual_target))
            checks.append(compare('complete_history_after_actual_source_ack', controller_state(baseline), controller_state(recorded)))
            checks.append(compare('ack_global_rng_unchanged', global_before, rng_state(cuda)))
        row = dict(step_id=tick, episode_id=int(packet['episode_id'][0]), planning=tick % 12 == 0,
            first_execution_order=order[0][0], bitwise_equal=all(c['bitwise_equal'] for c in checks), checks=checks)
        rows.append(row)
        if not row['bitwise_equal']:
            break
    return dict(passed=all(row['bitwise_equal'] for row in rows) and len(rows) >= 25,
        evaluated_ticks=len(rows), plans_compared=sum(row['planning'] for row in rows), rows=rows)


def load_source(entry, identity, ticks):
    if sha256(entry['path']) != entry['source_sha256']:
        raise ValueError('Real training RGB-D source hash changed')
    with h5py.File(entry['path']) as f:
        group = f['episodes'][entry['episode']]
        metadata = json.loads(f.attrs['metadata_json'])
        if len(group['obs/step_id']) < ticks:
            raise ValueError('Original episode is shorter than the registered gate prefix')
        obs = {k: torch.from_numpy(v[:ticks]) for k, v in group['obs'].items()}
        diag = {k: torch.from_numpy(group['transition/diagnostics_before/'+k][:ticks])
            for k in ('joint_position_rad', 'joint_velocity_rad_s')}
        image = torch.cat((torch.from_numpy(group['frames/rgb_u8'][:ticks]).float()/255,
            torch.from_numpy(group['frames/depth_m'][:ticks])), 2)
        depth_valid = torch.from_numpy(group['frames/depth_valid'][:ticks])
        target = torch.from_numpy(group['transition/target_after_rad'][:ticks])
        transition_step = torch.from_numpy(group['transition/step_id'][:ticks])
        after_time = torch.from_numpy(group['transition/time_after_s'][:ticks])
    if not torch.equal(obs['step_id'], torch.arange(ticks)) or not torch.equal(obs['step_id'], transition_step):
        raise ValueError('Source must start at zero and preserve all native transitions')
    if obs['episode_id'].unique().numel() != 1 or image.shape != (ticks, 3, 4, 252, 448):
        raise ValueError('One complete-prefix episode and full triple RGB-D frames required')
    if not torch.allclose(after_time[:-1].double(), obs['time_s'][1:].double(), atol=1e-6, rtol=0):
        raise ValueError('Actual execution acknowledgement is not aligned to the next observation')
    if not torch.equal(target[:-1], obs['actuator_target_rad'][1:]):
        raise ValueError('Actual source target does not equal next observation controller target')
    source_control_mode = metadata['action_control_mode']
    metadata['action_control_mode'] = 'absolute_joint_position_target_rad'
    samples = []
    for tick in range(ticks):
        packet = {k: value[tick:tick+1] for k, value in obs.items()}
        packet.update(image=image[tick:tick+1], depth_valid=depth_valid[tick:tick+1])
        samples.append((packet, {k: v[tick:tick+1] for k, v in diag.items()}, target[tick:tick+1]))
    return metadata, samples, dict(identity=identity, family=entry['family'],
        episode=entry['episode'], source_sha256=entry['source_sha256'], ticks=ticks,
        source_control_mode=source_control_mode, offline_controller_mode=metadata['action_control_mode'],
        target_acknowledgement='Actual source postprocessed target_after_rad, equal to the next actuator_target_rad; teacher-forced issued/applied target history, not a student rollout')


def gate(root, source_root, output, *, roles=('A', 'B'), episodes=1, ticks=25):
    from tools.tight_fresh_rollouts import completed_pair, read
    from visuotactile.native_controller import ResidualController
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.state_resources import lease_for
    from visuotactile.precision import state_fp32, snapshot
    if output.exists():
        raise FileExistsError('Preserve the complete prior gate identity and evidence')
    if tuple(roles) not in (('A',), ('B',), ('A', 'B')) or episodes not in (1, 2) or not 25 <= ticks <= 61:
        raise ValueError('Use explicit A/B roles, one/two original episodes, and 25-61 native ticks')
    session = read(root/'session.json')
    lease = lease_for(root, deadline=session['deadline'])
    pairs = {role: completed_pair(source_root, role) for role in roles}
    datasets = {role: read(source_root/role/'state_data/data.json') for role in roles}
    selected = {}
    for role, data in datasets.items():
        original = data['normalization_train_ids']
        if len(original) != 30 or len(set(original)) != 30 or not set(original) <= set(data['splits']['train']):
            raise ValueError('Exactly the original train30 must identify this diagnostic source')
        selected[role] = sorted(original)[:episodes]
        for identity in selected[role]:
            if data['entries'][identity].get('trajectory_kind') == 'expert_takeover_v1':
                raise ValueError('Use original training trajectories, not a correction or held-out source')
    output.mkdir(parents=True)
    source_files = ['tools/tight_fresh_forensics_gate.py', 'visuotactile/first_plan_forensics.py',
        'visuotactile/native_controller.py', 'visuotactile/models/native_dp.py', 'visuotactile/models/native_vision.py',
        'visuotactile/native_data.py', 'visuotactile/position_control.py',
        'visuotactile/precision.py', 'visuotactile/state_history.py']
    registration = dict(version='first_plan_forensics_cuda_bitwise_gate_v1', pairs=pairs,
        selected_training_ids=selected, ticks_per_episode=ticks, session_sha256=sha256(root/'session.json'),
        dataset_sha256={role: sha256(source_root/role/'state_data/data.json') for role in roles},
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={name: sha256(name) for name in source_files}, strict_bitwise=True,
        relaxed_tolerance=None, simulator_started=False, test_used=False, utc=time.time())
    atomic_json(output/'registration.json', registration)
    began = time.perf_counter(); runs = []; caught = None
    try:
        with lease.block():
            arithmetic = state_fp32()
            if not torch.cuda.is_available():
                raise ValueError('Actual CUDA is required; CPU mock tests do not pass this gate')
            dino = NativeDinoSpatial().cuda().eval()
        for role_index, role in enumerate(roles):
            saved = torch.load(pairs[role]['backend'], map_location='cpu', weights_only=False)
            for identity in selected[role]:
                metadata, samples, source = load_source(datasets[role]['entries'][identity], identity, ticks)
                directory = output/role/identity
                recorder = FirstPlanRecorder(directory/'first_plan_forensics', binding=dict(
                    frontend_sha256=pairs[role]['frontend_sha256'], backend_sha256=pairs[role]['backend_sha256'],
                    source=source, scheduler_index_mode='host', registration_sha256=sha256(output/'registration.json')))
                counters = [CountedDino(dino), CountedDino(dino)]
                with lease.block():
                    baseline = ResidualController(saved, metadata, dino=counters[0], scheduler_index_mode='host')
                    recorded = ResidualController(saved, metadata, dino=counters[1], scheduler_index_mode='host',
                        first_plan_recorder=recorder)
                result = run_sequence(baseline, recorded, samples, lease=lease, cuda=True,
                    enabled_first=role_index % 2 == 1)
                artifacts = recorder.summary()
                receipt_counts_ok = artifacts['episodes'] == 1 and len(list((directory/'first_plan_forensics').glob('*.json'))) == 1
                frame_counts_ok = counters[0].calls == counters[1].calls == result['evaluated_ticks']
                result.update(role=role, source=source, visual_calls=[x.calls for x in counters],
                    exactly_one_receipt=receipt_counts_ok, every_visual_frame_used=frame_counts_ok,
                    forensics=artifacts)
                result['passed'] = result['passed'] and receipt_counts_ok and frame_counts_ok
                atomic_json(directory/'parity.json', result)
                runs.append(dict(role=role, identity=identity, passed=result['passed'],
                    ticks=result['evaluated_ticks'], plans=result['plans_compared'], visual_calls=result['visual_calls'],
                    parity_path=str((directory/'parity.json').relative_to(output)), parity_sha256=sha256(directory/'parity.json')))
                del baseline, recorded, samples, counters, recorder
                if not result['passed']:
                    raise ValueError('Strict bitwise recorder parity failed; no admission and no relaxed rerun')
        arithmetic_after = snapshot()
    except Exception as error:
        caught = dict(type=type(error).__name__, message=str(error))
        arithmetic_after = snapshot()
    report = dict(registration_sha256=sha256(output/'registration.json'),
        passed=caught is None and len(runs) == len(roles)*episodes and all(r['passed'] for r in runs),
        runs=runs, failure=caught, seconds=time.perf_counter()-began, resource_wait_s=lease.wait_s,
        precision_before=locals().get('arithmetic'), precision_after=arithmetic_after,
        admission='Every byte, dtype, shape, RNG state, plan, command and history must match; no numerical tolerance.',
        scope='Actual CUDA complete DP+backend inference on every recorded RGB-D frame, with aligned teacher-forced actual source target acknowledgements. No simulator, online trajectory, training update, or throughput claim.')
    atomic_json(output/'result.json', report)
    if not report['passed']:
        raise RuntimeError('Forensics gate did not pass; inspect '+str(output/'result.json'))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True, help='Active resource lease/session root')
    parser.add_argument('--source-root', type=Path, required=True, help='Completed A/B student/data root')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--roles', nargs='+', choices=('A', 'B'), default=['A', 'B'])
    parser.add_argument('--episodes', type=int, choices=(1, 2), default=1)
    parser.add_argument('--ticks', type=int, default=25)
    args = parser.parse_args(); torch.set_num_threads(2)
    print(json.dumps(gate(args.root.resolve(), args.source_root.resolve(), args.output.resolve(),
        roles=tuple(args.roles), episodes=args.episodes, ticks=args.ticks), sort_keys=True))


if __name__ == '__main__':
    main()
