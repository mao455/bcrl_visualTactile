"""Bounded render-only failure diagnosis, never a source of control observations.

Only the explicit one-control-tick probe may invoke this after preserving its
failed packet. Observable-state equality is checked, not complete PhysX replay.
"""
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from .camera_failure_diagnostics import renderer_layout
from .training_resources import atomic_json


VERSION = 'failed_first_frame_render_only_v1'


def validate_render_probe(max_render_calls=0, *, max_control_ticks=None):
    if type(max_render_calls) is not int or not 0 <= max_render_calls <= 4:
        raise ValueError('failure_render_calls must be an integer from 0 through 4')
    if max_render_calls and (type(max_control_ticks) is not int or max_control_ticks != 1):
        raise ValueError('Extra render diagnosis is restricted to an explicit one-tick probe')
    return max_render_calls


def _fingerprints(value, prefix, result):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError('Opaque object array cannot establish observable-state equality')
        array = np.ascontiguousarray(value)
        header = json.dumps([array.dtype.str, list(array.shape)]).encode()
        result[prefix] = dict(kind='array', dtype=array.dtype.str, shape=list(array.shape),
            sha256=hashlib.sha256(header + array.tobytes()).hexdigest())
        if array.size <= 2048 and not prefix.startswith(('research_packet/', 'rng_')):
            result[prefix]['values'] = array.tolist()
    elif isinstance(value, dict):
        result[prefix + '/__keys__'] = sorted(map(str, value))
        for key, item in value.items():
            _fingerprints(item, prefix + '/' + str(key), result)
    elif isinstance(value, (tuple, list)):
        result[prefix + '/__sequence__'] = dict(kind=type(value).__name__, length=len(value))
        for index, item in enumerate(value):
            _fingerprints(item, prefix + '/' + str(index), result)
    elif value is None or isinstance(value, (str, bool, int, float, np.generic)):
        scalar = value.item() if isinstance(value, np.generic) else value
        # repr makes NaN/inf comparison stable and preserves integer/float type.
        result[prefix] = dict(kind=type(scalar).__name__, value=repr(scalar))
    else:
        raise TypeError('Unsupported observable state at ' + prefix + ': ' + type(value).__name__)


def observable_snapshot(env, *, rng_stream=None):
    """Strict field coverage for this task; missing fields refuse all extra renders."""
    fields = dict(diagnostics=env._diagnostics(),
        sim_time=env.sim.current_time, sim_step=env.sim.current_time_step_index,
        common_step_counter=env.common_step_counter, sim_step_counter=env._sim_step_counter,
        research_tick=env.research_tick, research_step=env.research_step,
        research_episode=env.research_episode, episode_length=env.episode_length_buf,
        current_targets=env._cur_targets, previous_targets=env._prev_targets,
        delay_queue=env._action_queue, action_control_mode=env.action_control_mode,
        robot_root=env.robot.data.root_state_w, robot_joint_position=env.robot.data.joint_pos,
        robot_joint_velocity=env.robot.data.joint_vel, actuator_target=env.robot.data.joint_pos_target,
        object_root=env.object.data.root_state_w, hole_root=env.hole.data.root_state_w,
        table_root=env.table.data.root_state_w, goal_root=env.goal_viz.data.root_state_w,
        research_packet=env.research_packet, research_packet_identity=id(env.research_packet),
        camera_frames=[camera.frame for camera in env.research_cameras],
        rng_python=random.getstate(), rng_numpy=np.random.get_state(), rng_torch_cpu=torch.get_rng_state())
    if rng_stream is not None:
        fields['rng_owned_stream'] = rng_stream.state
        fields['rng_owned_stream_active'] = rng_stream.active
    cuda_devices = []
    if torch.cuda.is_initialized():
        # Never initialize an unused CUDA device merely to inspect RNG.
        cuda_devices = [torch.cuda.current_device()]
        device = torch.device(env.device)
        if device.type == 'cuda' and device.index not in cuda_devices:
            raise RuntimeError('Probe CUDA device is not the current initialized device')
        fields['rng_torch_cuda_current'] = torch.cuda.get_rng_state(cuda_devices[0])
    result = {}
    for key, value in fields.items():
        _fingerprints(value, key, result)
    return dict(fields=result, observed_cuda_rng_devices=cuda_devices,
        owned_rng_stream_included=rng_stream is not None,
        scope='Current exposed task/robot/rigid-body/time/target/delay-queue/packet/frame and RNG fields only; '
            'does not establish equality of opaque PhysX, renderer internals or unobserved RNG generators')


def _sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _changed_fields(before, after):
    changed = []
    for key in sorted(set(before['fields']) | set(after['fields'])):
        left, right = before['fields'].get(key), after['fields'].get(key)
        if (isinstance(left, dict) and isinstance(right, dict)
                and left.get('kind') == right.get('kind') == 'array'):
            # Exact bytes include dtype/shape and NaN payload; display values
            # must not make unchanged NaNs compare unequal to themselves.
            different = left['sha256'] != right['sha256']
        else:
            different = left != right
        if different:
            changed.append(key)
    return changed


def run_render_probe(directory, env, *, max_render_calls, control_tick, max_control_ticks, rng_stream=None):
    """Write progress before every render; preserve original failure and packet."""
    validate_render_probe(max_render_calls, max_control_ticks=max_control_ticks)
    if max_render_calls == 0:
        return None
    if type(control_tick) is not int or control_tick != 0:
        raise ValueError('Render-only diagnosis is allowed only after failed tick zero')
    failure = Path(directory) / 'sensor_failure'
    packet_path, failure_path = failure / 'packet.npz', failure / 'report.json'
    report = json.loads(failure_path.read_text())
    if (not report.get('packet_saved') or report.get('control_tick') != 0
            or report.get('thresholds_changed') is not False or not packet_path.is_file()
            or _sha(packet_path) != report.get('packet_sha256')):
        raise RuntimeError('Original failed first packet must be preserved before extra rendering')
    destination = failure / 'render_only_probe'
    destination.mkdir(exist_ok=False)
    originals = dict(packet_sha256=_sha(packet_path), report_sha256=_sha(failure_path))
    status = dict(version=VERSION, diagnostic_only=True, max_render_calls=max_render_calls,
        control_tick=0, max_control_ticks=1, original_files=originals, started_utc=time.time(),
        renders_started=0, renders_completed=0, samples=[], state='snapshot_pending',
        original_failure_remains=True, control_frames_added=0, accepted_for_training=False,
        semantics='Read raw annotators after render-only calls. Render indices are not 60Hz observations. '
            'No step/reset/controller/camera.update, no failed-packet replacement; cached tiles may remain old.')
    path = destination / 'status.json'
    atomic_json(path, status)
    try:
        baseline = observable_snapshot(env, rng_stream=rng_stream)
        atomic_json(destination / 'before.json', baseline)
        status['state'] = 'ready'; atomic_json(path, status)
        for index in range(1, max_render_calls + 1):
            status.update(state='render_in_progress', renders_started=index)
            atomic_json(path, status)
            sample = dict(render_call_index=index, started_utc=time.time())
            render_error = None
            try:
                env.sim.render()
                status['renders_completed'] = index
            except Exception as error:
                render_error = repr(error)
                sample['render_error'] = render_error
            sample['render_finished_utc'] = time.time()
            after = observable_snapshot(env, rng_stream=rng_stream)
            changed = _changed_fields(baseline, after)
            sample['changed_fields'] = changed
            sample['observable_state_unchanged'] = not changed
            atomic_json(destination / f'after_{index:02d}.json', after)
            # Do not inspect more renderer buffers after a physical/time/RNG drift.
            if not changed and render_error is None:
                sample['renderer'] = renderer_layout(env)
                # Detect side effects of diagnostic annotator reads as well.
                after_read = observable_snapshot(env, rng_stream=rng_stream)
                changed = _changed_fields(baseline, after_read)
                sample['post_read_changed_fields'] = changed
                sample['observable_state_unchanged'] = not changed
                if changed:
                    atomic_json(destination / f'after_read_{index:02d}.json', after_read)
            sample['finished_utc'] = time.time()
            atomic_json(destination / f'sample_{index:02d}.json', sample)
            status['samples'].append(dict(render_call_index=index,
                observable_state_unchanged=sample['observable_state_unchanged'],
                changed_fields=changed, render_error=render_error))
            status['state'] = ('observable_state_changed' if changed else
                               'render_failed' if render_error is not None else 'ready')
            atomic_json(path, status)
            if changed or render_error is not None:
                break
        if status['state'] == 'ready':
            status['state'] = 'bounded_probe_complete'
    except Exception as error:
        status.update(state='diagnostic_failed', diagnostic_error=repr(error))
    finally:
        try:
            observed = dict(packet_sha256=_sha(packet_path), report_sha256=_sha(failure_path))
            status['original_files_after'] = observed
            status['original_files_unchanged'] = observed == originals
            if observed != originals:
                status['state'] = 'original_evidence_changed'
        except Exception as error:
            status.update(state='original_evidence_unreadable', evidence_error=repr(error))
        status['finished_utc'] = time.time()
        atomic_json(path, status)
    return status
