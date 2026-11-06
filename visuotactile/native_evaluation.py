"""Isolated native environment and honest Kit lifecycle instrumentation."""
import ctypes
import json
import os
import resource
import sys
from pathlib import Path
import time
import numpy as np
import torch
from .compact import sha256
from .native_data import multiply_quaternion,rotate
from .training_resources import atomic_json
_watch_client=None

def client_startup_controls(mode=None):
    if mode=='kit_managed':
        return dict(mode=mode,ctypes_preload=False,test_setter_called=False,
            reason='Let Kit initialize OmniClient through its extension lifecycle; inotify admission remains required')
    if mode not in (None,'legacy_preload_disabled'):
        raise ValueError('Unknown OmniClient startup mode')
    return disable_omniclient_watches()


def disable_omniclient_watches():
    global _watch_client
    import isaacsim
    library = Path(isaacsim.__file__).parent/'kit'/'extscore'/'omni.client.lib'/'bin'/'libomniclient.so'
    if not library.is_file():
        raise RuntimeError('OmniClient library not found: '+str(library))
    client = ctypes.CDLL(str(library))
    _watch_client = client  # Keep the explicitly loaded provider alive for Kit's lifetime.
    setter = client.testSetWatchesEnabled
    setter.argtypes = [ctypes.c_bool]
    setter.restype = None
    setter(False)
    return dict(library=str(library.resolve()), symbol='testSetWatchesEnabled', enabled=False,
        flag_readback=omniclient_watch_state())


def omniclient_watch_state():
    """Read back this build's setter storage only if its exact instruction matches."""
    if _watch_client is None:
        return None
    address=ctypes.cast(_watch_client.testSetWatchesEnabled,ctypes.c_void_p).value
    opcode=ctypes.string_at(address,8)
    if opcode[:3]!=b'\x40\x88\x3d' or opcode[-1:]!=b'\xc3':
        return dict(available=False,reason='Unknown setter machine code; no guessed memory access')
    offset=int.from_bytes(opcode[3:7],'little',signed=True)
    return dict(available=True,enabled=bool(ctypes.c_ubyte.from_address(address+7+offset).value))


def kit_process_snapshot():
    """Read-only process-local resource census; never changes shared limits."""
    proc = Path('/proc/self')
    watches = instances = 0
    for entry in (proc/'fdinfo').iterdir():
        try:
            lines = entry.read_text().splitlines()
        except OSError:
            continue
        count = sum(line.startswith('inotify wd:') for line in lines)
        watches += count
        instances += bool(count)
    return dict(pid=os.getpid(), utc=time.time(), fds=len(list((proc/'fd').iterdir())),
        threads=len(list((proc/'task').iterdir())), inotify_watches=watches,
        inotify_instances_with_watches=instances,
        inotify_limits={name: int((Path('/proc/sys/fs/inotify')/name).read_text())
            for name in ('max_user_watches','max_user_instances','max_queued_events')},
        rlimit_nofile=list(resource.getrlimit(resource.RLIMIT_NOFILE)),
        rlimit_nproc=list(resource.getrlimit(resource.RLIMIT_NPROC)))


def exit_completed_kit_worker(output):
    """Isolated-worker OS cleanup after durable results, not SDK shutdown success.

    Native crashes before this point remain nonzero exits. Parent must reap and
    verify this PID is absent from both /proc and the GPU process census.
    """
    output = Path(output)
    value = json.loads((output/'result.json').read_text())
    if value.get('software_complete') is not True or (output/'worker_error.json').exists():
        raise RuntimeError('Cannot cleanly exit an incomplete worker')
    atomic_json(output/'process_exit_requested.json', dict(pid=os.getpid(), utc=time.time(),
        result_sha256=sha256(output/'result.json'), method='os._exit after closed writers and fsync',
        sdk_graceful_shutdown=False, reason='Measured FileThread abort/segfault in Kit shutdown'))
    # HDF5/video writers have been closed by trial's finally block. Flush only
    # this worker's output files; never global sync or external process changes.
    for path in output.rglob('*'):
        if path.is_file() and not path.is_symlink():
            with path.open('rb') as stream:
                os.fsync(stream.fileno())
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)


def stable_grasp(arrays, initial_height):
    force = np.asarray(arrays['object_contact'])
    contact = np.linalg.norm(force, axis=-1) > .1
    two_fingers = contact.sum(-1) >= 2
    pose = torch.from_numpy(np.asarray(arrays['object']))
    palm = torch.from_numpy(np.asarray(arrays['palm']))
    inverse = palm[:, 3:].clone()
    inverse[:, 1:] *= -1
    relative_position = rotate(inverse, pose[:, :3]-palm[:, :3])
    relative_quaternion = multiply_quaternion(inverse, pose[:, 3:])
    stable = False
    longest = current = 0
    for active in two_fingers.reshape(-1):
        current = current+1 if active else 0
        longest = max(longest, current)
    for stop in range(12, len(pose)+1):
        start = stop-12
        if not two_fingers[start:stop].all() or not (pose[start:stop, 2] >= initial_height+.02).all():
            continue
        position_drift = (relative_position[start:stop]-relative_position[start]).norm(dim=-1).max()
        cosine = (relative_quaternion[start:stop]*relative_quaternion[start]).sum(-1).abs().clamp(0, 1)
        rotation_drift = (2*torch.acos(cosine)).max()
        stable |= bool(position_drift <= .01 and rotation_drift <= .2)
    return dict(object_contact=bool(contact.any()), max_fingers=int(contact.sum(-1).max()),
        two_finger_continuous_s=longest/120, stable_grasp_diagnostic=stable,
        object_contact_threshold_N=.1, stable_window_s=.2, scope='filtered normal forces; not native success')


def make_env(instrumented, lease, *, num_envs=1, seed=20261031, scene_queries=False, expert_speed=None,
             camera_content_validation=False):
    from evaluation.eval_isaacsim import _load_env_cfg
    from .observations import TaskCommand
    from .sim.replay_environment import RecordedReplayEnv
    from .sim.native_diagnostics import NativeDiagnosticEnv
    cfg = _load_env_cfg('Isaacsimenvs-PreciseAssembly-Direct-v0')
    cfg.seed = seed
    cfg.scene.num_envs, cfg.scene.env_spacing = num_envs, 8.
    cfg.sim.device = 'cpu'
    # Optional read-only collision admission queries. Default historical runners
    # retain their exact configuration; no integration/control property changes.
    if scene_queries:
        cfg.sim.enable_scene_query_support = True
    cfg.precise_assembly.problem = 'tight_insertion'
    cfg.student_obs.image_modality, cfg.student_obs.camera_backend = 'rgbd', 'standard'
    cfg.student_obs.camera_update_period_s = 0.
    cfg.student_obs.image_width = cfg.student_obs.image_input_width = 448
    cfg.student_obs.image_height = cfg.student_obs.image_input_height = 252
    cfg.sim.render.rendering_mode, cfg.sim.render.antialiasing_mode = 'quality', 'Off'
    env_class=NativeDiagnosticEnv if instrumented else RecordedReplayEnv
    extra={}
    if expert_speed is not None:
        assert instrumented
        from .sim.expert_speed_environment import SpeedLimitedDiagnosticEnv
        env_class=SpeedLimitedDiagnosticEnv;extra['speed_config']=expert_speed
    with lease.block():
        env = env_class(cfg, perception_version=2, **extra, camera_content_validation=camera_content_validation,
            task_command=TaskCommand('tight_insertion'), wrench_frame_mode='world_hold', camera_rig='wrist_bracket_close')
        if camera_content_validation:
            # Direct fresh reset never passed through restore_observable_state,
            # which was historically the only caller setting fixed camera poses.
            # Configure after sensor initialization and before the first reset/render.
            env.configure_camera_views()
        return env
