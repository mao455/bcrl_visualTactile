"""Registered execution choices; these never change task or control semantics.

Admission of GPU memory and live camera/contact validation are separate gates.
This module deliberately imports no simulator and does not initialize CUDA.
"""
import hashlib
import json
import os
import re


VERSION = 'tight_execution_backend_v1'
DESCRIPTOR_VERSION = 'tight_execution_backend_v2'
DEFAULT = dict(version=VERSION, physics_device='cpu', camera_backend='standard',
               kit_tasking_threads=2, physics_threads=2)
DESCRIPTOR_DEFAULTS = dict(renderer_descriptor_sets=360000, renderer_reserved_descriptors=900000)
DESCRIPTOR_SETTINGS = dict(renderer_descriptor_sets='/rtx/descriptorSets',
                           renderer_reserved_descriptors='/rtx/reservedDescriptors')


def execution_backend(value=None, *, available_cpus=None):
    """Resolve an explicit, serializable candidate, with no four-lane limit."""
    if value is not None and not isinstance(value, dict):
        raise ValueError('execution_backend must be a registered object')
    value = {} if value is None else dict(value)
    version = value.get('version', VERSION)
    if version not in (VERSION, DESCRIPTOR_VERSION):
        raise ValueError('Unknown execution backend version')
    defaults = dict(DEFAULT)
    if version == DESCRIPTOR_VERSION:
        defaults.update(DESCRIPTOR_DEFAULTS)
    unknown = set(value)-set(defaults)
    if unknown:
        raise ValueError('Unknown execution_backend fields: '+str(sorted(unknown)))
    result = dict(defaults, **value)
    if version == DESCRIPTOR_VERSION:
        for field in DESCRIPTOR_DEFAULTS:
            if type(result[field]) is not int or not 1 <= result[field] <= 2147483647:
                raise ValueError(field+' must be a positive signed 32-bit integer')
    device = result['physics_device']
    if not isinstance(device, str) or not re.fullmatch(r'cpu|cuda:(0|[1-9][0-9]*)', device):
        raise ValueError('physics_device must be cpu or an explicit cuda:N device')
    if result['camera_backend'] not in ('standard', 'tiled'):
        raise ValueError('Only standard or tiled real RGB-D cameras are supported')
    if available_cpus is None:
        available_cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    available_cpus = available_cpus or 1
    for field in ('kit_tasking_threads', 'physics_threads'):
        count = result[field]
        if type(count) is not int or not 1 <= count <= available_cpus:
            raise ValueError(f'{field} must be an integer between 1 and available CPUs ({available_cpus})')
    return result


def renderer_resource_readback(value, getter):
    """Read capacity only for v2; no extra setting access on the v1 path."""
    config = execution_backend(value)
    if config['version'] != DESCRIPTOR_VERSION:
        return None
    requested = {path:config[field] for field,path in DESCRIPTOR_SETTINGS.items()}
    observed = {path:getter(path) for path in requested}
    return dict(requested=requested, observed=observed,
        matched=all(type(observed[path]) is int and observed[path] == number for path,number in requested.items()),
        view_tile_limit_observed=getter('/rtx/viewTile/limit'),
        scope='Process-local renderer capacity; no image quality, camera calibration or physics change')


def require_renderer_resources(receipt):
    if receipt is not None and receipt['matched'] is not True:
        raise RuntimeError('Renderer resource settings differ from the registered request: '+str(receipt))


def backend_receipt(value, *, num_envs):
    config = execution_backend(value)
    if type(num_envs) is not int or num_envs <= 0:
        raise ValueError('num_envs must be a positive integer; resources determine the useful maximum')
    identity = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return dict(configuration=config, configuration_sha256=identity, num_envs=num_envs,
        camera_calibration=('same_rgbd_render_product_camera_params_v1' if config['camera_backend'] == 'standard'
                            else 'same_camera_1x1_render_product_camera_params_v1'),
        rng_contract='registered_device_batch_stream_v2',
        live_accepted=False,
        limitation='Execution candidate only; requires measured contact/control/sensor/task acceptance. '
                   'Native random draws are batch-owned, not independent lane streams or backend-exact replay.')


def startup_arguments(value, *, num_envs=None):
    config = execution_backend(value)
    args = ('--/renderer/multiGpu/enabled=false '
            f'--/plugins/carb.tasking.plugin/threadCount={config["kit_tasking_threads"]} '
            f'--/persistent/physics/numThreads={config["physics_threads"]} '
            '--/app/extensions/fsWatcherEnabled=false')
    if config['camera_backend'] == 'tiled':
        if type(num_envs) is not int or num_envs <= 0:
            raise ValueError('Tiled startup requires the registered lane count')
        # Replicator otherwise enforces an installation-specific tile ceiling.
        # Resource admission remains the runner's responsibility, not a lane cap.
        args += f' --/rtx/viewTile/limit={num_envs}'
    if config['version'] == DESCRIPTOR_VERSION:
        args += ''.join(f' --{path}={config[field]}' for field,path in DESCRIPTOR_SETTINGS.items())
    return args


# Stable public name for registration and runner callers.
normalize_backend = execution_backend
