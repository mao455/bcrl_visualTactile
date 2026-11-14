"""Pinned, per-instance renderer pose candidate; never edits the installed Lab.

This installed XformPrimView exposes USD mirroring only as a constructor option,
not as a public setter. Replacing an initialized sensor view would discard its
Fabric state. The explicit v3 candidate therefore changes the existing fixed
camera views' boolean only after exact source, class and state checks. Ordinary
Camera already constructs its view with this option enabled; old tiled paths
are untouched. Actual renderer poses still need live CameraParams validation.
"""
import hashlib
import inspect
from pathlib import Path

from .execution_backend import execution_backend, TILED_POSE_VERSION


PINNED_SOURCES = {
    'isaaclab.sensors.camera.camera.Camera': 'a3689cffa01d046b510f7564f7adad972edf6316568b0e323abc2c4f0c4ec354',
    'isaaclab.sensors.camera.tiled_camera.TiledCamera': 'f3e9205a16a3ba335f9376a57cf4bf0473ecced928d9ac12d9ddeae83961d506',
    'isaaclab.sim.views.xform_prim_view.XformPrimView': '5a05ccc9322e6f042f2ab669cbaf7fb60712726dfeb0d323913b78c87d3755b3',
}
FIELD = '_sync_usd_on_fabric_write'
RECEIPT_VERSION = 'instance_fixed_tiled_fabric_to_usd_v1'


def _qualified(cls):
    return cls.__module__ + '.' + cls.__name__


def _sources(camera):
    classes = (type(camera), type(camera).__mro__[1], type(camera._view))
    if set(map(_qualified, classes)) != set(PINNED_SOURCES):
        raise RuntimeError('Unreviewed TiledCamera/XformPrimView class implementation')
    rows = []
    for cls in classes:
        path = inspect.getsourcefile(cls)
        if path is None:
            raise RuntimeError('Cannot identify installed camera source')
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        name = _qualified(cls)
        if digest != PINNED_SOURCES[name]:
            raise RuntimeError('Installed camera source differs from reviewed SHA256: ' + name)
        rows.append(dict(class_name=name, path=str(Path(path).resolve()), sha256=digest))
    return rows


def _candidate(env):
    config = execution_backend(getattr(env, 'execution_backend', None))
    return config if config['version'] == TILED_POSE_VERSION else None


def _fixed_cameras(env, config):
    # This candidate was investigated on the two-fixed-plus-wrist rig only.
    if len(env.camera_rig.get('fixed_views', [])) != 2 or 'wrist' not in env.camera_rig:
        raise RuntimeError('Tiled pose candidate requires the reviewed two-fixed-plus-wrist rig')
    if len(env.research_cameras) != 3:
        raise RuntimeError('Tiled pose candidate requires exactly three camera sensors')
    fixed = list(env.research_cameras[:2])
    if len({id(camera._view) for camera in env.research_cameras}) != 3:
        raise RuntimeError('Camera views must be independent instances')
    for camera in fixed:
        view = camera._view
        if str(camera.device) != config['physics_device'] or str(view._device) != config['physics_device']:
            raise RuntimeError('Tiled pose candidate device differs from registered backend')
        if getattr(view, '_use_fabric', None) is not True:
            raise RuntimeError('Tiled pose candidate requires the reviewed CUDA Fabric path')
        if type(getattr(view, FIELD, None)) is not bool:
            raise RuntimeError('Missing or non-boolean instance USD mirroring field')
    return fixed


def enable_fixed_camera_pose_sync(env):
    """Enable before fixed-camera pose writes; a repeated call only verifies."""
    config = _candidate(env)
    if config is None:
        return None
    if getattr(env, '_tiled_fixed_camera_pose_sync_receipt', None) is not None:
        return fixed_camera_pose_sync_readback(env)
    cameras = _fixed_cameras(env, config)
    # Validate every instance and source before mutating any instance.
    sources = [_sources(camera) for camera in cameras]
    if any(getattr(camera._view, FIELD) is not False for camera in cameras):
        raise RuntimeError('Unregistered prior change to fixed camera USD mirroring')
    wrist = env.research_cameras[2]._view
    wrist_before = getattr(wrist, FIELD, None)
    if type(wrist_before) is not bool:
        raise RuntimeError('Missing or non-boolean wrist USD mirroring readback')
    changed = []
    try:
        for camera in cameras:
            setattr(camera._view, FIELD, True)
            changed.append(camera)
            if getattr(camera._view, FIELD) is not True:
                raise RuntimeError('Fixed camera USD mirroring write did not persist')
    except BaseException:
        for camera in changed:
            setattr(camera._view, FIELD, False)
        raise
    receipt = dict(version=RECEIPT_VERSION, backend_version=config['version'],
        mechanism='pinned instance boolean; no shared class/module or simulator installation mutation',
        private_field=FIELD, expected_fixed_count=2, wrist_field_before=wrist_before,
        sources=sources, fixed=[dict(view=i, prim_path=camera.cfg.prim_path,
            before=False, after=True) for i,camera in enumerate(cameras)],
        live_renderer_pose_accepted=False)
    env._tiled_fixed_camera_pose_sync_receipt = receipt
    env._tiled_fixed_camera_pose_sync_views = tuple(camera._view for camera in env.research_cameras)
    return fixed_camera_pose_sync_readback(env)


def fixed_camera_pose_sync_readback(env):
    """Fail closed if the registered instance setting or untouched wrist drifted."""
    config = _candidate(env)
    if config is None:
        return None
    receipt = getattr(env, '_tiled_fixed_camera_pose_sync_receipt', None)
    if receipt is None:
        raise RuntimeError('Registered fixed camera USD mirroring was not initialized')
    cameras = _fixed_cameras(env, config)
    original_views = getattr(env, '_tiled_fixed_camera_pose_sync_views', ())
    if (len(original_views) != 3 or any(camera._view is not original_views[i]
            for i,camera in enumerate(env.research_cameras))):
        raise RuntimeError('Registered camera view instance was replaced')
    observed = [getattr(camera._view, FIELD) for camera in cameras]
    wrist = getattr(env.research_cameras[2]._view, FIELD, None)
    if observed != [True, True] or type(wrist) is not bool or wrist != receipt['wrist_field_before']:
        raise RuntimeError('Registered camera USD mirroring readback changed')
    return dict(receipt, observed_fixed=observed, observed_wrist=wrist,
                instance_setting_matched=True,
                limitation='Instance setting readback only; actual rendered poses/content require a live gate')
