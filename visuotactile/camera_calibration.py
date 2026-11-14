"""Strict renderer calibration identities for ordinary and tiled sensor paths."""
import re
import numpy as np

from .camera_rigs import renderer_pose_ros


def ordered_camera_paths(camera, num_envs):
    """Use the sensor's exact tiled order and reject ambiguous lane mappings."""
    prims = getattr(camera, '_sensor_prims', None)
    if prims is None or len(prims) != num_envs:
        raise RuntimeError('Cannot establish one actual camera prim per tiled lane')
    paths = [str(prim.GetPath()) for prim in prims]
    lanes = []
    for path in paths:
        match = re.search(r'/env_([0-9]+)/', path)
        if match is None:
            raise RuntimeError('Camera path has no explicit environment identity: '+path)
        lanes.append(int(match.group(1)))
    if lanes != list(range(num_envs)) or len(set(paths)) != num_envs:
        raise RuntimeError('Tiled sensor order does not match environment lane order')
    return paths


def renderer_poses(annotators, num_envs):
    if len(annotators) != num_envs:
        raise RuntimeError('Renderer calibration must have one independent product per lane')
    poses = []
    for annotator in annotators:
        data = annotator.get_data()
        matrix = np.asarray(data.get('cameraViewTransform'))
        if matrix.size != 16 or not np.isfinite(matrix).all():
            raise RuntimeError('Missing or invalid current renderer camera view transform')
        poses.append(renderer_pose_ros(matrix))
    return np.stack(poses)
