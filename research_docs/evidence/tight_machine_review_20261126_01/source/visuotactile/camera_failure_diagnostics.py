"""Bounded, failure-only camera evidence. Never renders, steps, or relaxes acceptance."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch


PACKET_FIELDS = ('image', 'depth_m', 'depth_valid', 'camera_valid', 'camera_time_s',
                 'camera_frame', 'camera_pose_w', 'camera_parent_pose_w', 'time_s',
                 'episode_id', 'step')
MAX_PACKET_BYTES = 256 * 1024 * 1024
MAX_RENDERER_ARRAY_BYTES = 64 * 1024 * 1024


def _host_copy(value):
    """Bound an individual diagnostic copy, including Warp's GPU-to-host copy."""
    shape = tuple(value.shape)
    dtype = np.dtype(getattr(value.dtype, '__name__', str(value.dtype).removeprefix('torch.')))
    if int(np.prod(shape)) * dtype.itemsize > MAX_RENDERER_ARRAY_BYTES:
        raise ValueError('Renderer array exceeds diagnostic copy bound')
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    elif not isinstance(value, np.ndarray):
        value = value.numpy()
    return np.array(value, copy=True)


def channel_metrics(array):
    """Channel statistics include alpha; a zero RGB image can have valid alpha."""
    if array.ndim < 3:
        array = array[..., None]
    rows = []
    for channel in range(array.shape[-1]):
        values = array[..., channel]
        finite = np.isfinite(values)
        selected = values[finite]
        rows.append(dict(channel=channel, count=int(values.size), finite_count=int(finite.sum()),
            nonzero_count=int(np.count_nonzero(values)),
            min=float(selected.min()) if selected.size else None,
            max=float(selected.max()) if selected.size else None))
    return rows


def tiled_cached_comparison(raw, cached, *, width, height, columns):
    """Independent CPU tile indexing, including padded non-square atlases."""
    if cached.ndim != 4 or cached.shape[1:3] != (height, width) or columns < 1:
        raise ValueError('Unexpected cached tile geometry')
    if raw.ndim == 2:
        raw = raw[..., None]
    lanes, _, _, channels = cached.shape
    if raw.ndim != 3 or raw.shape[-1] < channels or raw.shape[1] != columns * width:
        raise ValueError('Raw atlas shape differs from cached tile geometry')
    if raw.shape[0] < ((lanes + columns - 1) // columns) * height:
        raise ValueError('Raw atlas lacks complete lane tiles')
    rows = []
    for lane in range(lanes):
        y, x = (lane // columns) * height, (lane % columns) * width
        tile = raw[y:y+height, x:x+width, :channels]
        expected = cached[lane]
        finite = np.isfinite(tile) & np.isfinite(expected)
        delta = np.abs(tile[finite].astype(np.float64) - expected[finite].astype(np.float64))
        rows.append(dict(lane=lane, raw_channel_metrics=channel_metrics(tile),
            cached_channel_metrics=channel_metrics(expected),
            equal_values=bool(np.array_equal(tile, expected, equal_nan=True)),
            unequal_values=int(np.count_nonzero(~((tile == expected) | (np.isnan(tile) & np.isnan(expected))))),
            finite_max_absolute_difference=float(delta.max()) if delta.size else None))
    return dict(lanes=rows, all_equal_values=all(row['equal_values'] for row in rows),
        unused_tile_slots=int(raw.shape[0] // height * columns - lanes),
        semantics='Read-time comparison, not proof of identical render frame; cached copy precedes raw read.')


def _plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def packet_metrics(arrays, active):
    """Same strict content terms, plus diagnostic finite/range summaries on CPU."""
    image, depth, valid = (arrays[key] for key in ('image', 'depth_m', 'depth_valid'))
    rows = []
    for lane in range(image.shape[0]):
        for view in range(image.shape[1]):
            rgb = torch.from_numpy(image[lane, view, :3])
            metric = torch.from_numpy(depth[lane, view])
            mask = torch.from_numpy(valid[lane, view]).bool()
            spatial = rgb.flatten(-2).std(-1, correction=0)
            finite_rgb = torch.isfinite(rgb)
            finite_depth = torch.isfinite(metric)
            positive = finite_depth & (metric > 0)
            coverage = (mask & positive).float().mean()
            accepted = (spatial.amax() > 1 / 255) & (coverage > .01) & finite_rgb.all()
            finite_values = rgb[finite_rgb]
            rows.append(dict(lane=lane, view=view, active=bool(active[lane]),
                rgb_min=float(finite_values.min()) if len(finite_values) else None,
                rgb_max=float(finite_values.max()) if len(finite_values) else None,
                rgb_spatial_std_by_channel=spatial.tolist(),
                rgb_spatial_std_max=float(spatial.amax()),
                rgb_finite_fraction=float(finite_rgb.float().mean()),
                metric_depth_finite_fraction=float(finite_depth.float().mean()),
                metric_depth_positive_fraction=float(positive.float().mean()),
                provided_depth_valid_fraction=float(mask.float().mean()),
                combined_depth_valid_fraction=float(coverage),
                rgb_content_component=bool(spatial.amax() > 1 / 255),
                depth_content_component=bool(coverage > .01),
                rgb_finite_component=bool(finite_rgb.all()),
                original_camera_valid=bool(arrays['camera_valid'][lane, view]),
                recomputed_cpu_content_valid=bool(accepted)))
    return rows


def renderer_layout(env):
    """Read existing buffers/prim bindings only; no camera.update or render call."""
    result = dict(calibration_bindings=_plain(getattr(env, '_camera_calibration_bindings', [])), cameras=[],
        read_started_utc=time.time(), max_array_copy_bytes=MAX_RENDERER_ARRAY_BYTES,
        read_semantics='Failure-only reads of current existing buffers after the failed packet was saved. '
            'No render/update/physics call. Annotators may advance asynchronously; timestamps are host read '
            'times, not camera exposures. These reads never replace the failed packet.',
        same_frame_as_failed_packet_verified=False)
    products = set()
    for camera in env.research_cameras:
        paths = [str(prim.GetPath()) for prim in getattr(camera, '_sensor_prims', [])]
        render_products = list(camera.render_product_paths)
        products.update(render_products)
        row = dict(sensor_class=type(camera).__name__, camera_paths=paths,
            image_render_products=render_products, cfg_prim_path=camera.cfg.prim_path,
            configured_resolution=[camera.cfg.width, camera.cfg.height])
        if hasattr(camera, '_tiling_grid_shape'):
            row['expected_tiling_grid_shape'] = list(camera._tiling_grid_shape())
            row['expected_tiled_image_shape'] = list(camera._tiled_image_shape())
        cached = getattr(camera, '_data', None)
        row['cached_output'] = {key: dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
            for key, value in getattr(cached, 'output', {}).items()}
        row['cached_intrinsics'] = _plain(getattr(cached, 'intrinsic_matrices', None))
        row['raw_annotators'] = {}
        for name, annotator in getattr(camera, '_annotators', {}).items():
            detail = dict(read_started_utc=time.time(),
                node_path=_plain(getattr(annotator, '_node_path', None)),
                render_products=_plain(getattr(annotator, '_render_products', None)))
            row['raw_annotators'][name] = detail
            try:
                # Copy cache first: a subsequent asynchronous annotator read can
                # describe a newer render. A mismatch is evidence, not a fix.
                output = getattr(cached, 'output', {})
                cached_value = output.get(name, output.get('rgb') if name == 'rgba' else None)
                cached_copy = _host_copy(cached_value) if cached_value is not None else None
                detail['cached_copy_finished_utc'] = time.time()
                raw = annotator.get_data()
                if isinstance(raw, dict):
                    raw = raw.get('data')
                detail.update(shape=list(raw.shape), dtype=str(raw.dtype))
                raw_copy = _host_copy(raw)
                detail.update(raw_copy_finished_utc=time.time(), channel_metrics=channel_metrics(raw_copy))
                if cached_copy is not None and hasattr(camera, '_tiling_grid_shape'):
                    detail['cpu_tiled_comparison'] = tiled_cached_comparison(raw_copy, cached_copy,
                        width=camera.cfg.width, height=camera.cfg.height, columns=camera._tiling_grid_shape()[0])
            except Exception as error:
                detail['read_error'] = repr(error)
        result['cameras'].append(row)
    result['camera_params'] = []
    for view, annotators in enumerate(getattr(env, '_camera_pose_annotators', [])):
        for lane, annotator in enumerate(annotators):
            try:
                result['camera_params'].append(dict(view=view, lane=lane, data=_plain(annotator.get_data())))
            except Exception as error:
                result['camera_params'].append(dict(view=view, lane=lane, read_error=repr(error)))
    for binding in result['calibration_bindings']:
        products.update(binding['render_products'])
    try:
        import carb
        import omni.usd
        settings = carb.settings.get_settings()
        result['settings'] = {key: _plain(settings.get(key)) for key in (
            '/rtx/viewTile/limit', '/rtx/viewTile/resolution/0', '/rtx/viewTile/resolution/1',
            '/rtx/viewTile/width', '/rtx/viewTile/height', '/rtx/descriptorSets', '/rtx/reservedDescriptors')}
        stage = omni.usd.get_context().get_stage()
        result['render_products'] = []
        for path in sorted(products):
            prim = stage.GetPrimAtPath(path)
            result['render_products'].append(dict(path=path, valid=bool(prim.IsValid()),
                attributes={attribute.GetName(): _plain(attribute.Get()) for attribute in prim.GetAttributes()
                            if any(term in attribute.GetName().lower() for term in ('tile', 'postprocess', 'enabled'))
                            or attribute.GetName() == 'resolution'},
                camera_targets=[str(path) for path in prim.GetRelationship('camera').GetTargets()],
                ordered_vars=[str(path) for path in prim.GetRelationship('orderedVars').GetTargets()]))
        result['camera_prim_properties'] = {}
        for row in result['cameras']:
            for path in row['camera_paths']:
                prim = stage.GetPrimAtPath(path)
                result['camera_prim_properties'][path] = {key: _plain(prim.GetAttribute(key).Get()) for key in (
                    'focalLength', 'horizontalAperture', 'verticalAperture', 'horizontalApertureOffset',
                    'verticalApertureOffset', 'clippingRange', 'projection')}
    except Exception as error:
        result['renderer_read_error'] = repr(error)
    result['read_finished_utc'] = time.time()
    return result


def save_failure(directory, packet, active, env, failure, *, control_tick, max_packet_bytes=MAX_PACKET_BYTES):
    """One local snapshot; an oversized packet is explicitly refused, never truncated."""
    destination = Path(directory) / 'sensor_failure'
    destination.mkdir(exist_ok=False)
    selected = {key: packet[key] for key in PACKET_FIELDS if key in packet and torch.is_tensor(packet[key])}
    size = sum(value.numel() * value.element_size() for value in selected.values())
    report = dict(version='camera_failure_snapshot_v2', control_tick=control_tick, snapshot_started_utc=time.time(),
        original_error=repr(failure), active=list(map(bool, active)), payload_bytes=size,
        max_payload_bytes=max_packet_bytes, packet_saved=False, thresholds_changed=False,
        semantics='Actual pre-action packet, no extra render/physics/control. Metric depth is already sanitized '
            'to zero where raw depth was invalid; depth_valid preserves that mask. CPU recomputation is '
            'diagnostic; original camera_valid remains authoritative. No missing frame substitution.')
    try:
        if size > max_packet_bytes:
            raise ValueError('Failure packet exceeds diagnostic byte bound; no partial image snapshot')
        arrays = {key: value.detach().cpu().numpy().copy() for key, value in selected.items()}
        archive = destination / 'packet.npz'
        np.savez(archive, **arrays)
        digest = hashlib.sha256()
        with archive.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        report.update(packet_saved=True, packet_sha256=digest.hexdigest(),
            arrays={key: dict(shape=list(value.shape), dtype=str(value.dtype)) for key, value in arrays.items()})
        report['content'] = packet_metrics(arrays, active)
    except Exception as error:
        report['packet_diagnostic_error'] = repr(error)
    try:
        report['renderer'] = renderer_layout(env)
    except Exception as error:
        report['renderer_diagnostic_error'] = repr(error)
    # JSON may expose nonfinite diagnostic values, not silently map them to valid content.
    report['snapshot_finished_utc'] = time.time()
    (destination / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    return report
