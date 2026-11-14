"""CPU-only audit of preserved Tiled failure snapshots; never imports Kit/torch.

Reads small JSON, empty numeric HDF5 and selected arrays from existing packet NPZ.
Writes metadata only. Raw image bytes are neither loaded nor published.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def rotation(q_wxyz):
    q = np.asarray(q_wxyz)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def changed(before, after):
    result = []
    for key in set(before['fields']) | set(after['fields']):
        a, b = before['fields'].get(key), after['fields'].get(key)
        if isinstance(a, dict) and isinstance(b, dict) and a.get('kind') == b.get('kind') == 'array':
            unequal = a['sha256'] != b['sha256']
        else:
            unequal = a != b
        if unequal:
            result.append(key)
    return sorted(result)


def poses(renderer, rigs):
    result = np.empty((8, 3, 7), dtype=np.float32)
    assert len(renderer['camera_params']) == 24
    seen = set()
    for item in renderer['camera_params']:
        lane, view = item['lane'], item['view']
        assert (lane, view) not in seen
        seen.add((lane, view))
        result[lane, view] = rigs.renderer_pose_ros(item['data']['cameraViewTransform'])
    return result


def pose_errors(pose, parent, origins, rig, rigs):
    fixed_pos, fixed_rot = [], []
    for view, (eye, target) in enumerate(rig['fixed_views']):
        # Match float32 eye+origin construction; desired optical axes are Z-up look-at.
        desired = np.asarray(eye, np.float32) + origins.astype(np.float32)
        q = rigs.optical_quat(np.asarray(eye), np.asarray(target), up=(0., 0., 1.))
        fixed_pos.append(np.linalg.norm(pose[:, view, :3] - desired, axis=-1).tolist())
        fixed_rot.append((rotation(q).inv() * rotation(pose[:, view, 3:])).magnitude().tolist())
    pr = rotation(parent[:, 3:])
    desired_wrist = parent[:, :3] + pr.apply(np.asarray(rig['wrist']['pos_parent_m']))
    wrist_pos = np.linalg.norm(pose[:, 2, :3] - desired_wrist, axis=-1)
    wrist_rot = ((pr * rotation(rig['wrist']['quat_parent_ros_wxyz'])).inv()
                 * rotation(pose[:, 2, 3:])).magnitude()
    return dict(fixed_position_error_m_by_view=fixed_pos,
                fixed_rotation_error_rad_by_view=fixed_rot,
                fixed_position_max_m=float(np.max(fixed_pos)),
                fixed_rotation_max_rad=float(np.max(fixed_rot)),
                wrist_mount_position_error_m=wrist_pos.tolist(),
                wrist_mount_rotation_error_rad=wrist_rot.tolist(),
                wrist_mount_position_max_m=float(wrist_pos.max()),
                wrist_mount_rotation_max_rad=float(wrist_rot.max()))


def renderer_summary(renderer):
    views = []
    products = {p['path']: p for p in renderer['render_products']}
    for view, camera in enumerate(renderer['cameras']):
        binding = renderer['calibration_bindings'][view]
        assert camera['camera_paths'] == binding['camera_paths']
        assert camera['image_render_products'] == binding['image_render_products']
        assert products[camera['image_render_products'][0]]['camera_targets'] == camera['camera_paths']
        for lane, rp in enumerate(binding['render_products']):
            assert products[rp]['camera_targets'] == [camera['camera_paths'][lane]]
        kinds = {}
        for kind, item in camera['raw_annotators'].items():
            comparison = item['cpu_tiled_comparison']
            kinds[kind] = dict(node_path=item['node_path'], shape=item['shape'],
                atlas_channel_metrics=item['channel_metrics'],
                raw_cache_all_equal=comparison['all_equal_values'],
                unused_tile_slots=comparison['unused_tile_slots'],
                lanes=[dict(lane=x['lane'], raw_channels={key: [m[key] for m in x['raw_channel_metrics']]
                            for key in ['count', 'finite_count', 'nonzero_count', 'min', 'max']},
                            raw_cache_equal=x['equal_values'], unequal_values=x['unequal_values'],
                            finite_max_absolute_difference=x['finite_max_absolute_difference'])
                       for x in comparison['lanes']])
        views.append(dict(view=view, sensor_class=camera['sensor_class'],
            camera_paths=camera['camera_paths'], image_render_products=camera['image_render_products'],
            calibration_render_products=binding['render_products'],
            configured_resolution=camera['configured_resolution'],
            tiling_grid=camera['expected_tiling_grid_shape'],
            atlas_resolution=camera['expected_tiled_image_shape'],
            raw=kinds))
    assert len(products) == 27
    return dict(settings=renderer['settings'], render_products_count=len(products),
        image_atlases=3, calibration_products=24, bindings_verified=True, views=views,
        read_is_same_frame_as_failed_packet_verified=renderer['same_frame_as_failed_packet_verified'])


def packet_metadata(root):
    failure = root / 'A/sensor_failure'
    report = read(failure / 'report.json')
    assert report['control_tick'] == 0 and report['thresholds_changed'] is False
    assert sha(failure / 'packet.npz') == report['packet_sha256']
    with np.load(failure / 'packet.npz', allow_pickle=False) as packet:
        arrays = {k: packet[k].copy() for k in ['camera_pose_w', 'camera_parent_pose_w',
                  'camera_time_s', 'camera_frame', 'camera_valid', 'time_s']}
    numeric = []
    for path in sorted((root / 'A').glob('lane_*_numeric.h5')):
        with h5py.File(path, 'r') as f:
            assert list(f) == ['episode'] and not list(f['episode'])
        numeric.append(dict(path=str(path), sha256=sha(path), episode_datasets=0))
    assert len(numeric) == 8
    return report, arrays, numeric


def audit(candidate, prior):
    owner = read(candidate / 'owner_started.json')
    plan = read(candidate / 'plan.json')
    assert sha(candidate / 'plan.json') == owner['plan_sha256']
    frozen = Path(owner['cwd'])
    for name, expected in plan['source_sha256'].items():
        assert sha(frozen / name) == expected, name
    spec = importlib.util.spec_from_file_location('audited_camera_rigs', frozen / 'visuotactile/camera_rigs.py')
    rigs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rigs)
    rig = rigs.rig_contract('wrist_bracket_close')
    current_report, current, numeric = packet_metadata(candidate)
    prior_report, previous, prior_numeric = packet_metadata(prior)
    initial = read(candidate / 'A/initial_observable.json')
    old_initial = read(prior / 'A/initial_observable.json')
    # Native grid lies on z=0. Its XY agrees with table centers and robot base offsets.
    origins = np.asarray(initial['table_pose_w'])[:, :3].copy()
    origins[:, 2] = 0
    base = np.asarray(initial['base_pose_w'])[:, :3]
    assert np.allclose(base - origins, [0., .8, 0.], atol=3e-7, rtol=0.)
    assert np.array_equal(np.asarray(old_initial['table_pose_w'])[:, :2], origins[:, :2])
    backend = read(candidate / 'execution_backend_observed.json')
    assert backend['requested']['configuration']['tiled_fixed_camera_usd_sync'] is True
    sources = {}
    for items in backend['tiled_fixed_camera_pose_sync']['sources']:
        for item in items:
            assert sha(item['path']) == item['sha256']
            sources[item['path']] = item['sha256']
    probe = candidate / 'A/sensor_failure/render_only_probe'
    status = read(probe / 'status.json')
    assert status['state'] == 'bounded_probe_complete' and status['renders_completed'] == 4
    assert status['control_frames_added'] == 0 and status['accepted_for_training'] is False
    assert status['original_files'] == status['original_files_after']
    assert status['original_files']['packet_sha256'] == sha(candidate / 'A/sensor_failure/packet.npz')
    assert status['original_files']['report_sha256'] == sha(candidate / 'A/sensor_failure/report.json')
    samples, snapshot_receipts = [], []
    before = read(probe / 'before.json')
    for index in range(5):
        item = current_report if index == 0 else read(probe / f'sample_{index:02d}.json')
        actual_pose = poses(item['renderer'], rigs)
        if index:
            after = read(probe / f'after_{index:02d}.json')
            diff = changed(before, after)
            assert not diff and not item['changed_fields'] and not item['post_read_changed_fields']
            snapshot_receipts.append(dict(render_call_index=index, compared_fields=len(before['fields']),
                changed_fields=diff, reported_observable_state_unchanged=item['observable_state_unchanged']))
        samples.append(dict(render_call_index=index,
            camera_params_pose_exact_failed_packet=bool(np.array_equal(actual_pose, current['camera_pose_w'])),
            pose_error=pose_errors(actual_pose, current['camera_parent_pose_w'], origins, rig, rigs),
            renderer=renderer_summary(item['renderer'])))
    assert all(x['camera_params_pose_exact_failed_packet'] for x in samples)
    hashes = []
    for root in [candidate, prior]:
        names = ['plan.json', 'owner_started.json', 'owner_finished.json', 'startup_timing.json',
                 'A/initial_observable.json', 'A/sensor_failure/report.json', 'A/sensor_failure/packet.npz']
        if root == candidate:
            names += ['execution_backend_observed.json'] + [str(x.relative_to(root)) for x in sorted(probe.glob('*.json'))]
        for name in names:
            path = root / name
            hashes.append(dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size))
    return dict(version='tiled_pose_sync_failure_review_v1', read_only_cpu=True,
        candidate_root=str(candidate), prior_root=str(prior), frozen_source_root=str(frozen),
        source_commit=plan['source_commit'], frozen_source_files_verified=len(plan['source_sha256']),
        installed_sources_verified=sources, desired_rig=rig, reconstructed_env_origins=origins.tolist(),
        origin_method='Native z=0 grid; XY table centers checked against robot base minus (0,0.8,0).',
        same_initial_observable=initial == old_initial,
        same_parent_pose_prior=bool(np.array_equal(current['camera_parent_pose_w'], previous['camera_parent_pose_w'])),
        owner_result=read(candidate / 'owner_finished.json'), startup=read(candidate / 'startup_timing.json'),
        execution_backend=backend, diagnostic_plan=dict(max_control_ticks=plan['max_control_ticks'],
            failure_render_calls=plan['failure_render_calls'], vision_microbatch=plan['vision_microbatch']),
        action_rows=0, numeric_files=numeric, prior_action_rows=0, prior_numeric_files=prior_numeric,
        initial_packet=dict(active=current_report['active'], content=current_report['content'],
            camera_time_s=current['camera_time_s'].tolist(), camera_frame=current['camera_frame'].tolist(),
            camera_pose_w=current['camera_pose_w'].tolist(), thresholds_changed=current_report['thresholds_changed']),
        prior_packet=dict(pose_error=pose_errors(previous['camera_pose_w'], previous['camera_parent_pose_w'], origins, rig, rigs),
            renderer=renderer_summary(prior_report['renderer'])),
        samples=samples, render_only_status=status, observable_snapshots=snapshot_receipts,
        observable_scope=before['scope'], source_evidence=hashes,
        limitations=['Engineering first-frame failure, not task performance or throughput acceptance.',
            'Four render calls are not four native 60Hz observations; original packet was not replaced.',
            'Post-render camera cache deliberately remains old; raw/cache mismatch in view0 is expected.',
            'Raw statistics and CPU tile comparisons are preserved live diagnostic reports, not newly read atlas bytes.',
            'CameraParams uses separate 1x1 products with verified camera bindings; image-product pose equivalence is not proven.',
            'Observable equality excludes opaque PhysX, renderer internals and unobserved RNG state.',
            'The initial observations match, but source versions differ; this is not an isolated causal speed comparison.'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--prior', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.candidate, args.prior)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as f:
        json.dump(result, f, separators=(',', ':'), allow_nan=False)
        f.write('\n')
