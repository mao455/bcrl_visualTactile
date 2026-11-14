from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from tools.tight_fresh_batch_eval import check_sensor_clock_with_evidence
from visuotactile import camera_failure_diagnostics as diagnostics


def packet():
    image = torch.zeros(2, 3, 4, 4, 4)
    image[:, :, :3, :, 2:] = 1
    image[:, :, 3] = .5
    return dict(image=image, depth_m=image[:, :, 3:].clone(),
        depth_valid=torch.ones(2, 3, 1, 4, 4, dtype=torch.bool),
        camera_valid=torch.ones(2, 3, dtype=torch.bool),
        camera_time_s=torch.zeros(2, 3, dtype=torch.float64),
        time_s=torch.zeros(2, dtype=torch.float64), camera_frame=torch.ones(2, 3, dtype=torch.long),
        camera_pose_w=torch.zeros(2, 3, 7), camera_parent_pose_w=torch.zeros(2, 7))


def fake_env():
    return SimpleNamespace(research_cameras=[], _camera_calibration_bindings=[], _camera_pose_annotators=[])


def test_failure_saves_exact_packet_and_failed_components_then_raises(tmp_path):
    value = packet()
    value['image'][0, 1, :3] = .3  # Constant nonblack image is still invalid.
    value['depth_valid'][1, 2] = False
    value['camera_valid'][0, 1] = False
    value['camera_valid'][1, 2] = False
    with pytest.raises(ValueError, match='Invalid real RGB-D content during backend probe'):
        check_sensor_clock_with_evidence(value, [True, True], [None, None],
            directory=tmp_path, env=fake_env(), control_tick=0)
    report = json.loads((tmp_path / 'sensor_failure/report.json').read_text())
    assert report['packet_saved'] and not report['thresholds_changed']
    assert report['control_tick'] == 0
    with np.load(tmp_path / 'sensor_failure/packet.npz') as saved:
        assert set(saved.files) == set(value)
        for key in saved.files:
            assert np.array_equal(saved[key], value[key].numpy())
    rows = {(row['lane'], row['view']): row for row in report['content']}
    assert not rows[0, 1]['rgb_content_component']
    assert rows[0, 1]['depth_content_component']
    assert rows[1, 2]['rgb_content_component']
    assert not rows[1, 2]['depth_content_component']
    assert rows[0, 0]['recomputed_cpu_content_valid']


def test_normal_path_never_calls_saver_or_writes(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Normal sensor validation must not save')
    monkeypatch.setattr(diagnostics, 'save_failure', forbidden)
    previous = [None, None]
    assert check_sensor_clock_with_evidence(packet(), [True, True], previous,
        directory=tmp_path, env=fake_env(), control_tick=0) is previous
    assert not list(tmp_path.iterdir())
    assert all(torch.equal(value, torch.ones(3, dtype=torch.long)) for value in previous)


def test_diagnostic_write_error_never_replaces_original_failure(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError('diagnostic disk write unavailable')
    monkeypatch.setattr(diagnostics, 'save_failure', broken)
    value = packet()
    value['camera_valid'][0, 0] = False
    with pytest.raises(ValueError, match='Invalid real RGB-D content during backend probe') as raised:
        check_sensor_clock_with_evidence(value, [True, True], [None, None],
            directory=tmp_path, env=fake_env(), control_tick=0)
    assert 'diagnostic disk write unavailable' in raised.value.__notes__[0]


def test_payload_budget_refuses_partial_snapshot(tmp_path):
    report = diagnostics.save_failure(tmp_path, packet(), [True, True], fake_env(),
        ValueError('failure'), control_tick=0, max_packet_bytes=1)
    assert not report['packet_saved']
    assert 'exceeds diagnostic byte bound' in report['packet_diagnostic_error']
    assert not (tmp_path / 'sensor_failure/packet.npz').exists()
    assert (tmp_path / 'sensor_failure/report.json').exists()
    with pytest.raises(FileExistsError):
        diagnostics.save_failure(tmp_path, packet(), [True, True], fake_env(), ValueError('new'), control_tick=1)


def test_renderer_layout_records_actual_bindings_and_cached_shapes():
    prim = SimpleNamespace(GetPath=lambda: '/World/envs/env_0/Camera')
    annotator = SimpleNamespace(get_data=lambda: np.zeros((756, 1344, 4), dtype=np.uint8))
    calibration = SimpleNamespace(get_data=lambda: {'cameraViewTransform': np.eye(4), 'cameraResolution': np.array([1, 1])})
    camera = SimpleNamespace(_sensor_prims=[prim], render_product_paths=['/Render/tiled'],
        cfg=SimpleNamespace(prim_path='/World/envs/env_.*/Camera', width=448, height=252),
        _data=SimpleNamespace(output={'rgb': torch.zeros(8, 252, 448, 3, dtype=torch.uint8)},
                              intrinsic_matrices=torch.eye(3)[None]),
        _tiling_grid_shape=lambda: (3, 3), _tiled_image_shape=lambda: (1344, 756),
        _annotators={'rgba': annotator})
    env = SimpleNamespace(research_cameras=[camera], _camera_pose_annotators=[[calibration]],
        _camera_calibration_bindings=[dict(camera_paths=['/World/envs/env_0/Camera'],
            render_products=['/Render/calibration'], image_render_products=['/Render/tiled'])])
    report = diagnostics.renderer_layout(env)
    row = report['cameras'][0]
    assert row['expected_tiling_grid_shape'] == [3, 3]
    assert row['raw_annotators']['rgba']['shape'] == [756, 1344, 4]
    assert row['cached_output']['rgb']['shape'] == [8, 252, 448, 3]
    assert report['calibration_bindings'][0]['render_products'] == ['/Render/calibration']
    assert report['camera_params'][0]['data']['cameraResolution'] == [1, 1]


def tiled_fixture():
    # Three lanes in a 2x2 atlas, last tile is unused. Each live lane/channel
    # differs, so a transposed or reused tile cannot accidentally pass.
    raw = np.zeros((6, 8, 4), dtype=np.uint8)
    for lane in range(3):
        y, x = (lane // 2) * 3, (lane % 2) * 4
        raw[y:y+3, x:x+4] = [11+lane, 31+lane, 71+lane, 255]
    cached = torch.from_numpy(np.stack([raw[(i//2)*3:(i//2+1)*3, (i%2)*4:(i%2+1)*4]
        for i in range(3)]).copy())
    camera = SimpleNamespace(_sensor_prims=[], render_product_paths=['/Render/tiled'],
        cfg=SimpleNamespace(prim_path='/World/env_.*/Camera', width=4, height=3),
        _data=SimpleNamespace(output={'rgba': cached}), _tiling_grid_shape=lambda: (2, 2),
        _tiled_image_shape=lambda: (8, 6),
        _annotators={'rgba': SimpleNamespace(get_data=lambda: raw,
            _node_path='/Render/tiled/SDGPipeline/Rgb', _render_products=['/Render/tiled'])})
    env = SimpleNamespace(research_cameras=[camera], _camera_pose_annotators=[], _camera_calibration_bindings=[])
    return raw, cached, camera, env


def test_raw_rgba_distinguishes_renderer_black_from_cached_lane_copy_failure():
    raw, cached, camera, env = tiled_fixture()
    cached[1, ..., :3] = 0
    raw[3:6, :4, :3] = 0
    cached[2, ..., :3] = 0
    detail = diagnostics.renderer_layout(env)['cameras'][0]['raw_annotators']['rgba']
    rows = detail['cpu_tiled_comparison']['lanes']
    assert rows[0]['equal_values']
    assert not rows[1]['equal_values'] and rows[1]['unequal_values'] == 3*4*3
    assert rows[1]['raw_channel_metrics'][0]['min'] == 12
    assert rows[1]['cached_channel_metrics'][0]['max'] == 0
    assert rows[2]['equal_values'] and rows[2]['raw_channel_metrics'][0]['nonzero_count'] == 0
    assert rows[2]['raw_channel_metrics'][3]['min'] == 255  # valid alpha, black RGB
    assert detail['cpu_tiled_comparison']['unused_tile_slots'] == 1
    assert detail['node_path'] == '/Render/tiled/SDGPipeline/Rgb'
    assert detail['read_started_utc'] <= detail['cached_copy_finished_utc'] <= detail['raw_copy_finished_utc']


def test_cached_snapshot_precedes_async_annotator_read_and_never_replaces_failed_packet(tmp_path):
    raw, cached, camera, env = tiled_fixture()
    def read_newer_buffer():
        # Emulate a renderer update between cache snapshot and annotator read.
        cached.fill_(99)
        return raw
    camera._annotators['rgba'].get_data = read_newer_buffer
    failed = packet()
    failed['image'].zero_(); failed['camera_valid'].zero_()
    report = diagnostics.save_failure(tmp_path, failed, [True, True], env, ValueError('black'), control_tick=0)
    detail = report['renderer']['cameras'][0]['raw_annotators']['rgba']
    assert detail['cpu_tiled_comparison']['all_equal_values']  # compared to the earlier copied cache
    assert bool((cached == 99).all())
    assert report['renderer']['same_frame_as_failed_packet_verified'] is False
    saved = np.load(tmp_path/'sensor_failure/packet.npz')
    assert not saved['image'].any() and not saved['camera_valid'].any()


def test_oversized_renderer_array_rejected_before_transfer():
    class Oversized:
        shape = (diagnostics.MAX_RENDERER_ARRAY_BYTES + 1,)
        dtype = np.uint8
        def numpy(self):
            raise AssertionError('Oversized array must not be copied')
    with pytest.raises(ValueError, match='copy bound'):
        diagnostics._host_copy(Oversized())
