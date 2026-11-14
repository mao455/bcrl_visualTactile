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
