from visuotactile.training_resources import ResourceGuard


def row(util=100, rolling=40):
    return dict(gpu_free_mib=12000,ram_available_bytes=32*1024**3,
                gpu_util_percent=util,rolling_60s_percent=rolling)


def test_average_gate_ignores_isolated_peak_but_checks_both_means(tmp_path):
    guard=ResourceGuard(tmp_path)
    guard.measured_s=100;guard.area=4000
    assert guard.safe(row(util=100))
    assert not guard.safe(row(rolling=49))
    guard.area=4900
    assert not guard.safe(row(util=0,rolling=20))
    guard.area=4000;guard.max_rolling=95
    assert guard.safe(row())  # a historical peak cannot cause permanent waiting


def test_average_gate_requires_telemetry_and_keeps_memory_reserve(tmp_path):
    guard=ResourceGuard(tmp_path)
    assert not guard.safe(row())
    guard.measured_s=2;guard.area=60
    assert guard.safe(row(rolling=None))  # no invented incomplete 60s mean
    assert not guard.safe(row(util=None))
    assert not guard.safe({**row(),'gpu_free_mib':3000})


def test_session_waits_measure_sleep_and_survive_budget_exit(tmp_path, monkeypatch):
    import json
    import pytest
    from visuotactile import residual_session
    (tmp_path / 'session.json').write_text(json.dumps(dict(deadline_epoch=400)))
    monkeypatch.setattr(residual_session.atexit, 'register', lambda callback: None)
    clock = dict(now=100.)
    monkeypatch.setattr(residual_session.time, 'time', lambda: clock['now'])
    monkeypatch.setattr(residual_session.time, 'monotonic', lambda: clock['now'])
    def delayed_sleep(duration):
        clock['now'] += duration + .25
    monkeypatch.setattr(residual_session.time, 'sleep', delayed_sleep)
    gate = residual_session.SessionGate(tmp_path)
    gate.pace(97.)
    assert gate.pacing_wait_s == 6.25
    clock['now'] = 500.
    times = iter((10., 10.2))
    monkeypatch.setattr(residual_session.time, 'monotonic', lambda: next(times))
    with pytest.raises(InterruptedError, match='budget'):
        gate.before()
    gate.save_waits()
    saved = json.loads(next(tmp_path.glob('waits_*.json')).read_text())
    assert saved['admission_wait_s'] == pytest.approx(.2)
    assert saved['total_wait_s'] == pytest.approx(6.45)


def test_stream_video_records_each_view_and_frame_without_raw_frame_cache(tmp_path):
    import av
    import numpy as np
    import torch
    from visuotactile.evaluate_baselines import RolloutVideo
    packet={'image':torch.zeros(1,3,4,32,448)}
    for i in range(3):packet['image'][0,i,i]=1
    path=tmp_path/'demo.mp4'
    video=RolloutVideo(path,'b0',1,49,['fixed_high','fixed_side','wrist_rgbd'])
    video.append(packet,0);video.append(packet,1)
    info=video.close(complete=True)
    assert info['frames']==2 and not info['terminal_image_available']
    assert not video.partial.exists()
    with av.open(str(path)) as c:
        assert c.streams.video[0].average_rate==60
        frames=list(c.decode(video=0))
    assert len(frames)==2
    rgb=frames[0].to_ndarray(format='rgb24')
    for i in range(3):
        center=rgb[80,i*448+224]
        assert center.argmax()==i and center[i]>245
