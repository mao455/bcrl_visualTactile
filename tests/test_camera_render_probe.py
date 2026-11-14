import hashlib
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.tight_fresh_batch_eval import check_sensor_clock_with_evidence
from visuotactile import camera_failure_diagnostics as failures
from visuotactile import camera_render_probe as probe
from visuotactile.environment_rng import RandomStream


def fixture_env(monkeypatch):
    packet=dict(image=torch.zeros(2,3,4,3,4),depth_m=torch.ones(2,3,1,3,4),
        depth_valid=torch.ones(2,3,1,3,4,dtype=torch.bool),camera_valid=torch.zeros(2,3,dtype=torch.bool),
        camera_time_s=torch.zeros(2,3),time_s=torch.zeros(2),camera_frame=torch.ones(2,3,dtype=torch.int64))
    z=torch.zeros(2,29)
    root=lambda:SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2,13)))
    env=SimpleNamespace(device='cpu',research_packet=packet,research_tick=0,
        research_step=torch.zeros(2,dtype=torch.int64),research_episode=torch.tensor([7,8]),
        episode_length_buf=torch.zeros(2,dtype=torch.int64),common_step_counter=0,_sim_step_counter=0,
        _cur_targets=z.clone(),_prev_targets=z.clone(),_action_queue=torch.zeros(2,3,29),
        action_control_mode='absolute_joint_position_target_rad',
        robot=SimpleNamespace(data=SimpleNamespace(root_state_w=torch.zeros(2,13),joint_pos=z.clone(),
            joint_vel=z.clone(),joint_pos_target=z.clone())),object=root(),hole=root(),table=root(),goal_viz=root(),
        research_cameras=[SimpleNamespace(frame=torch.ones(2,dtype=torch.int64))for _ in range(3)])
    env._diagnostics=lambda:dict(joint_position_rad=env.robot.data.joint_pos.clone(),native_stage=torch.zeros(2))
    env.sim=SimpleNamespace(current_time=0.,current_time_step_index=0,calls=0)
    def render():env.sim.calls+=1
    env.sim.render=render
    def forbidden(*args,**kwargs):raise AssertionError('No physics, controls, reset, or camera update permitted')
    env.step=env.reset=env.sim.step=forbidden
    for camera in env.research_cameras:camera.update=forbidden
    layout=lambda e:dict(raw_rgb_nonzero=e.sim.calls>=2,read_index=e.sim.calls)
    monkeypatch.setattr(failures,'renderer_layout',layout)
    monkeypatch.setattr(probe,'renderer_layout',layout)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    return env


def save_original(tmp_path,env):
    failures.save_failure(tmp_path,env.research_packet,[True,True],env,ValueError('original black'),control_tick=0)
    return {name:hashlib.sha256((tmp_path/'sensor_failure'/name).read_bytes()).hexdigest()
            for name in ('packet.npz','report.json')}


def test_render_recovers_raw_without_adding_frames_or_hiding_original_failure(tmp_path,monkeypatch):
    env=fixture_env(monkeypatch);original=env.research_packet;stream=RandomStream(31)
    with pytest.raises(ValueError,match='Invalid real RGB-D content'):
        check_sensor_clock_with_evidence(original,[True,True],[None,None],directory=tmp_path,env=env,
            control_tick=0,failure_render_calls=4,max_control_ticks=1,rng_stream=stream)
    assert env.sim.calls==4 and env.research_packet is original
    assert not original['camera_valid'].any() and bool((original['camera_frame']==1).all())
    root=tmp_path/'sensor_failure/render_only_probe';status=json.loads((root/'status.json').read_text())
    assert status['state']=='bounded_probe_complete' and status['original_files_unchanged']
    assert status['control_frames_added']==0 and not status['accepted_for_training']
    assert not json.loads((root/'sample_01.json').read_text())['renderer']['raw_rgb_nonzero']
    assert json.loads((root/'sample_02.json').read_text())['renderer']['raw_rgb_nonzero']
    assert all(row['observable_state_unchanged']for row in status['samples'])
    assert json.loads((root/'before.json').read_text())['owned_rng_stream_included']
    with np.load(tmp_path/'sensor_failure/packet.npz')as value:assert not value['image'].any()


@pytest.mark.parametrize('field',['joint','time','target','queue','rng','packet','camera_frame','owned_rng'])
def test_first_observable_drift_stops_before_next_render(tmp_path,monkeypatch,field):
    env=fixture_env(monkeypatch);stream=RandomStream(5);before=RandomStream.snapshot();save_original(tmp_path,env)
    def render():
        env.sim.calls+=1
        if field=='joint':env.robot.data.joint_pos[0,1]=.01
        elif field=='time':env.sim.current_time+=1/60
        elif field=='target':env._cur_targets[1,2]=.02
        elif field=='queue':env._action_queue[0,2,3]=1
        elif field=='rng':random.random()
        elif field=='packet':env.research_packet=dict(env.research_packet)
        elif field=='camera_frame':env.research_cameras[2].frame[1]+=1
        elif field=='owned_rng':stream.state=(('changed',),*stream.state[1:])
    env.sim.render=render
    try:status=probe.run_render_probe(tmp_path,env,max_render_calls=4,control_tick=0,max_control_ticks=1,rng_stream=stream)
    finally:RandomStream.restore(before)
    assert env.sim.calls==1 and status['state']=='observable_state_changed'
    assert status['samples'][0]['changed_fields'] and status['original_files_unchanged']
    sample=json.loads((tmp_path/'sensor_failure/render_only_probe/sample_01.json').read_text())
    assert 'renderer' not in sample


def test_missing_state_and_render_exception_leave_recoverable_status(tmp_path,monkeypatch):
    env=fixture_env(monkeypatch);save_original(tmp_path,env)
    del env._action_queue
    status=probe.run_render_probe(tmp_path,env,max_render_calls=2,control_tick=0,max_control_ticks=1)
    assert status['state']=='diagnostic_failed' and env.sim.calls==0
    assert '_action_queue' in status['diagnostic_error']
    other=tmp_path/'other';other.mkdir();env=fixture_env(monkeypatch);save_original(other,env)
    def render():
        # The on-disk state is useful even if a native abort prevents Python finally.
        status=json.loads((other/'sensor_failure/render_only_probe/status.json').read_text())
        assert status['state']=='render_in_progress' and status['renders_started']==1
        env.sim.calls+=1;raise RuntimeError('renderer failed')
    env.sim.render=render
    status=probe.run_render_probe(other,env,max_render_calls=4,control_tick=0,max_control_ticks=1)
    assert env.sim.calls==1 and status['state']=='render_failed'
    assert status['renders_completed']==0 and status['original_files_unchanged']


def test_annotator_side_effect_also_stops_probe(tmp_path,monkeypatch):
    env=fixture_env(monkeypatch);save_original(tmp_path,env)
    def layout(e):e.robot.data.joint_vel[1,3]=.3;return {'invalid':True}
    monkeypatch.setattr(probe,'renderer_layout',layout)
    status=probe.run_render_probe(tmp_path,env,max_render_calls=4,control_tick=0,max_control_ticks=1)
    assert status['state']=='observable_state_changed' and env.sim.calls==1
    assert (tmp_path/'sensor_failure/render_only_probe/after_read_01.json').exists()


def test_default_off_and_no_diagnostics_for_success(tmp_path,monkeypatch):
    env=fixture_env(monkeypatch)
    assert probe.run_render_probe(tmp_path,env,max_render_calls=0,control_tick=5,max_control_ticks=None) is None
    assert env.sim.calls==0 and not list(tmp_path.iterdir())
    env.research_packet['camera_valid'].fill_(True)
    check_sensor_clock_with_evidence(env.research_packet,[True,True],[None,None],directory=tmp_path,env=env,
        control_tick=0,failure_render_calls=4,max_control_ticks=1)
    assert env.sim.calls==0 and not list(tmp_path.iterdir())


@pytest.mark.parametrize('calls,limit',[(True,1),(1,True),(-1,1),(5,1),(1,None),(1,36),(1.0,1)])
def test_only_registered_bounded_one_tick_probe(calls,limit):
    with pytest.raises(ValueError):probe.validate_render_probe(calls,max_control_ticks=limit)


def test_tampered_packet_or_second_failure_cannot_start_render(tmp_path,monkeypatch):
    env=fixture_env(monkeypatch);save_original(tmp_path,env)
    with pytest.raises(ValueError,match='tick zero'):
        probe.run_render_probe(tmp_path,env,max_render_calls=1,control_tick=1,max_control_ticks=1)
    with (tmp_path/'sensor_failure/packet.npz').open('ab')as f:f.write(b'changed')
    with pytest.raises(RuntimeError,match='preserved'):
        probe.run_render_probe(tmp_path,env,max_render_calls=1,control_tick=0,max_control_ticks=1)
    assert env.sim.calls==0
