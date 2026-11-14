import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from visuotactile import inactive_render_products as render
from visuotactile.camera_rigs import renderer_pose_ros


class Texture:
    def __init__(self, lane, view):
        self.path=f'/Render/test_{view}_{lane}'
        self.camera=f'/World/envs/env_{lane}/Camera{view}'
        self.enabled=True
        self.calls=[]
        self.fail_disable=False
        self.fail_restore=False

    def get_render_product_path(self):return self.path
    def get_camera_path(self):return self.camera
    def get_width(self):return 448
    def get_height(self):return 252
    def get_updates_enabled(self):return self.enabled
    def set_updates_enabled(self, value):
        self.calls.append(value)
        if (value and self.fail_restore) or (not value and self.fail_disable):return
        self.enabled=value


def setup(count=2, gate=None):
    clock=dict(research_tick=0,sim_step=10,sim_time=10/120)
    bindings=[]
    for lane in range(count):
        for view in range(3):
            texture=Texture(lane,view)
            bindings.append(dict(lane=lane,view=view,render_product_path=texture.path,
                camera_path=texture.camera,handle=texture,async_rendering=False))
    guard=render.InactiveRenderProducts(bindings,profile=render.PROFILE,num_envs=count,
        clock=lambda:dict(clock),gate_limit_ticks=gate)
    return guard,clock,bindings


def packet(count=2, tick=0):
    pose=torch.from_numpy(renderer_pose_ros(np.eye(4))).expand(count,3,7).clone()
    return dict(camera_valid=torch.ones(count,3,dtype=torch.bool),
        camera_time_s=torch.full((count,3),tick/60,dtype=torch.float64),
        camera_frame=torch.full((count,3),tick+1,dtype=torch.int64),
        camera_pose_w=pose,time_s=torch.full((count,),tick/60,dtype=torch.float64),
        image=torch.arange(count*3*4*2*2).reshape(count,3,4,2,2).float())


def emit(guard, clock, number, lanes=None):
    for lane in range(guard.num_envs):
        if lanes is not None and lane not in lanes:continue
        for view in range(3):
            guard.note_drawable((lane,view),dict(frame_number=number,resolution=[448,252],
                view=np.eye(4),projection=np.eye(4)),dict(clock))


def test_profile_is_explicit_and_rejects_other_backends_before_importing_kit():
    assert render.validate_profile(None) is None
    assert render.validate_profile(render.PROFILE)==render.PROFILE
    for update in (dict(physics_device='cuda:0'),dict(camera_backend='tiled'),dict(views=True),dict(unknown=True)):
        with pytest.raises(ValueError,match='profile'):
            render.validate_profile(dict(render.PROFILE,**update))
    env=SimpleNamespace(device='cuda:0',cfg=SimpleNamespace(sim=SimpleNamespace(device='cuda:0'),
        student_obs=SimpleNamespace(camera_backend='standard')))
    with pytest.raises(ValueError,match='CPU standard'):render.InactiveRenderProducts.bind(env,render.PROFILE)
    assert not torch.cuda.is_initialized()


def test_missing_existing_product_does_not_create_or_reassign():
    calls=[]
    class Registry:
        def get(self,context,path):
            calls.append((context,path));raise ValueError('missing')
    manager=SimpleNamespace(_context=None,_hydra_textures=Registry())
    with pytest.raises(ValueError,match='missing'):render.existing_product(manager,'/Render/absent')
    assert calls==[(None,'/Render/absent')]
    manager._hydra_textures.get=lambda context,path:SimpleNamespace(path='/Render/other')
    with pytest.raises(ValueError,match='different path'):render.existing_product(manager,'/Render/absent')


@pytest.mark.parametrize('corruption',['shared','lane','async','disabled','resolution'])
def test_binding_rejects_shared_or_ambiguous_products(corruption):
    _,clock,bindings=setup()
    if corruption=='shared':bindings[1]['render_product_path']=bindings[0]['render_product_path']
    if corruption=='lane':bindings[1]['camera_path']='/World/envs/env_99/Camera1'
    if corruption=='async':bindings[1]['async_rendering']=True
    if corruption=='disabled':bindings[1]['handle'].enabled=False
    if corruption=='resolution':bindings[1]['handle'].get_width=lambda:224
    with pytest.raises(ValueError):
        render.InactiveRenderProducts(bindings,profile=render.PROFILE,num_envs=2,clock=lambda:clock)


def test_retired_lane_cannot_spoof_new_image_time_frame_or_calibration():
    guard,clock,bindings=setup();guard.begin_wave(0);emit(guard,clock,100)
    actual=packet();observed,proof=guard.observe(actual,control_tick=0)
    assert proof['active_lanes']==[0,1] and len(proof['products'])==6
    guard.retire([1],reason='initial_illegal',control_tick=0)
    assert all(b['handle'].enabled==(b['lane']==0) for b in bindings)
    clock.update(research_tick=1,sim_step=12,sim_time=.1);emit(guard,clock,101,lanes=[0])
    next_packet=packet(tick=1);masked,proof=guard.observe(next_packet,control_tick=1)
    assert proof['active_lanes']==[0] and len(proof['products'])==3
    assert torch.equal(masked['image'],next_packet['image']) and masked['image'] is next_packet['image']
    assert next_packet['camera_valid'].all()  # original packet never overwritten
    for key in ('camera_valid','camera_time_s','camera_frame','camera_pose_w'):
        assert torch.equal(masked[key][0],next_packet[key][0])
    assert not masked['camera_valid'][1].any()
    assert (masked['camera_time_s'][1]==-1).all() and (masked['camera_frame'][1]==-1).all()
    assert not masked['camera_pose_w'][1].any()
    with pytest.raises(ValueError,match='distinct active'):guard.retire([1],reason='natural_termination',control_tick=1)


def test_synthetic_read_counter_cannot_replace_actual_new_renderer_event():
    guard,clock,_=setup();guard.begin_wave(0);emit(guard,clock,100)
    guard.observe(packet(),control_tick=0)
    clock.update(research_tick=1,sim_step=12,sim_time=.1)
    with pytest.raises(ValueError,match='No new real renderer frame'):
        guard.observe(packet(tick=1),control_tick=1)
    emit(guard,clock,100)  # repeated drawable callback with old native frame
    with pytest.raises(ValueError,match='No new real renderer frame'):
        guard.observe(packet(tick=1),control_tick=1)
    emit(guard,clock,101);guard.observe(packet(tick=1),control_tick=1)


def test_current_physics_clock_and_actual_camera_params_are_both_required():
    guard,clock,_=setup();guard.begin_wave(0);emit(guard,clock,100)
    clock['sim_step']+=1
    with pytest.raises(ValueError,match='predates current physical'):
        guard.observe(packet(),control_tick=0)
    emit(guard,clock,101);bad=packet();bad['camera_pose_w'][0,0,0]+=.001
    with pytest.raises(ValueError,match='CameraParams'):
        guard.observe(bad,control_tick=0)
    actual=packet();actual['camera_pose_w'][0,0,3:]*=-1
    guard.observe(actual,control_tick=0)  # equivalent quaternion sign is accepted


def test_camera_and_packet_clocks_cannot_both_claim_a_stale_control_tick():
    guard,clock,_=setup();guard.begin_wave(0)
    clock['research_tick']=1;emit(guard,clock,100)
    with pytest.raises(ValueError,match='current 60Hz research tick'):
        guard.observe(packet(tick=0),control_tick=0)


def test_extra_render_cannot_substitute_for_a_control_step():
    guard,clock,_=setup();guard.begin_wave(0);emit(guard,clock,100)
    _,first=guard.observe(packet(),control_tick=0)
    assert first['advancement_since_previous_observation'] is None
    emit(guard,clock,101)  # A genuinely newer drawable, but no physics step.
    with pytest.raises(ValueError,match='exactly one control tick'):
        guard.observe(packet(),control_tick=1)
    assert guard.snapshot()['last_control_tick']==0


@pytest.mark.parametrize('physics_steps',[0,1,3])
def test_control_counter_and_new_drawable_cannot_hide_wrong_physics_steps(physics_steps):
    guard,clock,_=setup();guard.begin_wave(0);emit(guard,clock,100)
    guard.observe(packet(),control_tick=0)
    clock.update(research_tick=1,sim_step=10+physics_steps,sim_time=.1);emit(guard,clock,101)
    with pytest.raises(ValueError,match='exactly two 120Hz steps'):
        guard.observe(packet(tick=1),control_tick=1)


def test_float32_substep_roundoff_is_allowed_but_time_drift_is_not():
    guard,clock,_=setup();clock['sim_time']=123456.25
    guard.begin_wave(0);emit(guard,clock,100);guard.observe(packet(),control_tick=0)
    previous=clock['sim_time']
    clock.update(research_tick=1,sim_step=12,sim_time=previous+1/60+1e-6);emit(guard,clock,101)
    with pytest.raises(ValueError,match='original 1/60'):
        guard.observe(packet(tick=1),control_tick=1)
    clock['sim_time']=previous+2*float(np.float32(1/120));emit(guard,clock,102)
    _,proof=guard.observe(packet(tick=1),control_tick=1)
    assert proof['advancement_since_previous_observation']['physics_steps']==2
    assert proof['advancement_since_previous_observation']['sim_time_tolerance_s']<1e-7


def test_restored_wave_establishes_a_new_clock_baseline_after_multiple_reset_renders():
    guard,clock,_=setup();guard.begin_wave(0);emit(guard,clock,100)
    guard.observe(packet(),control_tick=0)
    clock.update(research_tick=1,sim_step=12,sim_time=.1);emit(guard,clock,101)
    guard.observe(packet(tick=1),control_tick=1)
    guard.retire([0,1],reason='natural_termination',control_tick=1)
    guard.begin_wave(1);emit(guard,clock,102);emit(guard,clock,103)
    _,proof=guard.observe(packet(tick=1),control_tick=0)
    assert proof['advancement_since_previous_observation'] is None
    clock.update(research_tick=2,sim_step=14,sim_time=.1+1/60);emit(guard,clock,104)
    _,proof=guard.observe(packet(tick=2),control_tick=1)
    assert proof['advancement_since_previous_observation']['research_ticks']==1


def test_wave_restore_precedes_new_reset_and_requires_new_frames():
    guard,clock,bindings=setup();guard.begin_wave(3);emit(guard,clock,100)
    guard.observe(packet(),control_tick=0)
    with pytest.raises(ValueError,match='active lanes'):guard.begin_wave(4)
    guard.retire([0,1],reason='natural_termination',control_tick=0)
    assert not any(b['handle'].enabled for b in bindings)
    guard.begin_wave(4)
    assert all(b['handle'].enabled for b in bindings)
    with pytest.raises(ValueError,match='No new real renderer frame'):guard.observe(packet(),control_tick=0)
    emit(guard,clock,101);guard.observe(packet(),control_tick=0)
    with pytest.raises(ValueError,match='one ordered'):guard.observe(packet(),control_tick=0)


def test_toggle_failure_restores_all_products_and_latches_failure():
    guard,clock,bindings=setup();guard.begin_wave(0);emit(guard,clock,100)
    guard.observe(packet(),control_tick=0);bindings[1]['handle'].fail_disable=True
    with pytest.raises(ValueError,match='disable readback'):guard.retire([0],reason='natural_termination',control_tick=0)
    assert all(b['handle'].enabled for b in bindings)
    assert guard.snapshot()['callback_error'] is not None
    with pytest.raises(ValueError,match='callback failed'):guard.observe(packet(tick=1),control_tick=1)
    guard.close();assert guard.snapshot()['closed']


def test_cleanup_attempts_every_restore_and_subscription_even_after_failure():
    guard,clock,bindings=setup();guard.begin_wave(0);emit(guard,clock,100)
    guard.observe(packet(),control_tick=0);guard.retire([0,1],reason='natural_termination',control_tick=0)
    bindings[0]['handle'].fail_restore=True;calls=[]
    guard._subscriptions=[SimpleNamespace(unsubscribe=lambda:calls.append('first')),
        SimpleNamespace(unsubscribe=lambda:calls.append('second'))]
    with pytest.raises(RuntimeError,match='cleanup failed'):guard.close()
    assert calls==['first','second'] and guard.snapshot()['closed']
    assert all(b['handle'].enabled for b in bindings[1:])


def test_artificial_retirement_is_finite_diagnostic_only():
    guard,clock,_=setup();guard.begin_wave(0);emit(guard,clock,100);guard.observe(packet(),control_tick=0)
    with pytest.raises(ValueError,match='finite diagnostic'):guard.retire([0],reason='diagnostic_censor',control_tick=0)
    guard,clock,_=setup(gate=1);guard.begin_wave(0);emit(guard,clock,100);guard.observe(packet(),control_tick=0)
    guard.retire([0],reason='diagnostic_censor',control_tick=0)
    with pytest.raises(ValueError,match='gate exhausted'):guard.observe(packet(tick=1),control_tick=1)


def test_callback_error_is_retained_and_prevents_all_future_consumption():
    guard,clock,_=setup();guard.begin_wave(0)
    with pytest.raises(ValueError,match='Hydra frame'):
        guard.note_drawable((0,0),dict(frame_number=float('nan')),clock)
    guard._latch(RuntimeError('native event failed'));emit(guard,clock,100)
    with pytest.raises(ValueError,match='native event failed'):guard.observe(packet(),control_tick=0)
    assert guard.snapshot()['last_control_tick']==-1


def test_exact_installed_sources_are_readable_without_cuda_or_simulator_start():
    if not all(render.Path(name).exists() for name in render.INSTALLED_SOURCES):
        pytest.skip('Machine-specific installed API receipt; runtime bind always requires these pins')
    assert render.verify_installed_sources()==render.INSTALLED_SOURCES
    assert not torch.cuda.is_initialized()


def test_installed_source_drift_is_rejected(monkeypatch,tmp_path):
    path=tmp_path/'installed.py';path.write_text('original\n')
    monkeypatch.setattr(render,'INSTALLED_SOURCES',{str(path):render.hashlib.sha256(path.read_bytes()).hexdigest()})
    render.verify_installed_sources();path.write_text('changed\n')
    with pytest.raises(ValueError,match='API source changed'):render.verify_installed_sources()
