import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from visuotactile import tiled_camera_pose_sync as sync
from visuotactile.execution_backend import (execution_backend, backend_receipt, startup_arguments,
    renderer_resource_readback, require_renderer_resources, DESCRIPTOR_VERSION, TILED_POSE_VERSION)


def candidate(**changes):
    return execution_backend(dict(version=TILED_POSE_VERSION, physics_device='cuda:0',
        camera_backend='tiled', tiled_fixed_camera_usd_sync=True, **changes))


def fixture_env(tmp_path, monkeypatch):
    class Camera:
        pass
    class TiledCamera(Camera):
        def set_world_poses_from_view(self, eye, target):
            self._view.fabric = eye.clone()
            if self._view._sync_usd_on_fabric_write:
                self._view.usd = eye.clone()
    class XformPrimView:
        def __init__(self):
            self._device = 'cuda:0'
            self._use_fabric = True
            self._sync_usd_on_fabric_write = False
            self.fabric = self.usd = None
    Camera.__module__ = 'isaaclab.sensors.camera.camera'
    TiledCamera.__module__ = 'isaaclab.sensors.camera.tiled_camera'
    XformPrimView.__module__ = 'isaaclab.sim.views.xform_prim_view'
    paths = {}; pins = {}
    for cls in (Camera,TiledCamera,XformPrimView):
        path = tmp_path/(cls.__name__+'.py'); path.write_text('reviewed '+cls.__name__)
        paths[cls] = str(path); pins[sync._qualified(cls)] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(sync, 'PINNED_SOURCES', pins)
    monkeypatch.setattr(sync.inspect, 'getsourcefile', lambda cls: paths[cls])
    cameras = []
    for i in range(3):
        c = TiledCamera(); c._view = XformPrimView(); c.device = 'cuda:0'
        c.cfg = SimpleNamespace(prim_path=f'/World/env_.*/Camera{i}'); cameras.append(c)
    env = SimpleNamespace(execution_backend=candidate(), research_cameras=cameras,
        camera_rig=dict(version=4,name='wrist_bracket_close',fixed_views=[((1.,2.,3.),(0.,0.,0.)),
            ((-1.,2.,3.),(0.,0.,0.))],wrist={}),device='cpu',scene=SimpleNamespace(env_origins=torch.zeros(2,3)))
    return env, paths


def test_v3_is_explicit_and_distinct_with_v2_capacity_readback():
    old = execution_backend(dict(version=DESCRIPTOR_VERSION,physics_device='cuda:0',camera_backend='tiled'))
    new = candidate()
    assert 'tiled_fixed_camera_usd_sync' not in old
    assert startup_arguments(new,num_envs=8)==startup_arguments(old,num_envs=8)
    assert backend_receipt(new,num_envs=8)['configuration_sha256']!=backend_receipt(old,num_envs=8)['configuration_sha256']
    receipt=renderer_resource_readback(new,{'/rtx/descriptorSets':360000,'/rtx/reservedDescriptors':900000}.get)
    require_renderer_resources(receipt)
    assert receipt['matched']
    for change in ({}, {'tiled_fixed_camera_usd_sync':False}, {'tiled_fixed_camera_usd_sync':1},
                   {'tiled_fixed_camera_usd_sync':True,'physics_device':'cpu'},
                   {'tiled_fixed_camera_usd_sync':True,'camera_backend':'standard'}):
        cfg=dict(version=TILED_POSE_VERSION,physics_device='cuda:0',camera_backend='tiled');cfg.update(change)
        with pytest.raises(ValueError,match='v3 requires'):
            execution_backend(cfg)
    with pytest.raises(ValueError,match='Unknown execution_backend fields'):
        execution_backend(dict(old,tiled_fixed_camera_usd_sync=True))


def test_actual_view_configuration_writes_fixed_usd_only_for_new_identity(tmp_path,monkeypatch):
    env,_=fixture_env(tmp_path,monkeypatch)
    # Execute the real project's method without importing Kit. This checks the
    # enable-before-pose-write ordering, not a mirrored copy of that method.
    path=Path(__file__).parents[1]/'visuotactile/sim/environment.py'
    tree=ast.parse(path.read_text()); cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='configure_camera_views')
    module=ast.Module(body=[method],type_ignores=[]); space={'torch':torch}
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),space)
    configure=space['configure_camera_views']
    old=execution_backend(dict(version=DESCRIPTOR_VERSION,physics_device='cuda:0',camera_backend='tiled'))
    env.execution_backend=old; configure(env)
    assert all(c._view.usd is None for c in env.research_cameras)
    env.execution_backend=candidate(); configure(env)
    assert torch.equal(env.research_cameras[0]._view.usd,torch.tensor([[1.,2.,3.],[1.,2.,3.]]))
    assert torch.equal(env.research_cameras[1]._view.usd,torch.tensor([[-1.,2.,3.],[-1.,2.,3.]]))
    assert env.research_cameras[2]._view.usd is None
    assert env.research_cameras[2]._view._sync_usd_on_fabric_write is False
    receipt=sync.fixed_camera_pose_sync_readback(env)
    assert receipt['observed_fixed']==[True,True] and receipt['observed_wrist'] is False
    assert len(receipt['sources'])==2 and all(len(x)==3 for x in receipt['sources'])
    assert receipt['instance_setting_matched'] and not receipt['live_renderer_pose_accepted']
    assert sync.enable_fixed_camera_pose_sync(env)==receipt  # repeated pose configuration is safe
    # The real research metadata emits the verified receipt, and fails if an
    # instance drifts, instead of silently serializing the requested boolean.
    metadata_method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='research_metadata')
    space.update(JOINT_NAMES_CANONICAL=['joint'],DISTAL_BODIES=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[metadata_method],type_ignores=[])),str(path),'exec'),space)
    env.action_control_mode='absolute_joint_position_target_rad';env.wrench_frame_mode='world_hold'
    env.initial_offset_record=None;env.robot=SimpleNamespace(joint_names=['joint'],
        data=SimpleNamespace(joint_pos_limits=torch.zeros(1,1,2)))
    env._perm_lab_to_canon=[0];env.research_contact_paths=[];env.physics_dt=1/120;env.step_dt=1/60
    env.cfg=SimpleNamespace(student_obs=SimpleNamespace(to_dict=lambda:{}))
    env.perception_version=1;env.camera_content_validation=True
    metadata=space['research_metadata'](env)
    assert metadata['tiled_fixed_camera_pose_sync']==receipt
    env.research_cameras[1]._view._sync_usd_on_fabric_write=False
    with pytest.raises(RuntimeError,match='readback changed'):space['research_metadata'](env)


@pytest.mark.parametrize('fault',['source','missing','integer','already_enabled','shared','cpu_view'])
def test_unreviewed_or_inconsistent_instances_fail_before_any_write(tmp_path,monkeypatch,fault):
    env,paths=fixture_env(tmp_path,monkeypatch)
    view=env.research_cameras[1]._view
    if fault=='source':Path(next(iter(paths.values()))).write_text('source drift')
    elif fault=='missing':del view._sync_usd_on_fabric_write
    elif fault=='integer':view._sync_usd_on_fabric_write=0
    elif fault=='already_enabled':view._sync_usd_on_fabric_write=True
    elif fault=='shared':env.research_cameras[1]._view=env.research_cameras[0]._view
    elif fault=='cpu_view':view._device='cpu'
    with pytest.raises(RuntimeError):sync.enable_fixed_camera_pose_sync(env)
    assert env.research_cameras[0]._view._sync_usd_on_fabric_write is False
    assert not hasattr(env,'_tiled_fixed_camera_pose_sync_receipt')


def test_readback_rejects_later_fixed_or_wrist_drift(tmp_path,monkeypatch):
    env,_=fixture_env(tmp_path,monkeypatch)
    with pytest.raises(RuntimeError,match='not initialized'):sync.fixed_camera_pose_sync_readback(env)
    sync.enable_fixed_camera_pose_sync(env)
    env.research_cameras[1]._view._sync_usd_on_fabric_write=False
    with pytest.raises(RuntimeError,match='readback changed'):sync.fixed_camera_pose_sync_readback(env)
    env.research_cameras[1]._view._sync_usd_on_fabric_write=True
    env.research_cameras[2]._view._sync_usd_on_fabric_write=True
    with pytest.raises(RuntimeError,match='readback changed'):sync.fixed_camera_pose_sync_readback(env)
    env.research_cameras[2]._view._sync_usd_on_fabric_write=False
    replacement=type(env.research_cameras[0]._view)();replacement._sync_usd_on_fabric_write=True
    env.research_cameras[0]._view=replacement
    with pytest.raises(RuntimeError,match='instance was replaced'):sync.fixed_camera_pose_sync_readback(env)


def test_old_backend_never_reads_camera_private_state():
    for cfg in (None,execution_backend(),execution_backend(dict(camera_backend='tiled',physics_device='cuda:0'))):
        env=SimpleNamespace(execution_backend=cfg)
        assert sync.enable_fixed_camera_pose_sync(env) is None
        assert sync.fixed_camera_pose_sync_readback(env) is None
