"""Exercise native reset/goal methods without loading Kit or integrating physics."""
import ast
import math
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch

SOURCE = Path(__file__).resolve().parents[1] / 'isaacsimenvs/tasks/precise_assembly/precise_assembly_env.py'


def native_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PreciseAssemblyEnv')
    nodes = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in
             ('_reset_peg_episode', '_write_goal_pose', '_clear_goal_trackers')]
    # Quaternion test inputs use identity rotations; reject unexpected input rather than approximate it.
    def apply(q, p):
        assert torch.equal(q, torch.tensor([1., 0., 0., 0.]).expand_as(q))
        return p
    def multiply(q, p):
        assert torch.equal(q, torch.tensor([1., 0., 0., 0.]).expand_as(q))
        return p
    scope = dict(torch=torch, math=math, TABLE_HALF_HEIGHT=.15, quat_apply=apply, quat_mul=multiply)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), scope)
    return scope


class KinematicBody:
    def __init__(self, n):
        self.poses = torch.full((n, 7), -99.)
    def write_root_pose_to_sim(self, pose, env_ids):
        self.poses[env_ids] = pose
    def write_root_velocity_to_sim(self, *args, **kwargs):
        raise AssertionError('PhysX does not accept velocity setters on kinematic bodies')


def environment():
    n = 3
    e = SimpleNamespace(device='cpu', cfg=SimpleNamespace(precise_assembly=SimpleNamespace(
        random_goal_max_successes=3, hole_x_range=(.04, .04), hole_y_range=(-.03, -.03),
        hole_yaw_range_deg=0., goal_xy_obs_noise=.002, goal_yaw_obs_noise_deg=.1),
        reset=SimpleNamespace(fixed_goal_pose=[.2, .1, .8, 1., 0., 0., 0.])),
        scene=SimpleNamespace(env_origins=torch.tensor([[0., 0., 0.], [2., 0., 0.], [4., 0., 0.]])),
        is_random_goal_env=torch.tensor([False, True, False]), _num_total_insertion_goals=2,
        env_max_goals=torch.full((n,), 2), _table_z_per_env=torch.full((n,), .38),
        _pih_hole_z_offset=0., hole_pos=torch.zeros(n, 3), hole_quat_wxyz=torch.zeros(n, 4),
        hole=KinematicBody(n), goal_viz=KinematicBody(n), goal_pos_obs_noise=torch.ones(n, 3),
        goal_yaw_obs_noise=torch.ones(n), _num_prelude_goals=0, _successes=torch.zeros(n, dtype=torch.long),
        _insert_pos_rel=torch.tensor([[0., 0., .186], [0., 0., .086]]),
        _insert_quat_wxyz=torch.tensor([[1., 0., 0., 0.], [1., 0., 0., 0.]]))
    for name in ('retract_phase', 'retract_succeeded', '_just_entered_retract', '_just_retracted'):
        setattr(e, name, torch.ones(n, dtype=torch.bool))
    for name in ('_closest_keypoint_max_dist', '_closest_fingertip_dist', '_near_goal_steps'):
        setattr(e, name, torch.ones(n))
    for name, value in native_methods().items():
        if name.startswith('_') and callable(value):
            setattr(e, name, MethodType(value, e))
    return e


def test_partial_reset_writes_kinematic_poses_preserves_other_env_and_tracks_goals():
    e = environment()
    ids = torch.tensor([0, 1])
    e._reset_peg_episode(ids)
    torch.testing.assert_close(e.hole.poses[0, :3], torch.tensor([.04, -.03, .53]))
    torch.testing.assert_close(e.hole.poses[1, :3], torch.tensor([2., 0., -1.]))
    torch.testing.assert_close(e.goal_viz.poses[0, :3], torch.tensor([.04, -.03, .716]))
    torch.testing.assert_close(e.goal_viz.poses[1, :3], torch.tensor([2.2, .1, .8]))
    assert (e.hole.poses[2] == -99).all() and (e.goal_viz.poses[2] == -99).all()
    assert e.env_max_goals.tolist() == [2, 3, 2]
    assert e.retract_phase.tolist() == [False, False, True]
    assert e._near_goal_steps.tolist() == [0., 0., 1.]
    assert torch.equal(e.goal_pos_obs_noise[1], torch.zeros(3))
    e._successes[0] = 1
    e._write_goal_pose(torch.tensor([0]))
    torch.testing.assert_close(e.goal_viz.poses[0, :3], torch.tensor([.04, -.03, .616]))


def test_reset_preserves_seeded_native_sampling_and_does_not_touch_global_config():
    a, b = environment(), environment()
    ids = torch.tensor([0, 1])
    torch.manual_seed(27150000)
    a._reset_peg_episode(ids)
    after_a = torch.get_rng_state().clone()
    torch.manual_seed(27150000)
    b._reset_peg_episode(ids)
    assert torch.equal(after_a, torch.get_rng_state())
    assert torch.equal(a.goal_pos_obs_noise, b.goal_pos_obs_noise)
    assert torch.equal(a.goal_yaw_obs_noise, b.goal_yaw_obs_noise)
    assert a.cfg.precise_assembly.goal_xy_obs_noise == .002
