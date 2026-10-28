import numpy as np
from scipy.spatial.transform import Rotation
from visuotactile.camera_rigs import renderer_pose_ros, optical_quat, rig_contract


def test_renderer_row_matrix_ros_roundtrip():
    # Independent non-identity world transform; catches transpose/GL/ROS errors.
    rot=Rotation.from_euler('xyz',[.3,-.7,1.2]).as_matrix()
    transform=np.eye(4);transform[:3,:3]=rot;transform[:3,3]=[.12,-.4,1.1]
    view=np.linalg.inv(transform).T
    pose=renderer_pose_ros(view)
    np.testing.assert_allclose(pose[:3],transform[:3,3],atol=1e-7)
    np.testing.assert_allclose(Rotation.from_quat(pose[[4,5,6,3]]).as_matrix(),rot@np.diag([1,-1,-1]),atol=1e-7)


def test_mount_looks_along_declared_parent_optical_direction():
    eye=np.array([.12,.16,.07]);target=np.array([0,-.14,.24])
    q=np.array(optical_quat(eye,target))
    np.testing.assert_allclose(Rotation.from_quat(q[[1,2,3,0]]).apply([0,0,1]),(target-eye)/np.linalg.norm(target-eye))
    assert len(rig_contract('wrist_bracket')['ids'])==3
    assert rig_contract('legacy_dual')['version']==2


def test_close_rig_moves_only_high_camera_preserves_real_wrist():
    old, new = rig_contract('wrist_bracket'), rig_contract('wrist_bracket_close')
    assert new['ids'] == old['ids'] == ['fixed_high','fixed_side','wrist_rgbd']
    assert new['version'] == 4 and old['version'] == 3
    assert new['wrist'] == old['wrist']
    assert new['fixed_focal_length'] == old['fixed_focal_length']
    assert new['fixed_views'][1] == old['fixed_views'][1]
    eye, target = map(np.asarray, old['fixed_views'][0])
    np.testing.assert_allclose(new['fixed_views'][0][0], target + .9*(eye-target))
    np.testing.assert_array_equal(new['fixed_views'][0][1],target)
