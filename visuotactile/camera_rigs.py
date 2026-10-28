"""Explicit observation-only camera layouts. No object state or task stages."""
import numpy as np
from scipy.spatial.transform import Rotation

RIGS = ('legacy_dual', 'focused_pair', 'wrist_side', 'wrist_opposite', 'wrist_dorsal', 'wrist_lower', 'wrist_overhead', 'wrist_bracket', 'third_fixed')
FIXED = (((.32,-.45,1.55),(0.,0.,.68)), ((-.58,-.06,1.05),(0.,0.,.66)))
WRISTS = {
    'wrist_side': ((.10,0.,.055),(0.,0.,.25)),
    'wrist_opposite': ((-.10,0.,.055),(0.,0.,.25)),
    'wrist_dorsal': ((0.,.11,.075),(0.,0.,.25)),
    'wrist_lower': ((0.,-.14,.035),(0.,-.09,.25)),
    'wrist_overhead': ((0.,.18,.09),(0.,-.10,.28)),
    'wrist_bracket': ((.12,.16,.07),(0.,-.14,.24)),
}


def optical_quat(eye, target, up=(0.,1.,0.)):
    """ROS optical axes x right, y down, z forward; wxyz in parent frame."""
    z=np.asarray(target)-eye;z=z/np.linalg.norm(z)
    x=np.cross(z,up);x=x/np.linalg.norm(x);y=np.cross(z,x)
    q=Rotation.from_matrix(np.stack((x,y,z),1)).as_quat()
    return tuple(q[[3,0,1,2]])


def rig_contract(name):
    if name not in RIGS: raise ValueError('Unknown camera rig')
    if name=='legacy_dual':return dict(version=2,name=name,ids=['global_oblique','local_opposite'])
    ids=['fixed_high','fixed_side']
    result=dict(version=3,name=name,ids=ids,fixed_views=FIXED,fixed_focal_length=40.)
    if name in WRISTS:
        eye,target=WRISTS[name]
        ids.append('wrist_rgbd')
        result['wrist']=dict(parent_body='iiwa14_link_7',pos_parent_m=eye,
            quat_parent_ros_wxyz=optical_quat(eye,target),convention='ROS optical +Z forward +X right +Y down')
    elif name=='third_fixed':ids.append('fixed_front')
    return result


def renderer_pose_ros(view_transform):
    """Replicator row-vector world->OpenGL view matrix to world ROS pose.

    Read from the same render product as RGB/depth. Unlike CPU USD XformCache,
    the annotator sees physics/Fabric parent motion in this installed Lab fork.
    """
    world_from_gl = np.linalg.inv(np.asarray(view_transform).reshape(4, 4)).T
    rotation = world_from_gl[:3, :3] @ np.diag([1., -1., -1.])
    q = Rotation.from_matrix(rotation).as_quat()
    return np.r_[world_from_gl[:3, 3], q[[3, 0, 1, 2]]].astype(np.float32)
