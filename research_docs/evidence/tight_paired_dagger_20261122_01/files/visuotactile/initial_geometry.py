"""Diagnostic OBB overlap of the original box colliders (no scene mutation).

This checks object/table/hole geometry, not the articulated hand's full mesh or
PhysX contact margins. Touching is permitted; positive geometric overlap is not.
"""
import xml.etree.ElementTree as ET
import numpy as np


def rotation(q):
    q=np.asarray(q)
    basis=np.eye(3)
    return (basis+2*np.cross(q[1:],np.cross(q[1:],basis)+q[0]*basis)).T


def boxes(path,pose):
    r=rotation(pose[3:]);out=[]
    for collision in ET.parse(path).getroot().findall('.//collision'):
        box=collision.find('geometry/box');origin=collision.find('origin')
        if box is None:raise ValueError('Initialization diagnostic requires audited box colliders')
        xyz=np.fromstring(origin.get('xyz','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        rpy=np.fromstring(origin.get('rpy','0 0 0'),sep=' ') if origin is not None else np.zeros(3)
        if np.any(rpy):raise ValueError('Nonzero collider origin rotation is not audited')
        out.append((np.asarray(pose[:3])+r@xyz,r,np.fromstring(box.get('size'),sep=' ')/2))
    return out


def overlaps(a,b):
    ca,ra,ha=a;cb,rb,hb=b
    axes=[*ra.T,*rb.T,*[np.cross(x,y) for x in ra.T for y in rb.T]]
    for axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-8:continue
        axis=axis/length
        margin=np.abs(ra.T@axis)@ha+np.abs(rb.T@axis)@hb-abs((cb-ca)@axis)
        if margin<=1e-6:return False
    return True


def inspect(metadata,initial):
    assets=metadata['config']['assets']
    obj=boxes(assets['object_urdf'],initial['object_pose_w'])
    result={}
    for name,filename in (('table','table_urdf'),('hole','receptive_urdf')):
        others=boxes(assets[filename],initial[name+'_pose_w'])
        result[name+'_overlap']=any(overlaps(a,b) for a in obj for b in others)
    result['valid_box_initialization']=not any(result.values())
    result['scope']='Original object/table/hole box colliders; not hand meshes or PhysX contact margins'
    return result
