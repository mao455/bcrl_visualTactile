"""Conservative initial clearance from collision meshes at live body poses.

Scene queries in the installed CPU tensor backend report inconsistent articulation
overlaps across resets. This opt-in gate uses no USD world pose, visual
mesh, physics step or state mutation. Every collision mesh must be accounted for.
"""
import numpy as np
from .initial_geometry import boxes, inspect, rotation


def separation(a,b):
    """Largest separating-axis gap of two OBBs; <=0 means possible overlap."""
    ca,ra,ha=a;cb,rb,hb=b
    axes=[*ra.T,*rb.T,*[np.cross(x,y) for x in ra.T for y in rb.T]]
    gaps=[]
    for axis in axes:
        length=np.linalg.norm(axis)
        if length<1e-8:continue
        axis=axis/length
        gaps.append(abs((cb-ca)@axis)-np.abs(ra.T@axis)@ha-np.abs(rb.T@axis)@hb)
    return float(max(gaps))


def mesh_inventory(env):
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    stage=omni.usd.get_context().get_stage()
    paths=env.robot.root_physx_view.link_paths[0]
    result=[]
    for index,path in enumerate(paths):
        body=stage.GetPrimAtPath(path)
        for prim in Usd.PrimRange(body,Usd.TraverseInstanceProxies()):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:continue
            approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
            if approximation not in ('convexHull','convexDecomposition','none'):
                raise ValueError('Unaudited collision approximation '+str(approximation))
            clouds=[]
            for mesh in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
                if mesh.IsA(UsdGeom.Mesh):
                    points=np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(),dtype=np.float64)
                    relative,_=UsdGeom.XformCache().ComputeRelativeTransform(mesh,body)
                    m=np.asarray(relative);clouds.append(points@m[:3,:3]+m[3,:3])
            if not clouds:raise ValueError('Unaccounted robot collider '+str(prim.GetPath()))
            vertices=np.concatenate(clouds)
            if len(vertices)<4 or not np.isfinite(vertices).all():raise ValueError('Invalid mesh vertices')
            low,high=vertices.min(0),vertices.max(0)
            result.append(dict(path=str(prim.GetPath()),body=path,index=index,
                center=(low+high)/2,half=(high-low)/2,approximation=approximation,vertices=len(vertices)))
    if len(result)<29:raise ValueError('Incomplete original robot collider inventory')
    return result


def clearance(env, *, lane=0):
    if not 0 <= lane < env.num_envs:raise ValueError('Invalid collision-admission lane')
    if not hasattr(env,'_initial_collision_mesh_inventory'):
        env._initial_collision_mesh_inventory=mesh_inventory(env)
    # This accessor updates kinematics without integrating simulation time.
    poses=env.robot.data.body_link_pose_w[lane].cpu().numpy()
    initial={k:v[lane].cpu().numpy() for k,v in env._diagnostics().items()}
    objects=boxes(env.cfg.assets.object_urdf,initial['object_pose_w'])
    report=[]
    for mesh in env._initial_collision_mesh_inventory:
        pose=poses[mesh['index']];r=rotation(pose[3:]);center=pose[:3]+r@mesh['center']
        # A 1mm envelope also keeps near-touching mesh bounds conservative.
        bound=(center,r,mesh['half']+.001)
        gap=min(separation(obj,bound) for obj in objects)
        report.append(dict(path=mesh['path'],body=mesh['body'],approximation=mesh['approximation'],
            vertex_count=mesh['vertices'],center_w=center.tolist(),half_with_margin=bound[2].tolist(),gap_m=gap))
    static=inspect(dict(config=dict(assets=env.cfg.assets.to_dict())),initial)
    return dict(legal=static['valid_box_initialization'] and all(r['gap_m']>0 for r in report),
        boxes=static,robot_mesh_bounds=report,minimum_gap_m=min(r['gap_m'] for r in report),
        version='live_collision_mesh_bounds_v1',margin_m=.001,
        definition='Object exact OBB vs every robot collision mesh bounding OBB at live link pose; static exact boxes',
        limitation='Conservative mesh bounds may reject free configurations; no claim of signed mesh penetration depth')
