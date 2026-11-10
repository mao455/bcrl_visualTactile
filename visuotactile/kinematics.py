"""Encoder-only URDF forward kinematics. No simulator body state is an input."""
import hashlib
import math
import xml.etree.ElementTree as ET

import torch


FINGERS = ("left_index_DP", "left_middle_DP", "left_ring_DP", "left_thumb_DP", "left_pinky_DP")


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return torch.tensor([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                         [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


class EncoderKinematics:
    """Fixed-base URDF tree, positions in robot base and rotations base<-link.

    Uses link frames (not centers of mass or contact points). Fixed joints remain
    in the chain even when the simulator merges them. No collision files loaded.
    """
    def __init__(self, urdf, joint_names, links=FINGERS):
        raw = open(urdf, "rb").read()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        root = ET.fromstring(raw)
        incoming = {j.find("child").get("link"): j for j in root.findall("joint")}
        all_links = {l.get("name") for l in root.findall("link")}
        roots = all_links - set(incoming)
        if len(roots) != 1:
            raise ValueError("Expected one fixed robot root")
        self.root = roots.pop()
        self.links = tuple(links)
        needed = set()
        for link in links:
            while link != self.root:
                needed.add(link)
                link = incoming[link].find("parent").get("link")
        self.edges = []
        available = {self.root}
        while needed:
            progress = False
            for child in sorted(needed):
                j = incoming[child]
                parent = j.find("parent").get("link")
                if parent not in available:
                    continue
                typ = j.get("type")
                if typ not in ("fixed", "revolute", "continuous") or j.find("mimic") is not None:
                    raise ValueError("Unsupported kinematic joint: " + j.get("name"))
                origin = j.find("origin")
                xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
                rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
                axis = j.find("axis")
                axis = torch.tensor([float(v) for v in (axis.get("xyz") if axis is not None else "1 0 0").split()])
                axis = axis / axis.norm()
                index = None if typ == "fixed" else list(joint_names).index(j.get("name"))
                self.edges.append((parent, child, torch.tensor(xyz), rpy_matrix(rpy), axis, index))
                available.add(child)
                needed.remove(child)
                progress = True
            if not progress:
                raise ValueError("Unresolvable URDF tree")

    def __call__(self, q):
        poses, _ = self._poses(q)
        return torch.stack([poses[k][0] for k in self.links], 1), torch.stack([poses[k][1] for k in self.links], 1)

    def _poses(self, q):
        eye = torch.eye(3, device=q.device, dtype=q.dtype).expand(len(q), 3, 3)
        poses = {self.root: (q.new_zeros(len(q), 3), eye)}
        ancestors = {self.root: []}
        for parent, child, xyz, rot, axis, index in self.edges:
            p, r = poses[parent]
            pos = p + (r @ xyz.to(q)[None, :, None]).squeeze(-1)
            orient = r @ rot.to(q)
            ancestors[child] = list(ancestors[parent])
            if index is not None:
                a = axis.to(q)
                ancestors[child].append((index, pos, (orient @ a).squeeze(-1)))
                x, y, z = a.unbind()
                zero = x*0
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                angle = q[:, index, None, None]
                rotation = eye + angle.sin()*skew + (1-angle.cos())*(skew @ skew)
                orient = orient @ rotation
            poses[child] = (pos, orient)
        return poses, ancestors

    def point_jacobian(self, q, points_base):
        """Analytic encoder Jacobian at one supplied world point per link.

        ``points_base`` has shape [batch, links, 3]. Points are contact witnesses,
        never interpreted as the link-frame origins. Useful for an external
        geometric distance oracle without differentiating that oracle's solver.
        """
        _, ancestors = self._poses(q)
        result = q.new_zeros(len(q), len(self.links), 3, q.shape[-1])
        for ordinal, link in enumerate(self.links):
            for index, origin, axis in ancestors[link]:
                result[:, ordinal, :, index] = torch.cross(axis, points_base[:, ordinal]-origin, dim=-1)
        return result


class HandCollisionGeometry:
    """Signed distance to the actual source collision hulls and object boxes.

    FCL GJK/EPA supplies exact convex support witnesses. Their envelope derivative
    is propagated through URDF FK; the discrete witness choice is stop-gradient.
    This is not an OBB test or a sampled penetration proxy. Source convex hulls
    are not claimed bitwise identical to PhysX's internally simplified/cooked
    hulls. The required recorded live inventory confirms convexHull semantics.
    The geometric surface excludes PhysX contact/rest offsets.
    """
    identity = 'urdf_convex_surface_fcl_witness_v1'

    def __init__(self, metadata, live_inventory):
        from pathlib import Path
        import numpy as np
        import trimesh
        import fcl
        from scipy.spatial import ConvexHull
        self._np, self._fcl = np, fcl
        assets = metadata['config']['assets']
        urdf = Path(assets['robot_urdf'])
        root = ET.parse(urdf).getroot()
        records = live_inventory['robot_mesh_bounds']
        live = [r for r in records if '/left_' in r['path']]
        if not live or any(r['approximation'] != 'convexHull' for r in live):
            raise ValueError('Complete recorded convexHull hand inventory required')
        self.links, self.shapes, self.assets = [], [], {}
        cooked = live_inventory.get('cooked_hulls')
        for link in root.findall('link'):
            if not link.get('name').startswith('left_'):
                continue
            for collision in link.findall('collision'):
                mesh = collision.find('geometry/mesh')
                if mesh is None:
                    raise ValueError('Unsupported hand collider; never silently omit it')
                path = (urdf.parent / mesh.get('filename')).resolve()
                raw = path.read_bytes()
                self.assets['hand_mesh:'+mesh.get('filename')] = hashlib.sha256(raw).hexdigest()
                vertices = np.asarray(trimesh.load_mesh(path, process=False).vertices, dtype=np.float64)
                scale = np.fromstring(mesh.get('scale', '1 1 1'), sep=' ')
                origin = collision.find('origin')
                xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
                rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
                vertices = (vertices * scale) @ rpy_matrix(rpy).double().numpy().T + xyz
                vertices = np.unique(vertices, axis=0)
                if cooked is not None:
                    vertices = np.asarray(cooked[link.get('name')]['vertices_link'], dtype=np.float64)
                hull = ConvexHull(vertices)
                # scipy facet ordering is not guaranteed outward; FCL requires it.
                faces = hull.simplices.copy()
                triangles = vertices[faces]
                normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
                reverse = np.einsum('ij,ij->i', normals, hull.equations[:, :3]) < 0
                faces[reverse] = faces[reverse][:, ::-1]
                flat = np.c_[np.full(len(faces), 3), faces].reshape(-1).astype(np.int32)
                self.shapes.append(fcl.Convex(vertices, len(faces), flat))
                self.links.append(link.get('name'))
        if len(live) != len(self.shapes):
            raise ValueError('Recorded live hand collision census differs from URDF')
        self.fk = EncoderKinematics(urdf, metadata['joint_order'], self.links)
        if self.fk.sha256 != metadata['fk_urdf_sha256']:
            raise ValueError('Collision FK differs from recorded asset')
        self.assets['robot_urdf'] = self.fk.sha256
        path = Path(assets['object_urdf'])
        self.assets['object_urdf'] = hashlib.sha256(path.read_bytes()).hexdigest()
        if cooked is not None:
            import json
            if live_inventory.get('cooked_hulls_asset_sha256') != self.assets:
                raise ValueError('Cooked hull source assets differ')
            if set(cooked) != set(self.links):
                raise ValueError('Cooked hull census differs')
            self.assets['cooked_hulls'] = hashlib.sha256(json.dumps(cooked, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.identity = 'physx_reconstructed_convex_surface_fcl_witness_v1'
        self.boxes = []
        for collision in ET.parse(path).getroot().findall('.//collision'):
            box, origin = collision.find('geometry/box'), collision.find('origin')
            if box is None:
                raise ValueError('Object collision is not the audited exact boxes')
            xyz = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            rpy = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ') if origin is not None else np.zeros(3)
            size = np.fromstring(box.get('size'), sep=' ')
            self.boxes.append((fcl.Box(*size), xyz, rpy_matrix(rpy).double().numpy()))
        self.finger_indices = [[i for i, link in enumerate(self.links) if ('left_'+finger+'_') in link]
                               for finger in ('index', 'middle', 'ring', 'thumb', 'pinky')]
        if any(not indices for indices in self.finger_indices):
            raise ValueError('Incomplete five-finger collision geometry')
        self.request = fcl.DistanceRequest(enable_nearest_points=True, enable_signed_distance=True,
                                          gjk_solver_type=fcl.GJKSolverType.GST_LIBCCD)

    def distances(self, q, object_pose_base, jacobian=False):
        """[B, colliders] signed metres; negative is surface penetration.

        The gradient is deliberately zero for the arm and object pose. Only
        predicted hand configuration is regularized. Jacobian, when requested,
        is [B, colliders, 22] for constrained local projection on CPU.
        """
        np, fcl = self._np, self._fcl
        from .initial_geometry import rotation
        if q.shape[-1] != 29 or object_pose_base.shape != (len(q), 7):
            raise ValueError('Explicit aligned 29D configuration and object pose required')
        if not torch.isfinite(q).all() or not torch.isfinite(object_pose_base).all():
            raise ValueError('Non-finite geometric input')
        q_geometry = torch.cat((q[:, :7].detach(), q[:, 7:]), -1)
        xyz, orient = self.fk(q_geometry)
        p, r = xyz.detach().double().cpu().numpy(), orient.detach().double().cpu().numpy()
        pose = object_pose_base.detach().double().cpu().numpy()
        values = np.empty((len(q), len(self.links)))
        witnesses = np.empty((len(q), len(self.links), 3)); normals = np.empty_like(witnesses)
        for row in range(len(q)):
            rotation_object = rotation(pose[row, 3:])
            objects = [fcl.CollisionObject(shape, fcl.Transform(rotation_object @ rot,
                       pose[row, :3] + rotation_object @ local)) for shape, local, rot in self.boxes]
            for index, shape in enumerate(self.shapes):
                hand = fcl.CollisionObject(shape, fcl.Transform(r[row, index], p[row, index]))
                candidates = []
                for obj in objects:
                    result = fcl.DistanceResult()
                    value = fcl.distance(hand, obj, self.request, result)
                    a, b = np.asarray(result.nearest_points)
                    if not np.isfinite(value) or not np.isfinite(a).all() or not np.isfinite(b).all():
                        raise ValueError('FCL produced invalid contact witnesses')
                    if abs(value) < 1e-10:
                        # Surface ties have no unique derivative; reject instead
                        # of manufacturing a contact gradient.
                        normal = np.zeros(3)
                    else:
                        normal = (a-b)/value
                        if not np.isclose(np.linalg.norm(normal), 1., atol=2e-4):
                            raise ValueError('FCL witness and signed distance disagree')
                    candidates.append((value, a, normal))
                value, point, normal = min(candidates, key=lambda c: c[0])
                values[row, index], witnesses[row, index], normals[row, index] = value, point, normal
        witness = torch.as_tensor(witnesses, device=q.device, dtype=q.dtype)
        normal = torch.as_tensor(normals, device=q.device, dtype=q.dtype)
        local = (orient.detach().transpose(-1, -2) @ (witness-xyz.detach())[..., None]).squeeze(-1)
        differentiable_point = xyz + (orient @ local[..., None]).squeeze(-1)
        result = torch.as_tensor(values, device=q.device, dtype=q.dtype)
        result = result + ((differentiable_point-differentiable_point.detach())*normal).sum(-1)
        if not jacobian:
            return result
        derivative = (self.fk.point_jacobian(q_geometry, witness)*normal[..., None]).sum(-2)[..., 7:]
        return result, derivative


def rotate_force_to_link(force_base, rotation_base_link):
    return (rotation_base_link.transpose(-1, -2) @ force_base[..., None]).squeeze(-1)
