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
        eye = torch.eye(3, device=q.device, dtype=q.dtype).expand(len(q), 3, 3)
        poses = {self.root: (q.new_zeros(len(q), 3), eye)}
        for parent, child, xyz, rot, axis, index in self.edges:
            p, r = poses[parent]
            pos = p + (r @ xyz.to(q)[None, :, None]).squeeze(-1)
            orient = r @ rot.to(q)
            if index is not None:
                a = axis.to(q)
                x, y, z = a.unbind()
                zero = x*0
                skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
                angle = q[:, index, None, None]
                rotation = eye + angle.sin()*skew + (1-angle.cos())*(skew @ skew)
                orient = orient @ rotation
            poses[child] = (pos, orient)
        return torch.stack([poses[k][0] for k in self.links], 1), torch.stack([poses[k][1] for k in self.links], 1)


def rotate_force_to_link(force_base, rotation_base_link):
    return (rotation_base_link.transpose(-1, -2) @ force_base[..., None]).squeeze(-1)
