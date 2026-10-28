"""Lossless storage v1: uint8 RGB, float32 depth, explicit post-frame references.

Perception/schema2 semantics are unchanged. MP4s are viewing derivatives ONLY.
Conversion is published only after every original field is reconstructed exactly.
"""
import hashlib
import json
from pathlib import Path
import os

import h5py
import numpy as np

IMAGE_FIELDS = {"obs/image": "pre_image", "obs/depth_m": "pre_depth",
                "obs/depth_valid": "pre_valid",
                "transition/observation_after/image": "post_image",
                "transition/observation_after/depth_valid": "post_valid"}


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rgb_u8(rgb):
    value = np.rint(rgb * np.float32(255)).astype(np.uint8)
    if not np.array_equal(value.astype(np.float32) / np.float32(255), rgb):
        raise ValueError("RGB is not exactly reversible uint8/255; conversion refused")
    return value


class VirtualArray:
    def __init__(self, group, name):
        self.g, self.name = group, name
        self.kind = IMAGE_FIELDS[name]
        base = group["frames/depth_m"]
        self.shape = (int(group.attrs["length"]), base.shape[1],
                      4 if self.kind.endswith("image") else 1, *base.shape[-2:])
        self.dtype = np.dtype(bool if self.kind.endswith("valid") else np.float32)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        tail = index[1:] if isinstance(index, tuple) else ()
        index = index[0] if isinstance(index, tuple) else index
        rows = np.arange(len(self))[index]
        scalar = np.ndim(rows) == 0
        rows = np.atleast_1d(rows)
        ids = rows if self.kind.startswith("pre") else self.g["frames/post_index"][:][rows]
        values = []
        for i in ids:
            if i < 0:
                values.append(np.zeros(self.shape[1:], dtype=self.dtype))
            elif self.kind.endswith("image"):
                rgb = self.g["frames/rgb_u8"][i].astype(np.float32) / np.float32(255)
                values.append(np.concatenate((rgb, self.g["frames/depth_m"][i]), axis=1))
            else:
                key = "depth_valid" if self.kind.endswith("valid") else "depth_m"
                values.append(self.g["frames/"+key][i])
        value = np.stack(values) if values else np.empty((0, *self.shape[1:]), dtype=self.dtype)
        value = value[0] if scalar else value
        return value[tail if scalar else (slice(None), *tail)] if tail else value


class GroupView:
    """Small read-only h5py-compatible facade used by existing Dataset code."""
    def __init__(self, group, prefix=""):
        self.g, self.prefix, self.attrs = group, prefix, group.attrs

    def __getitem__(self, key):
        name = self.prefix+key
        if name in IMAGE_FIELDS:
            return VirtualArray(self.g, name)
        obj = self.g[name]
        return GroupView(self.g, name+"/") if isinstance(obj, h5py.Group) else obj

    def __contains__(self, key):
        return self.prefix+key in IMAGE_FIELDS or self.prefix+key in self.g

    def items(self):
        obj = self.g[self.prefix.rstrip("/")] if self.prefix else self.g
        keys = set(obj.keys())
        keys.update(k[len(self.prefix):].split("/")[0] for k in IMAGE_FIELDS if k.startswith(self.prefix))
        keys.discard("frames")
        return [(k, self[k]) for k in sorted(keys)]

    def visititems(self, visitor):
        def visit(name, obj):
            if not name.startswith("frames/"):
                visitor(name, obj)
        self.g.visititems(visit)
        for name in IMAGE_FIELDS:
            visitor(name, VirtualArray(self.g, name))


class EpisodeGroups:
    def __init__(self, group, compact):
        self.g, self.compact = group, compact
    def __iter__(self):
        return iter(self.g)
    def __getitem__(self, key):
        return GroupView(self.g[key]) if self.compact else self.g[key]
    def items(self):
        return [(k, self[k]) for k in self]
    def __len__(self):
        return len(self.g)


class EpisodeFile:
    def __init__(self, path):
        self.file = h5py.File(path, "r")
        self.attrs = self.file.attrs
        version = int(self.attrs.get("storage_version", 0))
        if version not in (0, 1):
            self.file.close()
            raise ValueError("Unknown storage version")
        self.episodes = EpisodeGroups(self.file["episodes"], version == 1)
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.file.close()
    def __getitem__(self, key):
        if key == "episodes":
            return self.episodes
        if key.startswith("episodes/"):
            parts = key.split("/", 2)
            group = self.episodes[parts[1]]
            return group if len(parts) == 2 else group[parts[2]]
        return self.file[key]
    def get(self, key, default=None):
        return self[key] if key in self.file else default


def write_video(path, frames, fps=60, *, lossless=False):
    """RGB FFV1 test path is exact; ordinary MP4 is a labeled derivative."""
    import av
    count = 0
    with av.open(str(path), "w", format="matroska" if lossless else "mp4") as container:
        stream = None
        for array in frames:
            array = np.ascontiguousarray(array, dtype=np.uint8)
            if stream is None:
                stream = container.add_stream("ffv1" if lossless else "libx264", rate=fps)
                stream.width, stream.height = array.shape[1], array.shape[0]
                stream.pix_fmt = "bgr0" if lossless else "yuv420p"
                stream.codec_context.thread_count = 2
                if not lossless:
                    stream.options = {"crf": "18", "preset": "fast"}
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts = count
            for packet in stream.encode(frame):
                container.mux(packet)
            count += 1
        if stream is None:
            raise ValueError("Empty video")
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        decoded = sum(1 for _ in container.decode(video=0))
    if decoded != count:
        raise ValueError("Video frame count mismatch")
    return count


def convert_episode(source, destination, *, result=None):
    """One episode, no source mutation. Failed .partial files are retained."""
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    partial = destination/"data.h5.partial"
    with h5py.File(source, "r") as src, h5py.File(partial, "x") as dst:
        if int(src.attrs["schema_version"]) != 2 or len(src["episodes"]) != 1:
            raise ValueError("Compact storage currently requires one schema2 episode")
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs.update(storage_version=1, source_sha256=sha256(source),
                         video_role="viewing_derivative_not_training_RGB",
                         result_json=json.dumps(result or {}))
        name = next(iter(src["episodes"]))
        g, out = src["episodes"][name], dst.create_group("episodes/"+name)
        for k, v in g.attrs.items():
            out.attrs[k] = v
        n = int(g.attrs["length"])
        if n < 1:
            raise ValueError("No committed rows")
        def copy(name, obj):
            if isinstance(obj, h5py.Dataset) and name not in IMAGE_FIELDS:
                if len(obj) != n:
                    raise ValueError("Uncommitted trailing rows: "+name)
                out.create_dataset(name, data=obj[:], compression="gzip", compression_opts=4, shuffle=True)
        g.visititems(copy)
        post = g["transition/observation_after"]
        post_ids = np.full(n, -1, dtype=np.int64)
        extra = bool(post["available"][n-1])
        shape = g["obs/image"].shape
        frames = n+int(extra)
        opts = dict(compression="gzip", compression_opts=4, shuffle=True, chunks=(1,shape[1],1,*shape[-2:]))
        rgb = out.create_dataset("frames/rgb_u8", (frames,shape[1],3,*shape[-2:]), dtype="u1", **opts)
        depth = out.create_dataset("frames/depth_m", (frames,shape[1],1,*shape[-2:]), dtype="f4", **opts)
        valid = out.create_dataset("frames/depth_valid", depth.shape, dtype="?", **opts)
        for t in range(n):
            im = g["obs/image"][t]
            if not np.array_equal(im[:,3:], g["obs/depth_m"][t]):
                raise ValueError("Duplicate metric depth differs")
            rgb[t], depth[t], valid[t] = rgb_u8(im[:,:3]), im[:,3:], g["obs/depth_valid"][t]
            if post["available"][t]:
                post_ids[t] = t+1
                if t+1 < n:
                    for field in ("image","depth_valid","camera_time_s","camera_frame","episode_id","step_id"):
                        if not np.array_equal(post[field][t], g["obs/"+field][t+1]):
                            raise ValueError("Post/pre dedup misalignment: "+field)
            elif np.any(post["image"][t]) or np.any(post["depth_valid"][t]):
                raise ValueError("Unavailable post frame contains data")
        if extra:
            im = post["image"][n-1]
            rgb[n], depth[n], valid[n] = rgb_u8(im[:,:3]), im[:,3:], post["depth_valid"][n-1]
        out.create_dataset("frames/post_index", data=post_ids)
        out["frames"].attrs.update(extra_final_post=extra, missing_frame_index=-1)
    # Reconstruct every field (including duplicates and missing placeholders).
    compared = 0
    with h5py.File(source, "r") as src, EpisodeFile(partial) as dst:
        original, reconstructed = src["episodes"][name], dst["episodes"][name]
        def compare(key, obj):
            nonlocal compared
            if isinstance(obj, h5py.Dataset):
                other = reconstructed[key]
                if obj.shape != other.shape or obj.dtype != other.dtype:
                    raise ValueError("Roundtrip shape/dtype: "+key)
                for start in range(0,n,8):
                    if not np.array_equal(obj[start:start+8], other[start:start+8]):
                        raise ValueError("Roundtrip values: "+key)
                compared += 1
        original.visititems(compare)
    with h5py.File(partial) as f:
        rgb = f["episodes"][name]["frames/rgb_u8"]
        for view in range(2):
            write_video(destination/f"camera_{view}.mp4.partial", (x[view].transpose(1,2,0) for x in rgb))
    for name_on_disk in ("data.h5", "camera_0.mp4", "camera_1.mp4"):
        p = destination/(name_on_disk+".partial")
        with p.open("rb") as fd:
            os.fsync(fd.fileno())
        p.rename(destination/name_on_disk)
    return dict(episode=name, length=n, frame_count=frames, exact_fields=compared,
                extra_final_post=extra, source_sha256=sha256(source),
                files={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in destination.iterdir()})
