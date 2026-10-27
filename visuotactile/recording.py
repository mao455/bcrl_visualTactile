"""Episode-based HDF5 recorder; model inputs and privileged labels are separate."""
import json
import hashlib
import os
import pickle
from pathlib import Path
import h5py
import numpy as np


def model_fingerprint(model):
    """Parameter/buffer identity, independent of checkpoint filename."""
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()

def numpy_tree(value):
    if isinstance(value, dict):
        return {k: numpy_tree(v) for k, v in value.items()}
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value)


def flatten(value, prefix=""):
    result = {}
    for key, child in value.items():
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(child, dict):
            result.update(flatten(child, path))
        else:
            result[path] = child
    return result


class EpisodeWriter:
    def __init__(self, path, metadata):
        self.file = h5py.File(path, "x")
        self.file.attrs["schema_version"] = metadata.get("perception_version", 1)
        self.file.attrs["trace_version"] = metadata.get("trace_version",1)
        self.file.attrs["metadata_json"] = json.dumps(metadata, default=str)
        self.pending = {}
        self.count = 0

    def append(self, observation, transition, terminated, truncated, action_source):
        obs, tr = numpy_tree(observation), numpy_tree(transition)
        terminated, truncated = numpy_tree(terminated), numpy_tree(truncated)
        for i in range(len(obs["episode_id"])):
            key = (i, int(obs["episode_id"][i]))
            if int(tr["episode_id"][i]) != key[1] or tr["step_id"][i] != obs["step_id"][i]:
                raise ValueError("Observation/action episode or step misalignment")
            record = flatten({"obs": obs, "transition": tr})
            record = {k: v[i] for k, v in record.items()}
            record["terminated"] = terminated[i]
            record["truncated"] = truncated[i]
            group_name = f"episodes/env{key[0]:03d}_ep{key[1]:06d}"
            group = self.file.require_group(group_name)
            length = int(group.attrs.get("length",0))
            if int(obs["step_id"][i]) != length:
                raise ValueError("Episode must start at zero and record every control step once")
            if length == 0:
                group.attrs.update(complete=False,action_source=action_source,length=0,
                                   terminal_image_available=False)
            for name, value in record.items():
                arr = np.asarray(value)
                if length == 0:
                    ds = group.create_dataset(name,shape=(0,*arr.shape),maxshape=(None,*arr.shape),
                                              chunks=(1,*arr.shape),dtype=arr.dtype,
                                              compression="gzip",compression_opts=1)
                else:
                    ds = group[name]
                ds.resize(length+1,axis=0)
                ds[length] = arr
            # Publish length last: readers use only committed rows after a partial write.
            group.attrs["length"] = length+1
            self.pending[key] = group_name
            if bool(terminated[i] or truncated[i]):
                group.attrs["complete"] = True
                self.pending.pop(key)
                self.count += 1
            self.file.flush()

    def close(self, action_source):
        self.file.close()


def post_observation(packet,done):
    """Never attach automatic-reset imagery/encoders to the preceding episode."""
    import torch
    keys = ("image","depth_valid","camera_valid","camera_time_s","camera_frame",
            "proprio","time_s","episode_id","step_id")
    result = {"available":~done}
    for key in keys:
        if key in packet:
            value = packet[key]
            mask = done.reshape(-1,*([1]*(value.ndim-1)))
            result[key] = torch.where(mask,torch.zeros_like(value),value)
    return result


class EpisodeReader:
    """Inspection API: raw arrays, provenance and validity, not network features.

    Reads only committed rows. HDF5 flush is not a power-loss transaction;
    abrupt filesystem/process failure can still require HDF5 recovery.
    """
    def __init__(self,path):
        self.path = str(path)
        with h5py.File(self.path) as f:
            self.metadata = json.loads(f.attrs["metadata_json"])
            self.schema_version = int(f.attrs["schema_version"])
            self.trace_version = int(f.attrs.get("trace_version",1))
            self.episodes = list(f["episodes"])

    def attributes(self,episode):
        with h5py.File(self.path) as f:
            return dict(f["episodes"][episode].attrs)

    def read(self,episode,index=slice(None)):
        with h5py.File(self.path) as f:
            group = f["episodes"][episode]
            length = int(group.attrs["length"])
            if isinstance(index,int) and not 0 <= index < length:
                raise IndexError(index)
            selection = np.arange(length)[index]
            result = {}
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset):
                    cursor = result
                    parts = name.split("/")
                    for part in parts[:-1]:
                        cursor = cursor.setdefault(part,{})
                    cursor[parts[-1]] = obj[selection]
            group.visititems(visit)
            return result

    def steps(self,episode):
        for i in range(int(self.attributes(episode)["length"])):
            yield self.read(episode,i)

    def export_pickle(self,episode,path):
        """Trusted local inspection only: never load arbitrary/untrusted pickle."""
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        payload = dict(export_version=1,schema_version=self.schema_version,trace_version=self.trace_version,
                       metadata=self.metadata,episode=episode,attributes=self.attributes(episode),
                       arrays=self.read(episode),layout="nested arrays; first axis is control step")
        temporary = path.with_suffix(path.suffix+".partial")
        created = False
        try:
            with temporary.open("xb") as stream:
                created = True
                pickle.dump(payload,stream,protocol=5)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary,path)  # publish atomically; never overwrite an existing export
        finally:
            if created:
                temporary.unlink(missing_ok=True)
        return path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export one complete or incomplete episode for trusted local inspection")
    parser.add_argument("--data",required=True)
    parser.add_argument("--episode",required=True)
    parser.add_argument("--pkl",required=True)
    args = parser.parse_args()
    EpisodeReader(args.data).export_pickle(args.episode,args.pkl)


if __name__ == "__main__":
    main()
