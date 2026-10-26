"""Episode-based HDF5 recorder; model inputs and privileged labels are separate."""
import json
import h5py
import numpy as np

from . import SCHEMA_VERSION


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
        self.file.attrs["schema_version"] = SCHEMA_VERSION
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
            rows = self.pending.setdefault(key, [])
            rows.append(record)
            if bool(terminated[i] or truncated[i]):
                self._flush(key, complete=True, action_source=action_source)

    def _flush(self, key, complete, action_source):
        rows = self.pending.pop(key)
        group = self.file.create_group(f"episodes/env{key[0]:03d}_ep{key[1]:06d}")
        group.attrs.update(complete=complete, action_source=action_source, length=len(rows))
        group.attrs["terminal_image_available"] = False
        for name in rows[0]:
            arr = np.stack([row[name] for row in rows])
            group.create_dataset(name, data=arr, compression="gzip", compression_opts=1)
        self.count += 1
        self.file.flush()

    def close(self, action_source):
        for key in list(self.pending):
            self._flush(key, complete=False, action_source=action_source)
        self.file.close()
