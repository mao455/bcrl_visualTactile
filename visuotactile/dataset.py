"""Strict nominal-input allowlist; future actions are supervision only."""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .history import history_indices


class EpisodeDataset(Dataset):
    def __init__(self, path, history=4, stride=3, horizon=12, episodes=None,
                 allow_incomplete=False, allow_unsuccessful=False):
        self.path = str(path)
        self.history, self.stride, self.horizon = history, stride, horizon
        self.index, self.episodes = [], []
        with h5py.File(self.path) as f:
            if f.attrs["schema_version"] != 1:
                raise ValueError("Unsupported trajectory schema")
            self.metadata = json.loads(f.attrs["metadata_json"])
            for key, g in f["episodes"].items():
                if episodes is not None and key not in episodes:
                    continue
                if g.attrs["action_source"] != "released_rl_checkpoint":
                    continue
                if not allow_incomplete and not g.attrs["complete"]:
                    continue
                success = (g["transition/diagnostics_after/successes"][-1] >= g["transition/diagnostics_after/max_goals"][-1]
                           and bool(g["transition/diagnostics_after/retract_succeeded"][-1]))
                if not allow_unsuccessful and not success:
                    continue
                self.episodes.append(key)
                self.index.extend((key,t) for t in range(len(g["obs/step_id"])))
        if not self.index:
            raise ValueError("No eligible demonstrations; smoke-only data require explicit allow flags")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        key, t = self.index[i]
        idx, valid = history_indices(t, self.history, self.stride)
        with h5py.File(self.path) as f:
            g = f["episodes"][key]
            # Read contiguous slice then index: h5py rejects repeated padded indices.
            lo, hi = int(idx[0]), t+1
            obs = {k: torch.from_numpy(g[f"obs/{k}"][lo:hi][idx-lo]).float()
                   for k in ("image", "proprio")}
            actions = g["transition/executed_action"][t:t+self.horizon]
            n = len(actions)
            target = np.concatenate([actions, np.repeat(actions[-1:], self.horizon-n, axis=0)])
        return {"obs": {**obs, "valid": torch.from_numpy(valid)},
                "action_target": torch.from_numpy(target).float(),
                "action_valid": torch.arange(self.horizon) < n}

    def proprio_statistics(self):
        # Training episodes only; never fit normalization on validation trajectories.
        with h5py.File(self.path) as f:
            a = np.concatenate([f["episodes"][k]["obs/proprio"][:] for k in self.episodes])
        return torch.from_numpy(a.mean(0)), torch.from_numpy(a.std(0).clip(min=1e-3))
