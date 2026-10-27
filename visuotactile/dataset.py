"""Strict nominal-input allowlist; future actions are supervision only."""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from .history import history_indices
from .observations import nominal_keys


class EpisodeDataset(Dataset):
    def __init__(self, path, history=4, stride=3, horizon=12, episodes=None,
                 allow_incomplete=False, allow_unsuccessful=False, perception_version=None):
        self.path = str(path)
        self.history, self.stride, self.horizon = history, stride, horizon
        self.index, self.episodes = [], []
        with h5py.File(self.path) as f:
            self.version = int(f.attrs["schema_version"])
            if self.version not in (1,2) or (perception_version is not None and perception_version != self.version):
                raise ValueError("Unsupported trajectory schema")
            self.metadata = json.loads(f.attrs["metadata_json"])
            if self.metadata.get("perception_version",1) != self.version:
                raise ValueError("Trajectory metadata and schema versions disagree")
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
                if self.version == 2 and g["obs/image"].shape[1:3] != (len(self.metadata["camera_ids"]),4):
                    raise ValueError("Camera identity list and recorded image axes disagree")
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
            obs = {k: torch.from_numpy(np.asarray(g[f"obs/{k}"][lo:hi][idx-lo]))
                   for k in nominal_keys(self.version)}
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


class CorrectionEpisodeDataset(EpisodeDataset):
    """Real sensor history + causal model predictions, separate expert supervision.

    No loss or correction target representation is chosen here. Untrained probe
    predictions are allowed ONLY via an explicit engineering-test flag.
    """
    def __init__(self,*args,force_history_steps=12,allow_untrained_probe=False,**kwargs):
        super().__init__(*args,perception_version=2,**kwargs)
        source = self.metadata.get("nominal_plan_source")
        if source != "trained_checkpoint" and not (source == "untrained_perception_probe" and allow_untrained_probe):
            raise ValueError("Correction data require causal nominal predictions with explicit provenance")
        self.force_history_steps = force_history_steps

    def __getitem__(self,i):
        from .correction import remaining_nominal_chunk
        nominal = super().__getitem__(i)
        key,t = self.index[i]
        idx,valid = history_indices(t,self.force_history_steps)
        def read(g,name,indices):
            lo,hi = int(indices.min()),int(indices.max())+1
            return torch.from_numpy(np.asarray(g[name][lo:hi][indices-lo]))
        with h5py.File(self.path) as f:
            g = f["episodes"][key]
            contact_valid = read(g,"obs/contact_valid",idx).bool() & torch.from_numpy(valid)
            tactile = {}
            for source,dest in (("contact_substeps_w_N","force_w_N"),("contact_substeps_local_N","force_local_N"),
                                ("finger_position_base_m","position_base_m"),("finger_rotation_base","rotation_base"),
                                ("contact_time_s","time_s")):
                a = read(g,"obs/"+source,idx)
                tactile[dest] = a.flatten(0,1)
            motion = {k:read(g,"obs/"+k,idx) for k in ("proprio","actuator_target_rad","tracking_error_rad")}
            previous = read(g,"transition/executed_action",np.maximum(idx-1,0))
            command_valid = torch.from_numpy((idx>0)&valid)
            motion.update(previous_command=torch.where(command_valid[:,None],previous,0.),
                          command_valid=command_valid,valid=torch.from_numpy(valid))
            plan = torch.from_numpy(g["transition/nominal_predicted_chunk"][t:t+1])
            cursor = torch.from_numpy(g["transition/nominal_chunk_cursor"][t:t+1])
            chunk,chunk_valid = remaining_nominal_chunk(plan,cursor)
        inputs = dict(visual_history=nominal["obs"],tactile_history=tactile,
                      tactile_valid=contact_valid[:,None].expand(-1,2).flatten(),
                      nominal_chunk=chunk[0],nominal_valid=chunk_valid[0],motion_history=motion)
        return dict(inputs=inputs,supervision={"expert_command":nominal["action_target"][0]})
