"""Strict nominal-input allowlist; future actions are supervision only."""
import json
from .compact import EpisodeFile
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
        with EpisodeFile(self.path) as f:
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
                length = int(g.attrs["length"])
                if length == 0:
                    continue
                success = (g["transition/diagnostics_after/successes"][length-1] >= g["transition/diagnostics_after/max_goals"][length-1]
                           and bool(g["transition/diagnostics_after/retract_succeeded"][length-1]))
                if not allow_unsuccessful and not success:
                    continue
                if self.version == 2 and g["obs/image"].shape[1:3] != (len(self.metadata["camera_ids"]),4):
                    raise ValueError("Camera identity list and recorded image axes disagree")
                self.episodes.append(key)
                self.index.extend((key,t) for t in range(length))
        if not self.index:
            raise ValueError("No eligible demonstrations; smoke-only data require explicit allow flags")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        key, t = self.index[i]
        idx, valid = history_indices(t, self.history, self.stride)
        with EpisodeFile(self.path) as f:
            g = f["episodes"][key]
            # Read contiguous slice then index: h5py rejects repeated padded indices.
            lo, hi = int(idx[0]), t+1
            obs = {k: torch.from_numpy(np.asarray(g[f"obs/{k}"][lo:hi][idx-lo]))
                   for k in nominal_keys(self.version)}
            action_key = ("transition/command_clipped_pre_delay" if "command_clipped_pre_delay" in g["transition"]
                          else "transition/executed_action")
            actions = g[action_key][t:min(t+self.horizon,int(g.attrs["length"]))]
            n = len(actions)
            target = np.concatenate([actions, np.repeat(actions[-1:], self.horizon-n, axis=0)])
        return {"obs": {**obs, "valid": torch.from_numpy(valid)},
                "action_target": torch.from_numpy(target).float(),
                "action_valid": torch.arange(self.horizon) < n}

    def proprio_statistics(self):
        # Training episodes only; never fit normalization on validation trajectories.
        with EpisodeFile(self.path) as f:
            a = np.concatenate([f["episodes"][k]["obs/proprio"][:int(f["episodes"][k].attrs["length"])] for k in self.episodes])
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
        with EpisodeFile(self.path) as f:
            g = f["episodes"][key]
            contact_valid = read(g,"obs/contact_valid",idx).bool() & torch.from_numpy(valid)
            tactile = {}
            for source,dest in (("contact_substeps_w_N","force_w_N"),("contact_substeps_local_N","force_local_N"),
                                ("finger_position_base_m","position_base_m"),("finger_rotation_base","rotation_base"),
                                ("contact_time_s","time_s")):
                a = read(g,"obs/"+source,idx)
                tactile[dest] = a.flatten(0,1)
            motion = {k:read(g,"obs/"+k,idx) for k in ("proprio","actuator_target_rad","tracking_error_rad")}
            action_key = ("transition/command_clipped_pre_delay" if "command_clipped_pre_delay" in g["transition"]
                          else "transition/executed_action")
            previous = read(g,action_key,np.maximum(idx-1,0))
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


class EpisodeSequenceDataset(EpisodeDataset):
    """Ordered chunks from episode start, all native observations, no random windows.

    Chunk boundaries truncate storage/optionally gradients, never memory. Expert
    past issued commands are exposed separately as teacher-forcing history, with
    a validity mask; they are NOT disguised as a model prediction. The current
    default nominal encoder continues to condition on its own recorded predictions.
    """
    def chunks(self,episode,chunk_size=8):
        if episode not in self.episodes or chunk_size < 1:
            raise ValueError("Eligible episode and positive chunk size required")
        with EpisodeFile(self.path) as f:
            group = f["episodes"][episode]
            length = int(group.attrs["length"])
            steps = group["obs/step_id"][:length]
            if not np.array_equal(steps,np.arange(len(steps))):
                raise ValueError("Persistent training needs the full causal prefix from step zero")
            action_key = ("transition/command_clipped_pre_delay" if "command_clipped_pre_delay" in group["transition"]
                          else "transition/executed_action")
            actions = torch.from_numpy(group[action_key][:length]).float()
            for lo in range(0,len(steps),chunk_size):
                hi = min(lo+chunk_size,len(steps))
                obs = {k:torch.from_numpy(group["obs/"+k][lo:hi])[None] for k in nominal_keys(self.version)}
                obs["valid"] = torch.ones(1,hi-lo,dtype=torch.bool)
                index = torch.arange(lo,hi)[:,None]+torch.arange(self.horizon)
                target = actions[index.clamp(max=len(actions)-1)]
                previous_index = torch.arange(lo,hi)-1
                previous_valid = previous_index>=0
                previous = torch.where(previous_valid[:,None],actions[previous_index.clamp(min=0)],0.)
                yield dict(episode=episode,start=lo,stop=hi,episode_start=lo==0,obs=obs,
                           action_target=target,action_valid=index<len(actions),
                           expert_history=dict(previous_issued_command=previous,valid=previous_valid,
                               source="expert_teacher_forcing_only_not_nominal_prediction"))


def encode_episode_chunks(encoder,chunks,*,detach_between_chunks=False):
    """Fixed-weight sequential interface for training/replay, with explicit TBPTT.

    Caller owns loss/optimizer. If weights change mid-episode, carried states are
    TBPTT states produced by earlier weights (not a fresh full-prefix recompute).
    Never shuffle these chunks or initialize state at a random nonzero offset.
    """
    from .history import detach_state
    state = None
    expected, episode = 0,None
    for chunk in chunks:
        if chunk["episode_start"]:
            state,expected,episode = None,0,chunk["episode"]
        if episode != chunk["episode"] or chunk["start"] != expected:
            raise ValueError("Missing causal prefix or shuffled episode chunks")
        conditions,state = encoder(chunk["obs"],state,return_sequence=True,return_state=True)
        yield conditions,chunk,state
        expected = chunk["stop"]
        if detach_between_chunks:
            state = detach_state(state)


class CorrectionSequenceDataset(CorrectionEpisodeDataset):
    """One fresh control observation and its two physical samples per step.

    Supplies an ordered full prefix for future correction training/replay. No
    objective is selected, and expert labels stay outside the policy inputs.
    Stored nominal plans must identify the actual model and memory semantics.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,history=1,stride=1,force_history_steps=1,**kwargs)
        if self.metadata.get("memory_version") != 2 or not self.metadata.get("nominal_model_sha256"):
            raise ValueError("Persistent correction replay requires versioned, identified causal nominal plans")
        self._lookup = {key:i for i,key in enumerate(self.index)}

    def chunks(self,episode,chunk_size=8):
        from torch.utils.data import default_collate
        if episode not in self.episodes or chunk_size < 1:
            raise ValueError("Eligible episode and positive chunk size required")
        with EpisodeFile(self.path) as f:
            group = f["episodes"][episode]
            length = int(group.attrs["length"])
            if not np.array_equal(group["obs/step_id"][:length],np.arange(length)):
                raise ValueError("Persistent correction needs the complete ordered episode prefix")
        for lo in range(0,length,chunk_size):
            hi = min(lo+chunk_size,length)
            steps = [default_collate([self[self._lookup[(episode,t)]]]) for t in range(lo,hi)]
            yield dict(episode=episode,start=lo,stop=hi,episode_start=lo==0,steps=steps)


def replay_correction_chunks(model,chunks,*,detach_between_chunks=False):
    """Carry every branch across control steps; detach is not memory reset.

    Future training code owns its objective and optimizer. This helper performs
    no parameter update and can be compared with online Shadow at fixed weights.
    """
    from .history import detach_state
    state = None
    expected,episode = 0,None
    for chunk in chunks:
        if chunk["episode_start"]:
            state,expected,episode = None,0,chunk["episode"]
        if episode != chunk["episode"] or chunk["start"] != expected:
            raise ValueError("Missing causal prefix or shuffled correction chunks")
        candidates = []
        for step in chunk["steps"]:
            output,state = model(**step["inputs"],state=state,return_state=True)
            candidates.append(output)
        yield torch.stack(candidates,1),chunk,state
        expected = chunk["stop"]
        if detach_between_chunks:
            state = detach_state(state)
