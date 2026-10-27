"""Independent nominal checkpoint execution; no expert/privileged inference data."""
import time
import torch
from .history import OnlineHistory, reset_state
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.correction import compose_action
from .observations import nominal_keys, empty_plan_context


class NominalController:
    def __init__(self, model, num_envs):
        self.model = model.eval()
        cfg = model.cfg
        self.history = OnlineHistory(num_envs,cfg.history,cfg.stride,keys=nominal_keys(cfg.perception_version))
        self.episode_ids = torch.full((num_envs,),-1,dtype=torch.long)
        self.cursor = torch.full((num_envs,),cfg.replan_steps,dtype=torch.long)
        self.plan = torch.zeros(num_envs,cfg.horizon,29)
        self.inference_ms = []
        self.memory_ms = []
        self.memory = None
        self.memory_updates = torch.zeros(num_envs,dtype=torch.long)
        self.plan_generation_step = torch.full((num_envs,),-1,dtype=torch.long)
        self.plan_generation_time_s = torch.zeros(num_envs,dtype=torch.float64)
        self.previous_action = torch.zeros(num_envs,29)
        self.previous_valid = torch.zeros(num_envs,dtype=torch.bool)
        # Do not consume simulator's global RNG when sampling diffusion noise.
        self.generator = torch.Generator(device="cpu").manual_seed(0)

    @classmethod
    def load(cls,path,num_envs):
        state = torch.load(path,map_location="cpu",weights_only=False)
        if state["step"] < 1:
            raise ValueError("Checkpoint has no completed training updates")
        cfg = PolicyConfig.from_dict(state["config"])
        if (state.get("checkpoint_version",1) != cfg.perception_version
                or state["metadata"].get("perception_version",1) != cfg.perception_version):
            raise ValueError("Checkpoint model, header and sensor metadata versions disagree")
        # Loading after env.reset must not change the simulator's DR stream.
        # Only CPU RNG is touched; do not initialize a CUDA context for CPU inference.
        with torch.random.fork_rng(devices=[]):
            model = NominalDiffusionPolicy(cfg)
        model.load_state_dict(state["model"])
        controller = cls(model,num_envs)
        controller.metadata = state["metadata"]
        controller.smoke_only = state["smoke_only"]
        return controller

    @torch.no_grad()
    def act(self,packet):
        # OnlineHistory copies only versioned non-tactile keys plus lifecycle metadata.
        # Tactile, expert labels and diagnostics are never passed into nominal.
        changed = packet["episode_id"].cpu() != self.episode_ids
        if self.model.cfg.memory_version == 2 and (changed & (packet["step_id"].cpu() != 0)).any():
            raise ValueError("Persistent online memory requires the episode prefix from step zero")
        if self.model.cfg.perception_version == 2:
            # Construct only from our private predictions, never from supplied
            # packet action fields (which may have come from expert/correction).
            own = empty_plan_context(len(self.cursor),self.model.cfg.horizon)
            own["previous_nominal_valid"] = self.previous_valid & ~changed
            own["previous_nominal_action"] = torch.where(own["previous_nominal_valid"][:,None],self.previous_action,0.)
            indices = self.cursor[:,None]+torch.arange(self.model.cfg.horizon)
            mask = (indices < self.model.cfg.horizon) & ~changed[:,None] & self.previous_valid[:,None]
            own["prior_plan"] = torch.where(mask[...,None],self.plan.gather(1,indices.clamp(max=self.model.cfg.horizon-1)[...,None].expand(-1,-1,29)),0.)
            own["prior_plan_valid"] = mask
            packet = {**packet,**own}
        self.last_packet = packet
        self.history.append(packet)
        if self.model.cfg.memory_version == 2:
            self.memory = reset_state(self.memory,changed)
            self.memory_updates[changed] = 0
            current = {k:packet[k].cpu()[:,None] for k in nominal_keys(self.model.cfg.perception_version)}
            current["valid"] = torch.ones(len(self.cursor),1,dtype=torch.bool)
            start = time.perf_counter()
            self.memory = self.model.encoder(current,self.memory)
            self.memory_ms.append(1000*(time.perf_counter()-start))
            self.memory_updates += 1
        self.episode_ids = packet["episode_id"].cpu().clone()
        self.cursor[changed] = self.model.cfg.replan_steps
        due = self.cursor >= self.model.cfg.replan_steps
        if due.any():
            obs = {k:v[due] for k,v in self.history.batch().items()}
            start = time.perf_counter()
            self.plan[due] = (self.model.predict_from_condition(self.memory[due],generator=self.generator)
                              if self.model.cfg.memory_version == 2 else self.model.predict(obs,generator=self.generator))
            self.plan_generation_step[due] = packet["step_id"].cpu()[due]
            if "time_s" in packet:
                self.plan_generation_time_s[due] = packet["time_s"].cpu()[due].double()
            self.inference_ms.append(1000*(time.perf_counter()-start))
            self.cursor[due] = 0
        self.replanned = due.clone()
        action = self.plan[torch.arange(len(self.cursor)),self.cursor].clone()
        self.cursor += 1
        self.previous_action = action.clone()
        self.previous_valid[:] = True
        return compose_action(action)  # correction explicitly disabled
