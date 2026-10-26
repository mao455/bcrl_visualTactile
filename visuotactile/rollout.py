"""Independent nominal checkpoint execution; no expert/privileged inference data."""
import time
import torch
from .history import OnlineHistory
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.correction import compose_action


class NominalController:
    def __init__(self, model, num_envs):
        self.model = model.eval()
        cfg = model.cfg
        self.history = OnlineHistory(num_envs,cfg.history,cfg.stride)
        self.episode_ids = torch.full((num_envs,),-1,dtype=torch.long)
        self.cursor = torch.full((num_envs,),cfg.replan_steps,dtype=torch.long)
        self.plan = torch.zeros(num_envs,cfg.horizon,29)
        self.inference_ms = []
        # Do not consume simulator's global RNG when sampling diffusion noise.
        self.generator = torch.Generator(device="cpu").manual_seed(0)

    @classmethod
    def load(cls,path,num_envs):
        state = torch.load(path,map_location="cpu",weights_only=False)
        if state["step"] < 1:
            raise ValueError("Checkpoint has no completed training updates")
        # Loading after env.reset must not change the simulator's DR stream.
        # Only CPU RNG is touched; do not initialize a CUDA context for CPU inference.
        with torch.random.fork_rng(devices=[]):
            model = NominalDiffusionPolicy(PolicyConfig(**state["config"]))
        model.load_state_dict(state["model"])
        controller = cls(model,num_envs)
        controller.metadata = state["metadata"]
        controller.smoke_only = state["smoke_only"]
        return controller

    @torch.no_grad()
    def act(self,packet):
        # OnlineHistory copies only image/proprio plus lifecycle metadata.
        # Tactile, expert labels and diagnostics are never passed into nominal.
        self.history.append(packet)
        changed = packet["episode_id"].cpu() != self.episode_ids
        self.episode_ids = packet["episode_id"].cpu().clone()
        self.cursor[changed] = self.model.cfg.replan_steps
        due = self.cursor >= self.model.cfg.replan_steps
        if due.any():
            obs = {k:v[due] for k,v in self.history.batch().items()}
            start = time.perf_counter()
            self.plan[due] = self.model.predict(obs,generator=self.generator)
            self.inference_ms.append(1000*(time.perf_counter()-start))
            self.cursor[due] = 0
        action = self.plan[torch.arange(len(self.cursor)),self.cursor].clone()
        self.cursor += 1
        return compose_action(action)  # correction explicitly disabled
