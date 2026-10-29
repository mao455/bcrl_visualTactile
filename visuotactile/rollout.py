"""Independent nominal checkpoint execution; no expert/privileged inference data."""
import time
import torch
from .history import OnlineHistory, reset_state
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.correction import compose_action
from .observations import nominal_keys, empty_plan_context


class NominalController:
    def __init__(self, model, num_envs, *, retain_image_history=True):
        self.model = model.eval()
        self.device = next(model.parameters()).device
        cfg = model.cfg
        # memory2 already retains its causal recurrent state. Evaluation needs
        # lifecycle validation, not another large unused raw image window.
        keep = retain_image_history or cfg.memory_version != 2
        self.history = OnlineHistory(num_envs,cfg.history if keep else 1,cfg.stride if keep else 1,
                                     keys=nominal_keys(cfg.perception_version) if keep else ())
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
        self.generator = torch.Generator(device=self.device).manual_seed(0)

    @classmethod
    def load(cls,path,num_envs):
        state = torch.load(path,map_location="cpu",weights_only=False)
        if "variant" in state:
            return load_command_controller(path, num_envs)
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

    def encode_current(self, current, packet, memory):
        return self.model.encoder(current, memory)

    def condition_rows(self, due):
        return self.memory[due.to(self.device)]

    def synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

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
            own["previous_nominal_valid"] = self.previous_valid & ~changed & (self.model.cfg.own_plan_mode != "none")
            own["previous_nominal_action"] = torch.where(own["previous_nominal_valid"][:,None],self.previous_action,0.)
            indices = self.cursor[:,None]+torch.arange(self.model.cfg.horizon)
            mask = ((indices < self.model.cfg.horizon) & ~changed[:,None] & self.previous_valid[:,None]
                    & (self.model.cfg.own_plan_mode != "none"))
            own["prior_plan"] = torch.where(mask[...,None],self.plan.gather(1,indices.clamp(max=self.model.cfg.horizon-1)[...,None].expand(-1,-1,29)),0.)
            own["prior_plan_valid"] = mask
            packet = {**packet,**own}
        self.last_packet = packet
        self.history.append(packet)
        if self.model.cfg.memory_version == 2:
            self.memory = reset_state(self.memory,changed)
            self.memory_updates[changed] = 0
            self.synchronize()
            start = time.perf_counter()
            current = {k:packet[k].to(self.device)[:,None] for k in nominal_keys(self.model.cfg.perception_version)}
            current["valid"] = torch.ones(len(self.cursor),1,dtype=torch.bool,device=self.device)
            self.memory = self.encode_current(current, packet, self.memory)
            self.synchronize()
            self.memory_ms.append(1000*(time.perf_counter()-start))
            self.memory_updates += 1
        self.episode_ids = packet["episode_id"].cpu().clone()
        self.cursor[changed] = self.model.cfg.replan_steps
        due = self.cursor >= self.model.cfg.replan_steps
        if due.any():
            self.synchronize()
            start = time.perf_counter()
            if self.model.cfg.memory_version == 2:
                predicted = self.model.predict_from_condition(self.condition_rows(due),generator=self.generator)
            else:
                obs = {k:v[due].to(self.device) for k,v in self.history.batch().items()}
                predicted = self.model.predict(obs,generator=self.generator)
            self.plan[due] = predicted.cpu()
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


class TactileCommandController(NominalController):
    """B1 inference with independent visual, tactile and fusion episode states."""
    def encode_current(self, current, packet, memory):
        names = ("contact_substeps_w_N", "contact_substeps_local_N",
                 "finger_position_base_m", "finger_rotation_base", "contact_time_s")
        tactile = {k:packet[k].to(self.device)[:,None] for k in names}
        _, memory = self.model.encode(current, tactile, packet["contact_valid"].to(self.device)[:,None],
                                      memory, return_state=True)
        return memory

    def condition_rows(self, due):
        return self.memory["fusion"][due.to(self.device)]


def load_command_controller(path, num_envs, *, device="cpu", retain_image_history=False):
    """Strict offline-command checkpoint -> actual simulation controller."""
    from .models.tactile_dp import TactileCommandDiffusionPolicy
    saved = torch.load(path, map_location="cpu", weights_only=False)
    cfg = PolicyConfig.from_dict(saved["config"])
    contract = saved.get("history_contract", {})
    expected = dict(memory_version=2, horizon=12, replan_steps=6, command_hz=60,
                    tactile_physical_hz=120, own_plan_mode="none",
                    action_label="command_clipped_pre_delay",
                    camera_order=["fixed_high", "fixed_side", "wrist_rgbd"])
    if (saved.get("command_checkpoint_version") != 2 or cfg.own_plan_mode != "none"
            or any(contract.get(k) != v for k,v in expected.items())
            or cfg.views != 3 or cfg.horizon != 12 or cfg.replan_steps != 6
            or cfg.memory_version != 2 or cfg.perception_version != 2):
        raise ValueError("Incompatible command checkpoint / no-own-plan / control contract")
    if saved.get("optimizer_updates", 0) < 1:
        raise ValueError("Checkpoint has no completed training updates")
    if saved.get("variant") not in ("b0", "b1"):
        raise ValueError("Unknown command baseline variant")
    tactile = saved["variant"] == "b1"
    with torch.random.fork_rng(devices=[]):
        model = TactileCommandDiffusionPolicy(cfg) if tactile else NominalDiffusionPolicy(cfg)
    model.load_state_dict(saved["model"])
    model.to(device)
    controller = (TactileCommandController if tactile else NominalController)(
        model, num_envs, retain_image_history=retain_image_history)
    controller.metadata = saved["metadata"]
    controller.smoke_only = saved.get("diagnostic_only", False)
    controller.variant = saved["variant"]
    controller.checkpoint_epoch = saved["epoch"]
    controller.checkpoint_manifest_sha256 = saved["manifest_sha256"]
    return controller
