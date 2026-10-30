"""Released privileged expert, solely for explicit diagnostic reference runs."""
from types import SimpleNamespace
import torch


class ExpertReference:
    def __init__(self, env, checkpoint, metadata):
        from evaluation.eval_isaacsim import _configure_agent
        from isaacsimenvs.utils.rlgames_utils import register_rlgames_env
        from rl_games.torch_runner import Runner, _load_checkpoint_weights
        config=_configure_agent('Isaacsimenvs-PreciseAssembly-Direct-v0',
            'rl_games_sapg_cfg_entry_point',rl_device='cpu',num_envs=1,
            deterministic=True,games=1,extra_overrides={})
        self.wrapped=register_rlgames_env(env,rl_device='cpu',
            clip_obs=config['params']['env']['clip_observations'],clip_actions=1.)
        runner=Runner()
        runner.load(config)
        runner.reset()
        self.player=runner.create_player()
        self.player.set_weights(_load_checkpoint_weights(self.player,str(checkpoint)))
        self.player.has_batch_dimension=True
        self.player.reset()
        if self.player.intr_reward_coef_embd is not None:
            self.player.intr_reward_coef_embd.fill_(50.)
        self.metadata=metadata
        self.device=torch.device('cpu')
        self.variant='expert'
        self.smoke_only=True
        self.checkpoint_epoch=None
        self.model=SimpleNamespace(cfg=SimpleNamespace(inference_steps=0))
        self.generator=torch.Generator().manual_seed(0)
        self.memory_ms=[0.]
        self.replanned=torch.zeros(1,dtype=torch.bool)
        self.episode_id=None
        self.teacher_observation=None

    @torch.no_grad()
    def act(self,packet):
        episode=int(packet['episode_id'][0])
        if episode!=self.episode_id:
            self.player.reset()
            self.episode_id=episode
        actor=self.player.obs_to_torch(self.teacher_observation['policy'])
        if self.player.intr_reward_coef_embd is not None:
            actor=torch.cat((actor,self.player.intr_reward_coef_embd),dim=1)
        return self.player.get_action(actor,is_deterministic=True)

    def observe_execution(self,transition):
        pass
