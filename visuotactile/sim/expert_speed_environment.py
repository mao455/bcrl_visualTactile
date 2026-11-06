"""Versioned execution-only expert experiment; ordinary native env is unchanged."""
from .native_diagnostics import NativeDiagnosticEnv
from visuotactile.expert_speed import progress_target


class SpeedLimitedDiagnosticEnv(NativeDiagnosticEnv):
    def __init__(self,*args,speed_config,**kwargs):
        self.speed_config=dict(speed_config)
        assert self.speed_config['kind']=='closed_loop'
        assert self.speed_config.get('phase','all') in ('all','after_first_subgoal')
        super().__init__(*args,**kwargs)

    def _pre_physics_step(self,actions):
        previous=self._prev_targets[:,self._perm_lab_to_canon].clone()
        super()._pre_physics_step(actions)
        if self.action_control_mode=='absolute_joint_position_target_rad':
            # These are already processed targets. The shared native position
            # path applied them above; a retained expert speed config must not
            # apply alpha a second time during the position-interface gate.
            return
        assert self.action_control_mode=='native_command_clipped_pre_delay'
        native=self._cur_targets[:,self._perm_lab_to_canon].clone()
        active=self.speed_config.get('phase','all')=='all' or bool((self._successes>=1).all())
        arm=self.speed_config['arm'] if active else 1.
        hand=self.speed_config['hand'] if active else 1.
        executed=progress_target(previous,native,arm,hand)
        if arm!=1 or hand!=1:
            self._cur_targets[:]=executed[:,self._perm_canon_to_lab]
            self._prev_targets=self._cur_targets.clone()
        self.last_transition['speed_native_target_rad']=native
        self.last_transition['speed_executed_target_rad']=executed.clone()
        self.last_transition['target_after_rad']=executed.clone()
