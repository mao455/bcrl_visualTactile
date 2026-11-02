"""Read-only object-filtered contacts; imported only after AppLauncher."""
import torch
from pxr import Usd, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensor, ContactSensorCfg

from .replay_environment import RecordedReplayEnv


class NativeDiagnosticEnv(RecordedReplayEnv):
    def _setup_scene(self):
        super()._setup_scene()
        rigid = []
        for root in sim_utils.find_matching_prims(self.object.cfg.prim_path.replace('env_.*', 'env_0')):
            rigid.extend(prim for prim in Usd.PrimRange(root) if prim.HasAPI(UsdPhysics.RigidBodyAPI))
        if len(rigid) != 1:
            raise RuntimeError('Object contact identity requires exactly one audited rigid object')
        self.object_contact_filter = str(rigid[0].GetPath()).replace('/env_0/', '/env_.*/')
        self.object_contacts = []
        self.object_contact_samples = []
        for ordinal, path in enumerate(self.research_contact_paths):
            sensor = ContactSensor(ContactSensorCfg(prim_path=path, update_period=0., history_length=0,
                track_pose=False, debug_vis=False, filter_prim_paths_expr=[self.object_contact_filter]))
            self.scene.sensors['native_object_contact_'+str(ordinal)] = sensor
            self.object_contacts.append(sensor)

    def _sample_kinematics(self):
        super()._sample_kinematics()
        forces = []
        for sensor in self.object_contacts:
            matrix = sensor.data.force_matrix_w
            if matrix is None or matrix.shape != (self.num_envs, 1, 1, 3):
                raise RuntimeError('Object-specific contact matrix unavailable; no distance substitute')
            forces.append(matrix[:, 0, 0].clone())
        self.object_contact_samples = (self.object_contact_samples+[torch.stack(forces, 1)])[-2:]

    def _get_rewards(self):
        reward = super()._get_rewards()
        if len(self.object_contact_samples) != 2:
            raise RuntimeError('Missing object-specific physics contact sample')
        self.last_transition['object_contact_force_w'] = torch.stack(self.object_contact_samples, 1)
        self.last_transition['native_keypoints_max_dist'] = self._keypoints_max_dist.clone()
        self.last_transition['native_alignment_condition'] = self._is_success.clone()
        return reward
