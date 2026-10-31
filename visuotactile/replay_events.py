"""Explicit recorded exogenous events; never infer events from repeated actions."""
import numpy as np
import torch


MODES = ('original', 'delay_fixed', 'delay_wrench_fixed')
TOLERANCES = dict(command=3e-6, target_rad=3e-6, joint_rad=1e-3,
    joint_velocity_rad_s=1e-2, object_position_m=1e-3, object_angle_rad=1e-2,
    force_N=1e-5, torque_Nm=1e-6)


class RecordedEvents:
    def __init__(self, group, mode, *, extend=False, wrench_reconstruction='actual_world'):
        if mode not in MODES:
            raise ValueError('Unknown recorded event mode')
        self.mode, self.extend = mode, extend
        if wrench_reconstruction not in ('actual_world','native_request'):
            raise ValueError('Unknown wrench reconstruction')
        self.wrench_reconstruction=wrench_reconstruction
        indices=np.asarray(group['transition/control_trace/delay_index'][:])
        if not np.issubdtype(indices.dtype,np.integer):
            raise ValueError('Recorded delay indices must be integral')
        self.indices = torch.as_tensor(indices.copy()).long()
        self.force = torch.from_numpy(group['transition/applied_wrench/force_w_N'][:])
        self.torque = torch.from_numpy(group['transition/applied_wrench/torque_w_Nm'][:])
        self.request_force = torch.from_numpy(group['transition/external_force_after_dr_w_N'][:])
        self.request_torque = torch.from_numpy(group['transition/external_torque_after_dr_w_Nm'][:])
        if mode=='delay_wrench_fixed' and wrench_reconstruction=='native_request':
            self.source_quat=torch.from_numpy(group['transition/applied_wrench/object_quat_wxyz'][:])
            self.force_body=torch.from_numpy(group['transition/applied_wrench/force_b_N'][:])
            self.torque_body=torch.from_numpy(group['transition/applied_wrench/torque_b_Nm'][:])
        if (self.indices.ndim != 1 or not len(self.indices) or self.force.shape != (len(self.indices),2,1,3)
                or self.torque.shape != self.force.shape or self.request_force.shape != (len(self.indices),1,3)
                or self.request_torque.shape != self.request_force.shape
                or (self.indices < 0).any() or (self.indices >= 3).any()):
            raise ValueError('Recorded events require source-lane indices and two native physics samples')
        if not all(torch.isfinite(value).all() for value in (self.force,self.torque,self.request_force,self.request_torque)):
            raise ValueError('Nonfinite recorded wrench')
        if mode=='delay_wrench_fixed' and wrench_reconstruction=='native_request':
            if (self.source_quat.shape != (len(self.indices),2,1,4)
                    or self.force_body.shape != self.force.shape or self.torque_body.shape != self.torque.shape
                    or not all(torch.isfinite(value).all() for value in (self.source_quat,self.force_body,self.torque_body))
                    or not torch.allclose(self.source_quat.norm(dim=-1),torch.ones_like(self.source_quat[...,0]),atol=1e-5,rtol=0)):
                raise ValueError('Invalid source world_hold quaternion/body-buffer contract')

    def at(self, step):
        if step < 0 or (step >= len(self.indices) and not self.extend):
            raise IndexError('Recorded event sequence exhausted')
        source = step % len(self.indices)
        return dict(delay_index=self.indices[source:source+1],force=self.force[source],torque=self.torque[source],
            request_force=self.request_force[source:source+1],request_torque=self.request_torque[source:source+1],
            source_step=source,cycle=step//len(self.indices))


def pose_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    position = np.linalg.norm(actual[...,:3]-expected[...,:3],axis=-1)
    quat_actual = actual[...,3:7]/np.linalg.norm(actual[...,3:7],axis=-1,keepdims=True)
    quat_expected = expected[...,3:7]/np.linalg.norm(expected[...,3:7],axis=-1,keepdims=True)
    distance = np.minimum(np.linalg.norm(quat_actual-quat_expected,axis=-1),np.linalg.norm(quat_actual+quat_expected,axis=-1))
    return position,4*np.arcsin(np.clip(distance/2,0,1))


def bitwise_equal(actual, expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    return actual.shape==expected.shape and actual.dtype==expected.dtype and actual.tobytes()==expected.tobytes()


def error_summary(actual, expected, tolerance):
    error = np.abs(np.asarray(actual,dtype=np.float64)-np.asarray(expected,dtype=np.float64))
    rows = error.reshape(len(error),-1).max(-1)
    exceeded = np.flatnonzero(rows > tolerance)
    first = int(exceeded[0]) if len(exceeded) else None
    detail = None
    if first is not None:
        column = int(error[first].argmax())
        detail = dict(step=first,flat_component=column,source=float(np.asarray(expected)[first].reshape(-1)[column]),
            replay=float(np.asarray(actual)[first].reshape(-1)[column]),error=float(rows[first]))
    return dict(mean=float(error.mean()),maximum=float(error.max()),tolerance=tolerance,
        first_exceeded=first,first_difference=detail,bitwise_equal=bitwise_equal(actual,expected))
