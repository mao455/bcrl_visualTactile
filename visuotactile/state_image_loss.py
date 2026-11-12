"""Offline-only sample/x0 geometry objective, independent of simulator lifetime.

Link-frame FK is in the robot base frame; fingers are not contact points.
Weights are calibrated once on train-only predictions and never on validation.
"""
import torch

from .dp_objectives import masked_objectives
from .kinematics import EncoderKinematics, FINGERS, HandCollisionGeometry

LOSS_ID = 'position_geometry_sample_v1'
PALM_LOSS_ID = 'palm_command_sample_commit12'
DIRECT_HAND_GEOMETRY_ID = 'hand_surface_soft_hinge_v1'
COMPONENTS = ('position', 'rotation', 'fingertip', 'temporal')


def rotation_chord(a, b):
    # ||Ra-Rb||_F^2 / 8 == sin(theta/2)^2, smooth at zero and pi.
    return (a-b).square().sum((-2, -1))/8


class GeometryObjective:
    def __init__(self, metadata, mode=LOSS_ID, commit=12):
        if mode not in (LOSS_ID, PALM_LOSS_ID) or commit != 12:
            raise ValueError('Explicit supported geometry objective/commit required')
        self.mode, self.commit, self.gain = mode, commit, 1.
        self.metadata = metadata
        self.hand_geometry = None
        self.last_geometry_metrics = None
        self.fk = EncoderKinematics(metadata['config']['assets']['robot_urdf'],
            metadata['joint_order'], links=('iiwa14_link_7', *FINGERS))
        if self.fk.sha256 != metadata['fk_urdf_sha256']:
            raise ValueError('FK asset differs from recorded data')
        if mode == PALM_LOSS_ID:
            self.palm_fk = EncoderKinematics(metadata['config']['assets']['robot_urdf'],
                metadata['joint_order'], links=('iiwa14_link_7',))
        self.weights = {name: 0. for name in (('position', 'rotation') if mode == PALM_LOSS_ID else COMPONENTS)}

    def enable_hand_geometry(self, live_inventory, acceptance, loss_kind='projection_v1', **geometry_options):
        """Training admission is separate from offline implementation tests."""
        if acceptance.get('live_collision_parity') is not True:
            raise ValueError('Live collision parity has not admitted geometric training')
        if loss_kind not in ('projection_v1', DIRECT_HAND_GEOMETRY_ID):
            raise ValueError('Explicit supported hand geometry loss identity required')
        candidate = (HandGeometrySoftLoss if loss_kind == DIRECT_HAND_GEOMETRY_ID else HandGeometryProjection)(
            self.metadata, live_inventory, **geometry_options)
        if acceptance.get('geometry_identity') != candidate.geometry.identity:
            raise ValueError('Geometry acceptance identity differs')
        if acceptance.get('asset_sha256') != candidate.geometry.assets:
            raise ValueError('Geometry acceptance assets differ')
        if self.mode != PALM_LOSS_ID:
            raise ValueError('Hand projection is an explicit H24 palm/fit objective extension')
        if loss_kind == DIRECT_HAND_GEOMETRY_ID and (not acceptance.get('passed') or
                acceptance.get('loss_identity') != DIRECT_HAND_GEOMETRY_ID):
            raise ValueError('Direct soft geometry requires its own train-only scale acceptance')
        self.hand_geometry = candidate
        self.weights['hand_geometry'] = 0.

    def components(self, prediction, command, valid, scale, offset, geometry_context=None):
        self.last_geometry_metrics = None
        target = torch.where(valid[..., None], command*scale+offset, 0.)
        base = masked_objectives(prediction, target, valid)['B']
        if self.mode == PALM_LOSS_ID:
            # The same anchor's first contract.commit actions are actually issued.
            mask = valid & (torch.arange(valid.shape[1], device=valid.device)[None] < self.commit)
            if not mask.any():
                zero = prediction.sum()*0.
                values = dict(base=base, position=zero, rotation=zero)
                if self.hand_geometry is not None:
                    values['hand_geometry'] = zero
                return values
            # sample means x0: no SNR division and no DDIM graph. FK uses real
            # target radians, never noisy actions or next-tick measured state.
            q = (prediction[mask].float()-offset.float())/scale.float()
            xyz, rotation = self.palm_fk(q)
            with torch.no_grad(): exact_xyz, exact_rotation = self.palm_fk(command[mask].float())
            values = dict(base=base, position=(xyz[:, 0]-exact_xyz[:, 0]).square().sum(-1).mean(),
                rotation=.5*(rotation[:, 0]-exact_rotation[:, 0]).square().sum((-2, -1)).mean())
            if self.hand_geometry is not None:
                if geometry_context is None:
                    raise ValueError('Geometric loss requires separate aligned loss-only context')
                if isinstance(self.hand_geometry, HandGeometrySoftLoss):
                    values['hand_geometry'] = self._soft_hand_component(prediction, mask, scale, offset, geometry_context)
                    return values
                locations = mask.nonzero()
                selected = locations[torch.linspace(0, len(locations)-1,
                    min(len(locations), self.hand_geometry.maximum_rows), device=prediction.device).long()]
                row, tick = selected[:, 0], selected[:, 1]
                context = {name: geometry_context[name][row, tick] for name in ('object_pose_base', 'contact_mask', 'stage')}
                hand_prediction = (prediction[row, tick].float()-offset.float())/scale.float()
                value, report = self.hand_geometry.loss(hand_prediction, **context)
                values['hand_geometry'] = value
                self.last_geometry_metrics = report
            elif geometry_context is not None:
                raise ValueError('Loss-only context supplied without admitted geometry objective')
            return values
        # Select valid rows BEFORE FK. Invalid labels cannot affect its graph.
        q = (prediction[valid]-offset)/scale
        truth = command[valid]
        if not len(q):
            raise ValueError('No valid targets')
        xyz, rotation = self.fk(q)
        with torch.no_grad(): exact_xyz, exact_rotation = self.fk(truth)
        position = ((xyz[:, 0]-exact_xyz[:, 0])/.1).square().sum(-1).mean()
        orientation = rotation_chord(rotation[:, 0], exact_rotation[:, 0]).mean()
        finger = ((xyz[:, 1:]-exact_xyz[:, 1:])/.05).square().sum(-1).mean()
        # Match demonstrated differences, not zero velocity; equal Arm/Hand weight.
        pairs = valid[:, 1:] & valid[:, :-1]
        temporal = masked_objectives(prediction[:, 1:]-prediction[:, :-1],
            target[:, 1:]-target[:, :-1], pairs)['B']
        return dict(base=base, position=position, rotation=orientation, fingertip=finger, temporal=temporal)

    def _soft_hand_component(self, prediction, mask, scale, offset, context):
        """The caller selects at most two configurations across a whole update.

        The outer accumulator adds hand_geometry directly, without valid-tick
        or microbatch weighting. The complete update loss is therefore exactly
        sum(selected losses)/update_selected_count, including padded tails.
        A zero warmup coefficient skips all collision/FK work, not just its loss.
        """
        zero = prediction[..., 7:].sum()*0.
        if not context.get('active', True) or self.gain == 0 or self.weights['hand_geometry'] == 0:
            self.last_geometry_metrics = dict(skipped=True, reason='geometry_warmup_or_zero_weight',
                attempted=0, elapsed_s=0., loss_identity=DIRECT_HAND_GEOMETRY_ID)
            return zero
        selected = context.get('selected_mask')
        if selected is None or selected.shape != mask.shape or selected.dtype != torch.bool:
            raise ValueError('Explicit complete-update selected_mask must align with valid commit targets')
        if (selected & ~mask).any():
            raise ValueError('Geometry selection contains an invalid or uncommitted configuration')
        count = int(context['update_selected_count'])
        if count not in (1, 2):
            raise ValueError('One/two complete-update selections required')
        row, tick = selected.nonzero(as_tuple=True)
        if len(row) > count:
            raise ValueError('Microbatch selections exceed the complete-update geometry budget')
        if not len(row):
            self.last_geometry_metrics = dict(skipped=True, reason='no_complete_update_selection_in_microbatch',
                attempted=0, elapsed_s=0., loss_identity=DIRECT_HAND_GEOMETRY_ID)
            return zero
        physical = (prediction[row, tick].float()-offset.float())/scale.float()
        values, report = self.hand_geometry.per_configuration(physical,
            **{key: context[key][row, tick] for key in ('object_pose_base', 'contact_mask', 'stage')})
        report.update(selected_locations=torch.stack((row, tick), -1).tolist(),
            update_selected_count=count,
            normalization='sum selected row losses / complete-update selected count; outer accumulator adds directly')
        self.last_geometry_metrics = report
        return values.sum()/count

    def forward(self, model, condition, command, valid, generator, geometry_context=None, *, diffusion_inputs=None):
        if model.noise.config.prediction_type != 'sample' or model.horizon != 24:
            raise ValueError('New geometry objective is explicitly H24 sample only')
        target = torch.where(valid[..., None], command*model.action_scale+model.action_offset, 0.)
        if diffusion_inputs is None:
            noise = torch.randn(target.shape, device=target.device, generator=generator)
            ticks = torch.randint(100, (len(target),), device=target.device, generator=generator)
        else:
            noise, ticks = diffusion_inputs
            if (noise.shape != target.shape or noise.dtype != target.dtype or noise.device != target.device
                    or ticks.shape != (len(target),) or ticks.dtype != torch.long or ticks.device != target.device):
                raise ValueError('Precomputed diffusion draws must match this exact microbatch')
        prediction = model.denoiser(model.noise.add_noise(target, noise, ticks), ticks, global_cond=condition)
        values = self.components(prediction, command, valid, model.action_scale, model.action_offset, geometry_context)
        total = values['base']+self.gain*sum(self.weights[k]*values[k] for k in self.weights)
        return total, values, prediction


class HandGeometrySoftLoss:
    """Direct signed-surface penalties, evaluated once per selected configuration.

    Distances/tolerances are metres. Each squared hinge is divided by its fixed
    tolerance squared, making both loss and calibrated multiplier dimensionless.
    No local solve, feasible-set admission, predicted-contact gating or sample
    rejection is used. The arm/object are detached geometric context only.
    """
    identity = DIRECT_HAND_GEOMETRY_ID

    def __init__(self, metadata, live_inventory, penetration_tolerance_m=.001, contact_tolerance_m=.002):
        if penetration_tolerance_m != .001 or contact_tolerance_m != .002:
            raise ValueError('Direct geometry retains fixed 1mm penetration / 2mm required-contact tolerances')
        self.geometry = HandCollisionGeometry(metadata, live_inventory)
        self.penetration_tolerance_m = float(penetration_tolerance_m)
        self.contact_tolerance_m = float(contact_tolerance_m)
        self.contact_indices = [[i for i in indices if self.geometry.links[i].endswith(('_DP', '_elastomer'))]
                                for indices in self.geometry.finger_indices]
        if any(not indices for indices in self.contact_indices):
            raise ValueError('Every required finger needs actual distal collision geometry')

    def per_configuration(self, q_prediction_rad, object_pose_base, contact_mask, stage):
        import time
        started = time.perf_counter()
        contacts = contact_mask.detach().bool().to(q_prediction_rad.device)
        stages = stage.detach().long().to(q_prediction_rad.device)
        if contacts.shape != (len(q_prediction_rad), 5) or stages.shape != (len(q_prediction_rad),):
            raise ValueError('Aligned five-finger required contacts and stage required')
        if ((stages < 0) | (stages > 2)).any() or ((stages != 1) & contacts.any(-1)).any():
            raise ValueError('Only contact stage may prescribe fingertip contact; approach/release have no attraction')
        # One CPU signed-distance/Jacobian call, followed by its exact local
        # first derivative on the original device. No CUDA FK launch loop or
        # differentiation through GJK/EPA is needed; witnesses are stop-gradient.
        with torch.no_grad():
            distance, jacobian = self.geometry.distances(q_prediction_rad.detach().double().cpu(),
                object_pose_base.detach().double().cpu(), jacobian=True)
        distance = distance.to(q_prediction_rad)
        jacobian = jacobian.to(q_prediction_rad)
        hand_delta = q_prediction_rad[:, 7:]-q_prediction_rad[:, 7:].detach()
        distance = distance + torch.einsum('bch,bh->bc', jacobian, hand_delta)
        minima = [distance[:, indices].min(-1) for indices in self.contact_indices]
        fingertip = torch.stack([result.values for result in minima], -1)
        penetration = (-distance-self.penetration_tolerance_m).clamp_min(0)
        gap = (fingertip-self.contact_tolerance_m).clamp_min(0)
        pen_loss = (penetration/self.penetration_tolerance_m).square().mean(-1)
        contact_loss = ((gap/self.contact_tolerance_m).square()*contacts).sum(-1)/contacts.sum(-1).clamp_min(1)
        values = pen_loss+contact_loss
        if not torch.isfinite(values).all():
            raise FloatingPointError('Nonfinite direct hand geometry; do not silently discard a selected row')
        with torch.no_grad():
            batch = torch.arange(len(distance), device=distance.device)
            fingertip_jacobian = torch.stack([jacobian[batch,
                torch.as_tensor(indices, device=distance.device)[result.indices]]
                for indices, result in zip(self.contact_indices, minima)], 1)
            contact_gradient = (fingertip_jacobian*(2*gap/self.contact_tolerance_m**2*contacts/
                contacts.sum(-1, keepdim=True).clamp_min(1))[..., None]).sum(1)
            penetration_gradient = (jacobian*(-2*penetration/self.penetration_tolerance_m**2)[..., None]).mean(1)
        # Small scalar/row diagnostics are detached; difficult contact rows keep
        # their loss and gradient even when a local feasible projection fails.
        rows = [dict(stage=int(stages[i]), required_contacts=int(contacts[i].sum()),
            minimum_distance_m=float(distance[i].detach().min()),
            fingertip_distance_m=fingertip[i].detach().tolist(),
            required_contact_mask=contacts[i].tolist(),
            maximum_penetration_violation_m=float(penetration[i].detach().max()),
            maximum_required_contact_gap_violation_m=float((gap[i]*contacts[i]).detach().max()),
            penetrating_colliders=int((penetration[i]>0).sum()),
            violated_required_contacts=int(((gap[i]>0)&contacts[i]).sum()),
            penetration_loss=float(pen_loss[i].detach()), contact_loss=float(contact_loss[i].detach()),
            contact_hand_gradient_norm_per_rad=float(contact_gradient[i].norm()),
            penetration_hand_gradient_norm_per_rad=float(penetration_gradient[i].norm()),
            geometry_loss=float(values[i].detach()), included=True) for i in range(len(values))]
        return values, dict(rows=rows, attempted=len(rows), included=len(rows), dropped=0,
            contact_rows=int((stages==1).sum()), noncontact_rows=int((stages!=1).sum()),
            required_contacts=int(contacts.sum()), elapsed_s=time.perf_counter()-started,
            distance_evaluations=len(rows), collision_pairs_per_configuration=len(self.geometry.links)*len(self.geometry.boxes),
            identity=self.geometry.identity, loss_identity=self.identity, distance_unit='m', loss_unit='dimensionless',
            solver_calls=0, arm_gradient='detached', reduction='per row: mean collision penetration + mean required-finger gap')

    def loss(self, q_prediction_rad, object_pose_base, contact_mask, stage):
        values, report = self.per_configuration(q_prediction_rad, object_pose_base, contact_mask, stage)
        return values.mean(), report


class HandGeometryProjection:
    """Bounded nearest feasible hand projection; unavailable rows never get targets.

    All poses/contact arrangements are offline supervision, not policy inputs.
    The arm is fixed to the predicted arm context throughout the solve. Approach
    and release have no attraction term; only the contact stage preserves the
    specified object-filtered fingertip arrangement. No success labels are used.
    The local ball is an L-infinity joint bound, independent of the joint limits.
    """
    def __init__(self, metadata, live_inventory, penetration_tolerance_m=.001,
                 contact_tolerance_m=.002, local_radius_rad=.15, maximum_iterations=30,
                 maximum_rows=2):
        self.geometry = HandCollisionGeometry(metadata, live_inventory)
        self.limits = torch.as_tensor(metadata['joint_limits_rad']).reshape(-1, 29, 2)[0].double()
        self.penetration_tolerance_m = float(penetration_tolerance_m)
        self.contact_tolerance_m = float(contact_tolerance_m)
        self.local_radius_rad = float(local_radius_rad)
        self.maximum_iterations, self.maximum_rows = int(maximum_iterations), int(maximum_rows)
        if min(penetration_tolerance_m, contact_tolerance_m, local_radius_rad) <= 0:
            raise ValueError('Positive, preregistered fixed geometric tolerances required')
        # Contact sensors cover each distal body and its fixed elastomer, not
        # arbitrary proximal links that happen to touch the object.
        self.contact_indices = [[i for i in indices if self.geometry.links[i].endswith(('_DP', '_elastomer'))]
                                for indices in self.geometry.finger_indices]

    def _constraints(self, q, object_pose, contacts):
        distance, derivative = self.geometry.distances(q[None], object_pose[None], jacobian=True)
        d, j = distance[0].detach().numpy(), derivative[0].detach().numpy()
        np = self.geometry._np
        values = [d+self.penetration_tolerance_m]
        derivatives = [j]
        fingertip = []
        for active, indices in zip(contacts, self.contact_indices):
            index = indices[int(np.argmin(d[indices]))]
            fingertip.append(d[index])
            if active:
                values.append(np.asarray([self.contact_tolerance_m-d[index]]))
                derivatives.append(-j[index:index+1])
        return np.concatenate(values), np.concatenate(derivatives), d, np.asarray(fingertip)

    def project(self, q_prediction_rad, object_pose_base, contact_mask, stage):
        from scipy.optimize import minimize
        import numpy as np
        import time
        started = time.perf_counter()
        original = q_prediction_rad.detach().double().cpu()
        pose = object_pose_base.detach().double().cpu()
        contacts = contact_mask.detach().bool().cpu()
        stages = stage.detach().long().cpu()
        if contacts.shape != (len(original), 5) or stages.shape != (len(original),):
            raise ValueError('Explicit five-finger contact mask and aligned stage required')
        if ((stages < 0) | (stages > 2)).any() or ((stages != 1) & contacts.any(-1)).any():
            raise ValueError('Stages are approach=0, contact=1, release=2; only contact has required contacts')
        targets, reports = [], []
        for q, obj, required, phase in zip(original, pose, contacts, stages):
            low = torch.maximum(self.limits[7:, 0], q[7:]-self.local_radius_rad).numpy()
            high = torch.minimum(self.limits[7:, 1], q[7:]+self.local_radius_rad).numpy()
            report = dict(stage=int(phase), required_contacts=int(required.sum()), feasible=False,
                          reason='unresolved', iterations=0)
            if not torch.isfinite(q).all() or not torch.isfinite(obj).all():
                report['reason'] = 'nonfinite_context'
                targets.append(q); reports.append(report); continue
            if np.any(low > high):
                report['reason'] = 'joint_limits_incompatible_with_locality'
                targets.append(q); reports.append(report); continue
            origin = q[7:].numpy()
            cache = {}
            def evaluate(hand):
                key = hand.tobytes()
                if cache.get('key') != key:
                    candidate = torch.cat((q[:7], torch.from_numpy(hand)))
                    cache.update(key=key, result=self._constraints(candidate, obj, required.numpy()))
                return cache['result']
            before = self._constraints(q, obj, required.numpy())
            start = np.clip(origin, low, high)
            result = minimize(lambda hand: .5*np.square(hand-origin).sum(), start,
                jac=lambda hand: hand-origin, bounds=list(zip(low, high)), method='SLSQP',
                constraints=[dict(type='ineq', fun=lambda hand: evaluate(hand)[0], jac=lambda hand: evaluate(hand)[1])],
                options=dict(maxiter=self.maximum_iterations, ftol=1e-10, disp=False))
            after = evaluate(result.x)
            # Solver success alone is insufficient; independently enforce every
            # fixed constraint at 0.1um numerical tolerance, never loosen physics.
            feasible = bool(after[0].min() >= -1e-7 and np.all(result.x >= low-1e-10)
                            and np.all(result.x <= high+1e-10))
            reason = 'feasible' if feasible else ('penetration_constraint_unresolved' if after[2].min() < -self.penetration_tolerance_m-1e-7
                else 'required_contact_constraint_unresolved')
            target = torch.cat((q[:7], torch.from_numpy(result.x))) if feasible else q
            report.update(feasible=feasible, reason=reason, iterations=int(result.nit), solver_success=bool(result.success),
                solver_status=int(result.status), solver_message=str(result.message),
                local_solver_failure_is_not_global_infeasibility_proof=not feasible,
                before_min_distance_m=float(before[2].min()),
                after_min_distance_m=float(after[2].min()), before_fingertip_distance_m=before[3].tolist(),
                after_fingertip_distance_m=after[3].tolist(), required_contact_mask=required.tolist(),
                maximum_hand_change_rad=float(np.abs(result.x-origin).max()),
                arm_changed=False, minimum_constraint_margin_m=float(after[0].min()))
            targets.append(target); reports.append(report)
        return torch.stack(targets).to(q_prediction_rad), dict(rows=reports, elapsed_s=time.perf_counter()-started,
            feasible=sum(row['feasible'] for row in reports), attempted=len(reports), identity=self.geometry.identity)

    def loss(self, q_prediction_rad, object_pose_base, contact_mask, stage):
        target, report = self.project(q_prediction_rad, object_pose_base, contact_mask, stage)
        admitted = torch.tensor([row['feasible'] for row in report['rows']], device=q_prediction_rad.device)
        if admitted.any():
            loss = (q_prediction_rad[admitted, 7:]-target[admitted, 7:].detach()).square().mean()
        else:
            loss = q_prediction_rad[:, 7:].sum()*0.
        return loss, report


def geometry_loss_context(entry):
    """Read after-tick ground truth solely for geometry supervision.

    q_reached and the object pose share diagnostics_after[t]; the contact mask
    comes from the final matching 120Hz object-filtered sample, never unfiltered
    finger force (which may include the table or hole). An observed loss of
    object contact changes contact to release without attracting fingers back.
    """
    import h5py
    from .native_data import multiply_quaternion, rotate
    with h5py.File(entry['path']) as f:
        g = f['episodes'][entry['episode']]
        tr = g['transition']; after = tr['diagnostics_after']
        base = torch.from_numpy(after['base_pose_w'][:]).float()
        obj = torch.from_numpy(after['object_pose_w'][:]).float()
        times = torch.from_numpy(tr['time_after_s'][:])
        contact_times = torch.from_numpy(tr['contact_time_after_s'][:])[:, -1]
        if len(times) != len(base) or not torch.allclose(times, contact_times, atol=1e-6, rtol=0):
            raise ValueError('Geometry and contact timestamps differ')
        force = torch.from_numpy(tr['object_contact_force_w'][:])[:, -1]
        if force.shape != (len(base), 5, 3):
            raise ValueError('Aligned object-filtered five-finger normals required')
        before_times = torch.from_numpy(tr['time_before_s'][:])
        if not torch.allclose(times[:-1], before_times[1:], atol=1e-6, rtol=0):
            raise ValueError('Geometry supervision crosses a discontinuous transition')
        terminal = torch.from_numpy(g['terminated'][:]).bool() | torch.from_numpy(g['truncated'][:]).bool()
        if terminal[:-1].any():
            raise ValueError('Geometry context crosses an episode boundary')
    inverse = base[:, 3:].clone(); inverse[:, 1:] *= -1
    object_pose_base = torch.cat((rotate(inverse, obj[:, :3]-base[:, :3]), multiply_quaternion(inverse, obj[:, 3:])), -1)
    mask = force.norm(dim=-1) > .1
    ever = mask.any(-1).long().cumsum(0) > 0
    stage = torch.where(mask.any(-1), 1, torch.where(ever, 2, 0))
    return dict(object_pose_base=object_pose_base, contact_mask=mask, stage=stage,
                time_after_s=times, terminal=terminal)


def live_geometry_probe(env):
    """Bound a geometry failure without preventing the caller's Kit cleanup.

    This auxiliary gate cannot invalidate completed reset evidence or skip
    ``env.close()``/``app.close()``. Exceptions remain explicit blocking evidence,
    with their original traceback; they never admit geometric training.
    """
    import traceback
    started_steps = int(env._sim_step_counter)
    started_time = float(env.sim.current_time)
    try:
        report = _live_geometry_probe(env)
        report['status'] = 'PASSED' if report['live_collision_parity'] else 'BLOCKED'
        return report
    except Exception as error:
        return dict(complete=False, status='BLOCKED', live_collision_parity=False,
            geometry_identity=None, asset_sha256={}, inventory={}, rows=[],
            blockers=['geometry_probe_exception'], exception_class=type(error).__name__,
            exception_message=str(error), traceback=traceback.format_exc(),
            physics_steps=int(env._sim_step_counter)-started_steps,
            physics_elapsed_s=float(env.sim.current_time)-started_time,
            force=dict(accepted=False, calibration_perturbations=0,
                       reason='geometry_probe_precondition_failed'),
            scope='Auxiliary geometry gate failed; completed reset evidence is separate; no geometry training admission')


def _request_cooked_convex(api, stage_id, collision_prim_id, valid_result):
    """Capture one synchronous SDK request without combining failure causes.

    Copy vertex values inside the callback rather than retaining SDK-owned
    mesh wrappers. There are no retries, stage changes or app-update calls.
    This small pure wrapper is testable without importing Kit.
    """
    import numpy as np
    import traceback
    diagnostic = dict(stage_id=int(stage_id), collision_prim_id=int(collision_prim_id),
                      asynchronous=False, callback_count=0, callbacks=[], admitted=False)
    vertices = []
    def receive(result, convexes):
        diagnostic['callback_count'] += 1
        sample = dict(status_repr=repr(result), status_type=type(result).__name__,
                      status_int=None, hull_count=None, valid_status=False)
        diagnostic['callbacks'].append(sample)
        try:
            sample['status_int'] = int(result)
        except (TypeError, ValueError, OverflowError):
            pass
        try:
            sample['valid_status'] = bool(result == valid_result)
            sample['hull_count'] = len(convexes)
            sample['vertex_counts'] = [len(mesh.vertices) for mesh in convexes]
            if sample['valid_status'] and len(convexes) == 1:
                values = np.asarray(convexes[0].vertices, dtype=np.float64).copy()
                if values.ndim != 2 or values.shape[1] != 3 or len(values) < 4 or not np.isfinite(values).all():
                    raise ValueError('Invalid cooked convex vertex payload')
                vertices.append(values)
        except Exception as error:
            sample.update(exception_class=type(error).__name__, exception_message=str(error),
                          traceback=traceback.format_exc())
    try:
        returned = api.request_convex_collision_representation(stage_id=stage_id,
            collision_prim_id=collision_prim_id, run_asynchronously=False, on_result=receive)
        diagnostic.update(return_type=type(returned).__name__, return_repr=repr(returned)[:256])
    except Exception as error:
        diagnostic.update(exception_class=type(error).__name__, exception_message=str(error),
                          traceback=traceback.format_exc())
    if 'exception_class' in diagnostic:
        reason = 'request_exception'
    elif diagnostic['callback_count'] == 0:
        reason = 'synchronous_callback_missing'
    elif diagnostic['callback_count'] != 1:
        reason = 'unexpected_callback_count'
    elif 'exception_class' in diagnostic['callbacks'][0]:
        reason = 'callback_payload_exception'
    elif not diagnostic['callbacks'][0]['valid_status']:
        reason = 'sdk_nonvalid_result'
    elif diagnostic['callbacks'][0]['hull_count'] != 1:
        reason = 'unexpected_convex_count'
    elif len(vertices) != 1:
        reason = 'cooked_vertices_missing'
    else:
        reason = 'accepted'
        diagnostic['admitted'] = True
    diagnostic['reason'] = reason
    return diagnostic, vertices[0] if diagnostic['admitted'] else None


def _resolve_collision_mesh(stage, record):
    """Resolve one physical collider's unique mesh and its body transform."""
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    collider = stage.GetPrimAtPath(record['path'])
    if not collider or not collider.HasAPI(UsdPhysics.CollisionAPI):
        raise ValueError('Recorded collision owner is missing CollisionAPI')
    meshes = [prim for prim in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()) if prim.IsA(UsdGeom.Mesh)]
    if len(meshes) != 1:
        raise ValueError('Collider must have exactly one unambiguous Mesh descendant: '+str(len(meshes)))
    mesh = meshes[0]
    cursor = mesh
    while cursor != collider:
        if cursor.HasAPI(UsdPhysics.CollisionAPI):
            raise ValueError('Nested CollisionAPI changes collider ownership')
        cursor = cursor.GetParent()
    relative, reset = UsdGeom.XformCache().ComputeRelativeTransform(mesh, stage.GetPrimAtPath(record['body']))
    if reset:
        raise ValueError('Reset transform stack inside articulated collider is unsupported')
    matrix = np.asarray(relative)
    points = np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get(), dtype=np.float64)
    points_body = points@matrix[:3, :3]+matrix[3, :3]
    low, high = points_body.min(0), points_body.max(0)
    error = max(float(np.abs((low+high)/2-np.asarray(record['center'])).max()),
                float(np.abs((high-low)/2-np.asarray(record['half'])).max()))
    if error > 1e-8:
        raise ValueError('Resolved mesh transform does not reproduce collision inventory: '+str(error))
    return collider, mesh, matrix, dict(source_collision_path=str(collider.GetPath()),
        source_mesh_path=str(mesh.GetPath()), mesh_descendants=1,
        mesh_has_direct_collision_api=mesh.HasAPI(UsdPhysics.CollisionAPI),
        collision_settings_source='explicit_collision_owner; USD schemas are not assumed inherited',
        mesh_to_body_transform=matrix.tolist(), inventory_bounds_max_error_m=error,
        source_vertices=len(points))


def _cooking_mesh_stage(source_stage, collider, mesh):
    """Adapt an imported CollisionAPI Xform to the SDK's Mesh+API contract.

    This creates an unattached in-memory USD data stage, not a physics scene.
    Only the actual mesh's geometry and the owner's exact collision properties
    are copied. No live prim, collision attribute, transform or simulation state
    is authored. SDK representation is subsequently transformed using the
    separately verified source mesh-to-body transform.
    """
    import hashlib
    import numpy as np
    from pxr import Usd, UsdGeom, UsdPhysics
    proxy_stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(proxy_stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(proxy_stage, UsdGeom.GetStageUpAxis(source_stage))
    proxy = UsdGeom.Mesh.Define(proxy_stage, '/CollisionMesh').GetPrim()
    geometry_attributes = ('points', 'faceVertexCounts', 'faceVertexIndices', 'extent',
                           'normals', 'orientation', 'subdivisionScheme', 'doubleSided')
    values = {}
    for name in geometry_attributes:
        attribute = mesh.GetAttribute(name)
        value = attribute.Get()
        if value is None:
            continue
        proxy.CreateAttribute(name, attribute.GetTypeName(), custom=attribute.IsCustom()).Set(value)
        stored = proxy.GetAttribute(name).Get()
        if value != stored:
            raise ValueError('Cook adapter changed source geometry attribute '+name)
        array = np.asarray(value)
        values[name] = dict(type=str(attribute.GetTypeName()), dtype=str(array.dtype), shape=list(array.shape),
                            sha256=hashlib.sha256(array.tobytes()).hexdigest())
    UsdPhysics.CollisionAPI.Apply(proxy)
    UsdPhysics.MeshCollisionAPI.Apply(proxy)
    owners = [mesh]
    while owners[-1] != collider:
        owners.append(owners[-1].GetParent())
    owners.reverse()
    collision = {}
    for owner in owners:
        for schema in owner.GetAppliedSchemas():
            if 'Collision' in schema:
                proxy.AddAppliedSchema(schema)
        for attribute in owner.GetAttributes():
            name = attribute.GetName()
            if not name.startswith(('physics:', 'physx')):
                continue
            value = attribute.Get()
            if value is None:
                continue
            proxy.CreateAttribute(name, attribute.GetTypeName(), custom=attribute.IsCustom()).Set(value)
            if proxy.GetAttribute(name).Get() != value:
                raise ValueError('Cook adapter changed collision setting '+name)
            collision[name] = dict(type=str(attribute.GetTypeName()), value=str(value), source_owner=str(owner.GetPath()))
    # A convexHull shape's mesh-local geometry is cooked without the articulated
    # body transform; that same transform is applied to the returned vertices.
    if UsdPhysics.MeshCollisionAPI(proxy).GetApproximationAttr().Get() != 'convexHull':
        raise ValueError('Cook adapter requires the unchanged convexHull approximation')
    if not UsdPhysics.CollisionAPI(proxy).GetCollisionEnabledAttr().Get():
        raise ValueError('Disabled collider cannot be included in geometry training')
    provenance = dict(kind='unattached_mesh_schema_adapter', live_stage_modified=False,
        representation_provenance='same SDK reconstructed from exact live Mesh and effective collision properties',
        runtime_cooked_memory_read=False, full_physx_contact_equivalence_proven=False,
        geometry_attributes=values, collision_attributes=collision,
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(proxy_stage), up_axis=UsdGeom.GetStageUpAxis(proxy_stage),
        source_mesh_type=mesh.GetTypeName(), request_mesh_type=proxy.GetTypeName(),
        request_has_collision_api=proxy.HasAPI(UsdPhysics.CollisionAPI),
        source_points_sha256=hashlib.sha256(np.asarray(UsdGeom.Mesh(mesh).GetPointsAttr().Get()).tobytes()).hexdigest(),
        copied_points_sha256=hashlib.sha256(np.asarray(UsdGeom.Mesh(proxy).GetPointsAttr().Get()).tobytes()).hexdigest())
    if provenance['source_points_sha256'] != provenance['copied_points_sha256']:
        raise ValueError('Cook adapter point bytes differ')
    return proxy_stage, proxy, provenance


def _live_geometry_probe(env):
    """One zero-step cooked-geometry/contact-API gate inside the existing Kit.

    No robot state, physics parameter, contact setting or action is changed.
    This reconstructs the exact live mesh and collision properties through the
    same SDK cooking API, then checks transforms and live query consistency.
    It is not a read of runtime cooked memory or a force-model calibration.
    """
    import numpy as np
    import omni.usd
    import omni.physx
    from pxr import UsdGeom, UsdUtils, UsdPhysics, PhysicsSchemaTools
    from omni.physx.bindings._physx import PhysxCollisionRepresentationResult
    from .live_collision import mesh_inventory
    from .native_data import multiply_quaternion, rotate
    from .initial_geometry import rotation
    steps, seconds = int(env._sim_step_counter), float(env.sim.current_time)
    # research_metadata() is the sensor/control header only. The recorder and
    # state_benchmark attach the environment config separately; do the same.
    metadata = dict(env.research_metadata(), config=env.cfg.to_dict())
    inventory = dict(robot_mesh_bounds=[{key: value.tolist() if hasattr(value, 'tolist') else value
        for key, value in row.items()} for row in mesh_inventory(env)])
    source = HandCollisionGeometry(metadata, inventory)
    data = env._diagnostics()
    q = data['joint_position_rad'].detach().double().cpu()
    base, obj = data['base_pose_w'].detach().double().cpu(), data['object_pose_w'].detach().double().cpu()
    inverse = base[:, 3:].clone(); inverse[:, 1:] *= -1
    object_pose = torch.cat((rotate(inverse, obj[:, :3]-base[:, :3]), multiply_quaternion(inverse, obj[:, 3:])), -1)
    stage = omni.usd.get_context().get_stage()
    stage_id = UsdUtils.StageCache.Get().GetId(stage).ToLongInt()
    body_paths = env.robot.root_physx_view.link_paths[0]
    live_poses = env.robot.data.body_link_pose_w[0].detach().double().cpu().numpy()
    body_fk = EncoderKinematics(metadata['config']['assets']['robot_urdf'], metadata['joint_order'],
                               links=[path.rsplit('/', 1)[-1] for path in body_paths])
    bp, br = body_fk(q); lp, lr = source.fk(q)
    base_rotation = rotation(base[0, 3:].numpy())
    maximum_position_error = maximum_rotation_error = 0.
    for index, pose in enumerate(live_poses):
        expected_p = base[0, :3].numpy()+base_rotation@bp[0, index].numpy()
        expected_r = base_rotation@br[0, index].numpy()
        maximum_position_error = max(maximum_position_error, float(np.linalg.norm(expected_p-pose[:3])))
        maximum_rotation_error = max(maximum_rotation_error, float(np.abs(expected_r-rotation(pose[3:])).max()))
    cooked, rows, blockers, cook_requests = {}, [], [], []
    api = omni.physx.get_physx_cooking_interface()
    # Reuse the established actual-pose object-box query, including geometry
    # dimension checks and self-hit positive controls. An overlap_shape call on
    # the imported collision Xform is invalid, and raw USD poses may be stale.
    from .state_benchmark import collision_evidence
    live_query = collision_evidence(env)
    for record in inventory['robot_mesh_bounds']:
        if '/left_' not in record['path']:
            continue
        body = record['body'].rsplit('/', 1)[-1]
        link = ('left_hand_C_MC' if '/left_hand_C_MC/' in record['path'] else
                body.replace('_DP', '_elastomer') if 'elastomer/' in record['path'] else body)
        if link not in source.links or link in cooked:
            blockers.append('unmapped_or_duplicate_collider:'+record['path']); continue
        try:
            prim, mesh, matrix, resolution = _resolve_collision_mesh(stage, record)
            proxy_stage, proxy, adapter = _cooking_mesh_stage(stage, prim, mesh)
            proxy_id = UsdUtils.StageCache.Get().Insert(proxy_stage)
            try:
                request, vertices = _request_cooked_convex(api, proxy_id.ToLongInt(),
                    PhysicsSchemaTools.sdfPathToInt(proxy.GetPath()), PhysxCollisionRepresentationResult.RESULT_VALID)
            finally:
                UsdUtils.StageCache.Get().Erase(proxy_id)
            request.update(path=record['path'], link=link, prim_valid=bool(prim),
                prim_type=prim.GetTypeName(), is_mesh=prim.IsA(UsdGeom.Mesh),
                has_collision_api=prim.HasAPI(UsdPhysics.CollisionAPI),
                is_instance_proxy=prim.IsInstanceProxy(), is_instanceable=prim.IsInstanceable(),
                collision_enabled=UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(),
                approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get(),
                resolution=resolution, adapter=adapter)
        except Exception as error:
            import traceback
            request = dict(path=record['path'], link=link, admitted=False,
                reason='mesh_resolution_or_schema_adapter_failed', exception_class=type(error).__name__,
                exception_message=str(error), traceback=traceback.format_exc())
            vertices = None
        cook_requests.append(request)
        if not request['admitted']:
            blockers.append('cooked_hull_unavailable:'+record['path']); continue
        points_body = vertices@matrix[:3, :3]+matrix[3, :3]
        bi, li = body_paths.index(record['body']), source.links.index(link)
        # Use URDF's constant body<-source-link relationship, while separately
        # checking live body pose. This avoids absorbing an FK error into assets.
        points_base = points_body@br[0, bi].numpy().T+bp[0, bi].numpy()
        points_link = (points_base-lp[0, li].numpy())@lr[0, li].numpy()
        cooked[link] = dict(vertices_link=points_link.tolist(), live_path=record['path'], mesh_path=str(mesh.GetPath()),
                            body=record['body'], vertex_count=len(vertices))
        hits = [hit['collider'] for hit in live_query['external_hits'] if hit['collider'] == record['path']
                or hit['collider'].startswith(record['path']+'/')]
        rows.append(dict(link=link, object_overlap_hits=hits, cooked_vertices=len(vertices),
            query_source='verified actual-pose object boxes with object self-hit controls'))
    inventory['cooked_hulls'], inventory['cooked_hulls_asset_sha256'] = cooked, source.assets
    cooked_geometry = None
    if not blockers:
        cooked_geometry = HandCollisionGeometry(metadata, inventory)
        signed = cooked_geometry.distances(q, object_pose)[0].tolist()
        original = source.distances(q, object_pose)[0].tolist()
        for row in rows:
            index = source.links.index(row['link'])
            row.update(cooked_distance_m=signed[index], source_distance_m=original[index])
            if bool(row['object_overlap_hits']) != (signed[index] < -1e-6) and abs(signed[index]) > 1e-5:
                blockers.append('live_query_vs_cooked_distance:'+row['link'])
    if maximum_position_error > 1e-5 or maximum_rotation_error > 1e-4:
        blockers.append('live_link_pose_vs_urdf_fk')
    contact_api = []
    for sensor in env.object_contacts:
        raw = dict(prim_path=sensor.cfg.prim_path)
        try:
            view = sensor.contact_physx_view
            normal = view.get_contact_data(dt=1/120)
            friction = view.get_friction_data(dt=1/120)
            raw.update(normal_shapes=[list(x.shape) for x in normal], friction_shapes=[list(x.shape) for x in friction],
                       contact_count=int(normal[-2].sum()), available=True)
        except Exception as error:
            raw.update(available=False, error=type(error).__name__+': '+str(error))
        contact_api.append(raw)
    unchanged = steps == int(env._sim_step_counter) and seconds == float(env.sim.current_time)
    if not unchanged:
        raise RuntimeError('Read-only geometry gate unexpectedly advanced physics')
    return dict(complete=True, live_collision_parity=not blockers,
        geometry_identity=cooked_geometry.identity if cooked_geometry else None,
        asset_sha256=cooked_geometry.assets if cooked_geometry else source.assets,
        inventory=inventory, rows=rows, blockers=blockers, cooked_requests=cook_requests,
        live_object_query=live_query,
        stage_cache_id=int(stage_id), context_stage_id=int(omni.usd.get_context().get_stage_id()),
        representation_provenance='same SDK reconstructed from exact live Mesh and effective collision properties',
        scope='Exact source-input/schema-copy checks, live FK and current-pose overlap agreement; not a read of runtime cooked-memory buffers',
        limitation='Current reset pose may have no contacts; matching far-separation queries does not establish full PhysX contact/force equivalence',
        maximum_fk_position_error_m=maximum_position_error, maximum_fk_rotation_entry_error=maximum_rotation_error,
        control_steps=0, physics_steps=0, force=dict(accepted=False, calibration_perturbations=0,
            raw_contact_api=contact_api, reason='full_hand_environment_contact_coverage_and_action_response_missing',
            required='Object-filtered contacts for all hand bodies plus separate object-environment contacts, mass/inertia, timestamps, and bounded +/- action-response pairs',
            scope='Existing five distal normal sensors and zero-contact startup cannot identify internal force or calibrate a differentiable action-force model'))


def geometry_soft_acceptance(data_path, live_report):
    """One fixed six-state train-only direct-loss scale/gradient check.

    The same first two train episodes and 0/mid/three-quarter ticks as the old
    check are used, with preregistered [+0.08, -0.08, +0.08] rad perturbations.
    The middle open-hand perturbation tests actual required-contact attraction;
    noncontact rows are allowed zero penalty rather than invented gradients.
    This check does not assert feasible projection or successful learned control.
    """
    import hashlib
    import json
    from pathlib import Path
    import h5py
    path = Path(data_path)
    if path.is_dir():
        path = path/'data.json'
    data = json.loads(path.read_text())
    if data.get('supervision') != 'q_reached' or data.get('normalization_identity') != 'q_reached_train166_affine_v1':
        raise ValueError('Direct geometry calibration requires actual-configuration train-only statistics')
    if not live_report.get('live_collision_parity') or not live_report.get('inventory', {}).get('cooked_hulls'):
        return dict(passed=False, reason='accepted_collision_representation_missing', hand_geometry_weight=0.,
                    attempted=0, loss_identity=DIRECT_HAND_GEOMETRY_ID)
    statistics_path = path.parent/'statistics.pt'
    statistics = torch.load(statistics_path, map_location='cpu', weights_only=False)
    scale, offset = statistics['action_scale'].double(), statistics['action_offset'].double()
    rows, candidates, elapsed = [], [], 0.
    objective = None
    for identity in sorted(data['splits']['train'])[:2]:
        entry = data['entries'][identity]
        context = geometry_loss_context(entry)
        with h5py.File(entry['path']) as f:
            metadata = json.loads(f.attrs['metadata_json'])
            reached = torch.from_numpy(f['episodes'][entry['episode']]['transition/diagnostics_after/joint_position_rad'][:]).double()
        if objective is None:
            objective = HandGeometrySoftLoss(metadata, live_report['inventory'])
            if objective.geometry.assets != live_report['asset_sha256']:
                raise ValueError('Accepted collision assets differ from recorded train assets')
        for tick, perturbation in zip((0, len(reached)//2, 3*len(reached)//4), (.08, -.08, .08)):
            truth = reached[tick:tick+1]
            perturbed = truth.clone(); perturbed[:, 7:] += perturbation
            prediction = (perturbed*scale+offset).detach().requires_grad_()
            physical = (prediction-offset)/scale
            arguments = {k: context[k][tick:tick+1] for k in ('object_pose_base', 'contact_mask', 'stage')}
            geom, report = objective.loss(physical, **arguments)
            fit = masked_objectives(prediction, truth*scale+offset, torch.ones(1, dtype=torch.bool))['B']
            geom_gradient = torch.autograd.grad(geom, prediction, retain_graph=True)[0]
            physical_gradient = torch.autograd.grad(geom, physical, retain_graph=True)[0]
            fit_gradient = torch.autograd.grad(fit, prediction)[0]
            row = report['rows'][0]
            row.update(identity=identity, tick=tick, perturbation_hand_rad=perturbation,
                source_sha256=entry['source_sha256'], fit_loss=float(fit.detach()),
                fit_gradient_norm=float(fit_gradient.norm()), geometry_gradient_norm=float(geom_gradient.norm()),
                arm_geometry_gradient_norm=float(geom_gradient[:, :7].norm()),
                hand_geometry_gradient_norm=float(geom_gradient[:, 7:].norm()),
                gradients_finite=bool(torch.isfinite(geom_gradient).all() and torch.isfinite(fit_gradient).all()))
            if row['geometry_loss'] > 0 and row['geometry_gradient_norm'] > 0:
                candidates.extend((.05*row['fit_loss']/row['geometry_loss'],
                                   .05*row['fit_gradient_norm']/row['geometry_gradient_norm']))
                # A bounded deterministic gradient descent check is diagnostic;
                # it is not a projection or a physics rollout.
                changed = physical.detach()-.001*physical_gradient/physical_gradient.norm().clamp_min(1e-30)
                after, changed_report = objective.loss(changed, **arguments)
                row.update(descent_hand_step_norm_rad=float((changed[:, 7:]-physical.detach()[:, 7:]).norm()),
                    descent_geometry_loss=float(after), descent_decreased=bool(after<geom.detach()),
                    descent_contact_loss=changed_report['rows'][0]['contact_loss'])
                elapsed += changed_report['elapsed_s']
            elapsed += report['elapsed_s']; rows.append(row)
    finite = len(rows) == 6 and all(r['gradients_finite'] and r['arm_geometry_gradient_norm']==0 for r in rows)
    effective = [r for r in rows if r['geometry_loss']>0]
    contact = [r for r in rows if r['contact_loss']>0]
    passed = finite and bool(effective) and bool(contact) and all(
        r['hand_geometry_gradient_norm']>0 and r.get('descent_decreased', False) for r in effective)
    passed = passed and all(r['contact_hand_gradient_norm_per_rad']>0 and
        r.get('descent_contact_loss', float('inf'))<r['contact_loss'] for r in contact)
    weight = min(1., *candidates) if passed and candidates else 0.
    for row in rows:
        row['weighted_loss_fraction'] = weight*row['geometry_loss']/max(row['fit_loss'], 1e-30)
        row['weighted_gradient_fraction'] = weight*row['geometry_gradient_norm']/max(row['fit_gradient_norm'], 1e-30)
    passed = passed and 0 < weight <= 1 and all(r['weighted_loss_fraction'] <= .050000001 and
        r['weighted_gradient_fraction'] <= .050000001 for r in rows)
    return dict(passed=bool(passed), reason='six_fixed_train_soft_loss_scale_and_gradients_passed' if passed else 'direct_soft_acceptance_failed',
        loss_identity=DIRECT_HAND_GEOMETRY_ID, geometry_identity=objective.geometry.identity,
        live_collision_parity=True, asset_sha256=objective.geometry.assets,
        attempted=len(rows), included=len(rows), dropped=0, rows=rows, elapsed_s=elapsed,
        effective_geometry_rows=len(effective), effective_contact_attraction_rows=len(contact),
        hand_geometry_weight=weight if passed else 0., weight_unit='dimensionless', distance_unit='m',
        normalization_identity=data['normalization_identity'], statistics_sha256=hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
        weight_rule='min(1,0.05*Lfit/Lgeom,0.05*norm_grad_Lfit/norm_grad_Lgeom) over positive six-state train losses/gradients in normalized output coordinates; zero rows retained',
        constraints=dict(penetration_tolerance_m=.001, contact_tolerance_m=.002),
        training_selection='at most 2 configurations per complete optimizer update: 1 contact and 1 noncontact when available',
        warmup='skip all geometry computations while inactive or effective coefficient is zero',
        calibration_reads_validation=False, calibration_reads_test=False, solver_calls=0,
        limitation='Direct soft gradients do not guarantee feasible configurations or closed-loop task success; no internal-force term admitted')


def calibrate_geometry_update(acceptance, progress, fraction=.2):
    """Freeze one train-only complete-update output-gradient scale correction.

    The trainer computes the base gradient after global valid-tick weighting,
    and geometry after sum(selected losses)/complete-update selected count.
    Distinct microbatch outputs occupy disjoint coordinates, so their squared
    gradient norms add. These are hand output gradients, not model parameter
    gradients; the bound applies to this recorded update, not future updates.
    The caller verifies/pins the disposable run's source, input and RNG receipts.
    Neither input dictionary is modified and repeated calibration is rejected.
    """
    import copy
    import math
    if not acceptance.get('passed') or acceptance.get('loss_identity') != DIRECT_HAND_GEOMETRY_ID:
        raise ValueError('Accepted direct-soft six-state train calibration required')
    if 'complete_update_calibration' in acceptance:
        raise ValueError('Complete-update geometry calibration is once-only')
    if progress.get('disposable') is not True or progress.get('formal_updates') != 0:
        raise ValueError('Explicit disposable progress with zero formal updates required')
    batch = progress.get('actual_batch')
    if not isinstance(batch, int) or isinstance(batch, bool) or batch <= 0:
        raise ValueError('Complete nonempty optimizer batch required')
    sampling = progress.get('geometry_sampling', {})
    if sampling.get('selected_count') != 2 or sampling.get('selected_contact_count') != 1:
        raise ValueError('Two complete-update selections, one required-contact and one noncontact, required')
    if progress.get('geometry_solver_calls') != 0 or progress.get('hand_geometry_gain') != 1.:
        raise ValueError('Direct geometry without solver calls at unit calibration gain required')

    def number(value, name, positive=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite '+name+' required')
        if (value <= 0 if positive else value < 0):
            raise ValueError(('Positive ' if positive else 'Nonnegative ')+name+' required')
        return float(value)

    fraction = number(fraction, 'complete-update gradient fraction')
    if fraction > 1:
        raise ValueError('Complete-update gradient fraction cannot exceed one')
    initial = number(acceptance.get('hand_geometry_weight'), 'initial six-state geometry weight')
    base = number(progress.get('base_full_update_output_hand_gradient_norm'), 'complete-update base hand output norm')
    geometry = number(progress.get('geometry_full_update_output_hand_gradient_norm'), 'complete-update geometry hand output norm')
    applied = number(progress.get('loss_weights', {}).get('hand_geometry'), 'recorded calibration weight')
    if not math.isclose(applied, initial, rel_tol=1e-10, abs_tol=0.):
        raise ValueError('Disposable update must measure the accepted initial geometry coefficient')
    gradients = progress.get('geometry_probe_gradients')
    if not isinstance(gradients, list) or not gradients:
        raise ValueError('Complete selected-microbatch gradient records required')
    squared = 0.
    for row in gradients:
        arm = number(row.get('arm_norm'), 'geometry arm output gradient', positive=False)
        hand = number(row.get('hand_norm'), 'selected geometry hand output gradient', positive=False)
        if arm != 0:
            raise ValueError('Additional geometry arm gradient must be exactly zero')
        squared += hand*hand
    if not math.isclose(math.sqrt(squared), geometry, rel_tol=1e-6, abs_tol=1e-12):
        raise ValueError('Selected-microbatch gradient norms do not reconstruct the complete-update geometry norm')
    fit_loss = number(progress.get('components', {}).get('base'), 'complete-update fit loss')
    geometry_loss = number(progress.get('components', {}).get('hand_geometry'), 'complete-update geometry loss')
    weight = min(initial, fraction*base/geometry)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError('Calibration must retain a finite positive geometry coefficient')
    result = copy.deepcopy(acceptance)
    result['initial_hand_geometry_weight'] = initial
    result['hand_geometry_weight'] = weight
    for row in result['rows']:
        row['weighted_loss_fraction'] = weight*row['geometry_loss']/max(row['fit_loss'], 1e-30)
        row['weighted_gradient_fraction'] = weight*row['geometry_gradient_norm']/max(row['fit_gradient_norm'], 1e-30)
    result['reason'] = 'six_state_acceptance_and_once_complete_update_scale_passed'
    result['weight_rule'] = ('min(initial six-state train-only upper bound, fraction * complete-update base hand '
        'output gradient norm / unweighted geometry hand output gradient norm); one disposable train update')
    result['complete_update_calibration'] = dict(passed=True, disposable=True, formal_updates=0,
        update=progress.get('update'), actual_batch=batch, selected_count=2, selected_contact_count=1,
        initial_hand_geometry_weight=initial, final_hand_geometry_weight=weight, fraction=fraction,
        base_full_update_output_hand_gradient_norm=base,
        geometry_full_update_output_hand_gradient_norm=geometry,
        initial_weighted_gradient_ratio=initial*geometry/base,
        final_weighted_gradient_ratio=weight*geometry/base,
        final_weighted_geometry_loss=weight*geometry_loss,
        fit_loss=fit_loss, unweighted_geometry_loss=geometry_loss,
        final_weighted_loss_fraction=weight*geometry_loss/fit_loss,
        geometry_arm_gradient_norm=0., solver_calls=0, hand_geometry_gain=1.,
        calibration_reads_validation=False, calibration_reads_test=False,
        scope='Complete-update normalized hand output gradient; sum squared disjoint microbatch norms, not model parameter gradients',
        limitation='A fixed coefficient calibrated on one train update; no claim that every later update obeys this ratio')
    return result


def geometry_acceptance(data_path, live_report):
    """Frozen six train states, fixed perturbation, once-only 5% calibration.

    Call after the q_reached normalizer is written. Neither validation nor test
    is read. This is an implementation/scale acceptance, not task performance.
    Every representative state must satisfy the original fixed constraints;
    failing any state blocks this geometric cohort without a threshold retry.
    """
    import json
    import hashlib
    from pathlib import Path
    import h5py
    path = Path(data_path)
    if path.is_dir():
        path = path/'data.json'
    data = json.loads(path.read_text())
    if data.get('supervision') != 'q_reached' or data.get('normalization_identity') != 'q_reached_train166_affine_v1':
        raise ValueError('Geometry calibration must use separately identified actual-configuration normalization')
    if not live_report.get('live_collision_parity') or not live_report.get('inventory', {}).get('cooked_hulls'):
        return dict(passed=False, reason='live_cooked_geometry_gate_missing', live_collision_parity=False,
                    attempted=0, feasible=0, hand_geometry_weight=0.)
    statistics_path = path.parent/'statistics.pt'
    statistics = torch.load(statistics_path, map_location='cpu', weights_only=False)
    scale, offset = statistics['action_scale'].double(), statistics['action_offset'].double()
    rows, candidate_weights, elapsed = [], [], 0.
    for identity in sorted(data['splits']['train'])[:2]:
        entry = data['entries'][identity]
        context = geometry_loss_context(entry)
        with h5py.File(entry['path']) as f:
            metadata = json.loads(f.attrs['metadata_json'])
            reached = torch.from_numpy(f['episodes'][entry['episode']]['transition/diagnostics_after/joint_position_rad'][:]).double()
        objective = HandGeometryProjection(metadata, live_report['inventory'])
        if objective.geometry.assets != live_report['asset_sha256']:
            raise ValueError('Live cooked collision asset hashes differ from train data')
        for tick in (0, len(reached)//2, 3*len(reached)//4):
            truth = reached[tick:tick+1]
            perturbed = truth.clone(); perturbed[:, 7:] += .08
            prediction = (perturbed*scale+offset).detach().requires_grad_()
            physical = (prediction-offset)/scale
            geom, report = objective.loss(physical, **{key: context[key][tick:tick+1]
                for key in ('object_pose_base', 'contact_mask', 'stage')})
            fit = masked_objectives(prediction, truth*scale+offset, torch.ones(1, dtype=torch.bool))['B']
            geom_gradient = torch.autograd.grad(geom, prediction, retain_graph=True)[0]
            fit_gradient = torch.autograd.grad(fit, prediction)[0]
            row = report['rows'][0]
            row.update(identity=identity, tick=tick, source_sha256=entry['source_sha256'],
                fit_loss=float(fit.detach()), geometry_loss=float(geom.detach()),
                fit_gradient_norm=float(fit_gradient.norm()), geometry_gradient_norm=float(geom_gradient.norm()),
                arm_geometry_gradient_norm=float(geom_gradient[:, :7].norm()),
                hand_geometry_gradient_norm=float(geom_gradient[:, 7:].norm()),
                gradients_finite=bool(torch.isfinite(geom_gradient).all() and torch.isfinite(fit_gradient).all()))
            if row['geometry_loss'] > 0 and row['geometry_gradient_norm'] > 0:
                candidate_weights.extend((.05*row['fit_loss']/row['geometry_loss'],
                                          .05*row['fit_gradient_norm']/row['geometry_gradient_norm']))
            elapsed += report['elapsed_s']; rows.append(row)
    passed = len(rows) == 6 and all(r['feasible'] and r['gradients_finite'] and
        r['arm_geometry_gradient_norm'] == 0 and r['hand_geometry_gradient_norm'] > 0 for r in rows)
    weight = min(1., *candidate_weights) if passed and candidate_weights else 0.
    for row in rows:
        row['weighted_loss_fraction'] = weight*row['geometry_loss']/max(row['fit_loss'], 1e-30)
        row['weighted_gradient_fraction'] = weight*row['geometry_gradient_norm']/max(row['fit_gradient_norm'], 1e-30)
    passed = passed and 0 < weight <= 1 and all(r['weighted_loss_fraction'] <= .050000001 and
        r['weighted_gradient_fraction'] <= .050000001 for r in rows)
    return dict(passed=bool(passed), live_collision_parity=True,
        reason='six_fixed_train_projections_and_scale_passed' if passed else 'fixed_geometry_acceptance_failed',
        attempted=len(rows), feasible=sum(r['feasible'] for r in rows), rows=rows, elapsed_s=elapsed,
        hand_geometry_weight=weight if passed else 0., geometry_identity=objective.geometry.identity,
        asset_sha256=objective.geometry.assets, inventory=live_report['inventory'],
        statistics_sha256=hashlib.sha256(statistics_path.read_bytes()).hexdigest(),
        normalization_identity=data['normalization_identity'], perturbation_hand_rad=.08,
        weight_rule='min(1,0.05*Lfit/Lgeom,0.05*norm_grad_Lfit/norm_grad_Lgeom) over all six train states; gradients in normalized output coordinates',
        constraints=dict(penetration_tolerance_m=.001,contact_tolerance_m=.002,local_radius_rad=.15,
                         maximum_iterations=30,training_projection_rows_per_microbatch=2),
        calibration_reads_validation=False, calibration_reads_test=False,
        limitation='Acceptance uses fixed perturbed recorded configurations, not learned predictions or a closed-loop success claim; no internal-force term admitted')


def calibrated_weights(rows):
    # Each auxiliary term <=5% of base value AND output-gradient norm in the
    # worst of the frozen train batches. Sum bound <=20% by triangle inequality.
    return {key: min(1., *(min(.05*r['loss']['base']/max(r['loss'][key], 1e-12),
        .05*r['gradient']['base']/max(r['gradient'][key], 1e-12)) for r in rows))
        for key in COMPONENTS}


def calibrated_palm_weights(rows):
    """One train-only scale check: each palm term <=5% of base value/gradient.

    Position uses m² (lambda has units m^-2); rotation is dimensionless.
    This rule is frozen before inspecting validation; no repeated tuning.
    """
    return {key: min(min(.05*r['loss']['base']/max(r['loss'][key], 1e-12),
        .05*r['gradient']['base']/max(r['gradient'][key], 1e-12)) for r in rows)
        for key in ('position', 'rotation')}
