"""Opt-in standard-camera product retirement; no simulator creation or stepping.

This candidate is not live accepted. Install bindings are pinned because lookup
uses Replicator's existing-product registry. Ordinary execution imports nothing
here. Active cameras retain their original renderer, annotators and observation
path. Camera.frame is a read counter, never our evidence of a new exposure.
"""
import copy
import hashlib
import math
from pathlib import Path
import re
import threading


PROFILE = dict(version='standard_inactive_render_products_v1', physics_device='cpu',
    camera_backend='standard', views=3, retirement='whole_wave_monotone',
    frame_proof='hydra_drawable_event_and_host_physics_step_v1',
    inactive_packet='invalid_camera_time_frame_minus_one_pose_zero_v1')
_ISAAC = Path('/home/CNS2025331827/miniconda3/envs/isaaclab/lib/python3.11/site-packages/isaacsim/extscache')
_REP = _ISAAC/'omni.replicator.core-1.12.27+107.3.3.lx64.r.cp311/omni/replicator/core/scripts'
_HYDRA = _ISAAC/'omni.kit.hydra_texture-1.4.6+69cbf6ad.lx64.r.cp311/omni/hydratexture/_hydra_texture.pyi'
_LAB = Path('/home/CNS2025331827/docs/IsaacLab_DexAssemble-sharpa/source/isaaclab/isaaclab/sensors')
INSTALLED_SOURCES = {
    str(_HYDRA): 'd8b91af9acdf9546dc20e58893a5d6c205f65c0789cdc0f9ff2e9f2b441a6132',
    str(_REP/'utils/viewport_manager.py'): '9d233c062a552fcb77615c9244c9556797cebc1f39e672a9abfd3c9f34d05516',
    str(_REP/'annotators.py'): '4b7b872029493d23e4bbb2a2c6c08ceb353aa1273cba167594421d2eeb3d9379',
    str(_REP/'utils/annotator_utils.py'): '2ce643c50f044c1679dd8ca72db735fddf28edc559668990917dadbb5192a85d',
    str(_LAB/'camera/camera.py'): 'a3689cffa01d046b510f7564f7adad972edf6316568b0e323abc2c4f0c4ec354',
    str(_LAB/'sensor_base.py'): '4716a4b89ed1ac4f640f8744f1e2fcaac2e74d00c2193e0ce24e52c61e5659fc',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_profile(value):
    """Missing remains off; a candidate must register the complete identity."""
    if value is None:
        return None
    require(isinstance(value, dict) and value == PROFILE and
        type(value.get('views')) is int, 'Unknown inactive-render profile')
    return copy.deepcopy(PROFILE)


def verify_installed_sources():
    for name, expected in INSTALLED_SOURCES.items():
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected,
            'Installed inactive-render API source changed: '+name)
    return dict(INSTALLED_SOURCES)


def _clock(env):
    # Callback-time observation of the synchronous renderer, not a native
    # camera acquisition timestamp or a claim of full simulator-state replay.
    return dict(research_tick=int(env.research_tick), sim_step=int(env.sim.current_time_step_index),
        sim_time=float(env.sim.current_time))


def _matrix(value, name):
    import numpy as np
    array = np.asarray(value, dtype=float)
    require(array.size == 16 and np.isfinite(array).all(), 'Missing/nonfinite Hydra '+name)
    return array.reshape(4, 4).tolist()


def existing_product(manager, path):
    """Private pinned registry lookup only; a miss must never create a product."""
    product = manager._hydra_textures.get(manager._context, path)
    require(product.path == path, 'Existing product registry returned a different path')
    return product


class InactiveRenderProducts:
    """Owns only toggles and bounded latest event records, never frame buffers.

    Production must use bind(). The constructor is dependency-injected for CPU
    tests and is not an alternative to installed-source admission.
    """
    def __init__(self, bindings, *, profile, num_envs, clock, gate_limit_ticks=None):
        require(validate_profile(profile) is not None, 'Explicit inactive-render profile required')
        require(type(num_envs) is int and num_envs > 0, 'Positive environment count required')
        require(gate_limit_ticks is None or (type(gate_limit_ticks) is int and 1 <= gate_limit_ticks <= 120),
            'Diagnostic gate must have an explicit 1..120 control-tick limit')
        require(len(bindings) == 3*num_envs, 'Exactly three independent products per lane required')
        self.num_envs, self.clock = num_envs, clock
        self.gate_limit_ticks = gate_limit_ticks
        self.bindings = list(bindings)
        self._lock = threading.RLock()
        self._subscriptions = []
        self._latest, self._consumed = {}, {}
        self._error = None
        self._wave = None
        self._last_tick = -1
        self._last_clock = None
        self._active = [True]*num_envs
        self._transitions = []
        self._closed = False
        self.installed_sources = None
        products, cells = set(), set()
        for binding in bindings:
            lane, view = binding['lane'], binding['view']
            require(type(lane) is int and type(view) is int and 0 <= lane < num_envs and 0 <= view < 3,
                'Invalid product lane/view')
            key = (lane, view)
            path = binding['render_product_path']; camera = binding['camera_path']; handle = binding['handle']
            require(key not in cells and path not in products, 'Shared or duplicate render product rejected')
            require(re.search(r'/env_'+str(lane)+r'/', camera) is not None, 'Camera path/lane mismatch')
            require(handle.get_render_product_path() == path and handle.get_camera_path() == camera,
                'Existing Hydra product/camera binding differs')
            require((handle.get_width(), handle.get_height()) == (448, 252), 'Original camera resolution required')
            require(binding['async_rendering'] is False, 'Synchronous existing Hydra product required')
            require(handle.get_updates_enabled() is True, 'All products must initially be enabled')
            products.add(path); cells.add(key)
            self._latest[key] = None
            self._consumed[key] = None
        require(cells == {(lane, view) for lane in range(num_envs) for view in range(3)}, 'Incomplete product map')

    @classmethod
    def bind(cls, env, profile, *, gate_limit_ticks=None):
        """Bind existing initialized CPU-standard products; never call create()."""
        require(validate_profile(profile) is not None, 'Explicit inactive-render profile required')
        require(str(env.device) == 'cpu' and str(env.cfg.sim.device) == 'cpu' and
            env.cfg.student_obs.camera_backend == 'standard', 'Only actual CPU standard cameras admitted')
        require(len(env.research_cameras) == 3, 'Three real standard-camera views required')
        sources = verify_installed_sources()
        import carb.settings
        import omni.hydratexture
        from isaaclab.sensors import Camera
        from omni.replicator.core.scripts.utils import viewport_manager
        from .camera_calibration import ordered_camera_paths
        require(Path(viewport_manager.__file__).resolve() == (_REP/'utils/viewport_manager.py').resolve(),
            'Imported Replicator registry differs from pinned source')
        import sys
        require(Path(sys.modules[Camera.__module__].__file__).resolve() == (_LAB/'camera/camera.py').resolve(),
            'Imported standard Camera differs from pinned source')
        require(len(env._camera_pose_annotators) == 3, 'Original three-view CameraParams readers required')
        manager = viewport_manager.ViewportManager()
        bindings = []
        for view, camera in enumerate(env.research_cameras):
            require(type(camera) is Camera, 'Shared tiled or replaced sensor class rejected')
            paths = ordered_camera_paths(camera, env.num_envs)
            products = list(camera.render_product_paths)
            require(len(products) == env.num_envs, 'One independent product per camera lane required')
            require(len(env._camera_pose_annotators[view]) == env.num_envs and all(
                list(annotator._render_products) == [path] for annotator, path in
                zip(env._camera_pose_annotators[view], products, strict=True)),
                'CameraParams are not bound to the same real RGB-D products')
            for lane, (camera_path, product_path) in enumerate(zip(paths, products, strict=True)):
                # Exact existing path lookup. get_render_product/create can allocate
                # on a miss and are deliberately never used here.
                product = existing_product(manager, product_path)
                handle = product.hydra_texture
                bindings.append(dict(lane=lane, view=view, camera_path=camera_path,
                    render_product_path=product_path, handle=handle, owner=product,
                    async_rendering=carb.settings.get_settings().get(handle.get_settings_path()+'async')))
        result = cls(bindings, profile=profile, num_envs=env.num_envs,
            clock=lambda: _clock(env), gate_limit_ticks=gate_limit_ticks)
        result.installed_sources = sources
        try:
            for binding in bindings:
                key = (binding['lane'], binding['view']); handle = binding['handle']
                def callback(event, key=key, handle=handle):
                    try:
                        info = handle.get_frame_info(event.payload['result_handle'])
                        result.note_drawable(key, info, result.clock())
                    except BaseException as error:
                        result._latch(error)
                subscription = handle.get_event_stream().create_subscription_to_push_by_type(
                    omni.hydratexture.EVENT_TYPE_DRAWABLE_CHANGED, callback,
                    name='Tight inactive-render provenance '+binding['render_product_path'])
                result._subscriptions.append(subscription)
        except BaseException:
            result.close()
            raise
        return result

    def _latch(self, error):
        with self._lock:
            if self._error is None:
                self._error = repr(error)

    def _check(self):
        require(not self._closed, 'Inactive-render guard is closed')
        require(self._error is None, 'Hydra provenance callback failed: '+str(self._error))

    def note_drawable(self, key, info, clock):
        """Callback evidence only. Does not read annotators or advance Kit."""
        with self._lock:
            require(key in self._latest, 'Unexpected renderer product event')
            frame = info.get('frame_number')
            require(type(frame) in (int, float) and math.isfinite(frame) and frame >= 0,
                'Missing finite native Hydra frame number')
            require(tuple(info['resolution']) == (448, 252), 'Renderer resolution drift')
            require(set(clock) == {'research_tick', 'sim_step', 'sim_time'} and
                type(clock['research_tick']) is int and type(clock['sim_step']) is int and
                math.isfinite(clock['sim_time']), 'Incomplete host physics clock')
            old = self._latest[key]
            self._latest[key] = dict(sequence=1 if old is None else old['sequence']+1,
                frame_number=frame, view=_matrix(info.get('view'), 'view'),
                projection=_matrix(info.get('projection'), 'projection'), clock=dict(clock))

    def begin_wave(self, wave):
        """Restore BEFORE whole-wave physical reset, not afterwards."""
        self._check()
        require(type(wave) is int and wave >= 0 and (self._wave is None or wave > self._wave),
            'Wave identity must strictly advance')
        require(self._wave is None or not any(self._active), 'Cannot reset a wave with active lanes')
        self.restore()
        with self._lock:
            self._wave = wave; self._last_tick = -1
            self._last_clock = None
            self._active = [True]*self.num_envs
            self._consumed = copy.deepcopy(self._latest)
            self._transitions = []

    def observe(self, packet, *, control_tick):
        """Prove active freshness once per pre-action sample; return invalid-marked copy.

        The second return is a small receipt. Host event/physics-clock coherence
        is explicit evidence, not an intrinsic exposure-time sensor field.
        """
        import numpy as np
        import torch
        from .camera_rigs import renderer_pose_ros
        self._check()
        require(self._wave is not None and type(control_tick) is int and control_tick == self._last_tick+1,
            'Exactly one ordered pre-action sample per wave/control tick required')
        require(self.gate_limit_ticks is None or control_tick < self.gate_limit_ticks, 'Finite diagnostic gate exhausted')
        shape = (self.num_envs, 3)
        require(all(torch.is_tensor(packet[key]) and packet[key].device.type == 'cpu' for key in
            ('camera_valid', 'camera_time_s', 'camera_frame', 'camera_pose_w', 'time_s')),
            'Actual CPU sensor packet required')
        require(packet['camera_valid'].shape == shape and packet['camera_valid'].dtype == torch.bool and
            packet['camera_time_s'].shape == shape and packet['camera_frame'].shape == shape and
            packet['camera_frame'].dtype == torch.int64 and
            packet['camera_pose_w'].shape == (*shape, 7) and packet['time_s'].shape == (self.num_envs,),
            'Camera packet shape changed')
        ids = [lane for lane, active in enumerate(self._active) if active]
        require(ids and bool(packet['camera_valid'][ids].all()), 'Active camera content invalid or no active lane')
        require(bool((packet['camera_frame'][ids] >= 0).all()), 'Invalid active camera read counter')
        require(torch.isfinite(packet['camera_time_s'][ids]).all() and
            torch.allclose(packet['camera_time_s'][ids], packet['time_s'][ids,None], rtol=0, atol=1e-8),
            'Active camera/control clock mismatch')
        clock = self.clock(); evidence = []; advancement = None
        require(bool(torch.isfinite(packet['time_s'][ids]).all()) and bool(
            (torch.abs(packet['time_s'][ids]-clock['research_tick']/60) <= 1e-8).all()),
            'Packet clock does not describe the current 60Hz research tick')
        with self._lock:
            self._check()
            if self._last_clock is not None:
                prior_clock = self._last_clock
                require(clock['research_tick'] == prior_clock['research_tick']+1,
                    'Research clock did not advance exactly one control tick')
                require(clock['sim_step'] == prior_clock['sim_step']+2,
                    'Physics clock did not advance exactly two 120Hz steps')
                elapsed = clock['sim_time']-prior_clock['sim_time']
                # Core's PhysX callback can expose a float32 substep duration;
                # also allow double accumulation roundoff at large global times.
                # Integer tick/step deltas remain exact. This cannot admit an
                # extra or missing physics substep.
                tolerance = max(4*2**-23/60,
                    8*math.ulp(max(abs(clock['sim_time']), abs(prior_clock['sim_time']), 1.)))
                require(math.isfinite(elapsed) and abs(elapsed-1/60) <= tolerance,
                    'Physics time did not advance by the original 1/60 second')
                advancement = dict(research_ticks=1, physics_steps=2,
                    sim_time_delta_s=elapsed, sim_time_tolerance_s=tolerance)
            for binding in self.bindings:
                lane, view = binding['lane'], binding['view']
                if not self._active[lane]:
                    require(binding['handle'].get_updates_enabled() is False, 'Inactive product unexpectedly re-enabled')
                    continue
                require(binding['handle'].get_updates_enabled() is True, 'Active product was disabled')
                key = (lane, view); current = self._latest[key]; prior = self._consumed[key]
                require(current is not None and (prior is None or
                    (current['sequence'] > prior['sequence'] and current['frame_number'] > prior['frame_number'])),
                    'No new real renderer frame for active lane/view')
                require(current['clock'] == clock, 'Renderer event predates current physical state')
                actual = packet['camera_pose_w'][lane,view].numpy()
                rendered = renderer_pose_ros(current['view'])
                # Quaternion signs are equivalent. Fixed tolerances are an explicit
                # candidate gate; failure stops rather than substituting USD poses.
                require(np.isfinite(actual).all() and np.max(np.abs(actual[:3]-rendered[:3])) <= 2e-5 and
                    min(np.max(np.abs(actual[3:]-rendered[3:])), np.max(np.abs(actual[3:]+rendered[3:]))) <= 2e-5,
                    'Current CameraParams and actual renderer view differ')
                evidence.append(dict(lane=lane, view=view, render_product_path=binding['render_product_path'],
                    event_sequence=current['sequence'], renderer_frame=current['frame_number'],
                    camera_read_counter=int(packet['camera_frame'][lane,view])))
            for binding in self.bindings:
                key = (binding['lane'], binding['view'])
                if self._active[binding['lane']]:
                    self._consumed[key] = copy.deepcopy(self._latest[key])
            self._last_tick = control_tick
            self._last_clock = dict(clock)
        return self.mask_inactive(packet), dict(wave=self._wave, control_tick=control_tick,
            host_physics_clock=clock, advancement_since_previous_observation=advancement,
            active_lanes=ids, products=evidence,
            scope='New synchronous Hydra drawable plus host physics-step coherence and renderer/CameraParams pose agreement; not hardware exposure timestamps.')

    def mask_inactive(self, packet):
        """No image cache/copy: inactive pixels are explicitly unusable, not fresh."""
        import torch
        result = dict(packet)
        inactive = torch.tensor([not value for value in self._active], dtype=torch.bool)
        if bool(inactive.any()):
            for key, sentinel in (('camera_valid', False), ('camera_time_s', -1.), ('camera_frame', -1), ('camera_pose_w', 0.)):
                result[key] = packet[key].clone()
                result[key][inactive] = sentinel
        return result

    def retire(self, lanes, *, reason, control_tick):
        """Retire after final transition copying; never reactivate within a wave."""
        self._check()
        require(self._wave is not None and type(control_tick) is int and control_tick == self._last_tick and self._last_tick >= 0,
            'Retirement requires an already validated actual pre-action sample')
        require(reason in ('initial_illegal', 'unassigned', 'natural_termination', 'diagnostic_censor'),
            'Explicit retirement reason required')
        require(reason != 'diagnostic_censor' or self.gate_limit_ticks is not None,
            'Artificial retirement is allowed only in a finite diagnostic gate')
        require(reason not in ('initial_illegal', 'unassigned') or control_tick == 0,
            'Initial retirement only at the first sample')
        lanes = list(lanes)
        require(lanes and len(set(lanes)) == len(lanes) and all(type(lane) is int and
            0 <= lane < self.num_envs and self._active[lane] for lane in lanes), 'Only distinct active lanes can retire')
        try:
            for binding in self.bindings:
                if binding['lane'] in lanes:
                    binding['handle'].set_updates_enabled(False)
                    require(binding['handle'].get_updates_enabled() is False, 'Renderer disable readback failed')
        except BaseException as error:
            self._latch(error)
            self.restore()
            raise
        for lane in lanes:
            self._active[lane] = False
        self._transitions.append(dict(control_tick=control_tick, lanes=lanes, reason=reason))

    def restore(self):
        """Best-effort ALL owned products, then raise if any restoration failed."""
        errors = []
        for binding in self.bindings:
            try:
                binding['handle'].set_updates_enabled(True)
                require(binding['handle'].get_updates_enabled() is True, 'Renderer restore readback failed')
            except BaseException as error:
                errors.append(dict(product=binding['render_product_path'], error=repr(error)))
        if errors:
            self._latch(errors)
            raise RuntimeError('Unable to restore all owned renderer products: '+repr(errors))

    def snapshot(self):
        with self._lock:
            return dict(profile=copy.deepcopy(PROFILE), installed_sources=self.installed_sources,
                wave=self._wave, last_control_tick=self._last_tick, active=list(self._active),
                last_host_physics_clock=copy.deepcopy(self._last_clock),
                transitions=copy.deepcopy(self._transitions), callback_error=self._error, closed=self._closed,
                gate_limit_ticks=self.gate_limit_ticks, live_accepted=False,
                products=[dict(lane=b['lane'], view=b['view'], camera_path=b['camera_path'],
                    render_product_path=b['render_product_path'], enabled=b['handle'].get_updates_enabled(),
                    latest=copy.deepcopy(self._latest[(b['lane'],b['view'])])) for b in self.bindings])

    def close(self):
        errors = []
        try:
            self.restore()
        except BaseException as error:
            errors.append(repr(error))
        finally:
            # Drop callback ownership even on restoration failure. Never destroy,
            # detach, reset or render a product/sensor while closing this helper.
            for subscription in self._subscriptions:
                try:
                    subscription.unsubscribe()
                except BaseException as error:
                    errors.append(repr(error))
            self._subscriptions.clear()
            self._closed = True
        if errors:
            raise RuntimeError('Inactive renderer cleanup failed: '+repr(errors))
