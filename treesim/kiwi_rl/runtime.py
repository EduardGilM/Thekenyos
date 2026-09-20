import numpy as np
import warp as wp

from .control_warp import WarpSpotControl
from .physics import DeformableMonitor, FlexContactObserver
from .scene import load_scene_artifact
from .sensors_warp import WarpRGBDRig


class BatchedDeformableRuntime:
    def __init__(self, directory, worlds=2, block_steps=20, resolution=None, device='cuda:0'):
        import mujoco
        import mujoco_warp as mw
        if not isinstance(worlds, int) or not 1 <= worlds <= 64 or not isinstance(block_steps, int) or not 1 <= block_steps <= 100:
            raise ValueError('Invalid batching configuration')
        self.model, self.initial, self.manifest = load_scene_artifact(directory)
        self.dt = float(self.model.opt.timestep)
        self.gait_stride = round(.02 / self.dt)
        if abs(self.gait_stride * self.dt - .02) > 1e-9 or self.gait_stride % block_steps:
            raise ValueError('Physics blocks must exactly divide the 50 Hz gait interval')
        self.worlds, self.block_steps = worlds, block_steps
        wp.init()
        with wp.ScopedDevice(device):
            self.gpu_model = mw.put_model(self.model)
            self.data = mw.put_data(self.model, self.initial, nworld=worlds,
                                   nconmax=131072, nccdmax=131072, njmax=8192)
            self.device = self.data.qpos.device
            self.control = WarpSpotControl(self.model, self.data, self.manifest['robot'])
            self.monitor = DeformableMonitor(self.model, self.initial, self.data)
            self.contacts = FlexContactObserver(self.model, self.data)
            self.state_signature = int(mujoco.mjtState.mjSTATE_INTEGRATION)
            initial_state = np.empty(mujoco.mj_stateSize(self.model, self.state_signature))
            mujoco.mj_getState(self.model, self.initial, initial_state, self.state_signature)
            self.initial_state = wp.array(np.tile(initial_state, (worlds, 1)), dtype=float)
            self.control.apply()
            mw.step(self.gpu_model, self.data)
            self.contacts.record()
            self.refresh()
            self.monitor.record()
            self.check()
            with wp.ScopedCapture() as capture:
                for _ in range(block_steps):
                    self.control.apply()
                    mw.step(self.gpu_model, self.data)
                    self.contacts.record()
                    mw.kinematics(self.gpu_model, self.data)
                    mw.flex(self.gpu_model, self.data)
                    self.monitor.record()
            self.graph = capture.graph
            self.rig = None
            self.reset()
            if resolution is not None:
                self.rig = WarpRGBDRig(self.model, self.data,
                    cameras=tuple(c['name'] for c in self.manifest['cameras']), resolution=resolution)

    def refresh(self):
        import mujoco_warp as mw
        with wp.ScopedDevice(self.device):
            mw.kinematics(self.gpu_model, self.data)
            mw.com_pos(self.gpu_model, self.data)
            mw.com_vel(self.gpu_model, self.data)
            mw.camlight(self.gpu_model, self.data)
            mw.flex(self.gpu_model, self.data)

    def reset(self):
        import mujoco_warp as mw
        with wp.ScopedDevice(self.device):
            mw.reset_data(self.gpu_model, self.data)
            mw.set_state(self.gpu_model, self.data, self.initial_state, self.state_signature)
            self.control.set_targets(np.tile(self.control.contract.targets, (self.worlds, 1)))
            self.control.previous.zero_()
            self.control.commands.zero_()
            self.control.set_jaw_caps(np.full(self.worlds, min(.3, float(self.control.contract.limits[-1]))))
            self.monitor.reset()
            self.contacts.reset()
            mw.forward(self.gpu_model, self.data)
            self.refresh()
            self.monitor.record()
            self.check()
            self.step_index = 0
            if self.rig is not None:
                self.rig.invalidate()

    def check(self):
        if self.monitor.flags.numpy().any():
            self.monitor.check()
        if self.contacts.flags.numpy().any():
            self.contacts.check()

    def advance(self, control_dt_s=.04, gait_action_fn=None):
        if not np.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError('Control interval must be finite and positive')
        steps = round(control_dt_s / self.dt)
        if steps < 1 or abs(steps * self.dt - control_dt_s) > 1e-9 or steps % self.block_steps:
            raise ValueError('Control interval must contain complete physics blocks')
        with wp.ScopedDevice(self.device):
            for _ in range(steps // self.block_steps):
                if gait_action_fn is not None and self.step_index % self.gait_stride == 0:
                    self.refresh()
                    observations = self.control.observe().numpy()
                    self.control.set_gait_actions(gait_action_fn(observations))
                wp.capture_launch(self.graph)
                self.step_index += self.block_steps
                self.check()
            self.refresh()
        return self.step_index * self.dt

    def capture(self):
        if self.rig is None:
            raise RuntimeError('No camera rig was configured')
        self.refresh()
        self.rig.capture(self.gpu_model, self.data, self.step_index * self.dt)
        return {camera: self.rig.tensor(camera) for camera in self.rig.cameras}

    def report(self):
        return dict(scope='Experimental batched deformable runtime; not training-readiness acceptance',
                    training_ready=False, worlds=self.worlds, control_time_s=self.step_index * self.dt,
                    numerical=self.monitor.check(), contacts=self.contacts.check())
