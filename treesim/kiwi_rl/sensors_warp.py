import numpy as np
import warp as wp


@wp.kernel
def _metric_depth(raw: wp.array2d(dtype=float), offsets: wp.array(dtype=int), camera: int,
                  minimum_m: float, maximum_m: float, depth: wp.array3d(dtype=float),
                  valid: wp.array3d(dtype=wp.uint8)):
    world, pixel = wp.tid()
    width = depth.shape[2]
    y, x = pixel // width, pixel % width
    value = raw[world, offsets[camera] + pixel]
    available = wp.isfinite(value) and value >= minimum_m and value <= maximum_m
    depth[world, y, x] = wp.where(available, value, 0.)
    valid[world, y, x] = wp.uint8(available)


class WarpRGBDRig:
    def __init__(self, native_model, data, cameras=('hand_camera', 'body_camera'),
                 resolution=(128, 128), minimum_m=.05, maximum_m=4.):
        import mujoco_warp as mw
        self.worlds, self.device = data.qpos.shape[0], data.qpos.device
        self.width, self.height = resolution
        if any(not isinstance(size, int) or not 16 <= size <= 1024 for size in resolution):
            raise ValueError('Invalid camera resolution')
        if not np.isfinite([minimum_m, maximum_m]).all() or not 0 < minimum_m < maximum_m:
            raise ValueError('Invalid depth range')
        self.minimum_m, self.maximum_m = minimum_m, maximum_m
        self.cameras = tuple(cameras)
        if not self.cameras or len(set(self.cameras)) != len(self.cameras):
            raise ValueError('Camera names must be nonempty and unique')
        self.model_camera_ids = [native_model.camera(name).id for name in self.cameras]
        with wp.ScopedDevice(self.device):
            self.context = mw.create_render_context(native_model, nworld=self.worlds, cam_res=resolution,
                cam_active=list(self.cameras), render_rgb=True, render_depth=True, render_seg=False,
                use_textures=True, use_shadows=False, use_fast_math=False)
            self.rgb = [wp.zeros((self.worlds, self.height, self.width), dtype=wp.vec3) for _ in self.cameras]
            self.depth = [wp.zeros((self.worlds, self.height, self.width)) for _ in self.cameras]
            self.valid = [wp.zeros((self.worlds, self.height, self.width), dtype=wp.uint8) for _ in self.cameras]
        self.frame_id, self.timestamp_s, self.available = 0, None, False

    def invalidate(self):
        self.timestamp_s, self.available = None, False

    def capture(self, model, data, timestamp_s):
        import mujoco_warp as mw
        if not np.isfinite(timestamp_s) or timestamp_s < 0 or (self.timestamp_s is not None and timestamp_s < self.timestamp_s):
            raise ValueError('Sensor timestamps must be finite and monotonic')
        with wp.ScopedDevice(self.device):
            mw.render(model, data, self.context)
            for i in range(len(self.cameras)):
                mw.get_rgb(self.context, i, self.rgb[i])
                wp.launch(_metric_depth, dim=(self.worlds, self.height * self.width),
                    inputs=[self.context.depth_data, self.context.depth_adr, i,
                            self.minimum_m, self.maximum_m, self.depth[i], self.valid[i]], device=self.device)
        self.frame_id += 1
        self.timestamp_s = float(timestamp_s)
        self.available = True

    def tensor(self, camera):
        import torch
        if not self.available:
            raise RuntimeError('No sensor frame has been captured')
        index = self.cameras.index(camera)
        wp.synchronize_device(self.device)
        rgb = wp.to_torch(self.rgb[index]).permute(0, 3, 1, 2)
        depth = wp.to_torch(self.depth[index])[:, None] / self.maximum_m
        valid = wp.to_torch(self.valid[index])[:, None].to(rgb.dtype)
        return torch.cat((rgb, depth, valid), dim=1)
