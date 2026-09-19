from dataclasses import dataclass

import numpy as np
from scipy import ndimage


APPROACH_STANDOFF_M = .115


def pinhole_intrinsics(size, fov_deg):
    if size <= 0 or not np.isfinite(fov_deg) or not 1 < fov_deg < 179:
        raise ValueError('Invalid pinhole intrinsics')
    focal = size/(2*np.tan(np.deg2rad(fov_deg/2)))
    return np.array([[focal, 0., (size-1)/2], [0., focal, (size-1)/2], [0., 0., 1.]])


def optical_transform(position, forward, up):
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack([np.cross(forward, up), -np.asarray(up), forward])
    transform[:3, 3] = position
    return transform


def transform_points(points, transform):
    return np.asarray(points)@transform[:3, :3].T+transform[:3, 3]


def backproject_depth(depth_m, intrinsics):
    rows, cols = np.indices(depth_m.shape)
    return np.stack([(cols-intrinsics[0, 2])/intrinsics[0, 0]*depth_m,
                     (rows-intrinsics[1, 2])/intrinsics[1, 1]*depth_m, depth_m], axis=-1)


def project_points(points, intrinsics):
    points = np.asarray(points)
    valid = np.isfinite(points).all(axis=-1) & (points[..., 2] > 1e-6)
    z = np.where(valid, points[..., 2], 1.)
    pixels = np.stack([intrinsics[0, 0]*points[..., 0]/z+intrinsics[0, 2],
                       intrinsics[1, 1]*points[..., 1]/z+intrinsics[1, 2]], axis=-1)
    return pixels, valid


def brown_mask(image):
    r, g, b = image.astype(float).transpose(2, 0, 1)
    return (r > 1.18*g) & (g > 1.2*b) & (r > 30) & (g > 15)


def estimate_fruit(packet):
    cameras, intrinsics = packet['camera_to_base'], packet['intrinsics']
    rgb = packet['rgb']
    if np.asarray(rgb).shape[0] != 3 or np.asarray(cameras).shape[0] != 2 or np.asarray(intrinsics).shape[0] != 2:
        raise ValueError('Hand-only packets use one RGB camera and wrist ToF')
    estimates = []
    depth = packet['tof'][0]*3.
    valid = packet['tof'][1] > .5
    points = transform_points(backproject_depth(depth, intrinsics[1])[valid], cameras[1])
    in_rgb = transform_points(points, np.linalg.inv(cameras[0]))
    pixels, visible = project_points(in_rgb, intrinsics[0])
    pixels = np.rint(np.where(visible[:, None], pixels, -1.)).astype(int)
    height, width = depth.shape
    visible &= (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    color = brown_mask(rgb[:3].transpose(1, 2, 0))
    ids = np.flatnonzero(visible)
    ids = ids[color[pixels[ids, 1], pixels[ids, 0]]]
    foreground = np.zeros(depth.shape, dtype=bool)
    foreground.ravel()[np.flatnonzero(valid)[ids]] = True
    edges = np.zeros_like(valid)
    jump_x = valid[:, 1:] & valid[:, :-1] & (np.abs(np.diff(depth, axis=1)) > .025)
    jump_y = valid[1:] & valid[:-1] & (np.abs(np.diff(depth, axis=0)) > .025)
    edges[:, 1:] |= jump_x
    edges[:, :-1] |= jump_x
    edges[1:] |= jump_y
    edges[:-1] |= jump_y
    labels, count = ndimage.label(foreground & ~edges)
    for label in range(1, count+1):
        selected = points[labels[valid] == label]
        if len(selected) < 6:
            continue
        surface = np.median(selected, axis=0)
        centered = selected-selected.mean(axis=0)
        eigenvalues = np.linalg.eigvalsh(centered.T@centered/len(selected))
        if np.max(np.linalg.norm(selected-surface, axis=1)) > .09 or eigenvalues[-1] > 6*max(eigenvalues[-2], 1e-8):
            continue
        ray = surface-cameras[1, :3, 3]
        point = surface+.03*ray/max(np.linalg.norm(ray), 1e-6)
        if .4 < point[2] < 1.6:
            estimates.append(dict(point=point, source='hand_rgb_tof', pixels=len(selected), uncertainty_m=.02))
    return estimates


@dataclass
class BodyFirstServo:
    standoff_m: float = .5
    mode: str = 'body_first'

    def __post_init__(self):
        if self.mode not in ('fixed', 'body_first', 'simultaneous') or not 0 <= self.standoff_m <= .9:
            raise ValueError('Invalid body positioning experiment')
        self.reset()

    def reset(self):
        self.settled = 0
        self.positioned = self.closing = self.hand_tracking = False
        self.close_ticks = 0
        self.target = None
        self.hits = self.missing = self.ticks = 0
        self.capture_step = -1
        self.last = dict(phase='search', source='none', estimated_gap_m=None)

    def action(self, packet):
        action = np.zeros(10)
        action[-1] = -1.
        if self.closing:
            action[-1] = 1.
            return action
        if packet['age_s'] > .25:
            self.last = dict(phase='stale_stop', source='none', estimated_gap_m=None)
            self.settled = self.close_ticks = 0
            return action
        self.ticks += 1
        if self.target is not None:
            velocity = packet['body_velocity']
            self.target -= .1*(velocity[:3]+np.cross(velocity[3:], self.target))
        candidates = [c for c in estimate_fruit(packet) if abs(c['point'][1]) < .6 and c['point'][2] > .65]
        if self.hand_tracking:
            candidates = [c for c in candidates if c['source'] == 'hand_rgb_tof']
        if self.target is not None:
            candidates = [c for c in candidates if np.linalg.norm(c['point']-self.target) < .3]
        if not candidates:
            self.last = dict(phase='lost_stop', source='none', estimated_gap_m=None)
            self.settled = self.close_ticks = 0
            self.missing += 1
            if self.missing >= 5:
                self.target, self.hits, self.capture_step = None, 0, -1
            return action
        self.missing = 0
        tcp = packet['tcp_base']
        hand = [c for c in candidates if c['source'] == 'hand_rgb_tof']
        selected = min(hand or candidates, key=lambda c: np.linalg.norm(c['point']-(tcp if self.target is None else self.target)))
        self.hand_tracking = self.hand_tracking or selected['source'] == 'hand_rgb_tof'
        capture_step = packet.get('capture_step', self.ticks)
        fresh = capture_step != self.capture_step
        if fresh:
            self.hits += 1
            self.capture_step = capture_step
            self.target = selected['point'].copy() if self.target is None else .5*(self.target+selected['point'])
        target = self.target.copy()
        if self.hits < 3:
            self.last = dict(phase='confirm', source=selected['source'], estimated_gap_m=None)
            return action
        body_error = target[:2]-[self.standoff_m, 0.]
        positioned = np.linalg.norm(body_error) < .1 and np.linalg.norm(packet['body_velocity'][:2]) < .1
        self.settled = self.settled+1 if positioned else 0
        self.positioned = self.positioned or self.settled >= 3
        align = self.mode == 'body_first' and not self.positioned
        if self.mode != 'fixed' and (align or self.mode == 'simultaneous'):
            action[:2] = np.clip(1.3*body_error/[.4, .25], -.55, .55)
        forward = packet.get('hand_forward_base', packet['camera_to_base'][0, :3, 2])
        desired = target-tcp
        desired /= max(np.linalg.norm(desired), 1e-6)
        if not align:
            action[6:9] = np.clip(1.2*np.cross(forward, desired)/.6, -.25, .25)
        delta = target-tcp
        error = delta-APPROACH_STANDOFF_M*forward
        if not align and selected['source'] == 'hand_rgb_tof':
            action[3:6] = np.clip(2*error/.35, -.5, .5)
            ready = np.linalg.norm(error) < .012 and np.linalg.norm(packet['body_velocity'][:2]) < .1
            self.close_ticks = self.close_ticks+int(fresh) if ready else 0
            if self.close_ticks >= 3:
                self.closing = True
                action[:] = 0.
                action[-1] = 1.
        else:
            self.close_ticks = 0
        self.last = dict(phase='close' if self.closing else 'body' if align else 'fine', source=selected['source'],
                         estimate_base_m=target.tolist(), estimated_gap_m=float(np.linalg.norm(delta)),
                         body_error_m=float(np.linalg.norm(body_error)), candidates=len(candidates))
        return action
