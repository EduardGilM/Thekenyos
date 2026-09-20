"""Choose a fruit-facing arm pose at export, never in the actor observation path."""
import numpy as np


def set_camera_start_pose(model, data, robot, fruit_body, *, camera_distance_m=.25):
    """Move only six arm joints; reject collisions or an unusable camera view."""
    import mujoco
    from scipy.optimize import least_squares

    if not np.isfinite(camera_distance_m) or not .15 <= camera_distance_m <= .4:
        raise ValueError('Camera starting distance must be in [.15, .4] metres')
    names = robot['arm'][:6]
    joints = [model.joint(robot['prefix'] + name).id for name in names]
    qids = model.jnt_qposadr[joints]
    home = data.qpos[qids].copy()
    lower = np.where(model.jnt_limited[joints], model.jnt_range[joints, 0], home - np.pi)
    upper = np.where(model.jnt_limited[joints], model.jnt_range[joints, 1], home + np.pi)
    camera = model.camera('hand_camera').id
    fruit = model.body(fruit_body).id
    tcp = model.site('hand_tcp').id
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qvel[:] = data.qvel
    scratch.mocap_pos[:] = data.mocap_pos
    scratch.mocap_quat[:] = data.mocap_quat

    def residual(q):
        scratch.qpos[qids] = q
        mujoco.mj_forward(model, scratch)
        delta = scratch.xpos[fruit] - scratch.cam_xpos[camera]
        distance = np.linalg.norm(delta)
        forward = -scratch.cam_xmat[camera].reshape(3, 3)[:, 2]
        tcp_distance = np.linalg.norm(scratch.xpos[fruit] - scratch.site_xpos[tcp])
        return np.r_[2 * (forward - delta / max(distance, 1e-9)),
                     4 * (distance - camera_distance_m), 4 * (tcp_distance - (camera_distance_m - .05)), .02 * (q - home)]

    result = least_squares(residual, np.clip(home, lower, upper), bounds=(lower, upper), max_nfev=160)
    error = residual(result.x)
    local = scratch.cam_xmat[camera].reshape(3, 3).T @ (scratch.xpos[fruit] - scratch.cam_xpos[camera])
    half_fov = np.arctan(model.cam_sensorsize[camera] / (2 * model.cam_intrinsic[camera, :2]))
    angles = np.arctan2(np.abs(local[:2]), -local[2])
    arm_bodies = set(map(int, model.jnt_bodyid[joints]))
    for body in range(model.nbody):
        if int(model.body_parentid[body]) in arm_bodies:
            arm_bodies.add(body)
    contacts = [(int(c.geom1), int(c.geom2)) for c in scratch.contact
                if c.dist < 0 and (int(model.geom_bodyid[c.geom1]) in arm_bodies or
                                   int(model.geom_bodyid[c.geom2]) in arm_bodies)]
    if (local[2] >= 0 or np.any(angles > .8 * half_fov) or
            np.linalg.norm(error[:5]) > .12 or contacts):
        raise RuntimeError(f'No feasible camera starting pose: angles={angles}, residual={error[:5]}, contacts={contacts}')
    data.qpos[qids] = result.x
    mujoco.mj_forward(model, data)
    return dict(joint_positions_rad=dict(zip(names, map(float, result.x))),
                camera_fruit_distance_m=float(np.linalg.norm(local)),
                tcp_fruit_distance_m=float(np.linalg.norm(data.xpos[fruit] - data.site_xpos[tcp])),
                camera_angles_rad=angles.tolist(), collision_free=True,
                scope='export-time starting pose; fixed camera; no actor target input')
