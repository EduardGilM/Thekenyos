#!/usr/bin/env python
"""Scripted RidgebackFranka kiwi harvest on the pergola, recorded to MP4.

The ``--auto`` pipeline is apple-only (pergola autonomy is not implemented),
so this is a SCRIPTED sequence on ground-truth fruit poses — no perception:
drive under the canopy, reach up beneath a hanging kiwi, close the pincer,
pull down until the stem ruptures, carry the fruit over the bucket and
release.  The kiwi path deliberately has no grip-assist spring: the fruit is
held by finger-pad contact alone, so a grab can still slip.

    python scripts/pick_kiwi.py --video output/kiwi_pick.mp4 --picks 3
    python scripts/pick_kiwi.py --show                 # watch it live too

Remaining arguments are forwarded to grow_tree.py (e.g. --seed 7 --frames 2400).
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

import warp as wp  # noqa: E402

from treesim import robot as _robot          # noqa: E402
from treesim.picker import ArmIK, _inv_xform, _rot  # noqa: E402


class ScriptedPicker(_robot.RobotDriver):
    """Per-frame pick sequence, driven through RobotDriver's target mirrors."""

    UP = np.array([0.0, 0.0, 1.0])
    DOWN = np.array([0.0, 0.0, -1.0])
    FINGER_OPEN, FINGER_CLOSED = 0.04, 0.016
    MAX_PICKS = 3

    def __init__(self, sim, tm, rp):
        super().__init__(sim, tm, rp)
        self.ik = ArmIK(list(_robot._ARM_HOME.values()))
        rb = tm.robot_data
        self.chassis = int(rb["chassis"][0])
        self.wrist = int(rb["wrist"][0])
        self.arm_tq = np.asarray(rb["arm_tq"][0], dtype=int)
        self.arm_home = np.asarray(rb["arm_home"], dtype=float)
        self.fruit_bodies = np.asarray(tm.apple_bodies, dtype=int)
        self.apples = sim.apples
        self._q_goal = self.arm_home.copy()
        self._q_cmd = self.arm_home.copy()
        self.state = "SETTLE"
        self._t = 0
        self._fi = -1                # global fruit index being picked
        self._standoff = None        # chassis xy goal
        self._pre = None             # pre-grasp world point
        self._servo_off = np.zeros(3)
        self._retract = 0.0
        self._avoid: list = []       # fruit indices to skip
        self.picked = 0
        self.max_picks = ScriptedPicker.MAX_PICKS
        self.done = False
        self._frame = 0
        print("[pick] scripted pergola harvest starting")

    # ---- low-level helpers ---------------------------------------------------
    def _bq(self):
        return self.sim.body_q_np()

    def _yaw(self):
        return float(self.sim.joint_q_np()[int(self.planar_q[0]) + 3])

    def _chassis_pose(self):
        q = self._bq()[self.chassis]
        return q[:3].copy(), q[3:].copy()

    def _tcp(self):
        q = self._bq()[self.wrist]
        return q[:3] + _rot(q[3:], np.array([0.0, 0.0, 0.10]))

    def _ik_to(self, world_pt, approach_world):
        base, bq = self._chassis_pose()
        tgt = _inv_xform(base, bq, world_pt)
        app = _rot(np.array([-bq[0], -bq[1], -bq[2], bq[3]]), approach_world)
        if np.linalg.norm(app) < 1e-6:
            app = np.array([1.0, 0.0, 0.0])
        return self.ik.solve(tgt, app, self._q_cmd[:len(self.ik._arm_q)])

    def _goto(self, state):
        self.state = state
        self._t = 0
        print(f"[pick] {state.lower()} (frame {self._frame})", flush=True)

    def _fail(self, why):
        print(f"[pick] fruit {self._fi} failed: {why}", flush=True)
        if self._fi >= 0:
            self._avoid.append(self._fi)
            self._fi = -1
        self._q_goal = self.arm_home.copy()
        self._goto("SELECT")

    # ---- state machine ---------------------------------------------------------
    def update(self, viewer=None):
        self._frame += 1
        self._t += 1
        bq = self._bq()
        ds = int(self.planar_dof[0])
        vx = vy = w = 0.0
        chp = bq[self.chassis, :3]
        yaw = self._yaw()

        st = self.state
        if st == "SETTLE":
            if self._t > 100:
                self._goto("SELECT")

        elif st == "SELECT":
            if self.picked >= self.max_picks:
                self.done = True
                self._goto("DONE")
            else:
                pos = bq[self.fruit_bodies, :3]
                det = self.apples.detached
                best, bs = -1, 1e9
                for i in range(len(self.fruit_bodies)):
                    if i in self._avoid or det[i] or pos[i, 2] < 0.6:
                        continue
                    # keep clear of the four corner posts (+-1.5, +-2)
                    px, py = pos[i, 0], pos[i, 1]
                    if min(math.hypot(px - sx * 1.5, py - sy * 2.0)
                           for sx in (-1, 1) for sy in (-1, 1)) < 0.9:
                        continue
                    # standoff between robot and fruit: shoulder (chassis +0.15
                    # along the heading) ends ~0.30 m short of the fruit
                    d = pos[i, :2] - chp[:2]
                    n = np.linalg.norm(d)
                    if n < 1e-6:
                        continue
                    so = pos[i, :2] - 0.45 * d / n
                    score = np.linalg.norm(so - chp[:2])
                    if score < bs:
                        best, bs = i, score
                if best < 0:
                    print("[pick] no reachable fruit left", flush=True)
                    self.done = True
                    self._goto("DONE")
                else:
                    self._fi = best
                    fp = pos[best]
                    d = fp[:2] - chp[:2]
                    self._standoff = fp[:2] - 0.45 * d / np.linalg.norm(d)
                    self._pre = fp - 0.16 * self.UP
                    self._servo_off[:] = 0.0
                    self._retract = 0.0
                    self._goto("DRIVE")

        elif st == "DRIVE":
            fp = bq[self.fruit_bodies[self._fi], :3]
            err = self._standoff - chp[:2]
            dist = np.linalg.norm(err)
            yaw_t = math.atan2(fp[1] - chp[1], fp[0] - chp[0])
            dyaw = (yaw_t - yaw + math.pi) % (2 * math.pi) - math.pi
            if dist > 0.06:
                v = np.clip(1.6 * err, -0.7, 0.7)
                vx, vy = float(v[0]), float(v[1])
            w = float(np.clip(1.8 * dyaw, -1.2, 1.2))
            if dist < 0.06 and abs(dyaw) < 0.10:
                self._goto("REACH")
            elif self._t > 900:
                self._fail("drive timeout")

        elif st in ("REACH", "GRASP", "PULL", "TRANSPORT"):
            fp = bq[self.fruit_bodies[self._fi], :3]
            if st == "REACH":
                if self._t % 10 == 1:
                    q, err = self._ik_to(self._pre, self.UP)
                    if err > 0.10:
                        self._fail("pre-grasp unreachable")
                        return self._finish(vx, vy, w)
                    self._q_goal[:len(q)] = q
                self._q_goal[-2:] = self._q_cmd[-2:] = self.FINGER_OPEN
                if np.linalg.norm(self._tcp() - self._pre) < 0.05:
                    self._goto("GRASP")
                elif self._t > 400:
                    self._fail("reach stalled")
            elif st == "GRASP":
                if self._t % 8 == 1:
                    # integral droop compensation: position servos sag a few cm
                    self._servo_off = np.clip(
                        self._servo_off + 0.35 * (fp - self._tcp()), -0.09, 0.09)
                    q, err = self._ik_to(fp + self._servo_off, self.UP)
                    if err > 0.10:
                        self._fail("grasp unreachable")
                        return self._finish(vx, vy, w)
                    self._q_goal[:len(q)] = q
                if np.linalg.norm(self._tcp() - fp) < 0.05 or self._t > 200:
                    self._q_goal[-2:] = self._q_cmd[-2:] = self.FINGER_CLOSED
                    if self._t > 30:
                        self._goto("PULL")
                if self._t > 260:
                    self._fail("grasp stalled")
            elif st == "PULL":
                if self.apples.detached[self._fi]:
                    print(f"[pick] fruit {self._fi} detached", flush=True)
                    self._goto("TRANSPORT")
                    return self._finish(vx, vy, w)
                if self._t > 20:
                    self._retract = min(self._retract + 0.0028, 0.34)
                if self._t % 10 == 1:
                    q, _ = self._ik_to(fp - self._retract * self.UP, self.UP)
                    self._q_goal[:len(q)] = q
                if self._t > 380:
                    self._fail("stem too strong / grip slipped")
            else:  # TRANSPORT
                base, bq_ = self._chassis_pose()
                over = base + _rot(bq_, np.array([
                    _robot._BUCKET_CENTER_X, 0.0,
                    _robot._CHASSIS_Z + _robot._CHASSIS[2]
                    + _robot._BUCKET_WALL_H + 0.12]))
                if self._t % 15 == 1:
                    q, _ = self._ik_to(over, self.DOWN)
                    self._q_goal[:len(q)] = q
                d = np.linalg.norm(self._tcp() - over)
                fspeed = float(np.linalg.norm(
                    self.sim.state_0.body_qd.numpy()[self.fruit_bodies[self._fi], :3]))
                if (d < 0.08 and fspeed < 0.5) or self._t > 420:
                    self._goto("DROP")

        elif st == "DROP":
            if self._t == 1:
                self.apples.release(self._fi)
                self._q_goal[-2:] = self._q_cmd[-2:] = self.FINGER_OPEN
            if self._t > 70:
                self.picked += 1
                self._fi = -1
                self._q_goal = self.arm_home.copy()
                self._q_goal[-2:] = self.FINGER_OPEN
                self._goto("SELECT")

        elif st == "DONE":
            self._q_goal = self.arm_home.copy()
            if self._t > 90 and viewer is not None:
                try:
                    viewer.close()
                except Exception:
                    pass

        self._finish(vx, vy, w)

    def _finish(self, vx, vy, w):
        ds = int(self.planar_dof[0])
        th = self._target_host
        th[ds + 0], th[ds + 1], th[ds + 3] = vx, vy, w
        self.sim.control.joint_target_qd.assign(th)
        d = np.clip(self._q_goal - self._q_cmd, -0.045, 0.045)
        self._q_cmd = self._q_cmd + d
        self._tq_host[self.arm_tq] = self._q_cmd
        self.sim.control.joint_target_q.assign(self._tq_host)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", type=Path, default=Path("output/kiwi_pick.mp4"))
    p.add_argument("--picks", type=int, default=3)
    p.add_argument("--show", action="store_true",
                   help="also open the GL window (default: headless record only)")
    args, rest = p.parse_known_args()

    import shutil
    if shutil.which("ffmpeg") is None:
        p.error("ffmpeg must be installed to record MP4")
    args.video.parent.mkdir(parents=True, exist_ok=True)

    import grow_tree as demo

    _robot.RobotDriver = ScriptedPicker   # grow_tree builds this class for --robot

    # The stock base effort limits sit right at the wheel/ground friction cone
    # on this solver (the chassis only creeps); give the scripted demo headroom.
    _orig_make_config = demo.make_config

    def make_config(a):
        cfg = _orig_make_config(a)
        cfg.robot.drive_effort = 3000.0
        cfg.robot.turn_effort = 1200.0
        return cfg
    demo.make_config = make_config

    encoder = None
    frame = 0
    orig_render = demo.Sim.render

    def record_render(sim):
        nonlocal encoder, frame
        # gentle drift around the working volume between robot and canopy
        centre = np.array([0.9, -0.1, 0.9])
        a = math.radians(35 + 6 * sim.sim_time)
        eye = centre + np.array([3.1 * math.cos(a), 3.1 * math.sin(a), 1.1])
        d = centre - eye
        try:
            sim.viewer.set_camera(
                pos=wp.vec3(*map(float, eye)),
                yaw=math.degrees(math.atan2(d[1], d[0])),
                pitch=math.degrees(math.atan2(d[2], np.linalg.norm(d[:2]))))
        except Exception:
            pass
        orig_render(sim)
        frame += 1
        if frame % 2:
            return
        pixels = sim.viewer.get_frame().numpy()
        if encoder is None:
            h, w = pixels.shape[:2]
            encoder = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo",
                 "-pixel_format", "rgb24", "-video_size", f"{w}x{h}",
                 "-framerate", "30", "-i", "pipe:0", "-an",
                 "-vf", "scale=1280:-2", "-c:v", "libx264", "-preset", "fast",
                 "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                 str(args.video)],
                stdin=subprocess.PIPE)
        encoder.stdin.write(pixels.tobytes())

    demo.Sim.render = record_render

    ScriptedPicker.MAX_PICKS = args.picks
    sys.argv = [sys.argv[0], "--preset", "pergola", "--robot", "--foliage",
                "--no-robot-camera", "--device", "cpu", "--frames", "3000",
                "--seed", "42", *rest, "--viewer", "gl"]
    if not args.show:
        sys.argv.append("--headless")

    try:
        demo.main()
    finally:
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError("ffmpeg failed to encode the recording")
    if encoder is None:
        raise RuntimeError("no frames recorded")
    print(f"[video] {args.video} ({frame // 2} frames at 30 fps)")


if __name__ == "__main__":
    main()
