"""Orchard demo: walk between kiwis with the learned gait, harvest each with the RL arm policy,
deposit with the scripted carry, stow, and move on. Records one world for the video renderer.

Layers (all deployable except the reward oracle, which only reports):
  gait          RELIC actor, velocity commands from a simple stand-point route controller
  arm policy    privileged sequence teacher (approach, seated grasp, extraction)
  hold detector proprioceptive (jaw stall + hand travel)
  deposit       scripted joint-space carry and release, then stow
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def yaw_of(rotation):
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'checkpoint', 'gait-checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--order', type=int, nargs='*', help='Fruit indices to visit, default: nearest-first greedy')
    p.add_argument('--stand-distance', type=float, default=.55, help='Chassis-to-fruit horizontal offset at the harvest stand point (training geometry)')
    p.add_argument('--harvest-seconds', type=float, default=14.)
    p.add_argument('--settle-seconds', type=float, default=3.)
    p.add_argument('--max-walk-seconds', type=float, default=20.)
    p.add_argument('--deposit-trigger', choices=('oracle', 'proprioceptive'), default='proprioceptive')
    p.add_argument('--max-seconds', type=float, default=240.)
    p.add_argument('--fruit-roll-friction', type=float, default=None, help='Rolling friction for the kiwis (6-D contacts); stops torn fruit from rolling away')
    p.add_argument('--detach-requires-hold', action='store_true', help='The stem only yields to a secure grasp: knocked kiwis swing instead of being torn off')
    p.add_argument('--max-attempts', type=int, default=3, help='Harvest attempts per kiwi while it is still on the vine')
    p.add_argument('--freeze-legs-while-harvesting', action='store_true', help='Hold the current stance targets (no gait stepping) from stand through stow, like the fixed-base training')
    p.add_argument('--yaw-sign', type=float, default=1., help='Sign of the gait yaw-rate command that turns the robot counter-clockwise')
    p.add_argument('--vy-sign', type=float, default=1., help='Sign of the gait lateral command that moves the robot to its left')
    p.add_argument('--r84-template', type=Path, help='R84 proprioception recorded on the fixed-base training scene; leg/body entries are replayed from it while the gait stands, arm entries stay live')
    a = p.parse_args()
    import numpy as np
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        from treesim.kiwi_rl.fast_runtime import FastRuntime
        from treesim.kiwi_rl.fast_teacher import build_privileged_policy, privileged_observation
        from treesim.kiwi_rl.harvest_training import signals
        from treesim.kiwi_rl.sequence_curriculum import SequenceProgress
        from treesim.kiwi_rl.scripted_deposit import ScriptedDeposit, ProprioceptiveHoldDetector
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.ppo import load_checkpoint
        from treesim.kiwi_rl.reward_graph import SEQUENCE_PROFILE
        policy = build_privileged_policy(sequence=True).cuda()
        cfg = load_checkpoint(a.checkpoint, {'teacher': policy}, None)['meta'].get('config', {})
        policy.eval()
        gait = load_gait_artifact(a.gait_checkpoint).cuda().eval()
        rt = FastRuntime(a.scene, worlds=1, camera=None, task_profile=SEQUENCE_PROFILE,
                         arm_speed_rad_s=cfg.get('arm_speed_rad_s', .5), solver_iterations=100,
                         jaw_cap_Nm=cfg.get('jaw_cap_Nm', .6), absolute_jaw=cfg.get('absolute_jaw', True),
                         initial_jaw_rad=cfg.get('initial_jaw_rad', -1.4), jaw_rate_rad_s=cfg.get('jaw_rate_rad_s', 1.),
                         fruit_roll_friction=a.fruit_roll_friction, detach_requires_hold=a.detach_requires_hold)
        rt.prepare_settled_reset(gait)   # legs stay live: this robot walks
        rt.reset()
        fruits = rt.manifest['fruits']
        dt = rt.control_dt
        max_delta = rt.arm_speed_rad_s * dt
        stow_targets = wp.to_torch(rt._initial_targets)[12:18].clone()
        jaw_open = rt.jaw_hold_action()
        detector = ProprioceptiveHoldDetector(1, 'cuda', control_dt=dt)
        deposit = ScriptedDeposit(1, 'cuda', max_delta=max_delta)
        jaw_qid = int(rt.control.contract.qids[18])
        # The arm policy trained on a fixed base with frozen legs: its R84 input never saw
        # body velocity, gait actions or moving leg joints. Replay those entries from the
        # training template; keep the arm entries (targets 12:19, arm joint pos/vel) live.
        r84_template = torch.as_tensor(np.load(a.r84_template), device='cuda').float() if a.r84_template else None
        order = list(rt.manifest['robot']['observation_joints'])
        arm_slots = [i for i, name in enumerate(order) if name.startswith('arm_')]
        live_idx = list(range(12, 19)) + [34 + i for i in arm_slots] + [53 + i for i in arm_slots]
        live_idx = torch.tensor(live_idx, device='cuda')

        def policy_r84(obs):
            if r84_template is None:
                return obs
            mixed = r84_template.expand_as(obs).clone()
            mixed[:, live_idx] = obs[:, live_idx]
            return mixed

        def chassis_pose():
            pos = wp.to_torch(rt.data.xpos)[0, rt.chassis]
            rot = wp.to_torch(rt.data.xmat)[0, rt.chassis]
            return pos, rot

        def fruit_world(index):
            return wp.to_torch(rt.data.xpos)[0, int(rt.model.body(fruits[index]['body']).id)]

        def route_command(target_xy, target_yaw):
            """Far: turn toward the stand point and walk. Near: face the fruit and slide into place."""
            pos, rot = chassis_pose()
            yaw = yaw_of(rot)
            error = target_xy - pos[:2]
            dist = float(error.norm())
            bearing = math.atan2(float(error[1]), float(error[0]))
            near = dist < .6
            desired_yaw = target_yaw if near else bearing
            yaw_error = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))
            # Body-frame velocity toward the stand point (works sideways and backwards when near).
            ex, ey = float(error[0]), float(error[1])
            bx = math.cos(yaw) * ex + math.sin(yaw) * ey
            by = -math.sin(yaw) * ex + math.cos(yaw) * ey
            gain = 1. if near else 1.2
            vx, vy = gain * bx, gain * by
            if not near and abs(yaw_error) > .5:
                vx, vy = 0., 0.   # turn in place first
            vx = max(-.3, min(.45, vx)); vy = max(-.3, min(.3, vy))
            final_yaw_error = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
            return torch.tensor([[vx, a.vy_sign * vy, a.yaw_sign * max(-.7, min(.7, 1.8 * yaw_error))]], device='cuda'), dist, abs(final_yaw_error)

        states, events = [], []
        t = 0
        memory = torch.zeros(1, 64, device='cuda')

        hold_pose = {'xy': None, 'yaw': None}

        def hold_command():
            """Low-gain station keeping so the arm swing does not rotate or drift the standing robot."""
            if hold_pose['xy'] is None:
                return torch.zeros(1, 3, device='cuda')
            pos, rot = chassis_pose(); yaw = yaw_of(rot)
            e = hold_pose['xy'] - pos[:2]
            bx = math.cos(yaw) * float(e[0]) + math.sin(yaw) * float(e[1]); by = -math.sin(yaw) * float(e[0]) + math.cos(yaw) * float(e[1])
            ye = math.atan2(math.sin(hold_pose['yaw'] - yaw), math.cos(hold_pose['yaw'] - yaw))
            return torch.tensor([[max(-.15, min(.15, .8 * bx)), a.vy_sign * max(-.15, min(.15, .8 * by)), a.yaw_sign * max(-.4, min(.4, 1.5 * ye))]], device='cuda')

        def step(arm_actions, command):
            nonlocal t, memory
            if t % 2 == 0:
                states.append(rt.data.qpos.numpy()[0].copy())
            rt.control.set_commands(command.cpu().numpy())
            obs = rt.observe()
            rt.set_gait_actions(gait(obs))
            rt.step(arm_actions)
            t += 1
            return obs

        def slew_arm_to(targets_goal, max_seconds, jaw_action, settle_seconds=1.):
            """Drive the arm targets to a posture standing still; finish when they arrive, then settle."""
            for _ in range(round(max_seconds / dt)):
                tg = wp.to_torch(rt.control.targets)[:, 12:18]
                if float((targets_goal - tg).abs().max()) < .02:
                    break
                arm = ((targets_goal - tg) / max_delta).clamp(-1, 1)
                step(torch.cat((arm, torch.tensor([[jaw_action]], device='cuda')), -1), hold_command())
            for _ in range(round(settle_seconds / dt)):
                step(torch.cat((torch.zeros(1, 6, device='cuda'), torch.tensor([[jaw_action]], device='cuda')), -1), hold_command())

        remaining = list(range(len(fruits))) if not a.order else list(a.order)
        harvested = 0
        attempts = {}
        while remaining and t * dt < a.max_seconds:
            pos, rot = chassis_pose()
            if a.order:
                index = remaining.pop(0)
            else:
                index = min(remaining, key=lambda i: float((fruit_world(i)[:2] - pos[:2]).norm()))
                remaining.remove(index)
            fw = fruit_world(index)
            # Stand point: fruit `stand_distance` ahead of the chassis along its heading, like the training scene.
            approach = fw[:2] - pos[:2]
            heading = math.atan2(float(approach[1]), float(approach[0]))
            # Retries come in from a slightly different side and range so the same failure does not repeat.
            n = attempts.get(index, 0)
            heading += (0., .25, -.25, .5, -.5)[n % 5]
            stand = a.stand_distance + (0., -.04, .04, -.08, .08)[n % 5]
            stand_xy = fw[:2] - stand * torch.tensor([math.cos(heading), math.sin(heading)], device='cuda')
            events.append(dict(t=t * dt, event='walk', fruit=index, fruit_world=[round(float(x), 3) for x in fw], stand=[round(float(x), 3) for x in stand_xy]))
            hold_arm = torch.zeros(1, 7, device='cuda'); hold_arm[0, 6] = jaw_open
            walked = 0
            while walked * dt < a.max_walk_seconds:
                command, dist, yaw_err = route_command(stand_xy, heading)
                tg = wp.to_torch(rt.control.targets)[:, 12:18]
                arm = ((stow_targets - tg) / max_delta).clamp(-1, 1)
                step(torch.cat((arm, torch.tensor([[jaw_open]], device='cuda')), -1), command)
                walked += 1
                if dist < .05 and yaw_err < .1:
                    break
            pos, rot = chassis_pose(); hold_pose['xy'] = pos[:2].clone(); hold_pose['yaw'] = heading
            # Let the walk's sway die out with a zero command: stepping in place disturbs the grasp.
            for _ in range(round(a.settle_seconds / dt)):
                step(hold_arm, torch.zeros(1, 3, device='cuda'))
            if a.freeze_legs_while_harvesting:
                rt.freeze_legs = True   # stance targets stay where the gait left them
            # Harvest this fruit.
            rt.retarget_fruit(index)
            pos, rot = chassis_pose()
            fruit_local = rot.T @ (fruit_world(index) - pos)
            _, dist_end, yaw_end = route_command(stand_xy, heading)
            events.append(dict(t=t * dt, event='stand', fruit=index, walk_seconds=round(walked * dt, 2), stand_error_m=round(dist_end, 3),
                               yaw_error_rad=round(yaw_end, 3), chassis_height=round(float(pos[2]), 3),
                               fruit_in_chassis_frame=[round(float(x), 3) for x in fruit_local],
                               training_reference=[.55, -.005, .87]))
            progress = SequenceProgress(signals(rt), curriculum_stage=5, stall_steps=10 ** 6,
                                        max_steps=round(a.harvest_seconds / dt), control_dt=dt)
            detector.reset(torch.tensor([True], device='cuda')); deposit.reset(torch.tensor([True], device='cuda'))
            memory.zero_()
            outcome = 'timeout'
            with torch.no_grad():
                for k in range(round(a.harvest_seconds / dt)):
                    obs = rt.observe().clone()
                    pobs = policy_r84(obs)
                    mean, _, _, memory = policy(privileged_observation(rt, pobs, progress=progress), pobs, memory)
                    targets = wp.to_torch(rt.control.targets)
                    holding, extracted = detector.update(wp.to_torch(rt.data.qpos)[:, jaw_qid], targets[:, 18],
                                                         wp.to_torch(rt.data.site_xpos)[:, rt.tcp_site])
                    if a.deposit_trigger == 'proprioceptive':
                        actions = deposit.act(progress, targets[:, 12:18], mean.tanh(), engage=extracted, arm_q=progress.previous['arm_q'])
                    else:
                        actions = deposit.act(progress, targets[:, 12:18], mean.tanh())
                    # Station keeping only once the deposit swings the arm round; before that stand quietly.
                    step(actions, hold_command() if bool(deposit.active[0]) else torch.zeros(1, 3, device='cuda'))
                    now = signals(rt)
                    _, terminated, truncated, _ = progress.step(now, wp.to_torch(rt.task.failed).bool() | wp.to_torch(rt.task.success).bool())
                    if bool(progress.graph_completed['complete'][0]):
                        outcome = 'in_basket'; break
                    if bool(progress.failure[0]):
                        outcome = 'failure:' + ('physical' if bool(now['failed'][0]) else 'invalid_extract' if bool(progress.graph_invalid_extract[0])
                                                else 'ground_drop' if bool(progress.graph_ground_drop[0]) else 'unheld' if float(progress.unheld_time[0]) >= .12 else 'load_or_damage'); break
                    if bool(deposit.released[0]) and int(deposit.open_steps[0]) > 60:
                        outcome = 'released'; break
            attempts[index] = attempts.get(index, 0) + 1
            if outcome != 'in_basket' and not bool(now['detached'][0]) and attempts[index] < a.max_attempts:
                remaining.insert(0, index)   # still on the vine: walk up again and retry
                events.append(dict(t=t * dt, event='retry', fruit=index, attempt=attempts[index] + 1, after=outcome))
            harvested += outcome == 'in_basket'
            fl = rot.T @ (fruit_world(index) - chassis_pose()[0])
            events.append(dict(t=t * dt, event='harvest', fruit=index, outcome=outcome, steps=k + 1,
                               closest_m=round(float(progress.closest[0]), 3), max_load=round(float(now['max_load'][0]), 1),
                               stem_force=round(float(now['stem_force'][0]), 1), grasp_time=round(float(now['grasp_time'][0]), 2),
                               detached=bool(now['detached'][0]), bilateral=bool(now['bilateral'][0]), fruit_in_chassis_frame_end=[round(float(x), 3) for x in fl],
                               completed={k: bool(v[0]) for k, v in progress.graph_completed.items()}))
            # Stow the arm before walking on. After a deposit the hand is inside the basket:
            # retrace the collision-checked deposit path in reverse (up to the high posture,
            # back round to the handover posture) before returning to the stow pose. A direct
            # joint-space sweep from inside the basket drags the hand through the basket wall
            # and knocks the free-standing robot over.
            if bool(deposit.active[0]):
                slew_arm_to(deposit.high[None], 6., jaw_open, settle_seconds=.2)
                slew_arm_to(deposit.start_q, 10., jaw_open, settle_seconds=.2)
            slew_arm_to(stow_targets, 12., jaw_open)
            rt.freeze_legs = False
        rt.check()
        states.append(rt.data.qpos.numpy()[0].copy())
        a.output.mkdir(parents=True, exist_ok=True)
        report = dict(scene=str(a.scene), checkpoint=str(a.checkpoint), fruits=len(fruits), harvested=harvested,
                      events=events, frames=len(states), fps=25, simulated_seconds=t * dt, role='teacher',
                      freeze_legs_while_harvesting=a.freeze_legs_while_harvesting, stand_distance=a.stand_distance, fruit_roll_friction=a.fruit_roll_friction,
                      detach_requires_hold=a.detach_requires_hold, attempts=attempts,
                      reward_profile=SEQUENCE_PROFILE, controller=__doc__.strip().splitlines()[0])
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        np.savez_compressed(a.output / 'states.npz', qpos=np.array(states))
        for e in events:
            print(json.dumps(e))
        print(json.dumps(dict(fruits=len(fruits), harvested=harvested, seconds=round(t * dt, 1))))


if __name__ == '__main__':
    main()
