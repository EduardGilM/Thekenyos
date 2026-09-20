"""Full harvest cycle: RL policy through extraction, scripted deposit afterwards.

Reports per-stage completion and physical success over many worlds, and
records one world's states in the renderer's replay format.
"""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('scene', 'checkpoint', 'gait-checkpoint', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--worlds', type=int, default=64)
    p.add_argument('--seconds', type=float, default=20.)
    p.add_argument('--stochastic', action='store_true')
    p.add_argument('--fruit-damping', type=float, default=0.)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--role', choices=('teacher', 'student'), default=None, help='Defaults to the checkpoint role')
    p.add_argument('--deposit-trigger', choices=('oracle', 'proprioceptive'), default='oracle',
                   help='What starts the scripted deposit: the simulator grasp oracle, or joint-state evidence only')
    p.add_argument('--fruit-jitter-m', type=float, nargs=3, default=None, metavar=('DX', 'DY', 'DZ'))
    p.add_argument('--fruit-reach-fraction', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'))
    p.add_argument('--reset-jitter-rad', type=float, default=None)
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
        from treesim.kiwi_rl.sequence_curriculum import SequenceProgress, SEQUENCE_SCHEMA
        from treesim.kiwi_rl.scripted_deposit import ScriptedDeposit, ProprioceptiveHoldDetector
        from treesim.kiwi_rl.control import load_gait_artifact
        from treesim.kiwi_rl.ppo import load_checkpoint
        from treesim.kiwi_rl.reward_graph import SEQUENCE_PROFILE
        import torch as _torch
        peek = _torch.load(a.checkpoint, map_location='cpu', weights_only=True)['meta']
        role = a.role or peek.get('role', 'teacher')
        if role == 'student':
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from train_physical_smoke import build_policy
            policy = build_policy().cuda()
        else:
            policy = build_privileged_policy(sequence=True).cuda()
        saved = load_checkpoint(a.checkpoint, {role: policy}, None)
        cfg = saved['meta'].get('config', {})
        policy.eval()
        gait = load_gait_artifact(a.gait_checkpoint).cuda().eval()
        rt = FastRuntime(a.scene, worlds=a.worlds, camera='hand_camera' if role == 'student' else None,
                         resolution=tuple(cfg.get('resolution', (64, 48))), task_profile=SEQUENCE_PROFILE,
                         arm_speed_rad_s=cfg.get('arm_speed_rad_s', .5), solver_iterations=100,
                         jaw_cap_Nm=cfg.get('jaw_cap_Nm', .6), absolute_jaw=cfg.get('absolute_jaw', True),
                         initial_jaw_rad=cfg.get('initial_jaw_rad', -1.4), jaw_rate_rad_s=cfg.get('jaw_rate_rad_s', 1.),
                         fruit_damping=a.fruit_damping,
                         fruit_jitter_m=tuple(a.fruit_jitter_m if a.fruit_jitter_m is not None else cfg.get('fruit_jitter_m', (0., 0., 0.))),
                         fruit_reach_fraction=tuple(a.fruit_reach_fraction if a.fruit_reach_fraction is not None else cfg.get('fruit_reach_fraction', (0., 0.))),
                         fruit_sector_deg=cfg.get('fruit_sector_deg', 70.))
        rt.prepare_settled_reset(gait)
        rt.freeze_legs = True
        rt.reset_jitter_rad = a.reset_jitter_rad if a.reset_jitter_rad is not None else cfg.get('reset_jitter_rad', .04)
        torch.manual_seed(a.seed)
        rt.reset()
        steps = round(a.seconds / rt.control_dt)
        progress = SequenceProgress(signals(rt), curriculum_stage=5, stall_steps=10 ** 6, max_steps=steps,
                                    control_dt=rt.control_dt)
        deposit = ScriptedDeposit(a.worlds, 'cuda', max_delta=rt.arm_speed_rad_s * rt.control_dt)
        detector = ProprioceptiveHoldDetector(a.worlds, 'cuda', control_dt=rt.control_dt)
        jaw_qid = int(rt.control.contract.qids[18])
        agreement = dict(both=0, oracle_only=0, proprio_only=0)
        memory = torch.zeros(a.worlds, 64, device='cuda')
        done = torch.zeros(a.worlds, dtype=torch.bool, device='cuda')
        ends = {}
        states = []
        first_done = None
        rgbd = None
        with torch.no_grad():
            for t in range(steps):
                if t % 2 == 0:
                    states.append(rt.data.qpos.numpy()[0].copy())
                obs = rt.observe().clone()
                if role == 'student':
                    if t % 2 == 0 or rgbd is None:
                        rgbd = rt.pixels().clone()
                    mean, logstd, _, memory = policy(rgbd, obs, memory)
                else:
                    mean, logstd, _, memory = policy(privileged_observation(rt, obs, progress=progress), obs, memory)
                raw = mean + logstd.exp() * torch.randn_like(mean) if a.stochastic else mean
                targets = wp.to_torch(rt.control.targets)
                holding, extracted = detector.update(wp.to_torch(rt.data.qpos)[:, jaw_qid], targets[:, 18],
                                                     wp.to_torch(rt.data.site_xpos)[:, rt.tcp_site])
                oracle = progress.graph_completed['extract'] & progress.graph_valid_extract & progress.graph_grip
                agreement['both'] += int((oracle & extracted).sum()); agreement['oracle_only'] += int((oracle & ~extracted).sum()); agreement['proprio_only'] += int((~oracle & extracted).sum())
                if a.deposit_trigger == 'proprioceptive':
                    actions = deposit.act(progress, targets[:, 12:18], raw.tanh(), engage=extracted, arm_q=progress.previous['arm_q'])
                else:
                    actions = deposit.act(progress, targets[:, 12:18], raw.tanh())
                actions[done] = 0.
                rt.set_gait_actions(gait(obs))
                _, _, physical_done, _ = rt.step(actions)
                now = signals(rt)
                _, terminated, truncated, _ = progress.step(now, physical_done)
                detector.reset(terminated | truncated)
                ending = (terminated | truncated) & ~done
                for w in ending.nonzero().flatten().tolist():
                    if progress.graph_success[w]:
                        reason = 'success'
                    elif progress.failure[w]:
                        reason = 'failure:' + ('physical' if now['failed'][w] else 'invalid_extract' if progress.graph_invalid_extract[w]
                                               else 'ground_drop' if progress.graph_ground_drop[w] else 'unheld' if progress.unheld_time[w] >= .12 else 'other')
                    else:
                        reason = 'timeout'
                    ends[reason] = ends.get(reason, 0) + 1
                if role == 'student' and bool((terminated | truncated).any()):
                    rgbd = None  # fresh frame after any reset
                done |= terminated | truncated
                if first_done is None and bool(done[0]):
                    first_done = t
                    states.append(rt.data.qpos.numpy()[0].copy())
                if bool(done.all()):
                    break
        numerical = rt.check()
        final = signals(rt)
        settle = dict(hand_contact=float(final['touching'].float().mean()), settle_time_mean=float(final['settle_time'].mean()),
                      released=float(deposit.released.float().mean()), valid_release=float(progress.graph_valid_release.float().mean()),
                      ground_drop=float(progress.graph_ground_drop.float().mean()),
                      fruit_speed_mean=float(wp.to_torch(rt.data.cvel)[:, rt.fruit_body, 3:].norm(dim=-1).mean()))
        completed = {k: float(v.float().mean()) for k, v in progress.graph_completed.items()}
        # Success is the latched event (fruit inside the basket when the episode ended); finished
        # worlds keep being simulated here, so the final-frame flag can flicker afterwards.
        report = dict(checkpoint=str(a.checkpoint), scene=str(a.scene), worlds=a.worlds, stochastic=a.stochastic, role=role,
                      fruit_jitter_m=list(rt.fruit_jitter_m), fruit_reach_fraction=list(rt.fruit_reach_fraction), reset_jitter_rad=rt.reset_jitter_rad,
                      fruit_damping=a.fruit_damping, completed=completed, success=completed['complete'],
                      success_final_frame=float(progress.graph_success.float().mean()),
                      ends=ends, numerical=numerical, settle=settle, deposit_trigger=a.deposit_trigger, trigger_agreement_steps=agreement, controller=f'{role} policy through extraction; scripted joint-space deposit afterwards (deposit trigger uses the simulator grasp oracle)', frames=len(states), fps=25, simulated_seconds=(first_done + 1) * rt.control_dt if first_done is not None else steps * rt.control_dt,
                      terminated=bool(done[0]), reward_profile=SEQUENCE_PROFILE, schema=SEQUENCE_SCHEMA,
                      world0=dict(success=bool(progress.graph_completed['complete'][0]), completed={k: bool(v[0]) for k, v in progress.graph_completed.items()}))
        a.output.mkdir(parents=True, exist_ok=True)
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        np.savez_compressed(a.output / 'states.npz', qpos=np.array(states))
        print(json.dumps({k: report[k] for k in ('completed', 'success', 'ends', 'deposit_trigger', 'trigger_agreement_steps')}), 'numerical_flagged', numerical['flagged_world_count'])


if __name__ == '__main__':
    main()
