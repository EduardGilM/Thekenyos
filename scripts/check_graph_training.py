"""Physical reset, command sensitivity and progress checks for the current weighted graph."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import warp as wp
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.harvest_training import signals, EpisodeProgress
    from treesim.kiwi_rl.reward_graph import CONTROL_GRAPH_PROFILE as GRAPH_PROFILE
    torch.set_num_threads(1); wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)), torch.no_grad():
        rt = FastRuntime(os.environ['FAST_SCENE'], worlds=15, arm_speed_rad_s=.5, task_profile=GRAPH_PROFILE)
        gait = load_gait_artifact(os.environ['GAIT_CHECKPOINT']).cuda().eval()
        rt.prepare_settled_reset(gait)
        initial = signals(rt)
        start_q = wp.to_torch(rt.data.qpos).clone()
        p = EpisodeProgress(initial, reward_profile=GRAPH_PROFILE, stall_steps=400)
        action = torch.zeros((15,7),device='cuda')
        for i in range(7): action[1+2*i,i] = 1.; action[2+2*i,i] = -1.
        samples = []
        best = p.graph_score.clone()
        for tick in range(100):
            rt.set_gait_actions(gait(rt.observe()))
            _,_,done,_ = rt.step(action)
            now = signals(rt); p.step(now,done)
            best = torch.maximum(best,p.graph_score)
            if tick in (9,49,99):
                samples.append(dict(seconds=(tick+1)*.02, zero_distance=float(now['distance'][0]),
                    scores=p.graph_score.tolist(), failed=now['failed'].tolist()))
        rt.check()
        # Reset only one world, without modifying another world's physical state.
        before=wp.to_torch(rt.data.qpos).clone()
        mask=torch.arange(15,device='cuda')==0
        rt.reset(mask)
        torch.testing.assert_close(wp.to_torch(rt.data.qpos)[0],start_q[0],rtol=0,atol=0)
        torch.testing.assert_close(wp.to_torch(rt.data.qpos)[1:],before[1:],rtol=0,atol=0)
        drift=abs(samples[0]['zero_distance']-float(initial['distance'][0]))
        gain=max(samples[1]['scores'][1:])-samples[1]['scores'][0]
        result=dict(reset_distance=float(initial['distance'][0]),zero_drift_200ms=drift,
                    best_action_gain_vs_zero_1s=gain,samples=samples,
                    scope='scripted sensitivity probe, not learned harvesting')
        print(json.dumps(result),flush=True)
        if drift>.04: raise AssertionError('Settled reset still has a large startup transient')
        if gain<.005: raise AssertionError('No command produced useful positioning credit versus zero')


if __name__ == '__main__': main()
