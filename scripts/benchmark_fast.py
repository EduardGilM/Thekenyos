"""Measure policy transitions/s with batched rigid fruit, gait and optional camera."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    import torch
    import warp as wp
    from treesim.kiwi_rl.fast_runtime import FastRuntime
    from treesim.kiwi_rl.control import load_gait_artifact
    from treesim.kiwi_rl.training_log import TrainingLog, add_training_log_args
    from train_physical_smoke import build_policy
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path,required=True)
    p.add_argument('--gait-checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--worlds',type=int,default=64)
    p.add_argument('--steps',type=int,default=100)
    p.add_argument('--camera',action='store_true')
    add_training_log_args(p)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    gait=load_gait_artifact(a.gait_checkpoint).to('cuda:0').eval()
    policy=build_policy().to('cuda:0').eval() if a.camera else None
    runtime=FastRuntime(a.scene,worlds=a.worlds,camera='hand_camera' if a.camera else None)
    actions=torch.zeros(a.worlds,7,device='cuda:0')
    memory=torch.zeros(a.worlds,64,device='cuda:0')
    failures=torch.zeros(a.worlds,device='cuda:0',dtype=torch.bool)
    log=TrainingLog(a.output,dict(worlds=a.worlds,camera=a.camera,scope='throughput benchmark, no optimizer',
        rates={'physics_hz':1/runtime.dt,'policy_hz':1/runtime.control_dt}),
        wandb_mode=a.wandb_mode,wandb_project=a.wandb_project,wandb_entity=a.wandb_entity,wandb_name=a.wandb_name)
    try:
        def tick(index):
            nonlocal memory,actions,rgbd,failures
            r84=runtime.observe()
            with torch.no_grad():
                runtime.set_gait_actions(gait(r84))
                if policy is not None:
                    if index%2==0:
                        rgbd=runtime.pixels()
                    mean,_,_,memory=policy(rgbd,r84,memory)
                    actions=mean.tanh()
                _,_,done,_=runtime.step(actions)
                failures |= done
                if bool(done.any()):
                    runtime.reset(done)
                    memory *= (~done)[:,None]
                    if policy is not None:
                        rgbd = runtime.pixels()
        rgbd=None
        for i in range(4):
            tick(i)
        runtime.reset()
        memory.zero_()
        failures.zero_()
        torch.cuda.synchronize()
        start=time.monotonic()
        for i in range(a.steps):
            tick(i)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        numerical=runtime.check()
        if not bool(failures.any()):
            import numpy as np
            np.testing.assert_allclose(runtime.data.time.numpy(), a.steps*runtime.control_dt, rtol=1e-4, atol=1e-5)
        result=dict(worlds=a.worlds,steps=a.steps,camera=a.camera,elapsed_seconds=elapsed,
            transitions_per_second=a.worlds*a.steps/elapsed,
            physics_steps_per_second=a.worlds*a.steps*runtime.substeps/elapsed,
            terminated_worlds=int(failures.sum()),numerical=numerical,
            torch_peak_allocated_gb=torch.cuda.max_memory_allocated()/1e9,wandb_url=log.url)
        log.log({k:v for k,v in result.items() if isinstance(v,(int,float,bool))},step=1)
        (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    except BaseException:
        log.finish(success=False)
        raise
    log.finish()

if __name__ == '__main__':
    import torch
    import warp as wp
    wp.init()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), wp.ScopedStream(wp.stream_from_torch(stream)):
        main()
