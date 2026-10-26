"""Compare original and instrumented tasks in separate Kit processes.

Constant bounded actions are a regression probe, never demonstrations. Default
scene/control/DR are preserved; neither original task code nor physics is patched.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=["original","rendered","instrumented"],required=True)
    p.add_argument("--steps",type=int,default=32)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    from isaaclab.app import AppLauncher
    app=AppLauncher(headless=True,enable_cameras=args.mode!="original",device="cpu",
                    kit_args="--/renderer/multiGpu/enabled=false --/plugins/carb.tasking.plugin/threadCount=2 --/persistent/physics/numThreads=2").app
    import torch
    import numpy as np
    import isaacsimenvs
    from evaluation.eval_isaacsim import _load_env_cfg
    from isaacsimenvs.tasks.precise_assembly.precise_assembly_env import PreciseAssemblyEnv
    from .sim.environment import InstrumentedAssemblyEnv
    env=None
    try:
        cfg=_load_env_cfg("Isaacsimenvs-PreciseAssembly-Direct-v0")
        cfg.scene.num_envs=1; cfg.sim.device="cpu"; cfg.seed=17
        cfg.student_obs.image_modality="rgbd"; cfg.student_obs.camera_backend="standard"
        class RenderedOriginal(PreciseAssemblyEnv):
            def _get_observations(self):
                observations=super()._get_observations()
                self.sim.render()
                return observations
        cls={"original":PreciseAssemblyEnv,"rendered":RenderedOriginal,"instrumented":InstrumentedAssemblyEnv}[args.mode]
        env=cls(cfg)
        initial_obs,_=env.reset(seed=17)
        def snapshot():
            values=dict(joint=env.robot.data.joint_pos,target=env._cur_targets,successes=env._successes,
                        rng=torch.random.get_rng_state())
            for name in ("object","goal_viz","hole","table"):
                asset=getattr(env,name)
                values[name]=asset.data.root_state_w
                values[name+"_physx"]=asset.root_physx_view.get_transforms()
            return values
        initial={k:v.cpu().numpy().copy() for k,v in snapshot().items()}
        initial["policy"]=initial_obs["policy"].cpu().numpy().copy()
        np.savez(args.output/"initial.npz",**initial)
        rows={}
        action=torch.zeros(1,29)
        for _ in range(args.steps):
            obs,reward,term,trunc,_=env.step(action)
            for k,v in dict(**snapshot(),reward=reward,done=term|trunc,policy=obs["policy"]).items():
                rows.setdefault(k,[]).append(v.cpu().numpy().copy())
        np.savez(args.output/"trace.npz",**{k:np.stack(v) for k,v in rows.items()})
        (args.output/"result.json").write_text(json.dumps({"mode":args.mode,"steps":args.steps,"goal_visual_hidden":args.mode=="instrumented"}))
    except BaseException:
        import traceback
        message=traceback.format_exc(); print(message,flush=True)
        (args.output/"FAILED.txt").write_text(message)
        raise
    finally:
        if env is not None: env.close()
        app.close()


if __name__=="__main__":
    main()
