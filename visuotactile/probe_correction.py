"""Replay real contact episodes through causal correction inputs on CPU.

This is a shape/history/signal probe, NOT training or a simulation success test.
Only frozen nominal predictions supply intent. Recorded commands are committed
after each observation as the actually issued past commands, never future inputs.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from .rollout import NominalController
from .correction import CorrectionShadow
from .contracts import assert_policy_compatible


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--episodes",type=int,default=2)
    args = p.parse_args()
    if args.episodes < 1:
        p.error("episodes must be positive")
    torch.set_num_threads(2)
    nominal = NominalController.load(args.checkpoint,num_envs=1)
    shadow = CorrectionShadow(nominal,1)
    results = []
    with h5py.File(args.data) as f, torch.no_grad():
        assert_policy_compatible(nominal.metadata,json.loads(f.attrs["metadata_json"]))
        for name, group in f["episodes"].items():
            if not group.attrs["complete"]:
                continue
            force = group["obs/contact_substeps_w_N"][:]
            if not (np.linalg.norm(force,axis=-1)>.1).any():
                continue
            arrays = {k:group[f"obs/{k}"][:] for k in ("image","proprio","contact_substeps_w_N",
                      "contact_w_N","contact_valid","step_id")}
            commands = group["transition/executed_action"][:]
            contact_frames, maximum = 0, 0.
            for t in range(len(commands)):
                packet = {k:torch.as_tensor(v[t:t+1]) for k,v in arrays.items()}
                # Independent episode IDs for sequential offline replay, not task inputs.
                packet["episode_id"] = torch.tensor([len(results)])
                nominal.act(packet)
                log = shadow.observe(packet,nominal)
                contact_frames += int(log["contact_mask"].sum())
                maximum = max(maximum,float(log["untrained_candidate_diagnostic"].abs().max()))
                if t==0 and log["previous_command_valid"].any():
                    raise AssertionError("Command leaked across episode boundary")
                if t>0:
                    torch.testing.assert_close(log["previous_issued_command"][0],torch.as_tensor(commands[t-1]),rtol=0,atol=0)
                shadow.context.commit(torch.as_tensor(commands[t:t+1]))
            results.append(dict(episode=name,steps=len(commands),contact_frames=contact_frames,
                peak_normal_force_N=float(np.linalg.norm(force,axis=-1).max()),
                max_abs_untrained_candidate=maximum))
            if len(results)>=args.episodes:
                break
    if not results:
        raise ValueError("No complete real-contact episodes available")
    report = dict(kind="untrained CPU replay probe; no actions executed, no labels generated",
                  nominal_checkpoint=str(args.checkpoint),data=str(args.data),episodes=results,
                  finite_and_causal_checks_passed=True,trained_correction=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as file:
        json.dump(report,file,indent=2)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
