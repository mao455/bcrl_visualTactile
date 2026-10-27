"""Audit recorded trajectories and compare only identical recorded protocols.

This evaluates saved rollouts, not model predictions on expert observations.
Incomplete tails are shown explicitly and excluded from the episode denominator.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from .contracts import evaluation_contract, contract_hash


def audit_trajectory(path):
    episodes = []
    with h5py.File(path) as f:
        metadata = json.loads(f.attrs["metadata_json"])
        if int(f.attrs["schema_version"]) not in (1,2):
            raise ValueError("Unsupported trajectory schema")
        for name, g in f["episodes"].items():
            steps = g["obs/step_id"][:]
            if not np.array_equal(steps,np.arange(len(steps))):
                raise ValueError(f"{name}: incomplete/misaligned control sequence")
            for field in ("episode_id","step_id"):
                if not np.array_equal(g[f"obs/{field}"][:],g[f"transition/{field}"][:]):
                    raise ValueError(f"{name}: transition alignment failure")
            if len(np.unique(g["obs/episode_id"][:])) != 1:
                raise ValueError(f"{name}: crosses episode boundary")
            done = g["terminated"][:] | g["truncated"][:]
            if done[:-1].any() or bool(done[-1]) != bool(g.attrs["complete"]):
                raise ValueError(f"{name}: invalid terminal boundary")
            for key in ("obs/image","obs/proprio","obs/contact_substeps_w_N",
                        "transition/executed_action","transition/target_after_rad"):
                if not np.isfinite(g[key][:]).all():
                    raise ValueError(f"{name}: nonfinite {key}")
            if np.max(np.abs(g["transition/executed_action"][:])) > 1+1e-6:
                raise ValueError(f"{name}: native command out of bounds")
            if g["obs/contact_valid"][0] or not g["obs/contact_valid"][1:].all():
                raise ValueError(f"{name}: reset contact mask incorrect")
            # Current no-delay sensor contract: obs(t+1) must match pre-reset
            # terminal encoders/contact from transition(t), within this episode.
            if not (metadata["student_config"]["use_student_obs_delay"]):
                np.testing.assert_allclose(g["obs/proprio"][1:,29:58],
                    g["transition/diagnostics_after/joint_velocity_rad_s"][:-1],rtol=0,atol=0)
            np.testing.assert_allclose(g["obs/contact_substeps_w_N"][1:],
                g["transition/contact_substeps_after_w_N"][:-1],rtol=0,atol=0)
            if int(f.attrs["schema_version"]) == 2:
                np.testing.assert_array_equal(g["obs/actuator_target_rad"][1:],g["transition/target_after_rad"][:-1])
                np.testing.assert_allclose(g["obs/tracking_error_rad"][1:],
                    g["transition/target_after_rad"][:-1]-g["transition/diagnostics_after/joint_position_rad"][:-1],rtol=0,atol=0)
                previous_valid=g["obs/previous_nominal_valid"][:]
                if previous_valid[0] or g["obs/prior_plan_valid"][0].any():
                    raise ValueError(f"{name}: nominal plan history leaked across reset")
                if "nominal_predicted_chunk" in g["transition"]:
                    assert previous_valid[1:].all()
                    np.testing.assert_array_equal(g["obs/previous_nominal_action"][1:],g["transition/nominal_action"][:-1])
                    old=g["transition/nominal_predicted_chunk"][:-1]
                    start=g["transition/nominal_chunk_cursor"][:-1]+1
                    index=start[:,None]+np.arange(old.shape[1])
                    mask=index<old.shape[1]
                    expected=np.take_along_axis(old,np.minimum(index,old.shape[1]-1)[...,None],axis=1)
                    expected=np.where(mask[...,None],expected,0.)
                    np.testing.assert_array_equal(g["obs/prior_plan_valid"][1:],mask)
                    np.testing.assert_array_equal(g["obs/prior_plan"][1:],expected)
                elif previous_valid.any() or g["obs/prior_plan_valid"][:].any():
                    raise ValueError(f"{name}: unexplained nominal plan source")
            if "correction_shadow" in g["transition"]:
                if metadata["action_source"] == "nominal_bc":
                    np.testing.assert_array_equal(g["transition/executed_action"][:],g["transition/nominal_action"][:])
                shadow = g["transition/correction_shadow"]
                if shadow["previous_command_valid"][0] or not shadow["previous_command_valid"][1:].all():
                    raise ValueError(f"{name}: correction command reset mask invalid")
                np.testing.assert_array_equal(shadow["previous_issued_command"][1:],g["transition/executed_action"][:-1])
            diag = g["transition/diagnostics_after"]
            all_goals = bool(diag["successes"][-1] >= diag["max_goals"][-1])
            retract = bool(diag["retract_succeeded"][-1])
            force = np.linalg.norm(g["transition/contact_substeps_after_w_N"][:],axis=-1)
            episodes.append(dict(episode=name,steps=len(steps),complete=bool(g.attrs["complete"]),
                all_goals=all_goals,retract_succeeded=retract,success=all_goals and retract,
                action_source=str(g.attrs["action_source"]),
                max_normal_force_N_by_finger=force.max(axis=(0,1)).tolist(),
                force_samples_above_0p1N_fraction=float((force>.1).mean())))
    completed = [e for e in episodes if e["complete"]]
    successes = sum(e["success"] for e in completed)
    return dict(data=str(path),protocol_sha256=contract_hash(evaluation_contract(metadata)),
                alignment_checks_passed=True,completed_episodes=len(completed),
                incomplete_tails=len(episodes)-len(completed),successes=successes,
                success_rate=successes/len(completed) if completed else None,
                total_transitions=sum(e["steps"] for e in episodes),episodes=episodes,
                inference_source=metadata["action_source"],
                policy_provenance={k:metadata.get(k) for k in ("checkpoint","teacher_group","nominal_checkpoint","correction_mode")},
                limitation="Small-sample protocol-specific result; force is not pressure/slip; incomplete tails excluded")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    args = p.parse_args()
    runs = [audit_trajectory(path) for path in args.data]
    comparable = len({(r["protocol_sha256"],r["total_transitions"]) for r in runs}) == 1
    result = dict(runs=runs,comparable_protocol_and_budget=comparable,
                  comparison_note="Same initial seed/config/budget; policies can change later reset timing" if comparable
                  else "Do not compare rates: sensor/physics/task/seed/env-count or rollout budgets differ")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as file:
        json.dump(result,file,indent=2)
    print(json.dumps({"output":str(args.output),"comparable":comparable,
                      "results":[{k:r[k] for k in ("data","completed_episodes","successes","success_rate")} for r in runs]}))


if __name__ == "__main__":
    main()
