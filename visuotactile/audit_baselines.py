"""Bounded read-only diagnostics; reports never replace formal evaluation."""
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch
from diffusers import DDIMScheduler

from .compact import sha256
from .contracts import assert_execution_compatible
from .manifest import ManifestSequenceDataset
from .rollout import load_command_controller
from .train_dp import _condition, _move
from .training_resources import ResourceGuard, atomic_json


class DiagnosticBudget:
    """Short computation blocks, measured work/pacing and whole-card admission."""

    def __init__(self, output, device):
        self.device = torch.device(device)
        self.guard = ResourceGuard(output, poll_s=1, wait_timeout_s=180)
        self.work_seconds = self.pacing_seconds = 0.

    @contextmanager
    def block(self):
        self.guard.before_chunk()
        if shutil.disk_usage(self.guard.output).free < 12_000_000_000:
            raise RuntimeError("Diagnostic disk reserve below 12 GB")
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            duration = time.perf_counter()-started
            self.work_seconds += duration
            if self.device.type == "cuda":
                started = time.perf_counter()
                time.sleep(min(2*duration, 2.))
                self.pacing_seconds += time.perf_counter()-started
            self.guard.before_chunk()

    def summary(self):
        self.guard.sample()
        return dict(telemetry=self.guard.summary(), work_seconds=self.work_seconds,
                    pacing_seconds=self.pacing_seconds,
                    pacing="2x short compute block wall duration, capped at 2s; no long idle dilution")


def rms(values):
    values = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(values**2))) if values.size else None


def trace_statistics(data, initial, stop=None):
    commands = data["command"][:stop].astype(np.float64)
    joints = data["joint_after"][:stop].astype(np.float64)
    targets = data["target_rad"][:stop].astype(np.float64)
    delayed = data["delayed_command"][:stop].astype(np.float64)
    boundary = np.arange(1, len(commands)) % 6 == 0
    result = {}
    for name, selected in (("arm", slice(0,7)), ("hand", slice(7,29))):
        command = commands[:,selected]
        changes = np.diff(command, axis=0)
        movement = joints[:,selected]-data["joint_before"][:stop,selected]
        segments = command[:len(command)//6*6].reshape(-1,6,command.shape[-1]).mean(1)
        products = (segments[1:]*segments[:-1]).sum(-1)
        result[name] = dict(command_rms=rms(command), boundary_jump_rmse=rms(changes[boundary]),
            within_chunk_jump_rmse=rms(changes[~boundary]), delayed_jump_rmse=rms(np.diff(delayed[:,selected],axis=0)),
            opposing_segment_fraction=float(np.mean(products < 0)),
            command_net_to_path=float(np.linalg.norm(command.sum(0))/np.linalg.norm(command,axis=1).sum()),
            motion_net_to_path=float(np.linalg.norm(movement.sum(0))/max(np.linalg.norm(movement,axis=1).sum(),1e-12)),
            tracking_rmse_rad=rms(targets[:,selected]-joints[:,selected]),
            per_joint_tracking_rmse_rad=np.sqrt(((targets[:,selected]-joints[:,selected])**2).mean(0)).tolist())
    force = np.linalg.norm(data["contact_after_w_N"][:stop],axis=-1)
    heights = data["object_after"][:stop,2]-initial["object_initial_z_m"]
    result.update(contact_control_fraction=float(np.mean((force>.1).any(axis=(1,2)))),
                  contact_peak_N=float(force.max()), max_height_above_initial_m=float(heights.max()),
                  lift_above_5cm=bool((heights>.05).any()))
    return result


def audit_traces(root):
    evaluation = root/"evaluation"
    trials = json.loads((evaluation/"trials.json").read_text())["trials"]
    metadata = json.loads((evaluation/"environment.json").read_text())
    limits = np.asarray(metadata["joint_limits_rad"][0],dtype=np.float32)
    action = metadata["config"]["action"]
    rows = []
    for trial in trials:
        with np.load(evaluation/trial["trace"], allow_pickle=False) as data:
            commands = data["command"]
            previous = np.concatenate((np.asarray(trial["initial"]["actuator_target_rad"],dtype=np.float32)[None],
                                       data["target_rad"][:-1]))
            raw = limits[:,0]+.5*(data["delayed_command"]+1)*(limits[:,1]-limits[:,0])
            raw[:,:7] = previous[:,:7]+action["dof_speed_scale"]*metadata["control_dt"]*data["delayed_command"][:,:7]
            raw = np.clip(raw, limits[:,0],limits[:,1])
            mix = np.array([action["arm_moving_average"]]*7+[action["hand_moving_average"]]*22)
            expected = np.clip(mix*raw+(1-mix)*previous,limits[:,0],limits[:,1])
            rows.append(dict(variant=trial["variant"], seed=trial["seed"], steps=len(commands),
                success=trial["success"], reasons=trial["reasons"], first60=trace_statistics(data,trial["initial"],60),
                full=trace_statistics(data,trial["initial"]),
                command_submission_max_error=float(np.abs(commands-data["submitted_action"]).max()),
                native_target_formula_max_error=float(np.abs(expected-data["target_rad"]).max()),
                clock_max_error_s=float(np.abs(np.diff(data["time_before_s"])-metadata["control_dt"]).max()),
                max_successes=int(data["successes"].max()),trace_sha256=sha256(evaluation/trial["trace"])))
    aggregate = {}
    for variant in ("b0","b1"):
        selected = [row for row in rows if row["variant"]==variant]
        aggregate[variant] = dict(trials=len(selected), successes=sum(row["success"] for row in selected),
            native_timeouts=sum(row["reasons"]==["timeout"] for row in selected),
            contacted=sum(row["full"]["contact_peak_N"]>.1 for row in selected),
            lifted_5cm=sum(row["full"]["lift_above_5cm"] for row in selected),
            target_formula_max_error=max(row["native_target_formula_max_error"] for row in selected))
        for period in ("first60","full"):
            aggregate[variant][period] = {}
            for branch in ("arm","hand"):
                aggregate[variant][period][branch] = {
                    key:np.mean([row[period][branch][key] for row in selected],axis=0).tolist()
                    for key in selected[0][period][branch]}
    return dict(protocol="read existing formal traces; diagnostic contact has no partner identity",
                aggregate=aggregate, trials=rows)


def audit_development(evaluation):
    evaluation = Path(evaluation)
    report = json.loads((evaluation/"report.json").read_text())
    if not report["development"] or not report["complete"]:
        raise ValueError("Expected a completed development evaluation")
    trials = json.loads((evaluation/"trials.json").read_text())["trials"]
    rows = []
    for trial in trials:
        with np.load(evaluation/trial["trace"], allow_pickle=False) as data:
            distance = data["mean_finger_link_object_distance_m"]
            row = dict(variant=trial["variant"], seed=trial["seed"], steps=trial["steps"],
                success=trial["success"], reasons=trial["reasons"],
                pair_initial_max_error=trial["pair_initial_max_error"],
                first60=trace_statistics(data,trial["initial"],60),
                full=trace_statistics(data,trial["initial"]),
                native_lift_seen=bool(data["lifted_object"].any()),
                max_subgoals=int(data["successes"].max()),
                finger_link_object_distance_m=dict(first=float(distance[0]),
                    first60_min=float(distance[:60].min()),minimum=float(distance.min()),last=float(distance[-1])))
            if "predicted_chunk" in data:
                cursors = data["chunk_cursor"].astype(int)
                replans = np.flatnonzero(cursors == 0)[1:]
                overlap = data["predicted_chunk"][replans-1,6:]-data["predicted_chunk"][replans,:6]
                selected = data["predicted_chunk"][np.arange(len(cursors)),cursors]
                row.update(plan_overlap_arm_rmse=rms(overlap[:,:,:7]),
                    plan_overlap_hand_rmse=rms(overlap[:,:,7:]),
                    command_plan_max_error=float(np.abs(selected-data["command"]).max()),
                    generation_alignment_exact=bool(np.array_equal(data["plan_generation_step"],
                                                                   np.arange(len(cursors))-cursors)))
            if "action_history_valid" in data:
                row["history"] = dict(reset_invalid=not bool(data["action_history_valid"][0]),
                    subsequent_valid=bool(data["action_history_valid"][1:].all()),
                    issued_max_error=float(np.abs(data["previous_issued_command"][1:]-
                                                  data["command_clipped_pre_delay"][:-1]).max()),
                    delayed_max_error=float(np.abs(data["previous_delayed_command"][1:]-
                                                   data["delayed_command"][:-1]).max()),
                    issued_time_max_error_s=float(np.abs(data["previous_command_time_s"][1:]-
                                                        data["time_before_s"][:-1]).max()))
            rows.append(row)
    costs = [trial["timing"] for trial in trials]
    return dict(protocol="bounded development; never added to original formal trials", trials=rows,
        resources=report["resources"], process_wall_s=report["process_wall_s"],
        policy_and_env_work_s=sum(item["policy_s"]+item["env_step_s"] for item in costs),
        pacing_s=sum(item["pacing_s"] for item in costs),
        resource_gate_wall_s=sum(item["resource_gate_s"] for item in costs))


@torch.no_grad()
def sample_frozen(model, condition, initial_noise, inference_steps):
    scheduler = DDIMScheduler.from_config(model.sampler.config)
    scheduler.set_timesteps(inference_steps, device=condition.device)
    sample = initial_noise.clone()
    for timestep in scheduler.timesteps:
        prediction = model.denoiser(sample,timestep,global_cond=condition)
        sample = scheduler.step(prediction,timestep,sample,eta=0.).prev_sample
    return sample.clamp(-1,1)


def command_errors(prediction, target, valid):
    squared = (prediction-target).double().square()
    count = valid.sum().clamp_min(1)
    return dict(arm_rmse=float(((squared[:,:,:7]*valid[...,None]).sum()/(count*7)).sqrt()),
                hand_rmse=float(((squared[:,:,7:]*valid[...,None]).sum()/(count*22)).sqrt()),
                executed_first6_rmse=float(((squared[:,:6]*valid[:,:6,None]).sum()/(valid[:,:6].sum()*29)).sqrt()),
                first6_arm_rmse=float(((squared[:,:6,:7]*valid[:,:6,None]).sum()/(valid[:,:6].sum()*7)).sqrt()),
                per_arm_joint_rmse=((squared[:,:,:7]*valid[...,None]).sum((0,1))/count).sqrt().cpu().tolist(),
                horizon_rmse=((squared*valid[...,None]).sum((0,2))/(valid.sum(0).clamp_min(1)*29)).sqrt().cpu().tolist(),
                arm_sign_disagreement=float((((prediction[:,:,:7]*target[:,:,:7])<0)&valid[...,None]).sum()/(count*7)))


@torch.no_grad()
def audit_offline(root, manifest, output, device, budget):
    results = {}
    for variant in ("b0","b1"):
        path = root/variant/"best.pt"
        payload = torch.load(path,map_location="cpu",weights_only=False)
        with budget.block():
            controller = load_command_controller(path,1,device=device)
        model = controller.model
        scheduler = DDIMScheduler.from_config(model.sampler.config)
        scheduler.set_timesteps(8)
        result = dict(checkpoint_sha256=sha256(path), config=payload["config"],
            scheduler=dict(model.sampler.config), timesteps8=scheduler.timesteps.tolist(),
            alpha8=scheduler.alphas_cumprod[scheduler.timesteps].tolist(),
            alpha99=float(scheduler.alphas_cumprod[99]), episodes=[])
        for split in ("train","test"):
            identities = payload[split+"_episodes"][:2]
            dataset = ManifestSequenceDataset(manifest,ids=identities,stage="production",include_tactile=variant=="b1")
            for identity in identities:
                state = None
                conditions, targets, validity, steps = [], [], [], []
                for chunk in dataset.chunks(identity,8):
                    with budget.block():
                        data = _move(chunk,torch.device(device))
                        condition,state = _condition(model,variant,data,state)
                        selected = torch.arange(chunk["start"],chunk["stop"],device=device)%6==0
                        conditions.append(condition[0,selected].detach())
                        targets.append(data["action_target"][selected])
                        validity.append(data["action_valid"][selected])
                        steps.extend(np.arange(chunk["start"],chunk["stop"])[selected.cpu().numpy()].tolist())
                    del data,condition
                condition,target,valid = torch.cat(conditions),torch.cat(targets),torch.cat(validity)
                generator = torch.Generator(device=device).manual_seed(20261030)
                noise = torch.randn(target.shape,device=device,generator=generator)
                episode = dict(id=identity,split=split,observations=chunk['stop'],replans=len(steps),samplers={})
                predictions = {}
                for count in (8,20,50,100):
                    parts=[]
                    for start in range(0,len(condition),4):
                        with budget.block():
                            parts.append(sample_frozen(model,condition[start:start+4],noise[start:start+4],count))
                    prediction=torch.cat(parts)
                    predictions[count]=prediction
                    early=torch.tensor(steps,device=device)<60
                    episode["samplers"][str(count)]=dict(all=command_errors(prediction,target,valid),
                        first60=command_errors(prediction[early],target[early],valid[early]),
                        overlap_rmse=float((prediction[:-1,6:]-prediction[1:,:6]).square().mean().sqrt()))
                repeats=[]
                for noise_seed in range(4):
                    generator.manual_seed(noise_seed+314)
                    with budget.block():
                        repeats.append(sample_frozen(model,condition[:1],torch.randn(noise[:1].shape,device=device,generator=generator),8))
                repeated=torch.cat(repeats)
                episode["same_condition_noise_variation"] = dict(
                    first6_arm_std=float(repeated[:,:6,:7].std(0).mean()),
                    first6_hand_std=float(repeated[:,:6,7:].std(0).mean()))
                result["episodes"].append(episode)
                atomic_json(output/(variant+"_offline.json"),result)
                print(variant,split,identity,{key:round(value["all"]["arm_rmse"],4) for key,value in episode["samplers"].items()},flush=True)
        results[variant]=result
        del model,controller,payload,dataset
        if torch.device(device).type=="cuda":
            torch.cuda.empty_cache()
    return results


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--offline",action="store_true")
    parser.add_argument("--device",default="cuda:0")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    budget=DiagnosticBudget(args.output/"resources",args.device)
    plan=dict(source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
              diagnostic_source_sha256=sha256(__file__),
              versions={name:importlib.metadata.version(name) for name in ("torch","torchvision","diffusers","numpy","h5py")},
              protocol="frozen weights; complete causal prefixes; identical initial noise across DDIM step counts; no updates")
    atomic_json(args.output/"plan.json",plan)
    try:
        traces=audit_traces(args.root)
        atomic_json(args.output/"traces.json",traces)
        print(json.dumps(traces["aggregate"],indent=2),flush=True)
        environment=json.loads((args.root/"evaluation/environment.json").read_text())
        for variant in ("b0","b1"):
            saved=torch.load(args.root/variant/"best.pt",map_location="cpu",weights_only=False)
            assert_execution_compatible(saved["metadata"],environment)
        if args.offline:
            audit_offline(args.root,args.manifest,args.output,args.device,budget)
    finally:
        atomic_json(args.output/"budget.json",budget.summary())


if __name__=="__main__":
    main()
