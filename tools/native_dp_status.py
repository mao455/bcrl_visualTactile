#!/usr/bin/env python3
"""Read-only progress summary for a native three-way DP run."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def rate(rows, count=100):
    usable = [row for row in rows if "updates" in row and "elapsed_s" in row]
    if len(usable) < 2:
        return None
    usable = usable[-count:]
    delta_updates = usable[-1]["updates"] - usable[0]["updates"]
    delta_seconds = usable[-1]["elapsed_s"] - usable[0]["elapsed_s"]
    if delta_updates <= 0 or delta_seconds <= 0:
        return None
    return delta_updates / delta_seconds


def checkpoint_info(path: Path):
    if not path.exists():
        return None
    info = {"path": str(path), "mtime_epoch": path.stat().st_mtime, "bytes": path.stat().st_size}
    try:
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        for key in ("updates", "epoch", "cursor", "ema_step"):
            if key in payload:
                info[key] = payload[key]
    except Exception as error:
        info["read_error"] = repr(error)
    return info


def summarize(root: Path):
    status = read_json(root / "status.json", {}) or {}
    manifest = read_json(root / "manifest.json", {}) or {}
    session = read_json(root / "session.json", {}) or {}
    stage_name = status.get("stage") or manifest.get("current_stage")
    stage = (manifest.get("stages") or {}).get(stage_name, {}) if stage_name else {}
    output = Path(status["stage_output"]) if status.get("stage_output") else Path(stage.get("output", ""))
    rows = read_jsonl(output / "train.jsonl") if output else []
    last = rows[-1] if rows else {}
    current_updates = last.get("updates", status.get("updates"))
    target_updates = last.get("total_updates")
    if target_updates is None:
        configuration = read_json(output / "configuration.json", {}) or {}
        target_updates = configuration.get("total_updates")
    updates_per_second = rate(rows)
    remaining_seconds = None
    if updates_per_second and target_updates is not None and current_updates is not None:
        remaining_seconds = max(0, target_updates - current_updates) / updates_per_second
    resource = read_json(root / "resources" / "resources.json", {}) or {}
    work_rows = read_jsonl(root / "work.jsonl")
    pid = status.get("pid") or stage.get("pid")
    worker_rows = [row for row in work_rows if pid is not None and row.get("pid") == pid]
    work = worker_rows[-1] if worker_rows else (work_rows[-1] if work_rows else {})
    now = time.time()
    training_deadline = session.get("training_deadline")
    total_deadline = session.get("deadline")
    result = {
        "state": status.get("state", manifest.get("state")),
        "stage": stage_name,
        "stage_output": str(output) if output else None,
        "worker_pid": pid,
        "formal_training_started": bool(stage_name and stage_name.endswith("_formal")) or any(
            name.endswith("_formal") and value.get("state") in {"starting", "running", "complete"}
            for name, value in (manifest.get("stages") or {}).items()
        ),
        "updates": current_updates,
        "target_updates": target_updates,
        "updates_per_second_last_100": updates_per_second,
        "estimated_stage_seconds_remaining": remaining_seconds,
        "last_update_elapsed_s": last.get("elapsed_s"),
        "last_update_mtime_epoch": (output / "train.jsonl").stat().st_mtime if (output / "train.jsonl").exists() else None,
        "checkpoint": checkpoint_info(output / "last.pt") if output else None,
        "work": {key: work.get(key) for key in ("total_work_s", "total_wait_s", "total_pacing_s", "sleep_work_ratio")},
        "resource": {key: resource.get(key) for key in (
            "gpu_time_weighted_mean_percent", "max_complete_60s_mean_percent", "missing_samples",
            "missing_seconds", "latest", "thresholds")},
        "training_deadline_seconds_remaining": max(0, training_deadline - now) if training_deadline else None,
        "total_deadline_seconds_remaining": max(0, total_deadline - now) if total_deadline else None,
        "stages": {name: value.get("state") for name, value in (manifest.get("stages") or {}).items()},
    }
    if last:
        result["loss"] = last.get("loss")
        result["learning_rate"] = last.get("lr")
    if result["last_update_mtime_epoch"]:
        result["seconds_since_last_update"] = max(0, now - result["last_update_mtime_epoch"])
        result["stall_suspected"] = result["seconds_since_last_update"] > max(300, 5 * (1 / updates_per_second) if updates_per_second else 300)
    else:
        result["seconds_since_last_update"] = None
        result["stall_suspected"] = False
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()
    result = summarize(args.root)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print(f"state={result['state']} stage={result['stage']} pid={result['worker_pid']}")
    print(f"updates={result['updates']}/{result['target_updates']} rate={result['updates_per_second_last_100']}")
    print(f"eta_stage_s={result['estimated_stage_seconds_remaining']} formal_started={result['formal_training_started']}")
    print(f"work_s={result['work']['total_work_s']} wait_s={result['work']['total_wait_s']} pacing_s={result['work']['total_pacing_s']}")
    resource = result["resource"]
    print(f"gpu_mean={resource['gpu_time_weighted_mean_percent']} max60s={resource['max_complete_60s_mean_percent']} missing={resource['missing_samples']}")
    print(f"stall_suspected={result['stall_suspected']} training_deadline_s={result['training_deadline_seconds_remaining']} total_deadline_s={result['total_deadline_seconds_remaining']}")


if __name__ == "__main__":
    main()
