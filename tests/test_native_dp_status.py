import json
from pathlib import Path

from tools.native_dp_status import summarize


def test_status_summary_reports_rate_eta_and_work(tmp_path: Path):
    root = tmp_path / "run"
    output = root / "stages" / "S-NativeDP_formal" / "attempt_1"
    output.mkdir(parents=True)
    (root / "resources").mkdir()
    (root / "session.json").write_text(json.dumps({"training_deadline": 2000, "deadline": 3000}))
    (root / "status.json").write_text(json.dumps({
        "state": "running", "stage": "S-NativeDP_formal", "pid": 123,
        "stage_output": str(output),
    }))
    (root / "manifest.json").write_text(json.dumps({
        "state": "running", "stages": {"S-NativeDP_formal": {"state": "starting", "pid": 123, "output": str(output)}}
    }))
    (output / "configuration.json").write_text(json.dumps({"total_updates": 1000}))
    (output / "train.jsonl").write_text("\n".join(json.dumps({
        "updates": update, "total_updates": 1000, "elapsed_s": float(update),
    }) for update in range(1, 101)))
    (root / "work.jsonl").write_text(json.dumps({
        "pid": 123, "total_work_s": 80, "total_wait_s": 10, "total_pacing_s": 5,
    }) + "\n")
    (root / "resources" / "resources.json").write_text(json.dumps({
        "gpu_time_weighted_mean_percent": 32, "max_complete_60s_mean_percent": 46,
        "missing_samples": 0, "missing_seconds": 0, "latest": {}, "thresholds": {},
    }))

    result = summarize(root)

    assert result["formal_training_started"] is True
    assert result["updates"] == 100
    assert result["updates_per_second_last_100"] == 1
    assert result["estimated_stage_seconds_remaining"] == 900
    assert result["work"]["total_pacing_s"] == 5
    assert result["resource"]["max_complete_60s_mean_percent"] == 46
