"""Verify published experiment evidence using only the Python standard library."""
import csv
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[2]


def read(path):
    return json.loads(path.read_text())


def main():
    exported = read(ROOT / "export_manifest.json")
    for entry in exported["files"]:
        path = PROJECT / entry["published"]
        assert path.stat().st_size == entry["bytes"], path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], path
    training = read(ROOT / "audit/completed_training_status_20261122.json")
    local = {item["path"]: item for item in read(ROOT / "local_artifacts_manifest.json")["files"]}
    role_checks = {}
    for role in "AB":
        prefix = f"outputs/visuotactile/run/tight_entry_phase_20261121_01/{role}/"
        assert local[prefix + "training/best.pt"]["sha256"] == training["roles"][role]["dp_selected_sha256"]
        assert local[prefix + f"backend/{role}/training/best.pt"]["sha256"] == training["roles"][role]["rear_selected"]["checkpoint_sha256"]
        receipt = read(ROOT / f"{role}_selected_checkpoint_receipt.json")
        selected = receipt["updates"]
        state = receipt["entry_exposure_state"]
        assert receipt["epoch"] == 50 and selected == state["updates"] == 6000
        assert receipt["contract_sampler_seed"] == 7091
        counts = {identity: {int(t): 0 for t in values} for identity, values in state["counts"].items()}
        absolute = defaultdict(Counter)
        path = ROOT / "run" / role / "training/learning.receipts.jsonl.gz"
        with gzip.open(path, "rt") as handle:
            for update, line in enumerate(handle, 1):
                item = json.loads(line)
                assert item["updates"] == update
                if update > selected:
                    continue
                for identity, ticks in item["entry_tick_contributions"].items():
                    for tick in set(ticks):
                        counts[identity][tick] += 1
                for identity, ticks in item["effective_tick_contributions"].items():
                    absolute[identity].update(set(ticks))
                for visit in item["visits"]:
                    identity = visit["episode"]
                    if identity in state["entries"]:
                        entry = state["entries"][identity]
                        assert entry["anchor"] in visit["anchor_sampling"]["selected_anchors"]
                        assert set(item["entry_tick_contributions"][identity]) == set(entry["ticks"])
        assert update == training["roles"][role]["dp_actual_updates"] == 7200
        for identity, expected in state["counts"].items():
            assert counts[identity] == {int(t): n for t, n in expected.items()}
        values = [n for by_tick in counts.values() for n in by_tick.values()]
        assert len(counts) == 150 and len(values) == 956 and min(values) == max(values) == 50
        assert len(absolute) == 180 and sum(map(len, absolute.values())) == 35863
        assert min(n for counter in absolute.values() for n in counter.values()) >= 50
        with (ROOT / "audit" / f"{role}_new50_entry_selected_20261122.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 334 and len({row["identity"] for row in rows}) == 50
        for row in rows:
            tick, anchor, slot = (int(row[key]) for key in ("absolute_tick", "native_anchor", "slot"))
            assert 0 <= slot < 12 and anchor + slot == tick
            assert counts[row["identity"]][tick] == int(row["optimizer_steps"]) == 50
        rear_selected = training["roles"][role]["rear_selected"]["update"]
        rear_counts = defaultdict(Counter)
        with gzip.open(ROOT / "run" / role / "backend" / role / "training/loss.receipts.jsonl.gz", "rt") as handle:
            for rear_update, line in enumerate(handle, 1):
                item = json.loads(line)
                assert item["update"] == rear_update
                if rear_update <= rear_selected:
                    for identity, ticks in item["effective_tick_contributions"].items():
                        rear_counts[identity].update(set(ticks))
        assert rear_update == training["roles"][role]["rear_actual_updates"] == 22763
        assert len(rear_counts) == 180 and sum(map(len, rear_counts.values())) == 35863
        rear_min = min(n for values in rear_counts.values() for n in values.values())
        assert rear_min == training["roles"][role]["rear_selected_exposure"]["minimum_tick_steps"]
        role_checks[role] = {"actual_updates": update, "selected_updates": selected,
                             "new_entries": 50, "new_committed_slots": len(rows), "steps_per_slot": 50,
                             "rear_actual_updates": rear_update, "rear_selected_updates": rear_selected,
                             "rear_minimum_legal_tick_steps": rear_min}
    phase = read(ROOT / "audit/independent_phase.json")
    assert phase["unique_corrections"] == 50 and phase["nonboundary_corrections"] == 48
    example = [r for r in phase["rows"] if r["identity"] == "r10_B_layout10_train_28817004_contact"]
    assert len(example) == 2
    for row in example:
        assert (row["switch"], row["first_legal"], row["native_anchor"], row["commit_slot"]) == (108, 110, 108, 2)
        assert row["target_window_mask"][:3] == [False, False, True]
    groups = defaultdict(list)
    checks, episodes = 0, 0
    for wave in sorted((ROOT / "development").glob("wave[0-9][0-9]")):
        plan = read(wave / "plan.json")
        assert read(wave / "paired_initial_check.json")["passed"]
        assert plan["source_commit"].startswith("90b0a7c") and not plan["test_used"]
        assert plan["batch"]["vision_microbatch"] == 2
        cohort = plan["evaluation_scope"]["cohort"]
        rows = read(wave / "results.json")
        assert len(rows) == 6
        for row in rows:
            result = row["result"]
            assert result["software_complete"] and result["legal"]
            assert result["complete_student_policy"] == plan["pairs"][row["role"]]
            assert not result["expert_shadow_computed"]
            groups[(cohort, row["role"])].append(result)
        checks += 1
        episodes += len(rows)
    assert checks == 16 and episodes == 96
    summary = read(ROOT / "audit/development_summary_20261122.json")
    for group in summary["groups"]:
        values = groups[(group["cohort"], group["role"])]
        assert len(values) == group["episodes"] == 8
        assert sum(v["success"] for v in values) == group["successes"]
        assert sorted(v["seed"] for v in values if v["success"]) == group["success_seeds"]
    for contrast in summary["paired_contrasts"]:
        key, role = contrast["cohort"], contrast["role"]
        old = {v["seed"]: v["success"] for v in groups[(key, role + "_" + contrast["reference"]) ]}
        new = {v["seed"]: v["success"] for v in groups[(key, role + "_entry")]}
        assert old.keys() == new.keys()
        assert sorted(s for s in old if old[s] and new[s]) == contrast["retained"]
        assert sorted(s for s in old if old[s] and not new[s]) == contrast["lost"]
        assert sorted(s for s in old if not old[s] and new[s]) == contrast["added"]
    print(json.dumps({"exported_payload_files_verified": len(exported["files"]), "entry_training": role_checks,
                      "completed_development_episodes": episodes, "paired_initial_checks": checks,
                      "test_used": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
