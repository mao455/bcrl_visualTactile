"""Rebuild the development status from completed production result receipts."""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


PROJECT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
ROOT = PROJECT / "outputs/visuotactile/run/tight_entry_phase_20261121_01/development_20261122_01"


def main():
    groups = defaultdict(list)
    episodes, waves, sources = [], [], set()
    for wave in sorted(ROOT.glob("wave[0-9][0-9]")):
        plan = json.loads((wave / "plan.json").read_text())
        initial = json.loads((wave / "paired_initial_check.json").read_text())
        results = json.loads((wave / "results.json").read_text())
        assert plan["kind"] == "autonomous" and plan["test_used"] is False
        assert plan["batch"]["vision_microbatch"] == 2
        cohort = plan["evaluation_scope"]["cohort"]
        sources.add(plan["source_commit"])
        waves.append({"wave": wave.name, "episodes": len(results), "paired_initial_passed": initial["passed"]})
        for row in results:
            result = row["result"]
            assert result["complete_student_policy"] == plan["pairs"][row["role"]]
            item = {"cohort": cohort, "role": row["role"], "seed": result["seed"],
                    "success": result["success"], "legal": result["legal"],
                    "software_complete": result["software_complete"],
                    "native_subgoals": result["native_subgoals"], "steps": result["steps"],
                    "result_path": str(wave / row["case"] / "result.json")}
            groups[(cohort, row["role"])].append(item)
            episodes.append(item)
    assert len(waves) == 16 and len(episodes) == 96
    assert all(w["paired_initial_passed"] and w["episodes"] == 6 for w in waves)
    assert all(e["legal"] and e["software_complete"] for e in episodes)
    assert len({(e["cohort"], e["role"], e["seed"]) for e in episodes}) == 96
    summaries, contrasts = [], []
    for (cohort, role), values in sorted(groups.items()):
        assert len(values) == 8
        summaries.append({"cohort": cohort, "role": role, "episodes": len(values),
                          "successes": sum(v["success"] for v in values),
                          "subgoal_ge1": sum(v["native_subgoals"] >= 1 for v in values),
                          "success_seeds": sorted(v["seed"] for v in values if v["success"])})
    for cohort in sorted({e["cohort"] for e in episodes}):
        for role in ("A", "B"):
            candidate = {v["seed"]: v["success"] for v in groups[(cohort, role + "_entry")]}
            for reference in ("R2", "130_50"):
                baseline = {v["seed"]: v["success"] for v in groups[(cohort, role + "_" + reference)]}
                assert baseline.keys() == candidate.keys()
                contrasts.append({"cohort": cohort, "role": role, "reference": reference,
                                  "paired_episodes": len(baseline), "baseline_successes": sum(baseline.values()),
                                  "retained": sorted(s for s in baseline if baseline[s] and candidate[s]),
                                  "lost": sorted(s for s in baseline if baseline[s] and not candidate[s]),
                                  "added": sorted(s for s in baseline if not baseline[s] and candidate[s])})
    report = {"observed_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "root": str(ROOT), "source_commits": sorted(sources), "complete": True,
              "episodes": len(episodes), "legal_software_complete": len(episodes),
              "paired_initial_checks_passed": len(waves), "test_used": False,
              "scope": "Full autonomous DP + Transformer development; hidden-state exact parity and recovery causality are not established.",
              "groups": summaries, "paired_contrasts": contrasts, "waves": waves,
              "owner_complete": json.loads((ROOT / "project_complete.json").read_text()),
              "owner_exit": json.loads((ROOT / "project_exit.json").read_text())}
    (AUDIT / "development_summary_20261122.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (AUDIT / "development_episodes_20261122.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(episodes[0]))
        writer.writeheader()
        writer.writerows(episodes)
    print(json.dumps({"episodes": report["episodes"], "groups": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
