# Round09 completed autonomous behavior review

`report.json` is a compact projection of the completed private report. Source A08 and B06 results come from their completed source preflight; no synthetic mixed-source rollout was constructed. `summarize.py` is the exact executed script, without edits.

Actual report command (on the research machine):

```bash
/home/CNS2025331827/docs/bc+rl_RSS_fresh_collect09_20261114/.venv_isaacsim/bin/python \
  /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/cpu_audits/round09_behavior01/summarize.py \
  --session /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06 \
  --output /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/cpu_audits/round09_behavior01/report.json
```

The default comparison helper is `/home/CNS2025331827/docs/bc+rl_RSS_continuous_vision_20261114/tools/tight_fresh_round_compare.py`; its executed SHA is retained in the report. Reproduction requires retained local artifacts and matching helper sources. Use a new `--output` path: the script refuses to overwrite prior evidence. The archive path can replace the original script path; `--comparison-helper` can explicitly name the matching helper. Candidate execution used frozen commit `87577e84f5f9183ba739e483b34729029629bb99`.

The completed analysis read 15 closed candidate numeric records for 16 opportunities (A has one rejected initial state). It reused source behavior and read no source numeric records, checkpoints, RGB or DINO payloads. Numeric content SHA was deliberately not recomputed; small-file bindings, native timing, and before/after file stat stability bound this lightweight report. Export itself read only completed JSON reports and the script.

A sustained contact-and-height holding decreased from 3/7 to 2/7 legal starts; B increased from 4/8 to 5/8. Native lift fell from 1 to 0 for A and 2 to 1 for B. Alignment, first subgoal and insertion success remain zero. Old stable-window, rotation-permitting holding and native lift are separate criteria. Equal saved visible initial state does not establish equal hidden physics/RNG state; these are repeated development cases, not an independent held-out win. Round09 is a source for collecting its actual deviations, not an overall promoted policy.
