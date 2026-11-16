# 2026-11-16 diagnosis, no DAgger owner

Authoritative report: `research_docs/TIGHT_TRAINING_DIAGNOSIS_20261116.md`.
Run from the continuous_vision development tree with `.venv_isaacsim/bin/python`.

The actual sequence was cleanup plan → reviewed classified execution → stored-log
extraction → bounded numeric/expert and DP endpoint inference → I/O probes → plots.
`cleanup.py plan/execute` is one-shot and has already run. Do not rerun retirement.
Its complete local manifest is `outputs/visuotactile/run/tight_diagnostic_20261116_01/cleanup`.
All selected model/EMA tensors survive; historical optimizer resumption does not.

Read-only figure reproduction:

```bash
.venv_isaacsim/bin/python scripts/tight_diagnosis_20261116/extract.py
.venv_isaacsim/bin/python scripts/tight_diagnosis_20261116/stages_and_traces.py
.venv_isaacsim/bin/python scripts/tight_diagnosis_20261116/summary_figures.py
```

`endpoint_metrics.py` already completed36 fixed-input inferences in under10s.
It refuses an existing result; preserve that evidence instead of rerunning.
Recovery-onset legal masks and the original32-anchor inference batch are explicit.
An earlier incomplete diagnostic omitted the onset-window attachment; it was
stopped before public metrics. Its receipt is retained. A preliminary endpoint
height bin was corrected to the exact all-height/90%-contact holding definition;
previous derived numbers are archived locally, not mixed into final figures.

The explicit CPU expert diagnostic actually run was:

```bash
.venv_isaacsim/bin/python -m tools.tight_fresh_expert_counterfactual \
 --root outputs/visuotactile/run/tight_fresh_20261116_04/round_19 \
 --validation-numeric outputs/visuotactile/run/tight_fresh_20261115_09/new_family_15/r15_A_expanded_session20261115_09_early15_s29000120_grasp_early/numeric.h5 \
 --evaluation outputs/visuotactile/run/tight_fresh_20261116_04/round_19/development \
 --decompose-student \
 --output research_docs/evidence/tight_diagnosis_20261116/round19_command_chain.json
```

`io_probe.py --output <new.json> [--cold]` executes four disposable realB2 updates,
no checkpoint write or candidate adoption. `--cold` advises eviction ONLY of the
two closed train feature files; no global cache drop. This was run warm and cold.
The earlier bitwise-GPU-update demand failed; the ABBA probe records actual
nondeterministic variation even between identical eager runs. Formal loss/RNG
agree, numeric deltas remain disclosed. Do not read incomplete `io_probe.json`
as a passing experiment. Final acceptance:27 focused tests, real cold/warm probes.

`mapped_visual_io.json` is an opt-in profile accepted by existing formal cycle,
DP and rear input preparation. No old run config was changed, and no new long
cycle is registered. Dataset, optimizer ordering, fresh history, inference rate,
120/60/5Hz and execution physics remain unchanged. Full-cycle speed benefit is
not yet established.
