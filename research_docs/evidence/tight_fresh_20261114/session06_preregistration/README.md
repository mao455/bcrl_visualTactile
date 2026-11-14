# Next-family metadata preregistration

`initial_families.json` records 16 proposed training configurations (seeds
28400000–28400015) and eight development-only configurations (28500000–28500007).
Their float32 requested native-reset hashes and seeds are mutually disjoint and
do not duplicate the original registered families or prior expansion registries.
The pool is 99 training trajectories and six validation trajectories; its six
validation families and identities remain unchanged. These are configuration
checks, not physical executions or evidence of generalization. Mass/friction
diversity has not been added.

This metadata proof does not create session06 or a collection. The eventual
`session06_new16` admission registry must bind the actual new session's source
manifests and reproduce these requested reset fields. The proposed policy mix
remains A round05 / B round06; it is not altered by this metadata-only command.

Reproduce from the development tree, with a fresh output filename:

```bash
PY=/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python
RUN=/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run
"$PY" -m tools.tight_fresh_expansion_preregister \
  --source-pool "$RUN/tight_fresh_20261114_05/pool07_untrained" \
  --partition-manifest "$RUN/tight_state_20261107_01/composition_06/state_data/data.json" \
  --frozen-families "$RUN/tight_state_20261107_01/composition_06/frozen_families.json" \
  --registry-id session06_new16 \
  --train-seeds {28400000..28400015} \
  --development-seeds {28500000..28500007} \
  --output /tmp/tight_session06_initial_families_check.json
```

Only JSON registration metadata are read; no test trajectory, sensor payload,
simulation, model, or optimizer is opened. The original full split metadata are
needed solely to exclude all held-out families and requested initial states.

After a completed mixed collection, run `python -m
tools.tight_batch_collection_review --collection COLLECTION --output NEW.json`.
The review replays causal triggers from real student transitions, checks source
and registry bindings, separates fixed opportunities from causal opportunities,
and counts legal expert actions separately from accepted training data. It also
keeps native lift, two-finger contact, the old stable window, and near-hole
geometry separate. Fixed40 prefixes have no complete student-only 48-tick height
reference, so dependent prefix window counts and heights are null. Expert-window
coverage with a mixed-authority reference is descriptive, not a causal trigger.

## Execution wiring for the next cycle

Prepare the mixed follow-up session only after the old session's owned workers
and monitor have exited. Its `--source-root` is round05, `--role-source B=...`
selects round06, and `--data-source-root` is `pool07_untrained`. Both expansion
`--source` and `--session-root`, and the cycle's `--root`, must then point to the
new session. The collection is a new child of that session; the cycle's output
is a different new child. Do not use the untrained pool as the policy source.

Expansion uses the seeds above, `--registry-id session06_new16`,
`--schedule-cycle fixed40 fixed80 contact0 near60`, `--round 8`,
`--event-seed 28410000`, `--lanes 16`, and `--vision-microbatch 8`.
Omitting `--existing-identities` gives exactly 32 opportunities, one per A/B
student per registered new family. Launch the prepared collection through
`tools.tight_batch_owner --output COLLECTION --worker-tree FROZEN_TREE
--startup-seconds 300`, preserving its bounded owner and completion receipts.

The next cycle uses `--count 8 --seed 28500000 --evaluation-lanes 8
--vision-microbatch 8`. There are eight preregistered development cases per role;
eight environments avoid rendering an additional eight unused lanes. This is a
cohort-size choice, not an environment cap. Collection has sixteen cases per
role and therefore uses sixteen lanes. The same explicit accepted standard-CPU
`--backend-json` must be supplied to both expansion and cycle; it is not inherited
automatically from the collection. GPU-physics and tiled-camera probes do not
silently change this execution identity.

Cycle inputs are `--root SESSION06 --collection COLLECTION --output NEW_ROUND`,
with `--continue-rear --parallel-frontends --latest-repeat 2` and
`--reuse-index SESSION06/A/cache_index.json`. Epoch budgets remain an explicit
run decision. The original six validation families are unchanged; these eight
new development starts are separate from validation and from test. A starts
from its round05 parameters and B from round06, so differing prior training
budgets must remain visible. A/B frontend updates use the common enlarged pool;
source-dependent plans and rear features are refreshed before rear training.
