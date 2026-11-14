# GPU RGB recording failure, first collection wave

The failed `new_family_10` execution used source revision label `d30` in `bc+rl_RSS_fresh_io_20261115`. It stopped after 60.553 seconds, including 50.595 seconds of initialization. All 16 first-wave A starts were legal and each executed one student control step (two physics substeps). The first compact RGB conversion then failed. The other 48 registered opportunities never started.

No complete trajectory, verified capture, expert action label or accepted training data was produced. These are software interruptions; the task-success denominator is zero. One lane has a single numeric row and all 16 captures remain partial. `failure.json` records exact saved field names/values and distinguishes that one persisted row from the shared batch execution evidence for the other lanes.

Evidence is in the failed root's `partial_census.json`, `owner_started.json`, `owner_finished.json`, `owner_worker_started.json`, `owner_group_cleanup.json`, `worker.log`, per-case `initial_collision.json` / `capture_error.json`, and the first lane's closed `numeric.h5`. The inspection used interactive read-only commands; no dedicated audit script was saved. This export reuses those findings without a second scan or fabricated reproduction command.

Keep the failed root intact. After the RGB fix, use an explicit new recovery identity with the original 32 families, 64 opportunities and fixed deadline (`1794725918.9665225` Unix seconds, 2026-11-15 06:58:38.966522 UTC / 14:58:38.966522 UTC+08). No successful sample can be skipped and no training can start from this partial batch. New physical starts are not exact continuation of the interrupted simulator state.
