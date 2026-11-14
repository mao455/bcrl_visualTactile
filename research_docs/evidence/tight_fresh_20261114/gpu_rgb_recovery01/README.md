# GPU RGB recording failure, first collection wave

The failed `new_family_10` execution used source revision label `d30` in `bc+rl_RSS_fresh_io_20261115`. It stopped after 60.553 seconds, including 50.595 seconds of initialization. All 16 first-wave A starts were legal and each executed one student control step (two physics substeps). The first compact RGB conversion then failed. The other 48 registered opportunities never started.

No complete trajectory, verified capture, expert action label or accepted training data was produced. These are software interruptions; the task-success denominator is zero. One lane has a single numeric row and all 16 captures remain partial. `failure.json` records exact saved field names/values and distinguishes that one persisted row from the shared batch execution evidence for the other lanes.

Evidence is in the failed root's `partial_census.json`, `owner_started.json`, `owner_finished.json`, `owner_worker_started.json`, `owner_group_cleanup.json`, `worker.log`, per-case `initial_collision.json` / `capture_error.json`, and the first lane's closed `numeric.h5`. The inspection used interactive read-only commands; no dedicated audit script was saved. This export reuses those findings without a second scan or fabricated reproduction command.

Keep the failed root intact. After the RGB fix, use an explicit new recovery identity with the original 32 families, 64 opportunities and fixed deadline (`1794725918.9665225` Unix seconds, 2026-11-15 06:58:38.966522 UTC / 14:58:38.966522 UTC+08). No successful sample can be skipped and no training can start from this partial batch. New physical starts are not exact continuation of the interrupted simulator state.

## First recovered A wave

`first_wave.json` summarizes the completed first A16 wave of `new_family_11` from frozen source `01453ae7945665606979736ac71025a3d15b4802`, under the original deadline. All 16 initializations were legal, 11 reached a real expert takeover, and 10 successful suffixes entered the training manifest: fixed80 7/8, held0 2/4, held_root60 0/2, contact0 1/2. Accepted supervision contains 2643 expert action ticks; these are expert recoveries, with zero autonomous student successes.

All 16 captures passed the existing spawned verification of 186 source fields. No first-wave capture error or recurrence of the RGB conversion failure was recorded. This is an actual collection result; the observer only read small receipts and did not repeat numeric, RGB, checkpoint or payload-hash audits. The rest of the 64-opportunity batch was still running.

Wave execution/finalization took 1052.252 seconds and sequential admission another 29.848 seconds. The resulting 1082.100-second denominator yields 33.269 accepted trajectories/hour for this wave alone, excluding startup. The full profile retains overlapping writer/verification timings; they must not be summed into elapsed time. This does not isolate GPU speed or compare causally with a different cohort.

The 10 accepted collection entries await the next aggregate and training stage. The current aggregated training pool is still 170; this report does not claim 180 trained trajectories or completion of a new B2 cycle. The largest recorded wave component is `environment_s` (545.916 seconds), followed by controller computation (197.211), record submission (130.167), and finalization/audit (113.622). Environment time includes native stepping and associated sensor/render work, not pure physics alone. Observation ended after this first-wave summary; the next wave had already started normally.
