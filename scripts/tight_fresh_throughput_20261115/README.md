# Formal collector throughput, 2026-11-15

The old round14 trainers remain frozen and running. The unstarted round15 boundary launcher was stopped following the user's throughput priority; its original ledger remains. All new attempts below retain the original04 deadline (2026-11-16 02:05:20 +08:00).

Implemented: opt-in GPU observation/history/control batches, one shared CPU snapshot per record field, vector native expert mapping, persistent mixed A/B queue with indexed reset/refill, bounded asynchronous capture closure, disabled unassigned standard render products, cached fixed-camera extrinsics (current wrist renderer calibration remains live). Every active 60Hz observation still enters vision/history; DP planning remains 5Hz. Expert suffix student shadows are currently retained; removing those diagnostics is a separate decision after live profiling.

`prepare.py` registers distinct short attempts. Baseline `throughput_baseline15_01` uses frozen ba8ee36 and 2 lanes, two difficult training families × A11/B09, all causal early grasp takeover. The queue comparison uses the same opportunities/seeds and sensor quality. Native batch RNG ordering differs; compare actual steps and wall components, never assert hidden-state equivalence or attribute rescue changes solely to speed.

`check.py OUTPUT [cuda:0]` compares the real A11 checkpoint and stored sensor/state rows using a deterministic frozen-patch fixture, including planning ticks0/12 and lane retirement. Formal FP32 settings are mandatory. The initial CUDA fixture omitted them and failed a 2e-5rad check; after using the actual deployment precision policy, max difference is5.18e-7rad, no relaxed command tolerance. Vector bridge/independent reset and existing controller/dispatch checks:11 passed in1.85s. Live queue acceptance and throughput are still pending.

Tiled candidate now reserves3*N global RTX tiles for three views (the old setting reservedN). This is a hypothesis for the first-view-only black frames; it is not accepted until a real nonblack/current-camera run passes. No installed simulator files are modified. The formal fallback remains standard cameras.

New grasp evaluation option records300 control ticks for most cases, with two full-task opportunities per role in the fixed cohort. Additional hard/development checks use300ticks. A budget cutoff is explicitly censored and is not a native timeout or full-task failure. Raw trajectories and all prior successful data/models remain local and preserved.
