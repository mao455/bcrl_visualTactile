# Trajectory and policy contracts — schema 1

HDF5 root attributes: schema_version, metadata_json (full resolved task config, seed, source checkpoint, camera actual intrinsics/pose, contact paths/order, joint order, control/physics dt). Groups `episodes/envNNN_epNNNNNN` never cross a full reset. A group has complete, length, action_source, terminal_image_available=false. Incomplete captures are retained and excluded from ordinary training.

Each dataset has leading length T. Row t contains **pre-action observation**, action executed during [t,t+1], and **post-action/pre-auto-reset** diagnostics. Returned done-environment images belong to a new episode; they are never used as terminal images. Terminal joint state and contact are available. End-of-capture truncation is the group's complete=false; simulator timeout is the row's truncated flag.

| Field (per step) | Shape | Source / use |
| --- | --- | --- |
| obs/image | V,4,90,160 | Existing camera reader: RGB [0,1], window-normalized noisy depth [0,1]; nominal and correction |
| obs/proprio | 58 | Original canonical normalized encoder position 29 + encoder velocity rad/s 29; both policies |
| obs/full_proprio_diagnostic | 87 | Original student tensor, including previous target radians; diagnostic only |
| obs/contact_w_N | 5,3 | Five PhysX ContactSensors, aggregate normal world-axis force; correction only |
| obs/contact_substeps_w_N | 2,5,3 | Chronological 120-Hz samples before action; correction history/diagnostic |
| obs/contact_valid | scalar bool | False immediately after reset (no interval measured); true after native control step |
| obs/episode_id, step_id | scalar int64 | Independent full-reset and monotonic control counters, not task episode_length_buf |
| obs/time_s | scalar | Global simulated control clock; not wall clock |
| transition/raw_action | 29 | Caller output before adapter clipping; released player may already clip its network mean |
| transition/executed_action | 29 | Canonical native command after [-1,1] clip, BEFORE original action delay/smoothing |
| transition/target_before_rad, target_after_rad | 29 | Canonical actuator position targets immediately before/after original action pipeline |
| transition/contact_substeps_after_w_N | 2,5,3 | Same interval as action, saved before auto-reset |
| transition/diagnostics_after/joint_position_rad, joint_velocity_rad_s | 29 | True terminal encoder proxies, not reset state |
| transition/diagnostics_after/object_pose_w, goal_pose_w, hole_pose_w, table_pose_w | 7 | Simulator position + wxyz quaternion; privileged diagnostics/labels only; hole/table added in teacher03 and later |
| transition/diagnostics_after/successes, max_goals, retract_phase, retract_succeeded | scalar | Original state machine, diagnostic/evaluation only |
| transition/reward | scalar | Original reward, never inference input |
| terminated, truncated | scalar bool | Original native env.step flags kept distinct |
| transition/nominal_action, nominal_predicted_chunk, nominal_chunk_cursor | 29 / H,29 / scalar | Nominal rollout only; predicted plan and executed cursor, not expert future |
| transition/correction_shadow/untrained_candidate_diagnostic | 29 | Untrained forward output for integration checks; never submitted to env or used as training target |
| transition/correction_shadow/nominal_remaining_chunk, nominal_remaining_valid | H,29 / H bool | Causal model plan from current cursor; repeated end padding masked |
| transition/correction_shadow/previous_issued_command, previous_command_valid | 29 / bool | Actual command t−1, not current expert label; invalid at reset |
| transition/correction_shadow/contact_mask | bool | Valid observed distal force above exploratory threshold; not pressure/slip/safety |

The initial view count is 1. Native physical samples are saved for every control step, not only every network replan. Original action delay is preserved (default sampled lag 0..2 control ticks). The applied lag is not instrumented; before/after physical targets are authoritative and must not be reverse-labelled as an exact unobserved delayed command.

## Histories and splits

`history_indices(t,L,stride)` returns oldest→current indices. Left padding repeats the episode's first sample and supplies a false validity mask. `OnlineHistory` maintains a separate deque per environment and resets only when independent episode ID changes; duplicate/missing control observations raise an error. Subgoal changes do not clear history. The same index/mask convention is used by HDF5 datasets. Nominal GRU state is recomputed within each window with masked updates; it never persists across episode boundaries.

Training outputs are `obs={image,proprio,valid}`, `action_target=(H,29)`, `action_valid=(H,)`. Future native action targets are supervision only. End padding repeats the last action and is excluded from the epsilon loss. Normalization uses training episodes only; validation is whole-episode held out within one capture seed (not cross-seed generalization). Default dataset accepts completed all-goal AND retract-success teacher episodes; `--smoke-only` explicitly allows unsuccessful/incomplete real trajectories for software tests. This criterion targets insertion with retract enabled and random_goal_fraction=0, not all original task variants. It never generates labels.

New metadata adds actual camera positions relative to environment origins, origins themselves, correction mode and contact threshold. Input compatibility checks problem, mounting, actual orientation/intrinsics, joint order, physics/control dt, modality/preprocessing and sensor settings. Legacy schema-1 oblique captures lack explicit origins; their fixed declared mount (.6,−.8,1.1) is used with recorded orientation/K, based on the audited collector code. Simulator object/goal pose never supplies camera calibration.

Correction receives 12 consecutive 60-Hz motion/previous-command samples and 24 chronological force samples at 120 Hz, plus its own visual window. Nominal histories remain image/proprio only. Predicted chunks are logged to distinguish inference inputs from expert supervision. There is no correction training dataset/loss yet.

## Limitations and provenance gaps

- Current adapter explicitly rejects student camera/bundle delay, camera pose randomization and depth augmentation until timestamps/RNG/mount contracts are validated. They are not silently changed. Original teacher/action/object delays and disturbances stay on.
- Original camera read performs an extra camera.update; camera's internal timestamp is not a trustworthy physical exposure clock here. Store the control availability time and config. Camera skip interval other than zero is rejected by the adapter because the original RGB-D cache returns only depth on skipped reads.
- The saved RGB-D is the policy input; raw metric-depth example frames are separate .npy artifacts. Continuous raw RGB/depth calibration-grade streams and per-sample exposure timestamps are not yet recorded.
- Standard ContactSensor gives whole distal-body contact, including merged elastomer/fingertip collision shapes; not just the pad, no partner identity, tangential force, distributed pressure or contact moment. All five body channels respond in episodes02; calibration and saturation tests remain open.
- Aggregate normal force is not a slip measurement. Force peak statistics must not be called pressure or grasp-stability proof.
