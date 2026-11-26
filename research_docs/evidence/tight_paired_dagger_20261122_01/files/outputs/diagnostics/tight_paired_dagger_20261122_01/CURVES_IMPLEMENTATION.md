# Paired DAgger family/phase diagnostics

Implemented an exact opt-in `paired_family_phase_diagnostics_v1` profile. To enable before a stage starts, write `family_phase_diagnostics.json` at the training experiment root with `visuotactile.paired_curve_diagnostics.profile()` using `atomic_json`. Enable on both `common180_parents` and the subsequent main training root. Absent file preserves existing behavior. `tight_fresh_backend.prepare` copies the profile into `residual_pair`; the paired parent wrapper only removes the bounded stopping policy and preserves this new field.

All hooks are now applied; **do not apply dp_curve_hooks.patch again**. That patch is an audit record of the narrow DP hook changes.

The DP trainer records the already computed H24 diffusion sample-prediction error vectors, valid legal slot times, and real family identity after each successful AdamW/EMA update. It exports existing per-visit total objective losses by family from `learning.jsonl` separately. There is no invented decomposition of the geometry objective by phase. DP validation reuses its existing DDIM16 commit12 predictions, mask and anchors, with no extra model forwards, DDIM plans or RNG draws. Original family/phase labels are read once from numeric HDF5 metadata, not RGB tensors.

The Transformer trainer records already composed final command minus expert command on the same actual legal controls used by training. Chunk summaries remain pending until their actual optimizer update succeeds; tick-window summaries commit together after `window.step`. Its validation uses the predictions already returned by existing `residual_metrics`. Existing fixed train2 evaluation remains separate from all training-update data. Main and parent backends both target `u_expert`.

Shared offline phase labels come from unchanged `geometry_sampling.diagnostic_phases`: grasp_without_contact, contact_not_held, held_far, held_rotating, held_near_goal60, after_first_native_subgoal, contact_lost. These are descriptive after-transition labels, and never inputs, triggers, sampling weights, losses or model selection. Held-near60 is **not** the15mm entry gate or native confirmation. Contact-lost does not assert a physical fall. The phase definition is frozen with source identity and stage names in each output.

Each training directory exports:

- `family_phase_events.jsonl`: committed update, epoch, split, role, target, identity/family and per-phase squared-error sums/counts.
- `family_phase_curves.csv`: per epoch/split/family/phase and aggregate rows, sample counts, arm/hand RMSE and normalized equal MSE.
- `family_phase_curves.png`: small final training/evaluation and validation-phase plots.
- `family_phase_definition.json`: explicit target and limits.
- DP only: `family_objective_loss.csv`, actual whole-episode visit loss curves, with no fake phase assignment.

Event export ignores updates beyond the durable checkpoint update and replaces repeated update events after an explicit resume; it does not count aborted updates twice. DP overlapping supervised windows remain repeated samples in error diagnostics and are not claimed as distinct exposure. Actual distinct successful-update exposure remains the independent existing exposure ledgers.

DP train errors and validation errors use different noise/decoding protocols; compare trends within each split. A and B DP target spaces differ; do not use their nominal loss/error curves to rank A against B. The output does not claim autonomy or causality. Rear errors are command MSE diagnostics and do not decompose the FY, geometry or residual regularization training objective. Empty phases simply have no samples/row.

Validation: 37 CPU tests passed across fresh-backend contracts, full-trajectory training/resume, native inference-noise clocks, and7 new diagnostic tests. The actual opt-in DP test verified unchanged denoiser forward count, global and explicit noise RNG, exact AdamW/model/EMA state, loss receipts, and diagnostic gradient detachment. Default training tests pass without any config file. Real new-full440tick and old-validation257tick phase-label reads succeeded. Initial unittest invocation of pytest-style modules failed due its `tests` import path; corrected `PYTHONPATH=.:tests python -m pytest` passed all37. No training, GPU or Kit execution was launched by this task.
