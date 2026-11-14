# Round 06: independent CPU comparison with round 05

Use round06 B as the next real-student recovery-data candidate; keep round05 A and retain both round05 rollback models. This is a development choice, not a demonstrated final-success or generalization improvement. All four repeatedly reused development starts per model still fail the original task. They are not independent new families or exact hidden-state pairs.

| Model | Native lift | Old stable grasp | Rotation-permitting contact-height12 | Native near/alignment/first subgoal | Final success |
|---|---:|---:|---:|---:|---:|
| Round05 A | 1/4 | 2/4 | 2/4 | 0/4 each | 0/4 |
| Round06 A | 1/4 | 2/4 | 2/4 | 0/4 each | 0/4 |
| Round05 B | 1/4 | 1/4 | 1/4 | 0/4 each | 0/4 |
| Round06 B | 1/4 | 2/4 | 2/4 | 0/4 each | 0/4 |

B now has two trajectories that hold the object, turn it toward the first active goal and approach the hole. The following values belong to the **same actual completed control tick**, chosen by the smallest native keypoint error while a full causal contact-height window holds.

| B seed | Tick | Root angular difference | Root position distance | Native maximum keypoint error | Contact-height run containing that tick |
|---|---:|---:|---:|---:|---|
| 28140001 | 225 | 0.183145 rad (10.49 degrees) | 2.019731 mm | 17.439172 mm | ticks 211–284, 74 ticks (1.233 s); 15 ticks elapsed at snapshot |
| 28140002 | 316 | 0.219244 rad (12.56 degrees) | 4.601017 mm | 16.044691 mm | ticks 278–338, 61 ticks (1.017 s); 39 ticks elapsed at snapshot |

The original `near_goal` requires **maximum corresponding object/goal keypoint distance <=15 mm**, not root position distance <=15 mm. Both snapshots miss it; no near-goal tick occurs anywhere in either trajectory. Original alignment is the native accumulated near-goal-step condition, not a quaternion-angle threshold; neither trajectory reaches it or the first subgoal. No successful insertion occurred. Quaternion angles ignore shape symmetry. The minimum held angular errors are 0.173822/0.122729 rad at ticks 233/331, whose keypoint errors are 21.784833/29.149376 mm: those separate minima must not be combined with the better positions above.

Contact-height requires at least two object-filtered fingertips above 0.1 N in **both 120 Hz physics samples** of every 60 Hz control tick, object height at least 20 mm above the causal settled reference, continuously for 12 ticks. It allows rotation in the palm. It is distinct from native lift, old relative-palm stability and dropping. B seed01 passes this holding rule although native lift remains false. Both B trajectories lose the contact-height condition before their final timeout; that condition alone does not identify a drop.

Round05 B had only seed03 satisfying contact-height12; its best held keypoint error was 84.604345 mm and angle there 1.515002 rad. Round06 B seed03 regresses to no meaningful lift (maximum settled-reference rise 0.12 mm); seed00 still fails pickup. New B's benefit is therefore mixed and needs more actual student-prefix data and evaluation.

A remains mixed: seed03's longest contact-height run increases 26→84 ticks and best held keypoint error improves 216.477→59.270 mm, but angle at that better-position tick remains 2.574576 rad and the episode falls at 456 ticks. Seed02's longest holding run decreases 46→20 ticks, and best held keypoint error worsens 284.986→301.248 mm. Retain round05 A.

Training used 93 trajectories: 30 original +54 historical recoveries +9 current recoveries, not 93 independent initial families. The nine new records contain 1,850 legal expert-control ticks (3,038 full history ticks including student prefixes). Each completed DP line ran 10 epochs/1,020 episode updates; the current nine were sampled twice per epoch. They contributed 180 episode visits, 3,260 plan anchors and 73,280 supervised target slots: 13.48% of DP target slots. Backends completed 15 epochs; these records contributed 135 episode draws and 27,750 legal supervised ticks: 7.18% of backend action ticks.

Validation selected A DP epoch10 and B DP epoch3; deployed B therefore actually saw only 54 current-record visits/978 anchors/21,984 target slots during this continuation. Fixed independent validation selection error improves 3.58%/2.54% for A/B. A's selected current-correction arm/hand error improves 0.006838/0.056428→0.002783/0.020707 rad. B's epoch10 current-correction error improves too, but **epoch10 is not deployed epoch3**, so that final fitting result cannot establish selected B's correction fitting. A separate exact frozen train-only forward pass of selected B epoch3 measures new-correction arm/hand RMSE 0.00438868/0.0368985 rad (source 0.00716645/0.0658879), confirming improved fitting in the actually deployed weights; see [selected-B receipt](../round06_selected_B_train_fit.json). It does not add optimizer updates or new validation/test selection. Backends selected A epoch9/B epoch14; their plan/visual caches match the selected frontend hashes. The actual autonomous B trajectories above are the behavior evidence.

`autonomy.json` contains all 16 trajectory hashes, same-tick diagnostics, model identities and source verification. Source round05 commit `32b3988563daf7491662cad6a61edf8f0b3377e7` (156 registered files) and candidate round06 `3e648dbdeae3f900b1337de434b0679db5b0818d` (180 files) pass every frozen-source digest. Model pair, completed results/index, exact registered reset opportunities, no expert control, native 120/60/5 Hz, every-tick visual/history/backend counts and executed-target continuity were checked. Old stability function source is identical across these frozen trees. The reusable script reproduces all original numeric outputs exactly; four focused CPU tests pass. No simulator starts, optimizer updates or test-data access were added by this review.

Reproduce on the research machine from the development checkout (new output directory required):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv_isaacsim/bin/python tools/tight_fresh_round_compare.py \
  --source-root /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_02/round_05 \
  --candidate-root /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_05/round_06 \
  --source-tree /home/CNS2025331827/docs/bc+rl_RSS_fresh_round05_train_20261114 \
  --candidate-tree /home/CNS2025331827/docs/bc+rl_RSS_fresh_repeat06_20261114 \
  --output /path/to/new_cpu_review
```

The script imports helpers only from the fully verified candidate frozen tree; run it as the standalone file, not through a preloaded development package. Published JSON contains small summaries only; checkpoints and raw trajectories remain on the research machine. Full local learning report is retained under session05 `round06_cpu_review03/learning_round06.json` and referenced by hash in the public learning summary. Rollout wall times are descriptive measurements of these serial development executions, not a controlled speed comparison.
