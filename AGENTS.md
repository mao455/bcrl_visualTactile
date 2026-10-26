# Play2Perfect visual–tactile imitation research

Read `research_docs/README.md`, `CONTRACTS.md`, `ARCHITECTURE.md`, and the latest entries of `CHANGELOG.md` before changing this research system. This is a long-term research project, not a replacement of the original RL environment.

## Non-negotiable constraints
- Preserve original assets, collisions, limits, friction, gains, dynamics, coordinate transforms, reward/success criteria, reset, noise and delay mechanisms. Extend opt-in; do not edit vendored RL to implement BC.
- Nominal diffusion policy must never read tactile features, correction outputs, expert actions or privileged state. Explicit typed/allowlisted inputs, separate training supervision. Ordinary BC is NOT demonstration nominalization; pressure intent can remain in its targets.
- Tactile encoder and correction policy are separate modules. No fake losses/random labels; an untrained correction must not control the robot.
- Keep 120 Hz physics / 60 Hz original action processing. A 10 Hz policy can produce/replan chunks of 60 Hz actions; do not change decimation to change network frequency. Arm and hand action semantics differ.
- Trace each observation to a real sensor or declared external command. Simulation object/goal poses, success stage, reward, force disturbances and future state are diagnostics/labels only unless a separately validated estimator supplies them.
- Causal history, per-environment episode IDs and reset masks are required. Never use episode_length_buf==0 alone: it also resets at task subgoals.
- Keep full trajectories and action provenance. Never silently call unsuccessful expert rollouts successful demonstrations.
- User's background GPU workload is protected: no signals, renice, driver or global changes. Use low priority, two CPU threads, few environments. RTX cameras still use GPU; monitor and bound probes; do not claim zero contention.
- Every substantive change updates research_docs/CHANGELOG.md (paths, before/after, I/O, tests and actual results), contracts when changed, and current status. Clearly distinguish implemented, tested, and research hypotheses.
- Do not automatically freeze open architecture/hyperparameter choices. Do not claim performance from smoke tests.

## Environment
Use `.venv_isaacsim/bin/python` (project-local overlay of conda isaaclab), not base Python 3.13. Original Isaac Lab source is a local 2.3.2 fork. Use repo rl_games. Boot AppLauncher before importing Isaac Lab simulation/task modules. See prior `outputs/compat_check/REPORT.txt`.

**Critical CPU regression finding:** the original CPU no-render path resets kinematic table/hole/goal transforms to their spawn origin on the first physics step in this installed stack. Prior no-render success rates are not valid quality evidence. A rendered-original reference and the instrumented path are bitwise equal over the documented 32-step probe, including objects/goals/RNG/rewards. See `research_docs/AUDIT.md`. Do not silently disable rendering for CPU evaluation or claim global Isaac Lab compatibility from that bounded test.

Use `bash visuotactile/run_cpu.sh ...` for new tests (nice 19, two threads, affinity 22,23). Never signal unrelated jobs. Fresh output directories are mandatory. Student camera augmentation/delay/pose randomization are currently rejected, not silently disabled; original teacher/action DR is unchanged. New tests must preserve this distinction.

## Public repository workflow

Read `research_docs/GIT_WORKFLOW.md` before repository operations. `origin` is the user's public fork (`mao455/bcrl_visualTactile`); `upstream` is `kushal2000/play2perfect`. Never push to `upstream`. Before a public push, inspect the staged file list, ignored artifacts, credentials scan and license/provenance files. Do not upload `outputs/`, `pretrained_*`, checkpoints, trajectories, logs, virtual environments or private configuration. Keep the original commit history and MIT license. After each substantive code change, update the changelog and relevant contracts/status, run the focused tests, make a focused commit, and report its ID and GitHub URL. Pushing requires explicit authorization for that update.
