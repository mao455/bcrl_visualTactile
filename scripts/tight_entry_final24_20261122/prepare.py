"""Pre-register all test opportunities before any new outcomes are opened."""
import copy
import json
from pathlib import Path
import subprocess
import time

from tools.tight_fresh_rollouts import completed_pair
from scripts.tight_layout_20261116.pilot import reset_configuration
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
ROOT = PROJECT / 'outputs/visuotactile/run/tight_entry_final24_20261122_01'
EVIDENCE = PROJECT / 'research_docs/evidence/tight_entry_final24_20261122_01'


def main():
    assert not ROOT.exists(), 'Preserve every attempted identity'
    old = PROJECT / 'research_docs/evidence/tight_entry_phase_20261121_01/development'
    registered = json.loads((old / 'registration.json').read_text())
    pairs = {}
    for role in 'AB':
        for tag, run in [('130_50', 'tight_exposure50_20261119_01'),
                         ('entry', 'tight_entry_phase_20261121_01')]:
            name = role + '_' + tag
            pair = completed_pair(PROJECT / 'outputs/visuotactile/run' / run, role)
            assert pair == registered['pairs'][name]
            pairs[name] = pair
    template = json.loads((old / 'wave00/plan.json').read_text())
    previous = json.loads((old / 'session.json').read_text())
    now = time.time()
    session = {k: copy.deepcopy(previous[k]) for k in (
        'resource_policy', 'resources', 'expert_checkpoint', 'expert_sha256',
        'dino_checkpoint', 'dino_sha256', 'precision_policy', 'camera_content_validation',
        'domain', 'layout', 'visual_arithmetic', 'visual_stride', 'control_hz', 'planning_hz')}
    session.update(version='tight_entry_final24_frozen_v1', started_utc=now,
        deadline=now+6*3600, training_deadline=now+6*3600, reserve_bytes=12*2**30,
        resource_root=str(ROOT), owned_worker_roots=[str(ROOT)],
        authorization='User authorized frozen 24 initial states x four complete policies; zero optimizer updates and zero DAgger')
    seeds = list(range(28851000, 28851008)) + list(range(28852000, 28852016))
    sources = {str(p.relative_to(PROJECT)): sha256(p) for folder in ('tools', 'visuotactile')
               for p in sorted((PROJECT / folder).rglob('*.py'))}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    ROOT.mkdir(parents=True)
    EVIDENCE.mkdir(parents=True)
    atomic_json(ROOT / 'session.json', session)
    stages, wave_hashes, specifications = [], {}, []
    for index, seed in enumerate(seeds):
        cohort = 'reserved8' if index < 8 else 'fresh16'
        spec = dict(identity=f'entry_final24_{seed}', family=f'entry_final24_{seed}',
            split='test', seed=seed, event_seed=seed+100000, cohort='fresh',
            collision_gate='live_collision_mesh_bounds_v1',
            narrow_reset=reset_configuration(session['domain'], seed, .1), layout=session['layout'])
        specifications.append(spec)
        wave = ROOT / f'wave{index:02d}'
        wave.mkdir()
        names = list(pairs)
        names = names[index % 4:] + names[:index % 4]
        plan = copy.deepcopy(template)
        plan.update(pairs=pairs, cases=[dict(name=f'{name}_{spec["identity"]}', role=name,
            spec=copy.deepcopy(spec), video=index in (0, 8), diagnostic_cohort=cohort) for name in names],
            resource_root=str(ROOT), source_commit=commit, source_sha256=sources,
            test_used=True, holdout_contract='layout_final_test_v1', utc=now,
            cohort=cohort, visual_arithmetic=session['visual_arithmetic'], record_sampler_rng=True)
        plan['batch'].update(lanes=4, vision_microbatch=2, seed=seed,
            ordering='Four frozen policies on one initial state; cyclic lane rotation; one physical opportunity per policy')
        plan['evaluation_scope'] = dict(protocol='Frozen 130-50 vs entry; complete DP + Transformer',
            cohort=cohort, test_used=True, hidden_state_replay_claim=False,
            strict_success='unchanged native task success',
            hold_metrics='secondary diagnostics, never strict success')
        atomic_json(wave / 'session.json', session)
        atomic_json(wave / 'plan.json', plan)
        wave_hashes[str(wave)] = sha256(wave / 'plan.json')
        stages.append(dict(entry='heldout_evaluate', args=['--output', str(wave)]))
    job = PROJECT / 'configs/experiments/entry_final24/evaluate.json'
    job.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(job, dict(version='tight_project_job_v1', operation='evaluate',
        resource_root=str(ROOT), stages=stages))
    registration = dict(version='tight_entry_final24_frozen_v1', registered_utc=now,
        source_commit=commit, source_sha256=sources, pairs=pairs, seeds=seeds,
        reserved8=seeds[:8], fresh16=seeds[8:], specifications=specifications,
        wave_plan_sha256=wave_hashes, job=str(job), job_sha256=sha256(job),
        deadline=session['deadline'], episodes=96, previously_known_results=[],
        reuse='No executed plans found for reserved8; all 96 opportunities newly run',
        freshness_audit='Before registration, exact integer seed search over configs/research_docs/outputs JSON, CSV, JSONL found no fresh16 occurrences; reserved8 only in old reservation, no execution plan',
        primary='Within A and B separately: native complete insertion; entry minus 130-50 paired rate difference; reserved8, fresh16 and all24 reported regardless of direction',
        uncertainty=dict(method='95% paired percentile bootstrap over initial states; 200000 resamples; fixed seed 20261122; exact two-sided McNemar discordance test',
            limitations='small samples; bootstrap may be discrete; no pooling A/B or post-hoc seed selection; no superiority if interval includes zero'),
        secondary='native_subgoals >= 1; final success among those attaining >=1; legal/software incomplete separately, retained in intention-to-evaluate denominator',
        noise=dict(sampler_seed=7091, scheme='independent per-policy per-lane CUDA generator initialized at 7091, same 24x29 draw per native planning tick; no success-dependent reseeding',
            scheduler_index_mode='host', record_sampler_rng=True,
            hidden_state_or_future_environment_stream_exactness=False),
        lane_balance='cyclic four-role order: each role occupies each position twice in reserved8, four times in fresh16, six times overall',
        recovery='Never overwrite failed roots; reuse closed compliant individual results. Technical failure gets a distinct recovery namespace under same finite deadline, same model/config/noise; preserve and separately report all physical attempts, no outcome-driven retry',
        optimizer_updates=0, new_dagger_trajectories=0)
    atomic_json(ROOT / 'registration.json', registration)
    atomic_json(EVIDENCE / 'registration.json', registration)
    atomic_json(EVIDENCE / 'session.json', session)
    print(json.dumps(dict(root=str(ROOT), job=str(job), registration_sha256=sha256(ROOT/'registration.json'), episodes=96)))


if __name__ == '__main__':
    main()
