"""Register complete-policy development without changing historical artifacts."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
from tools.tight_fresh_rollouts import completed_pair
from scripts.tight_layout_20261116.pilot import reset_configuration
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def main():
    current = PROJECT / 'outputs/visuotactile/run/tight_entry_phase_20261121_01'
    original = PROJECT / 'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01/round02/learning'
    baseline = PROJECT / 'outputs/visuotactile/run/tight_exposure50_20261119_01'
    out = current / 'development_20261122_01'
    if out.exists():
        raise FileExistsError('Keep the registered physical opportunity identity')
    session = read(current / 'session.json')
    assert time.time() < session['deadline']
    pairs = {f'{role}_{arm}': completed_pair(root, role)
             for arm, root in [('130_50', baseline), ('R2', original), ('entry', current)]
             for role in 'AB'}
    old = PROJECT / 'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01/round02/evaluation'
    template = read(old / 'wave00/plan.json')
    specs = {case['spec']['seed']: copy.deepcopy(case['spec'])
             for wave in sorted(old.glob('wave*'))
             for case in read(wave / 'plan.json')['cases']}
    seeds = list(range(28850000, 28850008)) + list(range(28826000, 28826008))
    sources = {str(p.relative_to(PROJECT)): sha256(p)
               for folder in ('tools', 'visuotactile')
               for p in sorted((PROJECT / folder).rglob('*.py'))}
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PROJECT, text=True).strip()
    out.mkdir()
    session.update(resource_root=str(out), owned_worker_roots=[str(out)],
                   authorization='Authorized entry-fix complete-policy paired development; no new training or DAgger')
    atomic_json(out / 'session.json', session)
    stages = []
    for index, seed in enumerate(seeds):
        cohort = 'independent_development' if seed >= 28850000 else 'historical_development_regression'
        if cohort == 'independent_development':
            spec = dict(identity=f'entry_development_{seed}', family=f'entry_development_{seed}',
                        split='development', seed=seed, event_seed=seed + 100000,
                        cohort='fresh', collision_gate='live_collision_mesh_bounds_v1',
                        narrow_reset=reset_configuration(session['domain'], seed, .1), layout=session['layout'])
        else:
            spec = specs[seed]
        wave = out / f'wave{index:02d}'
        wave.mkdir()
        p = copy.deepcopy(template)
        names = list(pairs)
        names = names[index % len(names):] + names[:index % len(names)]
        p.update(pairs=pairs, cases=[dict(name=f'{name}_{spec["identity"]}', role=name,
                    spec=copy.deepcopy(spec), video=index in (0, 8), diagnostic_cohort=cohort) for name in names],
                 resource_root=str(out), source_commit=commit, source_sha256=sources,
                 visual_arithmetic=session['visual_arithmetic'], record_sampler_rng=True,
                 test_used=False, utc=time.time())
        p['batch'].update(lanes=6, vision_microbatch=2)
        p['evaluation_scope'] = dict(protocol='Same-wave full DP + Transformer for 130-50, original R2, native-entry A/B',
            cohort=cohort, test_used=False, hidden_state_replay_claim=False,
            strict_success='unchanged native task success',
            hold_metrics='30/60/120 ticks are diagnostics, never strict success')
        atomic_json(wave / 'session.json', session)
        atomic_json(wave / 'plan.json', p)
        stages.append(dict(entry='evaluate', args=['--output', str(wave)]))
    job = PROJECT / 'configs/experiments/entry_phase/development.json'
    atomic_json(job, dict(version='tight_project_job_v1', operation='evaluate', resource_root=str(out), stages=stages))
    atomic_json(out / 'registration.json', dict(pairs=pairs, seeds=seeds, episodes=len(seeds)*len(pairs),
        job=str(job), source_commit=commit, source_sha256=sources, deadline=session['deadline'],
        independent_development=seeds[:8], historical_development_regression=seeds[8:],
        reserved_final_test=list(range(28851000, 28851008)),
        selection='Training validation6-selected pairs fixed before development; no opened historical test used for selection',
        control_status='Original R2 shown as historical comparator until remaining equivalence gates close; rerun original sampler if any gate fails',
        utc=time.time()))
    print(json.dumps(dict(output=str(out), episodes=96, waves=16, job=str(job))), flush=True)


if __name__ == '__main__':
    main()
