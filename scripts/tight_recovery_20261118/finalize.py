"""Read-only scientific summaries and compact publication of this completed cycle.

Does not train, generate expert labels, restart workers or select/promote models.
"""
import csv
import gzip
import json
import shutil
import subprocess
from collections import defaultdict

from scripts.tight_recovery_20261118.run import ROOT, OUT, DATA, PARENT, read, write
from scripts.tight_layout_20261116.publication import table
from visuotactile.bounded_extension import summarize
from visuotactile.compact import sha256


def phase_groups():
    """Aggregate saved sufficient statistics; no additional GPU inference."""
    data = read(ROOT / 'learning/A/state_data/data.json')
    parent = set(read(PARENT / 'A/state_data/data.json')['splits']['train'])
    previous = set(read(DATA / 'A/state_data/data.json')['splits']['train'])
    original = set(data['normalization_train_ids'])
    groups = {
        'original30': original,
        'common72': parent,
        'historical_correction42': parent - original,
        'prior105_unseen33': previous - parent,
        'new_recovery25': set(data['splits']['train']) - previous,
    }
    rows = []
    for component, filename in [('DP', 'offline_phase_errors.csv'),
                                ('full_command', 'backend_phase_errors.csv')]:
        grouped = defaultdict(list)
        identities = defaultdict(set)
        with (OUT / 'offline' / filename).open() as handle:
            for row in csv.DictReader(handle):
                for group, members in groups.items():
                    if row['identity'] in members:
                        key = (row['role'], row['version'], group)
                        grouped[key].append(json.loads(row['bins']))
                        identities[key].add(row['identity'])
        for (role, version, group), bins in grouped.items():
            assert len(identities[role, version, group]) == len(groups[group])
            for phase, metrics in summarize(bins).items():
                rows.append(dict(component=component, role=role, version=version,
                                 group=group, trajectories=len(groups[group]),
                                 phase=phase, **metrics))
    comparisons = []
    for row in rows:
        if row['version'] == 'parent72':
            continue
        key_fields = ['component', 'role', 'group', 'phase']
        old = next((r for r in rows if r['version'] == 'parent72'
                    and all(r[k] == row[k] for k in key_fields)), None)
        metric = 'normalized_equal_mse'
        comparisons.append(dict(**row, parent_mse=old[metric] if old else None,
                                change_percent=(100 * (row[metric] / old[metric] - 1)
                                                if old and old[metric] and row[metric] is not None
                                                else None),
                                matched_parent_history=old is not None))
    table(OUT / 'offline/cohort_phase_errors.csv', rows)
    table(OUT / 'offline/cohort_comparison.csv', comparisons)
    write(OUT / 'offline/cohort_scope.json', dict(
        populations={k: sorted(v) for k, v in groups.items()},
        scope='Nested populations; do not sum common72 with its original30/correction42 subsets. '
              'Real recorded histories and legal expert labels only. DP target domains differ A/B. '
              'Parent full-command cache exists only on72: candidate errors on additional58 are '
              'absolute fitting diagnostics, NOT measured improvements over parent. '
              'Teacher forcing does not establish closed-loop recovery.'))


def receipts():
    """Keep exact selections, bindings and every update; omit large geometry debug."""
    paths = list(ROOT.glob('*.json')) + list((ROOT / 'learning/logs').glob('*.json'))
    for name in ['session.json', 'prepared.json', 'complete.json', 'owner_exit.json',
                 'training_sampling.json', 'bounded_extension.json', 'frozen_visual_hotset.json',
                 'backend_episode_sampling.json', 'full_trajectory_batch.json']:
        paths.append(ROOT / 'learning' / name)
    for role in 'AB':
        base = ROOT / 'learning' / role
        for name in ['continuation.json', 'backend_continuation.json', 'backend/base_identities.json',
                     'backend/cache_complete.json', f'backend/{role}/training/selected.json',
                     f'backend/{role}/nominal/index.json']:
            paths.append(base / name)
        destination = OUT / 'learning' / role / 'DP/learning.jsonl.gz'
        destination.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=str(destination), mode='wb', mtime=0) as out:
            with (base / 'training/learning.jsonl').open() as source:
                for line in source:
                    row = json.loads(line)
                    for visit in row['visits']:
                        visit.pop('geometry', None)
                    out.write((json.dumps(row, separators=(',', ':')) + '\n').encode())
    inventory = []
    for source in sorted(set(paths)):
        if not source.exists():
            continue
        destination = OUT / 'receipts' / source.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        inventory.append(dict(path=str(destination.relative_to(OUT)),
                              sha256=sha256(destination), bytes=destination.stat().st_size))
    write(OUT / 'receipt_inventory.json', inventory)
    rows = []
    for wave in ['pilot', 'collection', 'collection_recovery01', 'collection16_0',
                 'collection16_1'] + [p.name for p in sorted(ROOT.glob('comparison*')) if p.is_dir()]:
        directory = ROOT / wave
        row = dict(wave=wave, completed=(directory / 'owner_complete.json').exists())
        for name, key in [('owner_complete.json', 'owner_seconds'),
                          ('initialization.json', 'initialization_s')]:
            if (directory / name).exists():
                row[key] = read(directory / name).get('seconds')
        if (directory / 'plan.json').exists():
            row['lanes'] = read(directory / 'plan.json')['batch']['lanes']
        if (directory / 'kit_started.json').exists() and (directory / 'run_started.json').exists():
            row['kit_start_s'] = (read(directory / 'kit_started.json')['utc']
                                  - read(directory / 'run_started.json')['utc'])
        if (directory / 'wave_profiles.json').exists():
            for key in ['seconds', 'controller_s', 'environment_s', 'record_submit_s']:
                row['wave_' + key] = sum(w[key] for w in read(directory / 'wave_profiles.json'))
        rows.append(row)
    table(OUT / 'simulation_times.csv', rows)


def rear_exposure():
    """Count lawful control ticks at selected and final backend ages separately."""
    import h5py
    import numpy as np
    import torch
    from scripts.tight_balance_20261117.diagnose import phases
    from visuotactile.geometry_sampling import STAGES
    parent = set(read(PARENT / 'A/state_data/data.json')['splits']['train'])
    previous = set(read(DATA / 'A/state_data/data.json')['splits']['train'])
    rows = []
    for role in 'AB':
        data = read(ROOT / 'learning' / role / 'state_data/data.json')
        original = set(data['normalization_train_ids'])
        metrics = {}
        for identity in data['splits']['train']:
            entry = data['entries'][identity]
            correction = entry.get('trajectory_kind') == 'expert_takeover_v1'
            with h5py.File(entry['path']) as handle:
                transition = handle['episodes'][entry['episode']]['transition']
                phase = phases(transition)
                mask = (transition['dagger/expert_label_valid'][:].reshape(-1).astype(bool)
                        if correction else np.ones(len(phase), dtype=bool))
                first = np.flatnonzero(mask)
                group = ('original30' if identity in original else 'historical_correction42'
                         if identity in parent else 'prior105_unseen33' if identity in previous
                         else 'new_recovery25')
                metrics[identity] = dict(group=group, legal_ticks=int(mask.sum()),
                                         original_early_ticks=int((mask & (np.arange(len(mask)) < 150)).sum())
                                         if not correction else 0,
                                         onset24_ticks=min(24, len(first)) if correction else 0,
                                         **{s: int(((phase == k) & mask).sum()) for k, s in enumerate(STAGES)})
        folder = ROOT / 'learning' / role / 'backend' / role / 'training'
        for scope, name in [('selected', 'best.pt'), ('actual', 'last.pt')]:
            saved = torch.load(folder / name, map_location='cpu', weights_only=False, mmap=True)
            total = 0
            for identity, visits in saved['episode_counts'].items():
                source = metrics[identity]
                counts = {k: v * visits for k, v in source.items() if k != 'group'}
                total += counts['legal_ticks']
                rows.append(dict(role=role, scope=scope, group=source['group'], identity=identity,
                                 visits=visits, **counts))
            assert total == saved['supervised_ticks']
            del saved
    table(OUT / 'learning/backend_phase_exposure.csv', rows)


def family_exposure():
    grouped = {}
    for row in read(OUT / 'learning/episode_exposure.json'):
        key = (row['role'], row['scope'], row['family'])
        target = grouped.setdefault(key, dict(role=key[0], scope=key[1], family=key[2],
                                               trajectories=0, visits=0, windows=0, slots=0,
                                               early_slots=0, onset24_slots=0, base_objective_mass=0.))
        target['trajectories'] += 1
        for field in ['visits', 'windows', 'slots', 'early_slots', 'onset24_slots', 'base_objective_mass']:
            target[field] += row.get(field, 0)
    table(OUT / 'learning/family_exposure.csv', list(grouped.values()))


def cohort_curves():
    rows = []
    for role in 'AB':
        path = ROOT / 'learning' / role / 'training/validation.jsonl'
        for line in path.open():
            value = json.loads(line)
            for cohort, diagnostic in value.get('cohort_diagnostics', {}).items():
                for phase, metrics in diagnostic['phases'].items():
                    rows.append(dict(role=role, epoch=value['epoch'], updates=value['updates'],
                                     cohort=cohort, identities=diagnostic['ids'], phase=phase, **metrics))
    table(OUT / 'learning/dp_cohort_curves.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for row, role in enumerate('AB'):
        for column, phase in enumerate(['all', 'before150']):
            axis = axes[row, column]
            for cohort in ['original', 'historical', 'latest', 'validation']:
                values = [v for v in rows if v['role'] == role and v['phase'] == phase
                          and v['cohort'] == cohort]
                axis.plot([v['updates'] for v in values], [v['normalized_equal_mse'] for v in values],
                          label=cohort)
            axis.set_title(role + ' DP / ' + phase)
            axis.set_xlabel('Optimizer updates')
            axis.set_ylabel('Normalized MSE')
            axis.grid(alpha=.2)
            axis.legend(fontsize=8)
    fig.suptitle('Recorded fitting cohorts (original4/history4/new25); validation6 is separate', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'learning/dp_cohort_curves.png', dpi=140)
    plt.close(fig)


def paths():
    changes = read(OUT / 'evaluation/paired_changes.json')['rows']
    result = read(OUT / 'evaluation/results.json')
    outcomes = {}
    for cohort in ['training', 'new_development', 'all']:
        for role in 'AB':
            selected = [r for r in changes if r['role'] == role
                        and (cohort == 'all' or r['cohort'] == cohort)]
            outcomes[cohort + '_' + role] = dict(
                pairs=len(selected),
                retained=[r['seed'] for r in selected if r['baseline_success'] and r['candidate_success']],
                lost=[r['seed'] for r in selected if r['baseline_success'] and not r['candidate_success']],
                gained=[r['seed'] for r in selected if not r['baseline_success'] and r['candidate_success']],
                both_failed=[r['seed'] for r in selected if not r['baseline_success'] and not r['candidate_success']])
    write(OUT / 'evaluation/path_retention.json', dict(
        outcomes=outcomes, counts=result['counts'],
        scope='Same-wave observed starts, previously opened development. Changed paths are descriptive; '
              'not independent test, statistical significance, or isolated replay-factor causality.'))


def untriggered():
    rows = []
    for wave in ['pilot_admission_recovery02', 'collection_completed']:
        for entry in read(ROOT / wave / 'results.json'):
            result = entry['result']
            receipt = result['dagger']['trigger_receipt']
            if receipt['trigger_completed_tick'] is not None:
                continue
            rows.append(dict(wave=wave, role=entry['role'], seed=result['seed'],
                             success=result['success'], steps=result['steps'],
                             native_subgoals=result['native_subgoals'],
                             terminal_state=receipt['last'], reasons=result['reasons'],
                             scope='No takeover, no expert supervision; subgoal count is descriptive. '
                                   'The screen intentionally excludes post-subgoal states, but this '
                                   'alone does not prove a recoverable intervention existed.'))
    write(OUT / 'untriggered_cases.json', rows)


def media():
    """Export a few real recorded comparisons; selection is disclosed post hoc."""
    import cv2
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    available = []
    for part in range(4):
        folder = ROOT / f'comparison{part:02d}'
        plan = read(folder / 'plan.json')
        for case in plan['cases']:
            path = folder / case['name'] / 'autonomous.mp4'
            if path.exists():
                available.append((case, path))
    changes = read(OUT / 'evaluation/paired_changes.json')['rows']
    choices = []
    for role in 'AB':
        eligible = [r for r in changes if r['role'] == role and
                    sum(c['role'].startswith(role + '_') and c['spec']['seed'] == r['seed']
                        for c, _ in available) == 2]
        assert eligible
        # Prefer changed successes, then retained successes, then the first recorded family.
        eligible.sort(key=lambda r: (not (r['baseline_success'] != r['candidate_success']),
                                     not r['baseline_success'], r['seed']))
        choices.append((role, eligible[0]['seed']))
    panels, videos = [], []
    for case, source in available:
        if (case['role'].split('_')[0], case['spec']['seed']) not in choices:
            continue
        target = OUT / 'videos' / (case['role'] + '_' + str(case['spec']['seed']) + '.mp4')
        shutil.copyfile(source, target)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '2',
                        '-i', str(target), '-f', 'null', '-'], check=True)
        result = read(source.parent / 'result.json')
        cap = cv2.VideoCapture(str(target))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        assert frames == result['steps']
        ticks = sorted({min(frames - 1, tick) for tick in [72, 120, 180, 300, frames - 1]})
        strip = []
        for tick in ticks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
            ok, frame = cap.read()
            assert ok
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((576, 324))
            panel = Image.new('RGB', (576, im.height + 24), 'white')
            panel.paste(im, (0, 24))
            ImageDraw.Draw(panel).text((4, 4), f"{case['role']} {case['spec']['seed']} tick{tick}", fill='black')
            strip.append(panel)
        cap.release()
        panels.append(strip)
        videos.append(dict(case=case['name'], file=str(target.relative_to(OUT)),
                           sha256=sha256(target), frames=frames, success=result['success'], decoded_all=True))
    canvas = Image.new('RGB', (max(sum(p.width for p in row) for row in panels),
                               sum(max(p.height for p in row) for row in panels)), 'white')
    y = 0
    for row in panels:
        x = 0
        for panel in row:
            canvas.paste(panel, (x, y))
            x += panel.width
        y += max(p.height for p in row)
    canvas.save(OUT / 'evaluation/video_review.jpg', quality=88)
    write(OUT / 'evaluation/media.json', dict(videos=videos,
        selection='Post-outcome representative A/B recorded pairs, preferring changed successes. '
                  'Viewing only; no simulation rerun. All64 outcomes remain the evidence denominator.'))


if __name__ == '__main__':
    assert read(ROOT / 'tail_recovery02_complete.json')['complete']
    for action in ['phase_groups', 'receipts', 'rear_exposure', 'family_exposure', 'cohort_curves', 'paths', 'untriggered', 'media']:
        globals()[action]()
