"""Compare recorded autonomous risk states with the new lawful recovery entries.

CPU only. Marginal geometric support is not an observation match or expert label.
"""
import time

import h5py
import numpy as np

from scripts.tight_recovery_20261118.run import ROOT, OUT, DATA, read, write
from scripts.tight_recovery_20261118.diagnose import replay
from scripts.tight_cycle_20261118.analyze import features, buildtree, distance
from scripts.tight_layout_20261116.publication import table
from visuotactile.geometry_sampling import STAGES


def main():
    began = time.time()
    previous = read(DATA / 'A/state_data/data.json')
    current = read(ROOT / 'learning/A/state_data/data.json')
    old_ids = set(previous['splits']['train'])
    keys = ['palm_local_object_m', 'palm_local_rotation_rad',
            'object_speed_m_s', 'issued_history12_rms_rad']
    thresholds = {(r['phase'], r['metric']): r['p95']
                  for r in read(OUT / 'support105_reference.json')['rows']}
    populations = {'old105': [], 'new25_first24': []}
    for identity in current['splits']['train']:
        entry = current['entries'][identity]
        with h5py.File(entry['path']) as handle:
            transition = handle['episodes'][entry['episode']]['transition']
            values, phase = features(transition)
            valid = (transition['dagger/expert_label_valid'][:].reshape(-1).astype(bool)
                     if entry.get('trajectory_kind') == 'expert_takeover_v1'
                     else np.ones(len(phase), dtype=bool))
            if identity in old_ids:
                population = 'old105'
            else:
                population = 'new25_first24'
                ticks = np.flatnonzero(valid)[:24]
                valid[:] = False
                valid[ticks] = True
            populations[population].append((phase[valid], {k: values[k][valid] for k in keys}))
    trees = {}
    for population, records in populations.items():
        phase = np.concatenate([p for p, _ in records])
        for key in keys:
            values = np.concatenate([x[key] for _, x in records])
            for stage in set(phase):
                trees[population, int(stage), key] = buildtree(values[phase == stage], key)
    rows, cases = [], []
    for part in range(4):
        folder = ROOT / f'comparison{part:02d}'
        for case in read(folder / 'plan.json')['cases']:
            directory = folder / case['name']
            result = read(directory / 'result.json')
            with h5py.File(directory / 'numeric.h5') as handle:
                transition = handle['episode/transition']
                n = len(transition['target_after_rad'])
                receipt = replay(transition, n)
                start = receipt['trigger_completed_tick']
                cases.append(dict(case=case['name'], role=case['role'], seed=case['spec']['seed'],
                                  cohort=case['diagnostic_cohort'], success=result['success'],
                                  retrospective_screen=receipt,
                                  native_subgoals=result['native_subgoals']))
                if start is None:
                    continue
                # All of these actions were actually issued by the autonomous student.
                # The screen is replayed only to define a diagnostic window.
                values, phase = features(transition)
                window = (np.arange(n) >= start) & (np.arange(n) < start + 24)
                for stage in set(phase[window]):
                    mask = window & (phase == stage)
                    for key in keys:
                        threshold = thresholds.get((STAGES[int(stage)], key))
                        for population in populations:
                            tree = trees.get((population, int(stage), key))
                            if tree is None or threshold is None:
                                continue
                            distances = distance(tree, values[key][mask], key)
                            rows.append(dict(case=case['name'], role=case['role'],
                                             seed=case['spec']['seed'], cohort=case['diagnostic_cohort'],
                                             success=result['success'], screen_tick=start,
                                             screen_reason=receipt['reason'], phase=STAGES[int(stage)],
                                             metric=key, population=population, ticks=int(mask.sum()),
                                             median_distance=float(np.median(distances)),
                                             old105_leave_family_p95=threshold,
                                             within_reference_ticks=int((distances <= threshold + 1e-8).sum())))
    write(OUT / 'evaluation/recovery_screen_cases.json', cases)
    table(OUT / 'evaluation/recovery_state_support.csv', rows)
    write(OUT / 'evaluation/recovery_state_support_scope.json', dict(
        seconds=time.time() - began, old_legal_trajectories=105, new_entry_trajectories=25,
        new_entry_ticks=600, window='First24 actual autonomous transitions after retrospective causal screen',
        limitations='No new labels, intervention or simulation. Each unit is queried separately, so nearest '
                    'neighbors need not be the same recorded state. This is neither a joint observation/history '
                    'match nor proof of expert recoverability or policy error. Thresholds are unchanged old105 '
                    'phase leave-family p95. Training-family cases share families with the pool; development '
                    'families do not. Post-subgoal failures are outside this screen.'))


if __name__ == '__main__':
    main()
