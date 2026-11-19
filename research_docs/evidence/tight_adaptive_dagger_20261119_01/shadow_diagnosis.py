"""Read actual recorded shadow advice; never use prefix advice as supervision."""
import argparse
import collections
from pathlib import Path

import h5py
import numpy as np
import torch

from tools.tight_adaptive_dagger import PUBLIC, ROOT, read, write
from tools.tight_frozen_comparison import table
from visuotactile.geometry_sampling import STAGES
from scripts.tight_balance_20261117.diagnose import phases
from visuotactile.position_control import reachable_position_bounds, REACHABLE_POSITION_EXECUTION


def persistent(mask, width=6, earliest=48):
    run = 0
    for tick, value in enumerate(mask):
        run = run + 1 if tick >= earliest and value else 0
        if run >= width:
            return tick - width + 1
    return None


def analyze(round_index):
    torch.set_num_threads(2)
    root = ROOT / f'round{round_index:02d}/collection_combined'
    output = PUBLIC / f'round{round_index:02d}/shadow_diagnosis'
    output.mkdir(parents=True, exist_ok=True)
    plan = read(root / 'plan.json')
    rows, stage_rows, curves = [], [], []
    for case in plan['cases']:
        folder = root / case['name']
        result = read(folder / 'result.json')
        assert result['software_complete'] and result['legal']
        assert result['expert_shadow_computed']
        with h5py.File(folder / 'numeric.h5') as stream:
            t = stream['episode/transition']; d = t['dagger']
            controls = d['control_source'][:].reshape(-1).astype(bool)
            switches = np.flatnonzero(controls)
            prefix = int(switches[0]) if len(switches) else len(controls)
            actual = t['target_after_rad'][:]
            advice = d['expert_mapped_target_rad'][:]
            error = actual[:prefix] - advice[:prefix]
            assert np.isfinite(error).all()
            assert not d['expert_label_valid'][:prefix].any()
            phase = phases(t)
            arm = np.sqrt(np.mean(error[:, :7] ** 2, axis=-1))
            hand = np.sqrt(np.mean(error[:, 7:] ** 2, axis=-1))
            import json
            metadata = json.loads(stream.attrs['metadata_json'])
            limits = torch.as_tensor(metadata['joint_limits_rad'], dtype=torch.float32).reshape(-1,29,2)[0]
            bounds = reachable_position_bounds(torch.from_numpy(t['target_before_rad'][:prefix]), limits, REACHABLE_POSITION_EXECUTION)
            span = (bounds[...,1] - bounds[...,0]).clamp_min(1e-6).numpy()
            normalized = error / span
            normalized_arm = np.sqrt(np.mean(normalized[:,:7] ** 2, axis=-1))
            normalized_hand = np.sqrt(np.mean(normalized[:,7:] ** 2, axis=-1))
            row = dict(case=case['name'], role=case['role'], seed=case['spec']['seed'],
                       stage=case['adaptive_stage'], student_prefix_ticks=prefix,
                       takeover_tick=int(switches[0]) if len(switches) else None,
                       expert_rescued=result['dagger']['expert_takeover_success'],
                       first_arm_gap_001rad_6ticks=persistent(arm > .01),
                       first_hand_gap_005rad_6ticks=persistent(hand > .05),
                       first_arm_gap_quarter_span_6ticks=persistent(normalized_arm > .25),
                       first_hand_gap_quarter_span_6ticks=persistent(normalized_hand > .25),
                       prefix_arm_rmse_rad=float(np.sqrt(np.mean(error[:, :7] ** 2))),
                       prefix_hand_rmse_rad=float(np.sqrt(np.mean(error[:, 7:] ** 2))),
                       prefix_arm_fullspan_rmse=float(np.sqrt(np.mean(normalized[:,:7] ** 2))),
                       prefix_hand_fullspan_rmse=float(np.sqrt(np.mean(normalized[:,7:] ** 2))),
                       prefix_labels_masked=True)
            for name, tick in [('arm_gap', row['first_arm_gap_001rad_6ticks']),
                               ('hand_gap', row['first_hand_gap_005rad_6ticks'])]:
                row[name + '_phase'] = STAGES[int(phase[tick])] if tick is not None else None
                row[name + '_lead_to_takeover'] = prefix - tick if tick is not None and len(switches) else None
            for name, tick in [('arm_span_gap',row['first_arm_gap_quarter_span_6ticks']),('hand_span_gap',row['first_hand_gap_quarter_span_6ticks'])]:
                row[name + '_phase'] = STAGES[int(phase[tick])] if tick is not None else None
                row[name + '_lead_to_takeover'] = prefix - tick if tick is not None and len(switches) else None
            rows.append(row)
            for k, name in enumerate(STAGES):
                mask = phase[:prefix] == k
                if mask.any():
                    stage_rows.append(dict(case=case['name'], role=case['role'], stage=name,
                        ticks=int(mask.sum()), arm_squared_sum=float((error[mask,:7] ** 2).sum()),
                        hand_squared_sum=float((error[mask,7:] ** 2).sum())))
                    stage_rows[-1].update(arm_span_squared_sum=float((normalized[mask,:7]**2).sum()),
                                         hand_span_squared_sum=float((normalized[mask,7:]**2).sum()))
            for tick in range(0, prefix, 12):
                curves.append(dict(case=case['name'], role=case['role'], tick=tick,
                    phase=STAGES[int(phase[tick])], arm_advice_gap_rad=float(arm[tick]),
                    hand_advice_gap_rad=float(hand[tick])))
                curves[-1].update(arm_fullspan_gap=float(normalized_arm[tick]),hand_fullspan_gap=float(normalized_hand[tick]))
    totals = collections.defaultdict(collections.Counter)
    for r in stage_rows:
        totals[r['role'],r['stage']].update({k:r[k] for k in ('ticks','arm_squared_sum','hand_squared_sum','arm_span_squared_sum','hand_span_squared_sum')})
    summary = [dict(role=role,stage=stage,ticks=v['ticks'],
        arm_rmse_rad=(v['arm_squared_sum']/v['ticks']/7)**.5,
        hand_rmse_rad=(v['hand_squared_sum']/v['ticks']/22)**.5,
        arm_fullspan_rmse=(v['arm_span_squared_sum']/v['ticks']/7)**.5,
        hand_fullspan_rmse=(v['hand_span_squared_sum']/v['ticks']/22)**.5)
        for (role,stage),v in totals.items()]
    write(output/'records.json', rows)
    table(output/'records.csv',rows); table(output/'phases.csv',summary); table(output/'curves.csv',curves)
    write(output/'scope.json',dict(cases=len(rows),round=round_index,
        advice='Recorded expert mapper, actual previous target, continuously advanced shadow queue and original native delay draw',
        threshold='Six consecutive ticks after tick48. Original .01/.05rad screens retained; quarter-fullspan screen added because exact executor limits make raw arm threshold unattainable',
        normalization='Exact per-tick reachable endpoint width at actual previous command and joint limits; quarter span is descriptive and has no calibrated failure meaning',
        interpretation='Action disagreement only. Expert advice was not executed during student prefix; no counterfactual q_reached, harmful-error truth, optimality or recoverability claim',
        training_effect='None; all student-prefix supervision masks remain zero',
        physics_updates=0,optimizer_updates=0))
    print({'cases':len(rows),'phase_rows':len(summary),'output':str(output)},flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--round',type=int,required=True)
    analyze(p.parse_args().round)
