"""Focused numeric audit of admitted early corrections; never reads RGB payloads.

Actual labels/history are compared with closed source trajectories. This is not
a substitute for existing capture admission or a counterfactual expert label for
student prefixes. Timings use completed control transitions, not planning ticks.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from visuotactile.grasp_curriculum import holding_metrics, EarlyGraspTrigger, CONTACT_ONSET
from visuotactile.training_resources import atomic_json


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0] + 1) if len(indices) else None


def distribution(values):
    values = [v for v in values if v is not None]
    return dict(count=len(values), minimum=min(values), median=float(np.median(values)),
                maximum=max(values)) if values else dict(count=0)


def run(root, output, *, latest_only=False):
    began = time.time()
    read = lambda p: json.loads(p.read_text())
    data = read(root/'A/state_data/data.json')
    records = {role: torch.load(root/role/'state_data/numeric.pt', map_location='cpu',
                               mmap=True, weights_only=False)['records'] for role in 'AB'}
    latest = data['aggregation']['new_training_ids']
    originals = data['normalization_train_ids']
    assert len(originals) == 30 and set(latest + originals) <= set(data['splits']['train'])
    rows = []
    for identity in (latest if latest_only else originals + latest):
        entry = data['entries'][identity]
        with h5py.File(entry['path']) as stream:
            g = stream['episodes'][entry['episode']]
            result = json.loads(stream.attrs['result_json'])
            assert bool(g.attrs['complete']) and result['success'] and result['legal']
            pose = g['transition/diagnostics_after/object_pose_w'][:]
            palm = g['transition/diagnostics_after/palm_body_pose_w'][:]
            force = g['transition/contact_substeps_after_w_N'][:]
            target = g['transition/target_after_rad'][:]
            before = g['transition/target_before_rad'][:]
            reached = g['transition/diagnostics_after/joint_position_rad'][:]
            times = g['obs/time_s'][:]
            n = len(times)
            assert np.allclose(np.diff(times), 1/60, atol=1e-8, rtol=0)
            fingers = (np.linalg.norm(force, axis=-1) > .1).sum(-1).min(-1)
            touch = fingers > 0; two = fingers >= 2
            settled = np.arange(n) >= 48
            elevated = pose[:, 2] >= np.median(pose[24:48, 2]) + .02
            first_two = first(two)
            loss = first((~two) & (np.arange(n) >= first_two)) if first_two else None
            a, b = records['A'][identity], records['B'][identity]
            mask = a.get('supervision_mask', torch.ones(n, dtype=torch.bool)).numpy()
            assert np.array_equal(mask, b.get('supervision_mask', torch.ones(n, dtype=torch.bool)).numpy())
            for record in (a, b):
                assert np.array_equal(record['command'].numpy(), target)
                assert np.array_equal(record['u_expert'].numpy()[mask], target[mask])
                assert np.array_equal(record['q_reached'].numpy()[mask], reached[mask])
            assert np.array_equal(before[1:], target[:-1])
            switch = entry.get('takeover_tick')
            if switch is not None:
                d = g['transition/dagger']
                control = d['control_source'][:]
                assert not mask[:switch].any()
                assert np.array_equal(mask, d['expert_label_valid'][:])
                assert not (mask & ((control != 1) | ~d['switch_clean'][:])).any()
                assert np.array_equal(d['previous_issued'][1:], target[:-1])
                assert np.array_equal(d['actual_issued'][:], target)
                assert np.array_equal(d['actual_applied'][:], target)
                assert np.array_equal(g['transition/actuator_substep_target_rad'][:],
                                      np.repeat(target[:, None], 2, axis=1))
            first_label_index = int(np.flatnonzero(mask)[0])
            row = dict(identity=identity, family=entry['family'], source_group=entry.get('source_group'),
                cohort='latest_corrections' if identity in latest else 'original30', frames=n,
                simulation_duration_s=n/60, first_contact_completed_tick=first(touch),
                first_postsettle_contact_completed_tick=first(touch & settled),
                first_two_finger_completed_tick=first_two, first_two_finger_loss_completed_tick=loss,
                first_contact_lift_completed_tick=first(two & elevated),
                takeover_action_index=switch, takeover_simulation_s=switch/60 if switch is not None else None,
                first_legal_expert_action_index=first_label_index,
                first_legal_expert_simulation_s=first_label_index/60,
                expert_to_success_s=(n-switch)/60 if switch is not None else n/60,
                valid_expert_ticks=int(mask.sum()), precontact_expert_ticks=int((mask & (np.arange(n) < (first(touch) or n)-1)).sum()),
                legal_before_tick100=int(mask[:100].sum()), legal_before_tick150=int(mask[:150].sum()),
                hold60=holding_metrics(pose, force),
                label_checks='A q[t+1], B u_expert, both rear commands and prefix exclusion exact')
            if switch is not None:
                candidate = EarlyGraspTrigger(CONTACT_ONSET, 10**9)
                candidate_tick = None
                for tick in range(switch):
                    candidate_tick = candidate.observe(tick, pose[tick], palm[tick], force[tick])
                    if candidate_tick is not None: break
                row.update(contact_to_takeover_ticks=switch-first(touch) if first(touch) else None,
                    postsettle_contact_to_takeover_ticks=switch-first(touch & settled) if first(touch & settled) else None,
                    trigger_reason=entry['trigger_receipt']['reason'],
                    earlier_contact_trigger_replay=dict(tick=candidate_tick, actual_new_expert_steps=0,
                        scope='Causal replay of saved student prefix only; recovery not yet tested'),
                    prefix_contact_lift=bool((two & elevated)[:switch].any()),
                    prefix_hold60=holding_metrics(pose[:switch], force[:switch]))
            rows.append(row)
    # Count actual population, separately from stochastic optimizer exposure.
    population = {}
    for label, ids in [('all', data['splits']['train']), ('latest_corrections', latest), ('original30', originals)]:
        masks = [records['A'][i].get('supervision_mask', torch.ones(len(records['A'][i]['time']), dtype=torch.bool)) for i in ids]
        population[label] = dict(episodes=len(ids), legal_ticks=sum(int(m.sum()) for m in masks),
            legal_before_tick100=sum(int(m[:100].sum()) for m in masks),
            legal_before_tick150=sum(int(m[:150].sum()) for m in masks))
    exposure = {}
    for role in 'AB':
        path = root/role/'training/learning.jsonl'
        counts = Counter(); before100 = before150 = slots = 0; updates = 0
        if path.exists():
            for line in path.read_text().splitlines():
                try: row = json.loads(line)
                except json.JSONDecodeError: break  # live final append only
                updates = row['updates']
                for visit in row.get('episodes', row.get('visits', [])):
                    identity = visit['episode']; counts[identity] += 1
                    if identity not in latest: continue
                    anchors = visit['anchor_sampling']['selected_anchors']
                    m = records[role][identity]['supervision_mask'].numpy()
                    for anchor in anchors:
                        indices = np.arange(anchor, min(anchor+24, len(m)))
                        valid = indices[m[indices]]
                        slots += len(valid); before100 += int((valid < 100).sum()); before150 += int((valid < 150).sum())
        exposure[role] = dict(observed_updates=updates, latest_visits=dict((i,counts[i]) for i in latest),
            latest_target_slots=slots, latest_slots_before100=before100, latest_slots_before150=before150,
            snapshot_only=True, durable_checkpoint_reconciliation_pending=True)
    summary = {}
    fields = ['simulation_duration_s', 'first_contact_completed_tick', 'first_two_finger_completed_tick',
              'first_postsettle_contact_completed_tick', 'first_contact_lift_completed_tick',
              'takeover_simulation_s', 'expert_to_success_s', 'contact_to_takeover_ticks',
              'postsettle_contact_to_takeover_ticks']
    for cohort in ('original30', 'latest_corrections'):
        selected = [r for r in rows if r['cohort'] == cohort]
        summary[cohort] = {key: distribution([r.get(key) for r in selected]) for key in fields}
    atomic_json(output, dict(version='early_grasp_numeric_supervision_audit_v1', root=str(root),
        created=time.time(), seconds=time.time()-began, latest_only=latest_only, units=dict(control_hz=60, planning_hz=5,
        physics_hz=120, completed_tick='transition index + 1', action_index='zero-based preaction index'),
        rows=rows, summary=summary, population=population, observed_training=exposure,
        claims=dict(prefix_is_not_expert_supervision=True, latest_labels_match_actual_execution=True,
        no_counterfactual_expert_labels_on_prefix=True, test_read=False, rgb_payloads_reaudited=False)))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--latest-only', action='store_true', help='Reuse prior original30 audit; inspect only new numeric suffixes')
    args=parser.parse_args();run(args.root.resolve(), args.output.resolve(), latest_only=args.latest_only)
