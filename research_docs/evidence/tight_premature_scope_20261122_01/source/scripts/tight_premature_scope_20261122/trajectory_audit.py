"""Read-only train/development census of pre-confirmation physical events.

The CLI discovers only train180 and development96.  ``analyze`` is a pure
reader and can be applied separately to other registered sources by its caller.
It never starts a simulator, policy, expert, optimizer, or data collection.
All ticks are zero-based completed control transitions.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

import h5py
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
OUT = PROJECT/'outputs/diagnostics/tight_premature_scope_20261122_01/train_dev'
PUBLIC = PROJECT/'research_docs/evidence/tight_premature_scope_20261122_01/train_dev'
RULES = PROJECT/'outputs/diagnostics/tight_premature_scope_20261122_01/rules_registration.json'
DEV = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01/development_20261122_01'
STATE = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01'


def read(p):
    return json.loads(Path(p).read_text())


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def first(mask):
    rows = np.flatnonzero(mask)
    return int(rows[0]) if len(rows) else None


def runs(mask):
    a = np.asarray(mask, dtype=bool)
    edges = np.diff(np.r_[False, a, False].astype(int))
    return [dict(start=int(s), end=int(e-1), ticks=int(e-s))
            for s, e in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))]


def first_run(mask, n):
    return next((r['start'] for r in runs(mask) if r['ticks'] >= n), None)


def qconj(q):
    q = np.asarray(q, dtype=np.float64).copy()
    q[..., 1:] *= -1
    return q


def qmul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return np.concatenate((aw*bw-(av*bv).sum(-1, keepdims=True),
                           aw*bv+bw*av+np.cross(av, bv)), axis=-1)


def rotate(q, v):
    return qmul(qmul(q, np.concatenate((np.zeros_like(v[..., :1]), v), axis=-1)), qconj(q))[..., 1:]


def relative(parent, child):
    inv = qconj(parent[..., 3:])
    return np.concatenate((rotate(inv, child[..., :3]-parent[..., :3]), qmul(inv, child[..., 3:])), axis=-1)


def angle(a, b):
    a = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    return np.degrees(2*np.arccos(np.clip(np.abs((a*b).sum(-1)), 0, 1)))


def drop_over_window(z, n):
    out = np.full(len(z), np.nan)
    out[n:] = (z[:-n]-z[n:])*1000
    return out


def event(mask, n=1):
    if mask is None:
        return dict(known=False, onset_tick=None, confirmation_tick=None, runs=None)
    rr = [r for r in runs(mask) if r['ticks'] >= n]
    return dict(known=True, onset_tick=rr[0]['start'] if rr else None,
                confirmation_tick=rr[0]['start']+n-1 if rr else None, runs=rr)


def _array(g, key, missing):
    if key not in g:
        missing.append(key)
        return None
    return g[key][:]


def analyze(group, metadata, result, rules=None, include_commands=True):
    """Return (summary, per-tick columns), without selecting or changing rules.

    Fixed-final geometry is an unordered diagnostic; it never changes the
    recorded native outcome.  Missing native open-hand pose remains unknown.
    """
    from evaluation.problems import PROBLEM_REGISTRY
    from visuotactile.success_audit import keypoint_distance
    from visuotactile.kinematics import EncoderKinematics
    rules = read(RULES) if rules is None else rules
    tr = group['transition']; missing = []
    arr = lambda k: _array(tr, k, missing)
    obj = arr('diagnostics_after/object_pose_w')
    n = len(tr['step_id']); ticks = np.arange(n)
    aftergoals = arr('diagnostics_after/successes')
    beforegoals = arr('diagnostics_before/successes')
    counter = arr('native_success_diagnostic/near_goal_steps_after_goal_reset')
    key = arr('native_keypoints_max_dist')
    native_near = arr('native_success_diagnostic/near_goal')
    subgoal = first(aftergoals >= 1) if aftergoals is not None else None
    pre = ticks < (subgoal if subgoal is not None else n)
    gate = None if key is None else (key*1000 <= rules['native_gate_mm']) & pre
    gate_runs = runs(gate) if gate is not None else None
    entry = first(gate) if gate is not None else None
    gate_exits = ([r['end']+1 for r in gate_runs if r['end']+1 < n and pre[r['end']+1]]
                  if gate_runs is not None else None)
    palm = arr('diagnostics_after/palm_body_pose_w')
    base = arr('diagnostics_after/base_pose_w')
    hole = arr('diagnostics_after/hole_pose_w')
    fingers = arr('diagnostics_after/finger_position_w')
    force = arr('object_contact_force_w')
    force_norm = np.linalg.norm(force, axis=-1) if force is not None else None
    min_contact = (force_norm > rules['contact_loss']['all_object_force_below_N']).sum(-1).min(-1) if force is not None else None
    stable = min_contact >= rules['descent_observation']['stable_min_contact_fingers_both_substeps'] if min_contact is not None else None
    first_stable = first(stable & pre) if stable is not None else None
    postheld = pre & (ticks > first_stable) if first_stable is not None else np.zeros(n, bool)
    postgate = pre & (ticks >= entry) if entry is not None else np.zeros(n, bool)
    ft = np.linalg.norm(fingers-obj[:, None, :3], axis=-1).mean(-1)*1000 if fingers is not None and obj is not None else None
    final_key = None
    if obj is not None and hole is not None:
        cfg = metadata['config']; problem = PROBLEM_REGISTRY[cfg['precise_assembly']['problem']]
        rel = np.asarray(problem.final_insert_pose_rel_receptive, dtype=float)
        rel = np.r_[rel[:3], rel[6], rel[3:6]]
        final = np.concatenate((hole[:, :3]+rotate(hole[:, 3:], np.broadcast_to(rel[:3], (n, 3))),
                 qmul(hole[:, 3:], np.broadcast_to(rel[3:], (n, 4)))), axis=-1)
        final_key = keypoint_distance(obj.astype(np.float32), final.astype(np.float32), cfg['reward'])*1000
    withdrawn = ((final_key <= rules['withdrawn_final_geometry']['keypoint_tolerance_mm']) &
                 (ft > rules['withdrawn_final_geometry']['mean_fingertip_distance_mm']) & pre &
                 (counter < rules['withdrawn_final_geometry']['stage_counter_still_below'])) if final_key is not None and ft is not None and counter is not None else None
    premature = event(withdrawn, rules['withdrawn_final_geometry']['sustain_ticks'])
    flags = dict(never15mm=bool(entry is None) if gate is not None else None,
                 entered_left_before10=bool(gate_exits) if gate_exits is not None else None,
                 premature_withdrawn_fixed_final10=(premature['onset_tick'] is not None) if premature['known'] else None)
    category = ('premature_withdrawn_fixed_final10' if flags['premature_withdrawn_fixed_final10'] else
                'never15mm' if flags['never15mm'] else
                'entered_left_before10' if flags['entered_left_before10'] else
                'other' if all(v is not None for v in flags.values()) else 'unknown')
    w = rules['descent_observation']['window_ticks']
    objdrop = drop_over_window(obj[:, 2], w) if obj is not None else None
    palmdrop = drop_over_window(palm[:, 2], w) if palm is not None else None
    events = dict(fixed_final_withdrawn10=premature)
    columns = dict(tick=ticks, first_phase_before_confirmation=pre.astype(int),
                   native_subgoals_after=aftergoals, native_near_counter_after_reset=counter,
                   native_keypoint_mm=key*1000 if key is not None else None,
                   native_near_recorded=native_near, gate15mm_before_confirmation=gate,
                   fixed_final_keypoint_mm=final_key, fingertip_mean_mm=ft,
                   fixed_final_withdrawn_before_confirmation=withdrawn,
                   object_contact_min_fingers_both_substeps=min_contact,
                   object_contact_peak_N=force_norm.max(axis=(1,2)) if force is not None else None,
                   object_down_over12_mm=objdrop, palm_down_over12_mm=palmdrop)
    for name, values, threshold in [('object_down12', objdrop, rules['descent_observation']['object_down_mm']),
                                    ('palm_down12', palmdrop, rules['descent_observation']['palm_down_mm'])]:
        mask = values >= threshold if values is not None else None
        events[name+'_firstphase_after_stable_contact'] = event(mask & postheld if mask is not None else None)
        events[name+'_after_gate'] = event(mask & postgate if mask is not None else None)
    loss = (force_norm < rules['contact_loss']['all_object_force_below_N']).all(axis=(1,2)) if force is not None else None
    events['contact_loss_after_stable_contact'] = event(loss & postheld if loss is not None else None, rules['contact_loss']['sustain_ticks'])
    events['contact_loss_after_gate'] = event(loss & postgate if loss is not None else None, rules['contact_loss']['sustain_ticks'])
    columns['all_object_contact_loss_both_substeps'] = loss
    # Freeze the reference immediately before first gate.  If there is no gate,
    # use the closest first-phase approach only as a separately labelled anchor.
    anchor = entry if entry is not None else (int(np.argmin(np.where(pre, key, np.inf))) if key is not None and pre.any() else None)
    ref_candidates = np.flatnonzero(stable & (ticks < anchor)) if stable is not None and anchor is not None else []
    ref = int(ref_candidates[-1]) if len(ref_candidates) else None
    ref_scope = 'latest_stable_contact_before_first_gate' if entry is not None else 'latest_stable_contact_before_closest_firstphase_approach_no_gate'
    if ref is not None and obj is not None and palm is not None:
        op = relative(palm, obj)
        drift = np.linalg.norm(op[:, :3]-op[ref, :3], axis=-1)*1000
        rot = angle(op[:, 3:], np.broadcast_to(op[ref, 3:], (n,4)))
        relmask = ((drift >= rules['slip']['translation_mm']) | (rot >= rules['slip']['angle_deg'])) & pre & (ticks > ref)
        events['object_palm_relative_motion_from_stable_reference'] = event(relmask, rules['slip']['sustain_ticks'])
        columns['object_palm_reference_translation_mm'] = drift
        columns['object_palm_reference_angle_deg'] = rot
    else:
        events['object_palm_relative_motion_from_stable_reference'] = event(None)
    fk_receipt = None
    # This repository has no explicit task-native open-hand joint pose.  Zero
    # reset pose must never silently become an open-hand label.
    events['native_open_command'] = dict(known=False, onset_tick=None, confirmation_tick=None,
        runs=None, reason='No explicit task open-hand joint target was identified; reset zero is not assumed open')
    if include_commands:
        target = arr('target_after_rad'); qactual = arr('diagnostics_after/joint_position_rad')
        if target is not None:
            for joint in range(7, 29):
                columns[f'final_command_hand_q{joint}_rad'] = target[:,joint]
            if ref is not None:
                columns['final_command_hand_distance_from_reference_rad'] = np.linalg.norm(target[:,7:]-target[ref,7:],axis=-1)
        if qactual is not None:
            for joint in range(7,29):
                columns[f'actual_hand_q{joint}_rad'] = qactual[:,joint]
        if target is not None and base is not None:
            urdf = Path(metadata['config']['assets']['robot_urdf'])
            if not urdf.exists():
                urdf = PROJECT/'assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf'
            fk = EncoderKinematics(urdf, metadata['joint_order'], links=('iiwa14_link_7',))
            if fk.sha256 != metadata['fk_urdf_sha256'] or fk.root != metadata['fk_root']:
                raise ValueError('Recorded robot FK identity differs')
            with torch.no_grad():
                pos, _ = fk(torch.as_tensor(target, dtype=torch.float32))
                commandp = base[:, :3]+rotate(base[:, 3:], pos[:,0].numpy().astype(float))
                actualp = None
                if qactual is not None and palm is not None:
                    ap, _ = fk(torch.as_tensor(qactual, dtype=torch.float32))
                    actualp = base[:, :3]+rotate(base[:, 3:], ap[:,0].numpy().astype(float))
            cdrop = drop_over_window(commandp[:,2], w)
            columns['fk_final_target_palm_z_m'] = commandp[:,2]
            columns['fk_final_target_palm_down_over12_mm'] = cdrop
            events['command_palm_down12_after_gate'] = event((cdrop >= rules['descent_observation']['palm_down_mm']) & postgate)
            events['command_palm_down12_after_stable_contact'] = event((cdrop >= rules['descent_observation']['palm_down_mm']) & postheld)
            fk_receipt = dict(robot_urdf=str(urdf), sha256=fk.sha256, root=fk.root, palm_link='iiwa14_link_7',
                command_source='transition/target_after_rad; absolute final actuator target',
                actual_fk_palm_max_error_mm=float(np.linalg.norm(actualp-palm[:,:3],axis=-1).max()*1000) if actualp is not None else None)
        else:
            events['command_palm_down12_after_gate'] = event(None)
            events['command_palm_down12_after_stable_contact'] = event(None)
    success = result.get('success')
    outcome = ('complete_success' if success else 'first_subgoal_without_completion' if subgoal is not None else 'no_first_subgoal') if success is not None and aftergoals is not None else 'unknown'
    control_source = tr['dagger/control_source'][:] if 'dagger/control_source' in tr else None
    legal_expert = tr['dagger/expert_label_valid'][:] if 'dagger/expert_label_valid' in tr else None
    segment = tr['dagger/segment_id'][:] if 'dagger/segment_id' in tr else None
    columns['dagger_control_source_recorded'] = control_source
    columns['dagger_legal_expert_label_recorded'] = legal_expert
    columns['dagger_segment_id_recorded'] = segment
    for ev in events.values():
        for label in ('onset_tick','confirmation_tick'):
            t=ev[label]
            ev[label.replace('_tick','_context')] = (dict(
                native_keypoint_mm=float(key[t]*1000) if key is not None else None,
                recorded_native_counter_after_reset=int(counter[t]) if counter is not None else None,
                object_down_over12_mm=float(objdrop[t]) if objdrop is not None and np.isfinite(objdrop[t]) else None,
                palm_down_over12_mm=float(palmdrop[t]) if palmdrop is not None and np.isfinite(palmdrop[t]) else None,
                contact_min_fingers_both_substeps=int(min_contact[t]) if min_contact is not None else None,
                fixed_final_keypoint_mm=float(final_key[t]) if final_key is not None else None,
                fingertip_mean_mm=float(ft[t]) if ft is not None else None,
                dagger_control_source_recorded=int(control_source[t]) if control_source is not None else None,
                dagger_legal_expert_label_recorded=bool(legal_expert[t]) if legal_expert is not None else None,
                dagger_segment_id_recorded=int(segment[t]) if segment is not None else None) if t is not None else None)
    summary = dict(ticks=n, native_outcome=outcome, native_success=success,
        software_complete=result.get('software_complete'), legal=result.get('legal'),
        native_first_subgoal_tick=subgoal, native_max_subgoals=int(aftergoals.max()) if aftergoals is not None else None,
        native_counter_max_before_confirmation=int(counter[pre].max()) if counter is not None and pre.any() else None,
        native_force_consecutive=metadata['config']['termination']['force_consecutive_near_goal_steps'],
        native_success_steps=metadata['config']['termination']['success_steps'],
        gate_first_entry_tick=entry, gate_first_exit_before_confirmation_tick=gate_exits[0] if gate_exits else None,
        gate_runs_before_confirmation=gate_runs, gate_exits_before_confirmation=gate_exits,
        first_stable_contact_tick=first_stable, stable_reference_tick=ref, stable_reference_scope=ref_scope,
        exclusive_category=category, overlap_flags=flags, events=events, fk=fk_receipt,
        ordered_event_onsets=sorted([dict(event=k,onset_tick=e['onset_tick'],confirmation_tick=e['confirmation_tick'])
                                  for k,e in events.items() if e['onset_tick'] is not None],key=lambda x:(x['onset_tick'],x['event'])),
        trajectory_kind=metadata.get('trajectory_kind','expert_demonstration' if control_source is None and result.get('split')=='train' else 'autonomous'),
        missing_fields=missing, tick_index='zero-based completed control transition; event confirmation included',
        physical_interpretation='Relative motion and descent are observable events, not proof of accidental slip, expert recoverability, or causal policy defect')
    return summary, {k:v for k,v in columns.items() if v is not None}


def discover():
    ma = read(STATE/'A/state_data/data.json'); mb = read(STATE/'B/state_data/data.json')
    assert ma['splits']['train'] == mb['splits']['train'] and len(ma['splits']['train']) == 180
    jobs=[]; identities=set()
    for identity in ma['splits']['train']:
        a,b = ma['entries'][identity], mb['entries'][identity]
        assert (a['path'],a['episode'],a['source_sha256']) == (b['path'],b['episode'],b['source_sha256'])
        key=(str(Path(a['path']).resolve()),a['episode'])
        assert key not in identities; identities.add(key)
        jobs.append(dict(cohort='train',policy='expert_or_takeover',identity=identity,path=a['path'],
                         group='episodes/'+a['episode'],expected_sha256=a['source_sha256'],family=a['family']))
    for wave in sorted(DEV.glob('wave[0-9][0-9]')):
        assert read(wave/'paired_initial_check.json')['passed']
        for p in sorted(wave.iterdir()):
            if not p.is_dir() or not (p/'numeric.h5').exists(): continue
            policy=next((v for v in ('A_130_50','B_130_50','A_R2','B_R2','A_entry','B_entry') if p.name.startswith(v+'_')),None)
            assert policy is not None, p
            result=read(p/'result.json')
            jobs.append(dict(cohort='development',policy=policy,identity=f'{policy}_{result["seed"]}',path=str(p/'numeric.h5'),
                             group='episode',result_path=str(p/'result.json'),family=result['family'],seed=result['seed'],wave=wave.name))
    assert len(jobs)==276 and sum(j['cohort']=='development' for j in jobs)==96
    return jobs


def summarize(rows):
    out={}
    pools = {'all':rows,'train':[r for r in rows if r['cohort']=='train'],
             'development':[r for r in rows if r['cohort']=='development']}
    for policy in ('A_130_50','B_130_50','A_R2','B_R2','A_entry','B_entry'):
        pools['development/'+policy]=[r for r in rows if r['cohort']=='development' and r['policy']==policy]
    for name, pool in pools.items():
        flags={key:dict(true=sum(r['overlap_flags'][key] is True for r in pool),
                        false=sum(r['overlap_flags'][key] is False for r in pool),
                        unknown=sum(r['overlap_flags'][key] is None for r in pool)) for key in rows[0]['overlap_flags']}
        events={}
        for key in sorted(set(k for r in pool for k in r['events'])):
            ee=[r for r in pool if key in r['events']]
            successes=[r for r in ee if r['native_success'] is True]
            events[key]=dict(known=sum(r['events'][key]['known'] for r in ee),
                detected=sum(r['events'][key]['onset_tick'] is not None for r in ee),
                native_success_denominator=len(successes),
                detected_in_eventual_native_success=sum(r['events'][key]['onset_tick'] is not None for r in successes))
        out[name]=dict(n=len(pool),outcomes=dict(Counter(r['native_outcome'] for r in pool)),
                      exclusive=dict(Counter(r['exclusive_category'] for r in pool)),overlap_flags=flags,events=events,
                      missing_field_trajectories=sum(bool(r['missing_fields']) for r in pool))
    return out


def export_rows(rows, path):
    flat=[]
    for r in rows:
        one={k:r.get(k) for k in ('cohort','policy','identity','seed','family','wave','ticks','native_outcome','native_success','software_complete','legal',
              'native_first_subgoal_tick','native_max_subgoals','native_counter_max_before_confirmation','gate_first_entry_tick',
              'gate_first_exit_before_confirmation_tick','first_stable_contact_tick','stable_reference_tick','exclusive_category')}
        one.update(r['overlap_flags'])
        for name,e in r['events'].items():
            one[name+'_known']=e['known']; one[name+'_onset_tick']=e['onset_tick']; one[name+'_confirmation_tick']=e['confirmation_tick']
        flat.append(one)
    keys=list(dict.fromkeys(k for r in flat for k in r))
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(flat)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--quick',action='store_true'); parser.add_argument('--no-hashes',action='store_true')
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True); PUBLIC.mkdir(parents=True,exist_ok=True)
    rules=read(RULES); jobs=discover(); rows=[]; hashes=[]; total_ticks=0
    write(OUT/'source_registration.json',dict(scope='train180 raw deduplicated between A/B plus development96 only; excludes all terminal trajectories',
        rules_sha256=sha(RULES),jobs=jobs,classification_priority=['premature_withdrawn_fixed_final10','never15mm','entered_left_before10','other'],
        native_open_command='unknown: no explicit native task open-hand target found',
        stable_reference='latest >=2 contact-finger (both physics substeps) frame before first 15mm gate; no-gate cases labelled closest-approach reference'))
    trace_path=OUT/('quick_ticks.csv.gz' if args.quick else 'derived_ticks.csv.gz')
    tracefile=None; tracewriter=None
    if not args.quick: tracefile=gzip.open(trace_path,'wt',newline='')
    try:
        for i,j in enumerate(jobs):
            p=Path(j['path'])
            with h5py.File(p,'r') as f:
                metadata=json.loads(f.attrs['metadata_json'])
                result=read(j['result_path']) if j.get('result_path') else json.loads(f.attrs['result_json'])
                row, columns=analyze(f[j['group']],metadata,result,rules,include_commands=not args.quick)
            row.update({k:v for k,v in j.items() if k not in ('expected_sha256',)})
            rows.append(row);total_ticks+=row['ticks']
            if tracefile is not None:
                if tracewriter is None:
                    names=['cohort','policy','identity']+list(columns)
                    # Optional columns are represented by blank cells, never false.
                    names+=['object_palm_reference_translation_mm','object_palm_reference_angle_deg','fk_final_target_palm_z_m','fk_final_target_palm_down_over12_mm',
                            'final_command_hand_distance_from_reference_rad','dagger_control_source_recorded','dagger_legal_expert_label_recorded','dagger_segment_id_recorded']
                    names=list(dict.fromkeys(names));tracewriter=csv.DictWriter(tracefile,fieldnames=names);tracewriter.writeheader()
                for tick in range(row['ticks']):
                    values={k:(v[tick].item() if isinstance(v[tick],np.generic) else v[tick]) for k,v in columns.items()}
                    values.update(cohort=j['cohort'],policy=j['policy'],identity=j['identity']);tracewriter.writerow(values)
            if not args.quick and not args.no_hashes:
                actual=sha(p); expected=j.get('expected_sha256')
                if expected is not None and actual!=expected:raise ValueError(f'Raw source SHA mismatch: {p}')
                hashes.append(dict(path=str(p),sha256=actual,bytes=p.stat().st_size,expected_sha256=expected,expected_match=(actual==expected) if expected else None))
                if j.get('result_path'):
                    rp=Path(j['result_path']);hashes.append(dict(path=str(rp),sha256=sha(rp),bytes=rp.stat().st_size))
            if i%30==29: print(json.dumps(dict(progress=i+1,total=len(jobs),ticks=total_ticks)),flush=True)
    finally:
        if tracefile:tracefile.close()
    tag='quick_' if args.quick else ''
    write(OUT/(tag+'trajectories.json'),rows); write(OUT/(tag+'summary.json'),summarize(rows));export_rows(rows,OUT/(tag+'trajectories.csv'))
    write(OUT/(tag+'receipt.json'),dict(trajectories=len(rows),raw_train=sum(r['cohort']=='train' for r in rows),development=sum(r['cohort']=='development' for r in rows),ticks=total_ticks,
        terminal_trajectories_read=0,new_simulation=0,new_policy_inference=0,optimizer_updates=0,new_dagger=0,script_sha256=sha(__file__),rules_sha256=sha(RULES)))
    if not args.quick:
        write(OUT/'source_sha256.json',hashes)
        for name in ('source_registration.json','trajectories.json','summary.json','trajectories.csv','receipt.json','source_sha256.json','derived_ticks.csv.gz'):
            shutil.copy2(OUT/name,PUBLIC/name)
    print(json.dumps(summarize(rows),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
