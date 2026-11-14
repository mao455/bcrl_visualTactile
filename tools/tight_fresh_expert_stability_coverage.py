"""CPU-only coverage of the frozen grasp-window rule in admitted train successes."""
import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import h5py
import numpy as np
import torch

from tools.tight_fresh_takeover_pose_audit import window_components
from visuotactile.compact import sha256
from visuotactile.takeover_trigger import stable_windows
from visuotactile.training_resources import atomic_json

VERSION = 'train_expert_frozen_stability_coverage_v1'
CONDITIONS = ('two_contact_fingers_every_substep', 'height', 'translation', 'rotation')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def first(mask):
    indices = np.flatnonzero(mask)
    return int(indices[0]+1) if len(indices) else None


def isolated_windows(mask, segment):
    """Window endpoints with 12 actual transitions wholly inside one phase/segment."""
    mask, segment = np.asarray(mask), np.asarray(segment)
    require(mask.ndim == 1 and mask.dtype == np.bool_ and segment.shape == mask.shape,
        'Aligned Boolean phase and recorded segment required')
    full = np.zeros(len(mask), dtype=bool)
    for end in range(11, len(mask)):
        part = slice(end-11, end+1)
        full[end] = bool(mask[part].all() and (segment[part] == segment[end]).all())
    # The height reference is available only after completed transition48.
    eligible = full & (np.arange(len(mask)) >= 47)
    return eligible, dict(phase_ticks=int(mask.sum()),
        no_complete_same_phase_segment_window=int((mask & ~full).sum()),
        full_window_before_height_reference=int((mask & full & ~eligible).sum()),
        eligible_window_endpoints=int(eligible.sum()))


def angle(a, b):
    return 2*np.arccos(np.clip(np.abs((a*b).sum(-1)), 0., 1.))


def snapshot(index, arrays, stable, components, eligible):
    """Compare transition-after object with the goal BEFORE its native update."""
    obj, palm, goal = arrays['object'][index], arrays['palm'][index], arrays['goal_before'][index]
    component = components[index]
    return dict(completed_tick=index+1, stable_window=bool(stable[index]),
        eligible_same_phase_window=bool(eligible[index]),
        failed_conditions=[key for key in CONDITIONS if not component['conditions'][key]],
        palm_relative_root_max_translation_m=component['palm_relative_root_max_translation_m'],
        palm_relative_max_rotation_rad=component['palm_relative_max_rotation_rad'],
        height_margin_m=component['height_margin_m'],
        object_to_active_goal_root_distance_m=float(np.linalg.norm(obj[:3]-goal[:3])),
        object_to_active_goal_quaternion_angle_rad=float(angle(obj[3:], goal[3:])),
        object_to_palm_root_distance_m=float(np.linalg.norm(obj[:3]-palm[:3])),
        native_near_goal=bool(arrays['near'][index]), native_alignment=bool(arrays['align'][index]),
        subgoals_before=int(arrays['subgoals_before'][index]),
        subgoals_after=int(arrays['subgoals_after'][index]),
        goal_changed_during_transition=not np.array_equal(goal, arrays['goal_after'][index]))


def phase_summary(mask, segment, arrays, stable, components):
    eligible, denominator = isolated_windows(mask, segment)
    false = eligible & ~stable
    failures = {key:np.array([False if x is None else not x['conditions'][key] for x in components])
        for key in CONDITIONS}
    other_motion = false & ~failures['two_contact_fingers_every_substep'] & ~failures['height']
    stable_phase = eligible & stable
    first_goal = arrays['subgoals_before'] == 0
    before_near = first_goal & (np.cumsum(arrays['near'] & first_goal) == 0)
    contact_pass = ~failures['two_contact_fingers_every_substep']
    height_pass = ~failures['height']
    native_lift_window = np.array([i >= 11 and arrays['lifted'][i-11:i+1].all()
        for i in range(len(mask))])
    result = dict(denominator, stable_endpoints=int(stable_phase.sum()),
        false_endpoints=int(false.sum()), first_stable_completed_tick=first(stable_phase),
        last_stable_completed_tick=int(np.flatnonzero(stable_phase)[-1]+1) if stable_phase.any() else None,
        raw_full_history_stable_phase_endpoints=int((mask & stable).sum()),
        rejected_boundary_or_early_stable_endpoints=int((mask & stable & ~eligible).sum()),
        false_conditions_overlapping={key:int((false & value).sum()) for key,value in failures.items()},
        false_relative_motion_only_endpoints=int(other_motion.sum()),
        false_contact_and_height_pass_rotation_only_endpoints=int((other_motion & ~failures['translation'] & failures['rotation']).sum()),
        false_current_two_contacts_both_substeps=int((false & arrays['two']).sum()),
        false_with_native_lift_flag=int((false & arrays['lifted']).sum()),
        native_termination_flags={key:int((mask & value).sum()) for key,value in arrays['terminations'].items()},
        events={}, events_by_preaction_subgoals={}, prospective_screens={})
    for name, screen in dict(full_frozen_stable=stable,
            contact_height12=contact_pass & height_pass,
            contact_native_lift12=contact_pass & native_lift_window).items():
        selected = eligible & before_near & screen
        result['prospective_screens'][name] = dict(window_endpoints=int(selected.sum()),
            first_completed_tick=first(selected), additional_false_stable_endpoints=int((selected & ~stable).sum()))
    for label, event in dict(near_goal=arrays['near'], alignment=arrays['align'],
            subgoal_transition=arrays['subgoals_after'] > arrays['subgoals_before'],
            native_complete=arrays['complete']).items():
        selected = mask & event
        result['events'][label] = dict(ticks=int(selected.sum()),
            first_completed_tick=first(selected), eligible_ticks=int((selected & eligible).sum()),
            stable_ticks=int((selected & stable_phase).sum()),
            false_eligible_ticks=int((selected & false).sum()),
            excluded_boundary_or_early_ticks=int((selected & ~eligible).sum()))
        for stage in sorted(set(arrays['subgoals_before'].tolist())):
            stage_selected = selected & (arrays['subgoals_before'] == stage)
            result['events_by_preaction_subgoals'].setdefault(str(stage), {})[label] = dict(
                ticks=int(stage_selected.sum()), eligible_ticks=int((stage_selected & eligible).sum()),
                stable_ticks=int((stage_selected & stable_phase).sum()),
                false_eligible_ticks=int((stage_selected & false).sum()),
                excluded_boundary_or_early_ticks=int((stage_selected & ~eligible).sum()))
    start = np.flatnonzero(stable_phase & (arrays['subgoals_before'] == 0))
    approach = dict(available=False, reason='no_same_phase_first_goal_stable_window')
    if len(start):
        begin = int(start[0]); stop = np.flatnonzero(mask & arrays['near'] &
            (arrays['subgoals_before'] == 0) & (np.arange(len(mask)) >= begin) & (segment == segment[begin]))
        if len(stop):
            end = int(stop[0]); selected = (eligible & (segment == segment[begin]) &
                (np.arange(len(mask)) >= begin) & (np.arange(len(mask)) <= end) & (arrays['subgoals_before'] == 0))
            losses = np.flatnonzero(selected & ~stable)
            approach = dict(available=True, start_completed_tick=begin+1, near_goal_completed_tick=end+1,
                eligible_endpoints=int(selected.sum()), stable_endpoints=int((selected & stable).sum()),
                false_endpoints=len(losses), left_stable_window=bool(len(losses)),
                first_loss=snapshot(int(losses[0]), arrays, stable, components, eligible) if len(losses) else None,
                start=snapshot(begin, arrays, stable, components, eligible),
                near_goal=snapshot(end, arrays, stable, components, eligible),
                object_world_rotation_between_endpoints_rad=float(angle(arrays['object'][begin,3:],arrays['object'][end,3:])),
                palm_world_rotation_between_endpoints_rad=float(angle(arrays['palm'][begin,3:],arrays['palm'][end,3:])),
                false_relative_motion_only_endpoints=int((selected & other_motion).sum()))
        else:
            approach = dict(available=False, reason='no_later_same_segment_first_goal_near_event', start_completed_tick=begin+1)
    result['first_stable_to_first_later_near_goal'] = approach
    # All later endpoint changes are descriptive; no inferred intentionality or drop label.
    if stable_phase.any():
        later = eligible & (np.arange(len(mask)) > np.flatnonzero(stable_phase)[0])
        result['any_later_false_window'] = bool((later & ~stable).any())
    else:
        result['any_later_false_window'] = False
    return result


def load_episode(entry, identity, original, record, *, predicate=stable_windows, explain=window_components):
    path = Path(entry['path']); stat = path.stat()
    with h5py.File(path, 'r') as stream:
        metadata = json.loads(stream.attrs['metadata_json']); result = json.loads(stream.attrs['result_json'])
        group = stream['episodes'][entry['episode']]; transition = group['transition']
        require(metadata['initial_family'] == entry['family'] and metadata['dataset_split'] == 'train'
            and result['legal'] is True and result['success'] is True and bool(group.attrs['complete']),
            'Admitted successful training episode required before numeric payload access')
        paths = dict(object='diagnostics_after/object_pose_w', palm='diagnostics_after/palm_body_pose_w',
            object_before='diagnostics_before/object_pose_w', palm_before='diagnostics_before/palm_body_pose_w',
            goal_before='diagnostics_before/goal_pose_w', goal_after='diagnostics_after/goal_pose_w',
            subgoals_before='diagnostics_before/successes', subgoals_after='diagnostics_after/successes',
            near='native_success_diagnostic/near_goal', align='native_alignment_condition',
            complete='native_success_diagnostic/complete', lifted='diagnostics_after/lifted_object',
            force='object_contact_force_w', time_before='time_before_s', time_after='time_after_s')
        arrays = {key:transition[value][:] for key,value in paths.items()}
        n = entry['length']; require(n >= 48 and result['steps'] == n and len(arrays['object']) == n, 'Full episode length differs')
        require(arrays['force'].shape == (n,2,5,3) and arrays['object'].shape == arrays['palm'].shape == (n,7)
            and all(np.isfinite(value).all() for value in arrays.values()), 'Actual finite poses/two-substep object contacts required')
        require(np.array_equal(group['obs/time_s'][:],arrays['time_before']) and
            np.array_equal(record['time'].numpy(),arrays['time_before']) and
            np.array_equal(arrays['time_after'][:-1],arrays['time_before'][1:]) and
            np.allclose(arrays['time_after']-arrays['time_before'],1/60,atol=1e-9,rtol=0) and
            np.array_equal(transition['step_id'][:],np.arange(n)), 'Native transition/observation clock differs')
        for after,before in [('object','object_before'),('palm','palm_before'),('goal_after','goal_before'),('subgoals_after','subgoals_before')]:
            require(np.array_equal(arrays[after][:-1],arrays[before][1:]), 'Adjacent actual state continuity differs: '+after)
        episode = transition['episode_id'][:]
        require((episode == episode[0]).all() and not (group['terminated'][:-1] | group['truncated'][:-1]).any(), 'Episode boundary inside trajectory')
        arrays['terminations'] = {key:value[:].astype(bool) for key,value in transition['termination_reasons'].items()}
        require('dagger' not in transition if original else 'dagger' in transition, 'Original/correction provenance differs')
        if original:
            require(result.get('dagger') is None and group.attrs['action_source'] == 'frozen_expert_slow_execution', 'Original expert authority differs')
            segment = np.zeros(n,dtype=np.int64); phases = dict(original_expert=np.ones(n,dtype=bool)); authority = {}
            require(record['supervision_mask'].all() and np.array_equal(record['control_segment'].numpy(),segment), 'Original legal mask differs')
        else:
            require(entry['trajectory_kind'] == 'expert_takeover_v1' and result['dagger']['expert_takeover_success'] is True, 'Successful actual expert takeover required')
            authority = {key:transition['dagger/'+key][:] for key in ['control_source','expert_label_valid','switch_clean','segment_id']}
            source, valid, clean, segment = [authority[key] for key in ['control_source','expert_label_valid','switch_clean','segment_id']]
            require(source.shape == valid.shape == clean.shape == segment.shape == (n,) and np.isin(source,[0,1]).all()
                and np.array_equal(source==0,np.arange(n)<(source==0).sum()) and valid.dtype==clean.dtype==np.bool_
                and not (valid & ((source!=1) | ~clean)).any() and (np.diff(segment)>=0).all(), 'Recorded authority/legality differs')
            changes = (source[1:]!=source[:-1]) | (valid[1:]!=valid[:-1])
            require(not (changes & (segment[1:]==segment[:-1])).any(), 'Boundary lacks a new recorded segment')
            require(np.array_equal(record['supervision_mask'].numpy(),valid) and np.array_equal(record['control_segment'].numpy(),segment), 'Numeric action mask differs from raw expert authority')
            require(int(valid.sum()) == result['dagger']['expert_valid_ticks'] and int((source==0).sum()) == result['dagger']['student_ticks'], 'Result authority counts differ')
            phases = dict(student_prefix=source==0, expert_queue=(source==1)&~valid, legal_expert=valid)
        require(sum(int(mask.sum()) for mask in phases.values()) == n, 'Authority phases do not partition all transitions')
    require((path.stat().st_size,path.stat().st_mtime_ns)==(stat.st_size,stat.st_mtime_ns), 'Raw source changed during read')
    input_hashes = {key:digest(value) for key,value in arrays.items() if key!='terminations'}
    input_hashes.update({'authority/'+key:digest(value) for key,value in authority.items()})
    height = float(np.median(arrays['object'][24:48,2]))
    stable, contact = predicate(arrays['object'],arrays['palm'],arrays['force'],height)
    arrays['two'] = (contact.sum(-1)>=2).all(-1)
    components = [None]*11 + [explain(arrays['object'],arrays['palm'],arrays['force'],height,tick)
        for tick in range(12,n+1)]
    require(all(bool(stable[i])==row['stable_window'] for i,row in enumerate(components) if row is not None), 'Frozen predicate/component mismatch')
    summaries = {phase:phase_summary(mask,segment,arrays,stable,components) for phase,mask in phases.items()}
    switch = int(phases['student_prefix'].sum()) if not original else None
    return dict(identity=identity,family=entry['family'],kind='original' if original else 'correction',
        steps=n,success=True,round=entry.get('round_index'),rollin_role=entry.get('source_group'),
        source_manifest_sha256=entry['source_sha256'],read_numeric_fields_sha256=hashlib.sha256(json.dumps(input_hashes,sort_keys=True).encode()).hexdigest(),
        raw_size=stat.st_size,raw_mtime_ns=stat.st_mtime_ns,settled_height_m=height,
        height_reference_all_student_before_takeover=None if original else switch>=48,
        first_legal_expert_completed_tick=first(phases['legal_expert']) if not original else 1,
        exact_adjacent_before_after_continuity=True,phase=summaries)


def aggregate(rows, phase):
    selected = [row for row in rows if phase in row['phase']]; cells = [row['phase'][phase] for row in selected]
    additive = ('phase_ticks','no_complete_same_phase_segment_window','full_window_before_height_reference',
        'eligible_window_endpoints','stable_endpoints','false_endpoints','raw_full_history_stable_phase_endpoints',
        'rejected_boundary_or_early_stable_endpoints','false_relative_motion_only_endpoints',
        'false_contact_and_height_pass_rotation_only_endpoints','false_current_two_contacts_both_substeps','false_with_native_lift_flag')
    intervals = [cell['first_stable_to_first_later_near_goal'] for cell in cells if cell['first_stable_to_first_later_near_goal']['available']]
    stages = sorted(set().union(*(cell['events_by_preaction_subgoals'] for cell in cells)))
    screens = {}
    for name in ('full_frozen_stable','contact_height12','contact_native_lift12'):
        comparison = [cell['prospective_screens'][name] for cell in cells]
        screens[name] = dict(episodes_with_window=sum(v['window_endpoints']>0 for v in comparison),
            total_window_endpoints=sum(v['window_endpoints'] for v in comparison),
            additional_false_stable_endpoints=sum(v['additional_false_stable_endpoints'] for v in comparison),
            episodes_without_any_full_stable_before_near=sum(v['window_endpoints']>0 and
                cell['prospective_screens']['full_frozen_stable']['window_endpoints']==0 for v,cell in zip(comparison,cells)))
    return dict(episodes=len(cells),episodes_with_eligible_window=sum(x['eligible_window_endpoints']>0 for x in cells),
        episodes_with_stable_window=sum(x['stable_endpoints']>0 for x in cells),
        episodes_with_later_false_window=sum(x['any_later_false_window'] for x in cells),
        **{key:sum(x[key] for x in cells) for key in additive},
        false_conditions_overlapping={key:sum(x['false_conditions_overlapping'][key] for x in cells) for key in CONDITIONS},
        events={key:{metric:sum(x['events'][key][metric] for x in cells) for metric in
            ('ticks','eligible_ticks','stable_ticks','false_eligible_ticks','excluded_boundary_or_early_ticks')}
            for key in ('near_goal','alignment','subgoal_transition','native_complete')},
        events_by_preaction_subgoals={stage:{event:{metric:sum(
            cell['events_by_preaction_subgoals'].get(stage,{}).get(event,{}).get(metric,0) for cell in cells)
            for metric in ('ticks','eligible_ticks','stable_ticks','false_eligible_ticks','excluded_boundary_or_early_ticks')}
            for event in ('near_goal','alignment','subgoal_transition','native_complete')} for stage in stages},
        prospective_screens=screens,
        first_stable_to_first_later_near_goal=dict(available_episodes=len(intervals),
            unavailable_reasons=dict(Counter(x['first_stable_to_first_later_near_goal'].get('reason') for x in cells if not x['first_stable_to_first_later_near_goal']['available'])),
            episodes_leaving_stable_window=sum(x['left_stable_window'] for x in intervals),
            first_loss_conditions=dict(Counter(key for x in intervals if x['first_loss'] for key in x['first_loss']['failed_conditions'])),
            **{key:sum(x[key] for x in intervals) for key in ('eligible_endpoints','stable_endpoints','false_endpoints','false_relative_motion_only_endpoints')}),
        native_termination_flags={key:sum(x['native_termination_flags'][key] for x in cells)
            for key in sorted(set().union(*(x['native_termination_flags'] for x in cells)))} if cells else {})


def compact_case(row):
    """Retain per-episode denominators/evidence, with pooled detail in summary."""
    row=dict(row); row['phase']=dict(row['phase'])
    for name,cell in row['phase'].items():
        keys=('phase_ticks','no_complete_same_phase_segment_window','full_window_before_height_reference',
            'eligible_window_endpoints','stable_endpoints','first_stable_completed_tick',
            'last_stable_completed_tick','rejected_boundary_or_early_stable_endpoints')
        value={key:cell[key] for key in keys}
        if name!='expert_queue':
            value['prospective_screens']=cell['prospective_screens']
        if name in ('original_expert','legal_expert'):
            value['first_goal_events']=cell['events_by_preaction_subgoals']['0']
            interval=dict(cell['first_stable_to_first_later_near_goal'])
            if interval['available']:
                interval.pop('start');interval.pop('near_goal')
            value['first_stable_to_first_later_near_goal']=interval
        row['phase'][name]=value
    return row


def frozen_functions(execution_source, registered):
    """Load the recorded immutable predicate and explanation without changing files."""
    modules={};bindings={}
    for name,relative in [('trigger','visuotactile/takeover_trigger.py'),
            ('explanation','tools/tight_fresh_takeover_pose_audit.py')]:
        path=Path(execution_source).resolve()/relative
        require(sha256(path)==registered['source_sha256'][relative], 'Frozen diagnostic dependency changed: '+relative)
        spec=importlib.util.spec_from_file_location('frozen_stability_'+name,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        modules[name]=module;bindings[relative]=dict(path=str(path),sha256=sha256(path))
    # The explanation's imported dependency is bound to this same immutable function.
    modules['explanation'].stable_windows=modules['trigger'].stable_windows
    return modules['trigger'].stable_windows,modules['explanation'].window_components,bindings


def run(source, output, registration, *, execution_source):
    began = time.monotonic(); source=Path(source).resolve(); output=Path(output).absolute()
    if output.exists() or output.is_symlink():raise FileExistsError('Preserve existing coverage reports')
    torch.set_num_threads(2); registered=read(registration); project=Path(__file__).resolve().parents[1]
    require(str(source)==registered['source_root'], 'Frozen source root differs')
    predicate,explain,frozen=frozen_functions(execution_source,registered)
    for relative in ('A/state_data/data.json','B/state_data/data.json','A/state_data/numeric.pt'):
        path=source/relative
        require(sha256(path)==registered['protected'][str(path)], 'Registered source bytes changed: '+relative)
    data={role:read(source/role/'state_data/data.json') for role in ('A','B')}
    require(data['A']['entries']==data['B']['entries'] and data['A']['splits']==data['B']['splits'] and
        data['A']['normalization_train_ids']==data['B']['normalization_train_ids'], 'Matched actual train population required')
    d=data['A']; ids=d['splits']['train']; original=set(d['normalization_train_ids'])
    require(len(ids)==len(set(ids))==84 and len(original)==30 and original<=set(ids) and
        len({d['entries'][i]['family'] for i in original})==30 and not d['splits']['test'], 'Expected current84/original30 train population')
    require(d['splits']==registered['training_population'] and d['normalization_train_ids']==registered['normalization_train_ids'], 'Registered current training population differs')
    require(not (set(ids)&set(d['splits']['validation'])), 'Training and validation identities overlap')
    numeric_path=source/'A/state_data/numeric.pt'; numeric=torch.load(numeric_path,map_location='cpu',weights_only=False,mmap=True)
    require(numeric['data_sha256']==sha256(source/'A/state_data/data.json'), 'Numeric manifest binding differs')
    rows=[load_episode(d['entries'][i],i,i in original,numeric['records'][i],predicate=predicate,explain=explain) for i in ids]
    require(not torch.cuda.is_initialized(), 'CPU diagnostic unexpectedly initialized CUDA')
    result=dict(version=VERSION,complete=True,source_root=str(source),train_episodes=84,original_successes=30,legal_expert_suffixes=54,
        registration_sha256=sha256(registration),data_sha256={role:sha256(source/role/'state_data/data.json') for role in ('A','B')},
        numeric_sha256=sha256(numeric_path),summary={phase:aggregate(rows,phase) for phase in
            ('original_expert','legal_expert','student_prefix','expert_queue')},cases=[compact_case(row) for row in rows],
        boundary_audit=dict(exact_adjacent_physical_goal_state_continuity_episodes=len(rows),
            correction_reference_available_entirely_before_switch=sum(row['height_reference_all_student_before_takeover'] is True for row in rows),
            correction_reference_spans_expert_authority=sum(row['height_reference_all_student_before_takeover'] is False for row in rows)),
        frozen_diagnostic_dependencies=frozen,
        semantics=dict(stable_rule='Unchanged stable_windows:12 completed transitions; both physics samples >=2 object-contact fingertips >0.1N; object root z >=median completed25-48+0.02m; palm-relative root drift<=0.01m and relative quaternion angle<=0.2rad.',
            boundaries='Eligibility requires12 transitions wholly in the same declared authority/label phase AND recorded segment. Completed endpoints before48 are separate because the settled-height reference is not yet available. Student, masked expert queue and legal expert labels never share an eligible window.',
            reference='Exact full-trajectory median after-object-root z at indices24:48. If takeover occurs before48, this reference includes expert frames; such student-trigger qualification is not claimed.',
            approach='First same-phase/segment stable window on native first goal to first subsequent near_goal on that goal. This includes observed rotation/approach; it is not an isolated rotation-only stage. Native near_goal/alignment/subgoal/completion are recorded diagnostics, never new success criteria.',
            prospective_screens='Read-only comparisons at eligible endpoints on first goal strictly before its first near_goal flag. contact_height12 retains the exact existing12tick two-finger/two-substep>0.1N and +0.02m settled-height requirements, omitting relative translation/rotation conditions. contact_native_lift12 instead requires the recorded native lifted flag for all12ticks with the same contacts. Neither screen is deployed or a recommendation to change thresholds; success-only student prefixes cannot estimate failed-state false positives.',
            goal='Post-transition object is compared with pre-transition active goal, so a goal reset cannot substitute the next target. Raw root-origin quaternion angles ignore shape symmetry.',
            denominator='Successful training trajectories only: original30 and54 admitted successful suffixes, not new independent families or all collection opportunities. Student prefixes stop at their actual expert intervention and are censored at varying clocks; 5/54 or6/54 is not a full-rollout qualification rate. Per-tick counts and per-trajectory counts are distinct. Failure-condition counts overlap.',
            interpretation='False stable-window flag is not evidence of dropping, unintended slip or unrecoverability. Recorded expert success is not autonomous student success. Expert coverage alone cannot establish current student trigger qualification rate or justify a threshold change.',
            provenance='Actual required numeric HDF5 tensors are reread and hashed. Entire raw RGB-D/HDF5 bytes are not rehashed; source_manifest_sha256 records existing manifest provenance.'),
        source_sha256={str(Path(__file__).relative_to(project)):sha256(__file__)},
        cpu_threads=2,cuda_initialized=False,optimizer_updates=0,simulator_starts=0,validation_payloads_read=False,test_used=False,
        wall_seconds=time.monotonic()-began)
    atomic_json(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execution-source',type=Path,required=True)
    args=parser.parse_args(); result=run(args.source,args.output,args.registration,execution_source=args.execution_source)
    print(json.dumps(dict(output=str(args.output),summary=result['summary'],boundary_audit=result['boundary_audit']),indent=2))
