"""Recompute coverage and actual learning exposure of the first four DAgger rounds.

Read-only CPU audit of recorded complete student prefixes and actual expert
suffixes. The separate backend weighting study and subsequent sessions are
outside this fixed cohort. Public output contains relative source identities and
hashes, no raw trajectories, checkpoints, or machine-specific absolute paths.
"""
import argparse
import json
import subprocess
import time
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import h5py
import torch

from visuotactile.compact import sha256
from visuotactile.takeover_trigger import stable_windows
from tools.tight_fresh_phases import authority_phases
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def coverage_fractions(rows):
    accepted = [row for row in rows if row['accepted']]
    ticks = sum(row['expert_supervision']['legal_ticks'] for row in accepted)
    def fraction(numerator, denominator):
        return numerator / denominator if denominator else None
    return dict(ever_student_stable_successes=fraction(sum(row['student']['student_ever_stable'] for row in accepted), len(accepted)),
        stable_at_handover_successes=fraction(sum(row['student']['stable_at_takeover'] for row in accepted), len(accepted)),
        stable_at_handover_source_legal_ticks=fraction(sum(row['expert_supervision']['legal_ticks'] for row in accepted
            if row['student']['stable_at_takeover']), ticks),
        ever_student_stable_source_legal_ticks=fraction(sum(row['expert_supervision']['legal_ticks'] for row in accepted
            if row['student']['student_ever_stable']), ticks))


def compact_report(result):
    """Keep per-case provenance and key exposure; stable cases retain full detail."""
    keep_exposure = ('frontend_episode_updates', 'frontend_planning_anchor_exposures',
        'frontend_legal_target_slot_exposures', 'backend_completed_legal_ticks', 'backend_selected_legal_ticks')
    result['stable_student_cases'] = [dict(row) for row in result['stable_student_cases']]
    cases = []
    for row in result['cases']:
        item = {key:value for key,value in row.items() if key not in ('actual_training_exposure', 'source_population')}
        if row.get('actual_training_exposure'):
            item['actual_training_exposure_totals'] = {
                role:{key:sum(stage[key] for stage in stages) for key in keep_exposure}
                for role,stages in row['actual_training_exposure'].items()}
        cases.append(item)
    result['cases'] = cases
    for row in result['stable_student_cases']:
        row.pop('source_population', None)
    return result


def summary(selected):
    out=dict(opportunities=len(selected),accepted=sum(r['accepted'] for r in selected),
        unsuccessful_expert_attempts=sum(not r['expert_takeover_success'] for r in selected),
        categories=dict(Counter(r['student_phase_category'] for r in selected)),
        accepted_categories=dict(Counter(r['student_phase_category'] for r in selected if r['accepted'])),
        student_stage_counts={key:sum(bool(r['student'][key]) for r in selected) for key in [
            'student_contact_ever','student_two_fingers_ever','student_native_lift_ever',
            'student_rise_2cm_with_two_fingers','student_ever_stable','stable_at_takeover',
            'student_native_near_goal_ever','student_native_alignment_ever']},
        source_legal_ticks=sum(r['expert_supervision']['legal_ticks'] for r in selected if r['accepted']),
        source_legal_ticks_by_category={kind:sum(r['expert_supervision']['legal_ticks'] for r in selected if r['accepted'] and r['student_phase_category']==kind)
            for kind in ['no_completed_stable_grasp','formerly_stable_now_unstable','stable_at_takeover']})
    for role in ('A','B'):
        total=Counter()
        for r in selected:
            for exposure in r.get('actual_training_exposure',{}).get(role,[]):
                total.update({k:v for k,v in exposure.items() if k!='stage'})
        out.setdefault('four_completed_rounds_exposure_by_trained_role',{})[role]=dict(total)
    return out


def report(root, evidence):
    torch.set_num_threads(2)
    began=time.monotonic()
    rows=[];source_reports={};collections=[]
    for round_index in range(1,5):
        collection=root/f'takeover_{round_index:02}'
        receipt_path=evidence/f'takeover_{round_index:02}.json';receipt=read(receipt_path)
        phases_path=evidence/('takeover_phases.json' if round_index==1 else f'takeover{round_index:02}_phases.json')
        phases=read(phases_path);phase_cases={c['case']:c for c in phases['cases']}
        plan=read(collection/'plan.json');entries=read(collection/'correction_entries.json')
        require(receipt['plan_sha256']==phases['source_plan_sha256']==sha256(collection/'plan.json'), "Coverage audit failed: receipt['plan_sha256']==phases['source_plan_sha256']==sha256(collection/'plan.json')")
        require(receipt['results_sha256']==sha256(collection/'results.json'), "Coverage audit failed: receipt['results_sha256']==sha256(collection/'results.json')")
        require(receipt['entries_sha256']==sha256(collection/'correction_entries.json'), "Coverage audit failed: receipt['entries_sha256']==sha256(collection/'correction_entries.json')")
        source_reports[receipt_path.name]=sha256(receipt_path);source_reports[phases_path.name]=sha256(phases_path)
        for c in receipt['cases']:
            name=c['case']; directory=collection/name; actual=read(directory/'result.json')
            trace=directory/'numeric.h5';numeric_sha=sha256(trace)
            require(numeric_sha==phase_cases[name]['numeric_sha256'], "Coverage audit failed: numeric_sha==phase_cases[name]['numeric_sha256']")
            with h5py.File(trace) as f:
                t=f['episode/transition'];source=t['dagger/control_source'][:];legal=t['dagger/expert_label_valid'][:].astype(bool)
                student=source==0; expert=source==1;switch=int(student.sum())
                require(switch==c['takeover_tick'] and np.array_equal(student,np.arange(len(student))<switch), "Coverage audit failed: switch==c['takeover_tick'] and np.array_equal(student,np.arange(len(student))<switch)")
                require(not legal[student].any() and not legal[expert][:2].any(), 'Coverage audit failed: not legal[student].any() and not legal[expert][:2].any()')
                pose=t['diagnostics_after/object_pose_w'][:];palm=t['diagnostics_after/palm_body_pose_w'][:]
                force=t['object_contact_force_w'][:]; goal=t['diagnostics_after/goal_pose_w'][:]
                settled=float(np.median(pose[24:48,2]));stable,contact=stable_windows(pose,palm,force,settled)
                two=(contact.sum(-1)>=2).all(-1);subgoal=t['diagnostics_after/successes'][:]
                near=t['native_success_diagnostic/near_goal'][:];align=t['native_alignment_condition'][:]
                authority=authority_phases(source,stable,near,align,subgoal)
                if 'authority' in phase_cases[name]:require(authority==phase_cases[name]['authority'], "Coverage audit failed: authority==phase_cases[name]['authority']")
                angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1))
                distance=np.linalg.norm(pose[:,:3]-goal[:,:3],axis=-1)
                lifted=t['diagnostics_after/lifted_object'][:].astype(bool)
                late=np.arange(len(pose))>=48
                studentstable=student&stable; sidx=np.flatnonzero(studentstable)
                stablelast=bool(stable[switch-1]);everstable=bool(studentstable.any())
                kind='stable_at_takeover' if stablelast else 'formerly_stable_now_unstable' if everstable else 'no_completed_stable_grasp'
                before=t['diagnostics_before/object_pose_w'][switch];after=pose[switch-1]
                require(np.array_equal(before,after), 'Coverage audit failed: np.array_equal(before,after)')
                labels=int(legal.sum())
                actual_counts=actual['dagger']
                require(labels==actual_counts['expert_valid_ticks'], "Coverage audit failed: labels==actual_counts['expert_valid_ticks']")
                if c['accepted']:
                    require(name in entries and c['actual_exposure']['supervised_ticks']==labels, "Coverage audit failed: name in entries and c['actual_exposure']['supervised_ticks']==labels")
                else: require(name not in entries, 'Coverage audit failed: name not in entries')
                last=int(switch-1)
                captured=dict(student_contact_ever=bool((contact.any(axis=(1,2))&student).any()),
                    student_two_fingers_ever=bool((two&student).any()),student_native_lift_ever=bool((lifted&student).any()),
                    student_rise_2cm_with_two_fingers=bool((late&student&two&(pose[:,2]>=settled+.02)).any()) if switch>=48 else None,
                    student_ever_stable=everstable,stable_at_takeover=stablelast,student_stable_window_end_ticks=(sidx+1).tolist(),
                    student_stable_goal_angle_min_rad=float(angle[studentstable].min()) if everstable else None,
                    student_stable_goal_distance_min_m=float(distance[studentstable].min()) if everstable else None,
                    student_goal_angle_min_rad=float(angle[student].min()),
                    student_goal_distance_min_m=float(distance[student].min()),
                    student_native_near_goal_ever=bool((near&student).any()),student_native_alignment_ever=bool((align&student).any()),
                    student_max_subgoals=int(subgoal[student].max()),switch_goal_angle_rad=float(angle[last]),
                    switch_goal_distance_m=float(distance[last]),switch_two_contact_fingers=bool(two[last]),
                    switch_rise_over_settled_m=float(pose[last,2]-settled) if switch>=48 else None,
                    switch_native_lift=bool(lifted[last]),
                    max_student_rise_over_settled_m=float((pose[student&late,2]-settled).max()) if (student&late).any() else None)
                expert_bins=dict(legal_ticks=labels,first12=min(12,labels),first24=min(24,labels),
                    expert_stable_legal_ticks=int((legal&stable).sum()),expert_native_near_goal_legal_ticks=int((legal&near).sum()),
                    expert_native_alignment_legal_ticks=int((legal&align).sum()),
                    expert_after_first_student_stable_legal_ticks=labels if everstable else 0)
                firststable=np.flatnonzero(expert&stable)
                if len(firststable):expert_bins['first_expert_stable_after_switch_ticks']=int(firststable[0]-switch+1)
                row=dict(case=name,round=round_index,role=c['role'],family=actual['family'],accepted=c['accepted'],
                    expert_takeover_success=actual['dagger']['expert_takeover_success'],software_complete=actual['software_complete'],legal=actual['legal'],
                    takeover_tick=switch,steps=len(source),student_phase_category=kind,student=captured,
                    authority=authority,expert_supervision=expert_bins,source_population=c.get('actual_exposure'),
                    trigger=c.get('trigger'),native_terminal_reasons=c['reasons'],numeric_sha256=numeric_sha,
                    capture_sha256=c['capture_sha256'],
                    settled_reference_causal_within_student_prefix=switch>=48,
                    student_prefix_supervised=False)
                rows.append(row)
        collections.append(dict(round=round_index,opportunities=len(receipt['cases']),accepted=receipt['accepted'],
            failed=len(receipt['cases'])-receipt['accepted'],source_commit=receipt['source_commit'],
            exposure=receipt['retained_exposure']))
    require(len(rows)==72 and sum(r['accepted'] for r in rows)==53, "Coverage audit failed: len(rows)==72 and sum(r['accepted'] for r in rows)==53")
    by_name={r['case']:r for r in rows}; accepted={r['case'] for r in rows if r['accepted']}
    latest=read(root/'round_04/A/state_data/data.json')
    require(set(latest['splits']['train'])-set(latest['normalization_train_ids'])==accepted, "Coverage audit failed: set(latest['splits']['train'])-set(latest['normalization_train_ids'])==accepted")
    latest_numeric=torch.load(root/'round_04/A/state_data/numeric.pt',map_location='cpu',weights_only=False)['records']
    for name in accepted:
        require(latest['entries'][name]['source_sha256']==by_name[name]['capture_sha256'], "Coverage audit failed: latest['entries'][name]['source_sha256']==by_name[name]['capture_sha256']")
        require(len(latest_numeric[name]['time'])==by_name[name]['steps'], "Coverage audit failed: len(latest_numeric[name]['time'])==by_name[name]['steps']")
        require(int(latest_numeric[name]['supervision_mask'].sum())==by_name[name]['expert_supervision']['legal_ticks'], "Coverage audit failed: int(latest_numeric[name]['supervision_mask'].sum())==by_name[name]['expert_supervision']['legal_ticks']")
    del latest_numeric
    stage_paths=['round_01_v3','round_02_v2','round_03','round_04'];stages=[]
    for stage_i, stage in enumerate(stage_paths,1):
        learn_path=evidence/f'round{stage_i:02}_learning.json'; learn=read(learn_path)
        source_reports[learn_path.name]=sha256(learn_path)
        stage_summary=dict(stage=stage,roles={})
        for role in ('A','B'):
            directory=root/stage/role; report=learn['roles'][role]; training=directory/'training'
            require(report['learning_receipts_sha256']==sha256(training/'learning.jsonl'), "Coverage audit failed: report['learning_receipts_sha256']==sha256(training/'learning.jsonl')")
            data=read(directory/'state_data/data.json'); corrections=set(data['splits']['train'])-set(data['normalization_train_ids'])
            require(corrections<=accepted, 'Coverage audit failed: corrections<=accepted')
            counts=Counter();per=defaultdict(Counter);last=dict(anchors_seen=0,target_ticks=0)
            for line in (training/'learning.jsonl').read_text().splitlines():
                row=json.loads(line);name=row['episode'];counts[name]+=1
                if name in accepted:
                    b=per[name];b['frontend_episode_updates']+=1;b['frontend_history_frames']+=row['visual_frames']
                    b['frontend_planning_anchor_exposures']+=row['anchors_seen']-last['anchors_seen']
                    b['frontend_legal_target_slot_exposures']+=row['target_ticks']-last['target_ticks']
                last={k:row[k] for k in last}
            require(dict(counts)==report['actual_episode_counts'], "Coverage audit failed: dict(counts)==report['actual_episode_counts']")
            rear=directory/'backend'/role/'training'
            best=torch.load(rear/'best.pt',map_location='cpu',weights_only=False,mmap=True)
            chosen_counts=best['episode_counts'];chosen_epoch=best['epoch']
            require(chosen_epoch==report['backend_selected']['epoch'], "Coverage audit failed: chosen_epoch==report['backend_selected']['epoch']")
            require(sha256(rear/'best.pt')==report['backend_selected']['checkpoint_sha256'], "Coverage audit failed: sha256(rear/'best.pt')==report['backend_selected']['checkpoint_sha256']")
            for name in corrections:
                count=report['backend_actual_episode_counts'][name];b=per[name];base=by_name[name]['expert_supervision']
                b['backend_completed_episode_draws']=count;b['backend_selected_episode_draws']=chosen_counts[name]
                for key,val in base.items():
                    if key=='first_expert_stable_after_switch_ticks':continue
                    b['backend_completed_'+key]=count*val;b['backend_selected_'+key]=chosen_counts[name]*val
                by_name[name].setdefault('actual_training_exposure',{}).setdefault(role,[]).append(dict(stage=stage,**dict(b)))
            stage_summary['roles'][role]=dict(frontend_selected_epoch=report['selected_epoch'],frontend_completed_epoch=report['latest_epoch'],
                backend_selected_epoch=chosen_epoch,backend_completed_epoch=max(report['backend_actual_episode_counts'].values()),
                corrections=len(corrections),correction_exposure=dict(sum((Counter(v) for v in per.values()),Counter())),
                frontend_sha256=report['checkpoint_sha256'],backend_selected_sha256=report['backend_selected']['checkpoint_sha256'])
            del best
        stages.append(stage_summary)

    overall=summary(rows)
    stable_cases=[r for r in rows if r['student']['student_ever_stable']]
    for r in rows:
        if not r['accepted']:require('actual_training_exposure' not in r, "Coverage audit failed: 'actual_training_exposure' not in r")
    result=dict(version='four_round_rear_phase_coverage_review_v1',collections=collections,summary=overall,
        accepted_summary=summary([r for r in rows if r['accepted']]),
        unsuccessful_summary=summary([r for r in rows if not r['accepted']]),
        fractions=coverage_fractions(rows),
        by_round={str(i):summary([r for r in rows if r['round']==i]) for i in range(1,5)},
        by_rollin_role={role:summary([r for r in rows if r['role']==role]) for role in ('A','B')},
        by_phase={phase:summary([r for r in rows if r['student_phase_category']==phase]) for phase in overall['categories']},
        stable_student_cases=stable_cases,stages=stages,cases=rows,evidence_sha256=source_reports,
        scope='Four completed DAgger training rounds only. The separate recovery-weight comparison is excluded from these four-round optimizer totals. Abandoned attempts are not part of this selected model ancestry; total GPU work is not claimed.',
        interpretation=dict(expert_failures='19 actual legal attempts were not rescued within this run. This does not establish that these states are intrinsically unrecoverable.',
            late_coverage='Four successful records follow some student stable-grasp window; three have already lost the stability criterion at switch. Only one is still stable at actual handover. Expert-completed rotate/align/insert suffixes in earlier pickups are not student-origin late-stage coverage.',
            stable_rule='Existing12 completed ticks, both physics samples at least2 object-contact fingertips >0.1N, height>=2cm above median completed25–48, palm-relative drift<=1cm and relative rotation<=0.2rad. Diagnostics only; no task-success change.',
            angles='Current object-to-goal quaternion angle is reported at the same tick as stability, ignoring object symmetry. No new arbitrary turned-correctly threshold is imposed. Native near-goal/alignment are reported separately.',
            training='Population legal ticks exclude full student prefix and actual2tick expert queue flush. Completed exposure and selected-checkpoint exposure are separate. Repeated exposure cannot create new physical recovery states.',
            goals='All53 accepted corrections are pooled into both A andB within the same30 training families; they are not53 independent initial families. Validation6/test22 excluded.'),
        next_collection_design=dict(name='current_full_student_event_conditioned_late_recovery',
            rationale='Most current corrections begin before any student stable grasp; extending fixed late clocks produced14 failed rescues in24 opportunities and only2 stable-window-triggered rescues. Current stable-at-switch source has only1 case/86 legal expert ticks.',
            proposed_opportunities=12,roles={'A':6,'B':6},
            families=['formal_051','formal_046','formal_007','formal_053','formal_050','formal_029'],
            family_assignment='Identical family/order/trigger mapping for both roles. First3 families get immediate-stability handover; last3 get post-stability continuation. Uses original training starts only, no validation/development/test initial states.',
            strata='Per role:3 handovers immediately after first completed existing stable-grasp window;3 at first subsequent loss of that same condition, or after60 further actual student ticks if stability persists. Categorize loss-triggered versus still-stable delayed handovers separately.',
            fallback='No fixed-time expert rescue for a never-stable case in this late-coverage cohort. Record it as student failed to qualify and retain full autonomous trace; never count it as later-stage correction.',
            proposal_status='Historical design suggestion from coverage audit; the subsequent reviewed selection and actual registration are authoritative.',
            before_launch='Bind the selected complete reviewed pair and register its actual pair/source hashes and a concrete bounded cohort. If the selected pair still rarely qualifies, do not inflate the cohort with late unsuccessful pickups; report low eligibility and prioritize acquisition stability.',
            supervision='Complete actual student prefix, fresh observed actual expert suffix, existing queue-clean mask. Successful suffixes only enter legal action supervision; failed/nonqualifying traces remain diagnostic evidence.',
            completion='Report qualification/trigger/loss/at-switch stability and angle/distance, expert rescue outcomes and failures, exact source/learning exposure; then update DP, refresh all dependent visual/plan caches, train backend and evaluate full student on unchanged independent development cohort. No autonomous benefit assumed.'),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parent.parent,text=True).strip(),
        source_sha256={name:sha256(Path(__file__).resolve().parent.parent/name) for name in [
            'tools/tight_fresh_rear_coverage_report.py','tools/tight_fresh_phases.py','visuotactile/takeover_trigger.py']},
        wall_seconds=time.monotonic()-began,cpu_threads=2,cuda_initialized=torch.cuda.is_initialized(),optimizer_updates=0,simulator_starts=0,test_used=False)

    return compact_report(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--evidence', type=Path,
        default=Path(__file__).resolve().parent.parent/'research_docs/evidence/tight_fresh_20261114')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = report(args.root.resolve(), args.evidence.resolve())
    atomic_json(args.output, result)
    print(json.dumps(dict(output=str(args.output), summary=result['summary'], fractions=result['fractions']), indent=2))


if __name__ == '__main__':
    main()
