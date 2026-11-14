"""Read only completed training captures; never inspect RGB/DINO/expert future."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np
from visuotactile.takeover_trigger import contact_height_windows


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def review():
    base=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
    roots=[base/'tight_fresh_20261114_05/takeover_06_tasking4_01',
        base/'tight_fresh_20261114_05/takeover_07_batch8_m8_01',base/'tight_fresh_20261114_06/new_family_08']
    fields=['dagger/control_source','diagnostics_after/object_pose_w','object_contact_force_w',
        'native_keypoints_max_dist','diagnostics_before/goal_pose_w','diagnostics_after/lifted_object']
    report=dict(version='held_recheck_student_prefix_retrospective_v1',started=time.time(),cohorts=[],
        read_fields=['episode/transition/'+key for key in fields],gpu_starts=0,optimizer_updates=0,test_trajectories_opened=0,
        source_trigger_sha256=digest(Path('visuotactile/takeover_trigger.py')),
        scope='Only results-index committed closed captures; student prefix only. Historical families, policies and batch execution differ. Delay counterfactuals after actual expert switch are censored, not failures or rescued data.',
        definitions=dict(held='12 completed transitions, each with two fingers >0.1N in both physics samples and object >=2cm above median completed25..48; rotation allowed',
            distance='Native max keypoint distance, includes orientation; not object-center distance to hole',
            held_loss='Loss of this collection condition only; never inferred native dropped/fall',
            next_action_index='Completed transition count N schedules zero-based action N'))
    for root in roots:
        assert json.loads((root/'result.json').read_text())['software_complete']
        rows=json.loads((root/'results.json').read_text());plan=json.loads((root/'plan.json').read_text())
        cases={case['name']:case for case in plan['cases']};summary=Counter();delays=defaultdict(Counter);details=[]
        for row in rows:
            result=row['result'];summary['attempts']+=1
            item=dict(case=row['case'],role=row['role'],legal=bool(result.get('legal')),
                result_index_sha256=digest(root/'results.json'))
            details.append(item)
            if not item['legal']:summary['illegal']+=1;continue
            assert result['software_complete'];summary['legal']+=1
            n=result['dagger']['student_ticks'];path=root/row['case']/'numeric.h5';stat=path.stat()
            with h5py.File(path) as stream:
                transition=stream['episode/transition'];source=transition['dagger/control_source'][:]
                assert (source[:n]==0).all() and (source[n:]==1).all()
                pose=transition['diagnostics_after/object_pose_w'][:n];force=transition['object_contact_force_w'][:n]
                distance=transition['native_keypoints_max_dist'][:n];goal=transition['diagnostics_before/goal_pose_w'][:n]
                lift=transition['diagnostics_after/lifted_object'][:n]
            switched=bool(result['dagger']['takeover_triggered'])
            item.update(student_ticks=n,expert_switched=switched,native_lift=bool(lift.any()),
                native_termination=result['reasons'],native_termination_authority='expert' if switched else 'student',
                schedule=cases[row['case']].get('expansion_schedule'),
                old_trigger=result['dagger']['configuration'].get('trigger'),
                numeric=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns,
                    sha256_recomputed=False))
            summary['native_lift']+=item['native_lift'];summary['no_expert']+=not switched
            if n<48:
                item['held_reference_available']=False;summary['reference_unavailable']+=1;continue
            item['held_reference_available']=True;height=float(np.median(pose[24:48,2]))
            instantaneous=((np.linalg.norm(force,axis=-1)>.1).sum(-1)>=2).all(-1)&(pose[:,2]>=height+.02)
            held=contact_height_windows(pose,force,height);held[:47]=False;hits=np.flatnonzero(held)
            item['ever_held']=bool(len(hits));summary['ever_held']+=item['ever_held']
            if not len(hits):summary['no_held']+=1;continue
            angle=2*np.arccos(np.clip(np.abs((pose[:,3:]*goal[:,3:]).sum(-1)),0,1));first=int(hits[0])
            def state(index):return dict(completed_tick=int(index+1),held=bool(held[index]),
                native_keypoint_distance_m=float(distance[index]),quaternion_angle_rad=float(angle[index]),
                height_above_reference_m=float(pose[index,2]-height))
            item.update(first_held=state(first),held_at_boundary=bool(held[-1]),boundary=state(n-1),delays={})
            summary['held_at_boundary']+=bool(held[-1])
            for delay in (12,60,120):
                due=first+delay
                if due>=n:
                    delays[delay]['student_future_censored']+=1
                    item['delays'][str(delay)]=dict(status='censored_by_expert' if switched else 'native_ended_before_delay')
                    continue
                delays[delay]['observed']+=1;delays[delay]['held_at_due']+=bool(held[due])
                continuous=bool(instantaneous[first:due+1].all());delays[delay]['uninterrupted_held']+=continuous
                later=hits[hits>=due];delays[delay]['later_held_any']+=bool(len(later))
                item['delays'][str(delay)]=dict(status='student_observed',at_due=state(due),
                    uninterrupted_condition=continuous,instantaneous_condition_missing_ticks=int((~instantaneous[first:due+1]).sum()),
                    first_later_current_held=state(int(later[0])) if len(later) else None)
        report['cohorts'].append(dict(root=str(root),plan_sha256=digest(root/'plan.json'),
            results_sha256=digest(root/'results.json'),summary=dict(summary),delay_counts=dict(delays),cases=details))
    prior=base/'tight_fresh_20261114_06/cpu_audits/collection08_review_01/collection08.json'
    coverage=json.loads(prior.read_text());fixed=[row for row in coverage['cases'] if row.get('accepted') and row['schedule'].startswith('fixed')]
    report['fixed_source_scope']=dict(source_report=str(prior),source_sha256=digest(prior),
        total_valid_expert_ticks=coverage['summary']['accepted_valid_expert_ticks'],
        fixed40_80_valid_expert_ticks=sum(row['label_alignment']['legal_expert_ticks'] for row in fixed),
        fixed40_80_held_expert_ticks=sum(row['coverage_ticks']['valid_expert']['completed_contact_height_window'] for row in fixed),
        meaning='4244/5795=73.24% are from early-takeover trajectories including their full expert suffixes; not 73.24% early-stage actions. These suffixes include939 held-window expert ticks.')
    report['ended']=time.time()
    return report


if __name__=='__main__':
    output=Path(__file__).with_name('review.json')
    with output.open('x') as stream:json.dump(review(),stream,indent=2);stream.write('\n')
