"""Read numeric training failures and completed development traces once."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from visuotactile.grasp_curriculum import EARLY_GRASP, EarlyGraspTrigger, holding_metrics
from visuotactile.training_resources import atomic_json


def read(path): return json.loads(Path(path).read_text())


def behavior(directory, role=None, *, horizon=None):
    directory = Path(directory); plan = read(directory/'plan.json'); rows = []
    for case in plan['cases']:
        if role is not None and case['role'] != role: continue
        path = directory/case['name']; result = read(path/'result.json')
        row = dict(case=case['name'], role=case['role'], seed=case['spec']['seed'],
                   family=case['spec']['family'], legal=result['legal'],
                   diagnostic_cohort=case.get('diagnostic_cohort','registered_original_cohort'),
                   grasp_stage_censored=result.get('grasp_stage_censored',False))
        if result['legal']:
            with h5py.File(path/'numeric.h5') as f:
                t = f['episode/transition']; pose = t['diagnostics_after/object_pose_w'][:]
                force = t['object_contact_force_w'][:]
                if horizon is not None:pose=pose[:horizon];force=force[:horizon]
                fingers=(np.linalg.norm(force,axis=-1)>.1).sum(-1).min(-1)
                settled=np.arange(len(force))>=48
                def first(mask):
                    found=np.flatnonzero(mask)
                    return int(found[0]+1) if len(found) else None
                elevated=pose[:,2]>=float(np.median(pose[24:48,2]))+.02 if len(pose)>=48 else np.zeros(len(pose),bool)
                row.update(first_contact_completed_tick=first((fingers>0)&settled),
                    first_two_finger_completed_tick=first((fingers>=2)&settled),
                    first_contact_height_completed_tick=first((fingers>=2)&elevated&settled),
                    brief_hold12=holding_metrics(pose,force,window=12),
                    hold60=holding_metrics(pose, force), hold120=holding_metrics(pose, force, window=120),
                    native_lift=bool(t['diagnostics_after/lifted_object'][:].any()),
                    native_alignment=bool(t['native_alignment_condition'][:].any()),
                    native_near=bool(t['native_success_diagnostic/near_goal'][:].any()),
                    native_subgoals=int(t['diagnostics_after/successes'][:].max()), insertion=result['success'])
        rows.append(row)
    def summarize(selected):
      counts = {}
      for key in sorted({r['role'] for r in selected}):
        legal = [r for r in selected if r['role'] == key and r['legal']]
        counts[key] = dict(legal=len(legal), illegal=sum(r['role']==key and not r['legal'] for r in selected),
            contact=sum(r['first_contact_completed_tick'] is not None for r in legal),
            contact_height_lift=sum(r['first_contact_height_completed_tick'] is not None for r in legal),
            brief_hold12=sum(r['brief_hold12']['sustained'] for r in legal),
            grasp_stage_censored=sum(r['grasp_stage_censored'] for r in legal),
            hold60=sum(r['hold60']['sustained'] for r in legal), hold120=sum(r['hold120']['sustained'] for r in legal),
            **{name:sum(bool(r[name]) for r in legal) for name in
               ('native_lift','native_near','native_alignment','native_subgoals','insertion')})
      return counts
    return dict(directory=str(directory), rows=rows, counts=summarize(rows),holding_horizon=horizon,
                counts_by_cohort={c:summarize([r for r in rows if r['diagnostic_cohort']==c])
                    for c in sorted({r['diagnostic_cohort'] for r in rows})},
                semantics='Real autonomous full students; holding diagnostic differs from native lift/success; no hidden-state exact replay')


def retrospective(collection):
    collection=Path(collection); plan=read(collection/'plan.json'); cases={c['name']:c for c in plan['cases']}
    rows=[]
    for result_row in read(collection/'results.json'):
        case=cases[result_row['case']]; result=result_row['result']
        if (case['expansion_schedule']!='held0' or not result['legal']
                or result['dagger']['takeover_triggered']): continue
        trigger=EarlyGraspTrigger(EARLY_GRASP,10**9)
        with h5py.File(collection/case['name']/'numeric.h5') as f:
            t=f['episode/transition']; pose=t['diagnostics_after/object_pose_w'][:]
            palm=t['diagnostics_after/palm_body_pose_w'][:]; force=t['object_contact_force_w'][:]
            assert (t['dagger/control_source'][:]==0).all()
            for tick in range(len(pose)):
                if trigger.observe(tick,pose[tick],palm[tick],force[tick]) is not None: break
        rows.append(dict(case=case['name'],role=case['role'],family=case['spec']['family'],
            seed=case['spec']['seed'],trigger=trigger.receipt(), rescue_demonstrated=False))
    return dict(configuration=EARLY_GRASP, failed_training_prefixes=rows,
        count=len(rows), triggered=sum(r['trigger']['trigger_completed_tick'] is not None for r in rows),
        interpretation='Historical training failures only; candidate causal times, zero new rescues or autonomous successes')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--source-a',type=Path,required=True);p.add_argument('--source-b',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Preserve prior evidence')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output,dict(retrospective=retrospective(a.collection),
        sources=dict(A=behavior(a.source_a/'development','A'),B=behavior(a.source_b/'development','B')),
        task_success_changed=False, test_opened=False))


if __name__=='__main__': main()
