"""Small reproducible public receipts from actual fresh takeover artifacts."""
import argparse
from pathlib import Path
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read


def takeover(collection,output):
    result=read(collection/'result.json');plan=read(collection/'plan.json')
    if not result.get('software_complete'):raise ValueError('Collection is incomplete')
    rows=read(collection/'results.json');entries=read(collection/'correction_entries.json')
    planned={c['name'] for c in plan['cases']}
    if len(rows)!=len(planned) or {x['case'] for x in rows}!=planned:raise ValueError('Opportunity accounting differs')
    cases=[]
    for row in rows:
        x=row['result'];accepted=bool(x.get('accepted_for_training'))
        if accepted!=(row['case'] in entries):raise ValueError('Retained subset differs')
        if x.get('legal') and not read(collection/row['case']/'capture_audit.json')['passed']:
            raise ValueError('Capture audit failed')
        cases.append(dict(case=row['case'],role=row['role'],takeover_tick=x.get('takeover_tick'),
            legal=x.get('legal'),software_complete=x.get('software_complete'),success=x.get('success'),
            accepted=accepted,steps=x.get('steps'),reasons=x.get('reasons'),native_subgoals=x.get('native_subgoals'),
            contact=x.get('contact'),actual_exposure=x.get('actual_exposure'),
            frontend_sha256=plan['pairs'][row['role']]['frontend_sha256'],
            backend_sha256=plan['pairs'][row['role']]['backend_sha256'],
            capture_sha256=entries[row['case']]['source_sha256'] if accepted else None))
    retained=[c for c in cases if c['accepted']]
    counts={role:dict(opportunities=sum(c['role']==role for c in cases),
        legal=sum(c['role']==role and bool(c['legal']) for c in cases),
        expert_successes=sum(c['role']==role and bool(c['success']) for c in cases),
        accepted=sum(c['role']==role for c in retained)) for role in ['A','B']}
    exposure={key:sum(c['actual_exposure'][key] for c in retained)
        for key in ['history_ticks','supervised_ticks','supervised_anchors','supervised_window_targets']}
    atomic_json(output,dict(source_commit=plan['source_commit'],plan_sha256=sha256(collection/'plan.json'),
        results_sha256=sha256(collection/'results.json'),entries_sha256=sha256(collection/'correction_entries.json'),
        counts=counts,accepted=len(retained),retained_exposure=exposure,cases=cases,test_used=False,
        meaning='Actual expert rescues after complete student prefixes; not autonomous student successes. All failed attempts retained locally.',
        learning_exposure='Source population only; optimizer exposure is recorded by the following trainers.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();takeover(a.collection,a.output)

if __name__=='__main__':main()
