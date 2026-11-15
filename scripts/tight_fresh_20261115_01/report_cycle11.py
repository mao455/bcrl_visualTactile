#!/usr/bin/env python3
"""One completed behavior review; reuse round09 evidence and closed numeric fields."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

TREE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(TREE))
from tools import tight_fresh_phases as phases
from tools import tight_fresh_visual_closeout as motion
from tools import tight_fresh_round_compare as compare
from tools.tight_fresh_contact_height_coverage import causal_contact_height


def summarize(root,baseline,*,round_index=11):
    bindings={}
    def read(path):
        raw=path.read_bytes();bindings[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def stat(path):
        value=path.stat()
        return dict(device=value.st_dev,inode=value.st_ino,bytes=value.st_size,mtime_ns=value.st_mtime_ns)
    def token(path):
        path=Path(path)
        if path.suffix=='.h5':
            assert path.name=='numeric.h5' and path.parent.parent==development
            return 'stat_only:'+json.dumps(stat(path),sort_keys=True)
        raw=path.read_bytes();return hashlib.sha256(raw).hexdigest()
    candidate=root/f'round_{round_index:02d}';stages=root/f'round_{round_index:02d}_stages';development=candidate/'development'
    assert read(stages/'finished.json')['software_complete']
    assert read(development/'result.json')['software_complete']
    old=read(baseline)['candidate'];plan=read(development/'plan.json')
    task=read(root/'metadata.json')['config']
    oldplan=read(Path(old['development'])/'plan.json')
    population={role:compare.compare_role_plans(oldplan,plan,role) for role in ('A','B')}
    # Same diagnostic arithmetic as the previously reviewed round09 report.
    # Only this report process replaces numeric hashing with closed-file stability.
    phases.sha256=motion.sha256=token
    compare.causal_contact_height=causal_contact_height
    rows=[]
    for indexed in read(development/'results.json'):
        result=indexed['result'];directory=development/indexed['case']
        if not result['legal']:
            rows.append(dict(role=indexed['role'],case=indexed['case'],legal=False));continue
        assert result['software_complete'] and not result.get('dagger')
        assert result['complete_student_policy']==plan['pairs'][indexed['role']]
        path=directory/'numeric.h5';before=stat(path)
        phase=phases.case(directory)
        assert phase['authority']['expert_ticks']==0 and phase['steps']==result['steps']
        contact=motion.contact_motion(directory,phase);orientation=compare.held_orientation(directory)
        if phase['native_near_goal_first_tick'] is not None:
            import h5py
            with h5py.File(path,'r') as stream:
                diagnostic=stream['episode/transition/native_success_diagnostic']
                phase['near_goal_retention']=dict(near_ticks=int(diagnostic['near_goal'][:].sum()),
                    maximum_recorded_count=int(diagnostic['near_goal_steps_after_goal_reset'][:].max()),
                    required_count=task['termination']['success_steps'],
                    consecutive_required=task['termination']['force_consecutive_near_goal_steps'],
                    goal_mode=task['precise_assembly']['goal_mode'])
        assert stat(path)==before
        phase.pop('numeric_sha256')
        rows.append(dict(**phase,role=indexed['role'],contact_motion=contact,held_orientation=orientation,
            numeric_stat=before,wall_seconds=indexed['timing']['total_seconds']))
    new=dict(development=str(development),execution=compare.execution_identity(plan),pairs=plan['pairs'],cases=rows,
        totals={role:compare.autonomy_totals([row for row in rows if row['role']==role],batch=True,explicit=True) for role in ('A','B')})
    receipts={}
    for path in sorted(stages.glob('*.json')):
        value=read(path)
        if 'returncode' in value and 'seconds' in value:
            assert value['returncode']==0
            receipts[value['stage']]=dict(start=value['started'],seconds=value['seconds'])
    def span(names):
        values=[receipts[name] for name in names]
        return max(row['start']+row['seconds'] for row in values)-min(row['start'] for row in values)
    backend={}
    for role in ('A','B'):
        path=candidate/role/'backend'/role/'training/finished.json'
        if path.exists():backend[role]={key:value for key,value in read(path).items() if not isinstance(value,(dict,list))}
    return dict(version='closed_cycle_review_v1',round_index=round_index,complete=True,observed_at=datetime.datetime.now().astimezone().isoformat(),
        source=old,candidate=new,population=population,
        matched_legal=compare.matched_legal_comparison(old,new),
        backend_training=backend,stage_receipts=receipts,
        seconds=dict(collection=read(root/f'new_family_{round_index:02d}/result.json')['elapsed_seconds'],
            visual_cache=receipts['visual_cache']['seconds'],frontends=span(['frontend_A','frontend_B']),
            backend_inputs=span(['backend_cache_A','backend_cache_B']),
            backend_training=span(['backend_train_A','backend_train_B']),autonomous=receipts['development']['seconds'],
            aggregate_to_evaluation=span(list(receipts)),
            collection_end_to_aggregate_gap=receipts['aggregate']['start']-read(root/f'new_family_{round_index:02d}/owner_finished.json')['ended']),
        scope=dict(candidate_numeric_files=sum(row['legal'] for row in rows),old_numeric_files=0,
            models_opened=0,images_opened=0,learning_jsonl_opened=0,new_simulations=0,new_updates=0),
        limits=['Repeated development cohort, not independent held-out performance or equal hidden-state replay.',
            'Rotation-permitting holding, original stable window, native lift, alignment and task success remain separate.',
            'Angles compare object/goal root quaternions and ignore object symmetry; proximity does not establish alignment.',
            'Parallel-stage times are elapsed envelopes; collection interruption gap is reported separately.'],
        input_sha256=bindings,helper_sha256={str(path.relative_to(TREE)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [Path(__file__),TREE/'tools/tight_fresh_phases.py',TREE/'tools/tight_fresh_visual_closeout.py',
             TREE/'tools/tight_fresh_round_compare.py',TREE/'tools/tight_fresh_contact_height_coverage.py']})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--round',type=int,default=11)
    args=parser.parse_args();assert not args.output.exists(),'Preserve existing review'
    value=summarize(args.root.resolve(),args.baseline.resolve(),round_index=args.round)
    with args.output.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(candidate=value['candidate']['totals'],seconds=value['seconds']),indent=2))
