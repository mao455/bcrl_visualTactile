"""Summarize already-recorded correction metrics; JSON-only, no evaluation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--private',type=Path,required=True)
    args=parser.parse_args();session=args.session.resolve();bindings={}
    def read(relative):
        path=session/relative;assert path.suffix=='.json'
        payload=path.read_bytes();bindings[relative]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        return json.loads(payload)
    summary=json.loads((args.evidence/'report.json').read_text())
    entries=read('new_family_09/correction_entries.json');ids=set(entries)
    held={i for i in ids if entries[i]['expansion_schedule'] in ('held60','held120')}
    assert len(ids)==52 and len(held)==13
    report=dict(version='round09_recorded_correction_fit_v1',created_utc=time.time(),source_commit=summary['source_commit'],
        metric_scope='Already measured DDIM16 predictions on complete teacher-forced real prefixes; legal first12 slots of each H24 chunk.',
        aggregation=dict(macro='Arithmetic mean of episode RMSE, equal weight per trajectory; before and after use exactly the same identities.',
            full52_pooled='Original logged metric concatenates legal commit12 errors before RMSE; copied without reweighting.',
            held13_pooled=None,held13_pooled_reason='Per-episode receipts lack the exact legal commit12 error count needed for pooled weighting; no numeric payloads reopened.'),
        interpretation='These are training-data fit errors, not autonomous success or proof of holding, turning, alignment or insertion improvement.',
        cross_role_comparison='A targets actual q[t+1], one control step later (1/60 s), not final settled joints. B targets expert actually executed absolute command u. Their errors are not interchangeable or direct cross-label rankings.',
        method='87577e8 legacy single-trajectory fixed-anchor training; no B2/random-anchor/eager-I/O training in this round.',
        roles={},input_bindings=bindings,model_payloads_opened=0,numeric_payloads_opened=0,rgb_payloads_opened=0,
        new_evaluations=0,optimizer_updates=0)
    for role in 'AB':
        prefix='round_09/'+role+'/training/'
        initial=read(prefix+'initial_metrics.json');last=read(prefix+'validation.json')
        for name in ('initial_metrics.json','validation.json'):
            assert bindings[prefix+name]['sha256']==summary['input_bindings'][prefix+name]['sha256']
        assert last['epoch']==10 and last['updates']==2220
        before={r['identity']:r for r in initial['correction']['episodes']}
        after={r['identity']:r for r in last['correction']['episodes']}
        assert len(before)==len(initial['correction']['episodes'])==52 and set(before)==set(after)==ids
        for identity in ids:
            for field in ('profile','supervised_anchors','skipped_noise_draws','total_noise_draws','first_anchor','last_anchor','full_history_frames'):
                assert before[identity]['offline_planning'][field]==after[identity]['offline_planning'][field]
            assert all(math.isfinite(row[key]) and row[key]>0 for row in (before[identity],after[identity]) for key in ('arm_rmse','hand_rmse'))
        row=dict(label='q[t+1]' if role=='A' else 'u_expert_actual_absolute_command',
            measured_epoch=10,measured_updates=2220,selected_epoch=summary['roles'][role]['selected']['epoch'],
            matches_selected=role=='A',selected_new_correction_metrics_available=role=='A',
            selected_new_correction_metrics=None,groups={})
        for name,cohort in (('latest52',ids),('held13',held)):
            group=dict(episodes=len(cohort),families=len({entries[i]['family'] for i in cohort}),
                scope='All legal expert suffix predictions for these trajectories; not a per-tick held-state mask.',by_joint_group={})
            for joint,key in (('arm','arm_rmse'),('hand','hand_rmse')):
                old=statistics.mean(before[i][key] for i in cohort);new=statistics.mean(after[i][key] for i in cohort)
                group['by_joint_group'][joint]=dict(before_macro_rmse_rad=old,after_macro_rmse_rad=new,
                    relative_change=new/old-1,improved_episodes=sum(after[i][key]<before[i][key] for i in cohort),
                    unchanged_episodes=sum(after[i][key]==before[i][key] for i in cohort),
                    worsened_episodes=sum(after[i][key]>before[i][key] for i in cohort))
            group['both_arm_and_hand_improved']=sum(all(after[i][k]<before[i][k] for k in ('arm_rmse','hand_rmse')) for i in cohort)
            if name=='latest52':
                group['logged_pooled']={stage:{k:values[k] for k in ('arm_rmse_rad','hand_rmse_rad','selection')}
                    for stage,values in (('before',initial['correction']),('after',last['correction']))}
            row['groups'][name]=group
        if role=='A':row['selected_new_correction_metrics']=row['groups']
        row['per_episode']=[dict(identity=i,family=entries[i]['family'],held_schedule=i in held,
            before={k:before[i][k] for k in ('arm_rmse','hand_rmse')},after={k:after[i][k] for k in ('arm_rmse','hand_rmse')}) for i in sorted(ids)]
        report['roles'][role]=row
    report['summary_binding']=dict(path='report.json',sha256=sha(args.evidence/'report.json'))
    report['script']=dict(path=Path(__file__).name,sha256=sha(Path(__file__)))
    report['actual_command']='.venv_isaacsim/bin/python research_docs/evidence/tight_fresh_20261114/round09_front_learning/fit_summary.py --session <session06> --evidence research_docs/evidence/tight_fresh_20261114/round09_front_learning --private <session06>/cpu_audits/round09_front_learning01'
    for path in (args.private/'fit_report.json',args.evidence/'fit_report.json'):
        with path.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({role:row['groups'] for role,row in report['roles'].items()},indent=2))


if __name__=='__main__':main()
