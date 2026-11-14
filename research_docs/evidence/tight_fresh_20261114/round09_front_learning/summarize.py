"""Completed round09 frontend receipts only; never open model/numeric/RGB payloads."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from tools.tight_fresh_learning_report import exposure_rows,learning_sources
from tools.tight_fresh_stage_exposure import update_counts


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--session',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();session=a.session.resolve();directory=Path(__file__).resolve().parent;bindings={}
    def read(relative,lines=False):
        path=session/relative;payload=path.read_bytes();assert path.suffix in ('.json','.jsonl')
        bindings[relative]=dict(sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload))
        return [json.loads(line) for line in payload.splitlines()] if lines else json.loads(payload)
    stage=read('round_09_stages/registration.json')
    assert stage['source_commit']=='87577e84f5f9183ba739e483b34729029629bb99'
    census=read('cpu_audits/collection09_final_review_01/final128.json')
    accepted=read('new_family_09/correction_entries.json')
    cases={r['case']:r for r in census['cases'] if r['accepted']}
    assert set(accepted)==set(cases) and len(accepted)==52
    held={i for i,r in cases.items() if r['schedule'] in ('held60','held120')}
    assert len(held)==13
    report=dict(version='round09_frontend_json_receipts_v1',created_utc=time.time(),source_commit=stage['source_commit'],
        frontend_complete=True,backend_scope='Not audited here; still running when this report was requested',
        method=dict(optimizer_trajectory_batch=1,training_anchor_rule='existing deterministic legal planning_anchors on commit12 grid',
            horizon=24,planning_hz=5,control_hz=60,full_real_history_each_visit=True,latest_repeat=2,
            b2_trained=False,random_training_anchors=False,rotating_geometry_candidate_trained=False,
            eager_visual_io=False,offline_metrics_anchor_batch=32),
        populations=dict(training_trajectories=170,latest_accepted=52,latest_families=len({r['family'] for r in cases.values()}),
            latest_raw_legal_expert_ticks=sum(r['valid_expert_controls'] for r in cases.values()),
            latest_held_trajectories=13,latest_held_families=len({cases[i]['family'] for i in held}),
            latest_held_raw_legal_expert_ticks=sum(cases[i]['valid_expert_controls'] for i in held)),roles={},
        semantics=dict(target_slots='Actual legal H24 action-label occurrences from cumulative learning-row differences; overlapping horizons and repeated visits count again.',
            held_group='All legal suffix actions of accepted held60/held120 trajectories; not only ticks while held.',
            selected_identity='First strictly minimal logged validation selection, following frozen trainer rule; finished selected SHA agrees with backend preparation. Checkpoint metadata/weight bytes not reopened.',
            loss_scope='Exposure counts are not gradient contribution, geometry sample counts or distinct independent demonstrations.',
            policy_scope='Teacher-forced offline learning evidence; no claim of improved autonomous holding, orientation, alignment or insertion.'),
        forbidden_payload_reads=dict(weights=0,numeric=0,rgb=0,heldout_payloads=0),optimizer_updates_by_report=0)
    per_episode={}
    def compact(value):return {k:value[k] for k in ('selection','arm_rmse_rad','hand_rmse_rad')}
    for role in 'AB':
        prefix='round_09/'+role+'/';data=read(prefix+'state_data/data.json');cfg=read(prefix+'training/configuration.json')
        finished=read(prefix+'training/finished.json');initial=read(prefix+'training/initial_metrics.json')
        rows=read(prefix+'training/learning.jsonl',True);history=read(prefix+'training/validation.jsonl',True)
        last_json=read(prefix+'training/validation.json');backend=read(prefix+'backend/prepared.json')
        continuation=read(prefix+'continuation.json')
        assert finished['complete'] and cfg['version']=='fresh30_continuous_episode_latest_repeat_v1'
        assert not cfg.get('full_trajectory_batch') and not cfg.get('training_sampling') and not cfg.get('frozen_visual_io')
        assert backend['source_frontend_sha256']==finished['checkpoint_sha256']
        assert cfg['data_sha256']==bindings[prefix+'state_data/data.json']['sha256']
        training=set(data['splits']['train']);corrections,current,groups=learning_sources(data)
        assert current==set(accepted) and len(training)==170 and len(data['splits']['validation'])==6
        assert all(data['entries'][i]['source_sha256']==accepted[i]['source_sha256'] for i in current)
        assert all(data['entries'][i]['expansion_schedule']==cases[i]['schedule'] for i in current)
        assert len(rows)==finished['updates']==2220 and [r['epoch'] for r in history]==list(range(1,11))
        assert last_json==history[-1]
        selected=min(history,key=lambda r:r['validation']['selection'])
        result=dict(selected_checkpoint_sha256=finished['checkpoint_sha256'],source_checkpoint_sha256=continuation['sha256'],
            source_root=Path(continuation['source_root']).name,normalization_identity=cfg['normalization_identity'],
            completed_frontend_seconds=finished['wall_seconds'],initial_validation=compact(initial['validation']),
            selected={},last={},validation_history=[dict(epoch=r['epoch'],updates=r['updates'],**compact(r['validation'])) for r in history])
        per_episode[role]={}
        for name,epoch in (('selected',selected),('last',history[-1])):
            limit=epoch['updates'];counts,total=update_counts(rows,training,limit)
            observed,exposure=exposure_rows(rows[:limit],training,corrections,source_groups=groups)
            assert dict(observed)==epoch['sampling_exposure']
            assert all(total[k]==epoch[k] for k in total)
            if name=='last':
                assert dict(observed)==finished['sampling_exposure'] and all(total[k]==finished[k] for k in total)
            def bucket(ids):
                output=dict(trajectories=len(ids),visits=sum(counts[i]['visits'] for i in ids),
                    anchors=sum(counts[i]['anchors'] for i in ids),legal_H24_target_slots=sum(counts[i]['target_slots'] for i in ids),
                    visits_per_trajectory=sorted(set(counts[i]['visits'] for i in ids)))
                output['share_all_legal_H24_target_slots']=output['legal_H24_target_slots']/total['target_ticks']
                return output
            result[name]=dict(epoch=epoch['epoch'],optimizer_updates=limit,episode_visits=limit,anchors=total['anchors_seen'],
                legal_H24_target_slots=total['target_ticks'],validation=compact(epoch['validation']),
                latest52=bucket(current),held13=bucket(held),exposure_by_source=exposure,
                latest_by_schedule={s:bucket({i for i in current if cases[i]['schedule']==s}) for s in sorted({cases[i]['schedule'] for i in current})})
            result[name]['validation_relative_change_vs_source']={k:epoch['validation'][k]/initial['validation'][k]-1
                for k in ('selection','arm_rmse_rad','hand_rmse_rad')}
            if 'correction' in epoch:result[name]['latest_correction_fit']=compact(epoch['correction'])
            else:result[name]['latest_correction_fit']=None
            per_episode[role][name]={i:dict(counts[i],schedule=cases[i]['schedule'],family=cases[i]['family']) for i in sorted(current)}
        result['correction_fit_note']='Only final epoch correction cohort was measured; B selected epoch9 has no correction metric receipt.'
        report['roles'][role]=result
    report['input_bindings']=bindings
    report['script']=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__)))
    report['command']='PYTHONPATH=<reviewed-source> <python> '+str(Path(__file__).resolve())+' --session '+str(session)+' --public '+str(a.public)
    private=dict(report,per_episode=per_episode)
    def exclusive(path,value):
        with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    exclusive(directory/'report.json',private)
    public=dict(report);public['private_receipt']=dict(path=str((directory/'report.json').relative_to(session)),sha256=sha(directory/'report.json'))
    public['script']['path']=str(Path(__file__).relative_to(session))
    public['command']='PYTHONPATH=<reviewed-source> <python> <session>/'+public['script']['path']+' --session <session> --public <public-evidence-directory>'
    a.public.mkdir(parents=True,exist_ok=False);exclusive(a.public/'report.json',public)
    print(json.dumps(dict(populations=report['populations'],roles={r:{k:report['roles'][r][k] for k in ('selected','last')} for r in 'AB'}),indent=2))


if __name__=='__main__':main()
