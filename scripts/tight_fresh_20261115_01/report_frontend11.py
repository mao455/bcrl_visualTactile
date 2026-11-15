#!/usr/bin/env python3
"""Report completed frontends from small receipts, without models/raw payloads."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def summarize(root):
    bindings={}
    def read(path,lines=False):
        payload=path.read_bytes()
        bindings[str(path.relative_to(root))]=hashlib.sha256(payload).hexdigest()
        return [json.loads(line) for line in payload.splitlines()] if lines else json.loads(payload)
    def metric(value):
        return {key:value[key] for key in ('arm_rmse_rad','hand_rmse_rad','selection')}
    candidate=root/'round_11';stages=root/'round_11_stages';roles={}
    for role in ('A','B'):
        directory=candidate/role;training=directory/'training'
        done=read(training/'finished.json');assert done['complete']
        data=read(directory/'state_data/data.json');latest=data['aggregation']['new_training_ids']
        rows=read(training/'validation.jsonl',lines=True)
        selected=min(rows,key=lambda row:row['validation']['selection']);last=rows[-1]
        initial=read(training/'initial_metrics.json');coverage=read(training/'sampling_coverage.json')
        roles[role]=dict(complete=True,epochs=done['epochs'],optimizer_updates=done['updates'],
            episode_visits=done['visits_seen'],selected_epoch=selected['epoch'],
            selected_checkpoint_sha256=done['checkpoint_sha256'],
            selection_source='First minimum of completed validation receipts; checkpoint tensors not reopened',
            stage_seconds=read(stages/('frontend_'+role+'.json'))['seconds'],
            initial_validation=metric(initial['validation']),selected_validation=metric(selected['validation']),
            validation_selection_relative_reduction=1-selected['validation']['selection']/initial['validation']['selection'],
            initial_new_correction=metric(initial['correction']),
            last_epoch_new_correction=metric(last['correction']),
            selected_new_correction=metric(selected['correction']) if 'correction' in selected else None,
            selected_correction_scope='Missing means this selected epoch did not evaluate that cohort; never substitute last-epoch metrics',
            new_corrections=len(latest),
            selected_new_correction_visits={identity:selected['sampling_exposure'][identity] for identity in latest},
            last_new_correction_visits={identity:coverage['episodes'][identity]['visits'] for identity in latest},
            last_sampling_coverage=done['visit_sampling_coverage'])
    status=read(stages/'status.json')
    evaluation=candidate/'development/result.json'
    autonomous_complete=read(evaluation).get('software_complete') is True if evaluation.exists() else False
    return dict(version='frontend11_small_receipts_v1',observed_at=datetime.datetime.now().astimezone().isoformat(),
        active_stage=status['stage'],roles=roles,visual_cache_seconds=read(stages/'visual_cache.json')['seconds'],
        actual_frontend_stage_seconds=sum(value['stage_seconds'] for value in roles.values()),
        models_opened=0,raw_payloads_opened=0,learning_jsonl_opened=0,
        autonomous_evaluation_complete=autonomous_complete,
        interpretation='Within-role offline metrics only; A labels q[t+1], B expert targets. Distinct ticks include episode identity. No autonomous benefit or speedup claim.',
        input_sha256=bindings)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(summarize(args.root),indent=2))
