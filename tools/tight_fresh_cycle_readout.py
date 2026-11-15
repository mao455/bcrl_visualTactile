"""One compact readout of an actually completed fresh collection/training cycle.

Reuses admission/completion receipts. Reads numeric autonomy once; never rescans
RGB-D payloads, restarts models, or releases the supervised-night barrier.
"""
import argparse
import json
from pathlib import Path
import time

import torch

from tools.tight_fresh_grasp_evidence import behavior
from visuotactile.training_resources import atomic_json


def read(path): return json.loads(Path(path).read_text())
def scalars(value): return {k:v for k,v in value.items() if isinstance(v,(str,int,float,bool)) or v is None}


def phase_exposure(directory, saved, records, identities, *, rear=False):
    """Actual legal action slots, including overlapping DP target windows."""
    from visuotactile.target_windows import window_mask
    totals=dict(all_legal=0,before100=0,before150=0,latest_legal=0,latest_before100=0,latest_before150=0)
    def add(identity,ticks,multiplier=1):
        n=len(ticks)*multiplier;early100=int((ticks<100).sum())*multiplier;early150=int((ticks<150).sum())*multiplier
        totals['all_legal']+=n;totals['before100']+=early100;totals['before150']+=early150
        if identity in identities:
            totals['latest_legal']+=n;totals['latest_before100']+=early100;totals['latest_before150']+=early150
    if rear:
        for identity,count in saved['episode_counts'].items():
            add(identity,records[identity]['supervision_mask'].nonzero().flatten(),count)
        assert totals['all_legal']==saved['supervised_ticks']
    else:
        with (directory/'learning.jsonl').open() as stream:
            for line in stream:
                row=json.loads(line)
                if row['updates']>saved['updates']:break
                for visit in row['visits']:
                    identity=visit['episode'];record=records[identity]
                    anchors=torch.tensor(visit['anchor_sampling']['selected_anchors'])
                    mask=window_mask(record['supervision_mask'],record['control_segment'],anchors,24)
                    indices=anchors[:,None]+torch.arange(24)
                    assert int(mask.sum())==visit['target_ticks']
                    add(identity,indices[mask])
        assert totals['all_legal']==saved['target_ticks']
    totals['scope']='Selected checkpoint training only; legal target exposures, not unique states. DP overlapping horizon slots count repeatedly; rear each visited legal control tick once. No loss-weight mass is called additional data.'
    return totals


def training(directory, identities, *, rear=False, records=None):
    done=read(directory/'finished.json')
    saved=torch.load(directory/'best.pt',map_location='cpu',mmap=True,weights_only=False)
    counts=saved['episode_counts' if rear else 'sampling_exposure']
    history=saved['history'][-1]
    result=dict(completed=scalars(done),selected_epoch=saved['epoch'],selected_updates=saved['updates'],
        selected_latest_visits={identity:counts[identity] for identity in identities},
        selected_metrics={k:scalars(history[k]) for k in ('train','validation','correction') if isinstance(history.get(k),dict)})
    if rear:
        result['selected_supervised_ticks']=saved['supervised_ticks']
        weighting=saved['configuration'].get('fresh_recovery_weighting')
        if weighting is not None and records is not None:
            from visuotactile.fresh_recovery import GRASP_STAGE_WEIGHTING_VERSION
            assert weighting['version']==GRASP_STAGE_WEIGHTING_VERSION
            ticks=weighting['ticks'];counts=saved['episode_counts']
            early=sum(int((records[i]['supervision_mask'].cpu() & (torch.arange(len(records[i]['robot']))<ticks)).sum())*n
                      for i,n in counts.items())
            total=saved['supervised_ticks'];weight=weighting['backend_weight'];mass=total+(weight-1)*early
            result['selected_early_weight_mass']=dict(contract=weighting,raw_early_ticks=early,
                all_raw_ticks=total,all_loss_weight_mass=mass,early_loss_weight_mass=weight*early,
                early_weight_fraction=weight*early/mass,weights_are_not_additional_observations=True)
            result['completed_weighted_exposure']=done.get('fresh_recovery_exposure')
        result['completed_latest_exposure']={i:done['episode_sampling_exposure']['episodes'][i] for i in identities}
        result['latest_fit_initial']=read(directory/'latest_fit_initial.json')['summary']
        result['latest_fit_selected']=read(directory.parent/'offline/latest_fit_selected.json')['summary']
        result['latest_fit_scope']='Same newly selected DP inputs; old versus selected rear on real saved expert suffixes. Teacher-forced, not autonomous.'
    else:
        result['completed_latest_visits']={i:done['sampling_exposure'][i] for i in identities}
        result['completed_distinct_time_coverage']=done.get('visit_sampling_coverage')
        if saved.get('visit_sampling_coverage') is not None:
            from visuotactile.fresh_visit_sampling import Coverage
            result['selected_distinct_time_coverage']=Coverage(saved['visit_sampling_coverage']).summary()
    if records is not None:
        result['selected_phase_exposure']=phase_exposure(directory,saved,records,identities,rear=rear)
    return result


def run(candidate, collection, *, evaluation=None):
    ledger=candidate.with_name(candidate.name+'_stages')
    coordinator_complete=(ledger/'finished.json').exists() and read(ledger/'finished.json')['software_complete']
    if not coordinator_complete:
        assert evaluation is not None, 'Interrupted coordinator requires an explicit separately completed evaluation'
        for stage in ['frontend_A','frontend_B']+[f'backend_{part}_{role}' for role in 'AB' for part in ('cache','train','offline')]:
            assert read(ledger/(stage+'.json'))['returncode']==0
    evaluation=candidate/'development' if evaluation is None else Path(evaluation)
    assert read(evaluation/'result.json')['software_complete']
    collected=read(collection/'result.json');assert collected['software_complete']
    data=read(candidate/'A/state_data/data.json');identities=data['aggregation']['new_training_ids']
    records=torch.load(candidate/'A/state_data/numeric.pt',map_location='cpu',mmap=True,weights_only=False)['records']
    cases=[]
    for case in read(collection/'plan.json')['cases']:
        result=read(collection/case['name']/'result.json');dagger=result.get('dagger',{})
        trigger=dagger.get('trigger_receipt',{});coverage=dagger.get('recovery_coverage',{}).get('population',{})
        tick=trigger.get('actual_scheduled_takeover_tick')
        cases.append(dict(case=case['name'],role=case['role'],family=case['spec']['family'],legal=result['legal'],
            accepted=result.get('accepted_for_training',False),expert_success=dagger.get('expert_takeover_success',False),
            takeover_tick=tick,takeover_sim_seconds=None if tick is None else tick/60,
            first_contact_completed_tick=trigger.get('first_contact_completed_tick'),
            first_legal_expert_tick=coverage.get('first_valid_expert_tick'),
            legal_expert_ticks=dagger.get('expert_valid_ticks',0),duration_sim_seconds=result.get('duration_sim_s'),
            reason=trigger.get('reason')))
    stages={}
    # Successful receipts survive the controlled resume in the original ledger.
    for p in ledger.glob('*.json'):
        receipt=read(p)
        if isinstance(receipt,dict) and receipt.get('returncode')==0 and 'seconds' in receipt:
            stages[receipt['stage']]=dict(seconds=receipt['seconds'],started=receipt['started'],ended=receipt['utc'])
    autonomous=behavior(evaluation,horizon=300)
    from tools.tight_fresh_cycle_throughput import union_seconds
    groups=dict(frontend=('frontend_A','frontend_B'),backend_inputs=('backend_cache_A','backend_cache_B'),
        backend_training=('backend_train_A','backend_train_B'),visual_cache=('visual_cache',),
        autonomous_evaluation=('development',))
    wall={name:union_seconds([(stages[k]['started'],stages[k]['started']+stages[k]['seconds'])
                              for k in keys if k in stages]) for name,keys in groups.items()}
    evaluation_result=read(evaluation/'result.json')
    wall['collection_worker']=collected['elapsed_seconds']
    wall['collection_start_to_evaluation_end']=evaluation_result['utc']-(collected['utc']-collected['elapsed_seconds'])
    if not coordinator_complete:wall['autonomous_evaluation']=evaluation_result['elapsed_seconds']
    # Keep training-family and development counts separate; native task outcomes
    # come from the complete actual trace, while holding uses first300 controls.
    return dict(complete=True,coordinator_complete=coordinator_complete,utc=time.time(),candidate=str(candidate),collection=str(collection),evaluation=str(evaluation),
        collection_result=collected,collection_cases=cases,new_training_ids=identities,
        new_trajectories=len(identities),corrected_family_ids=sorted({r['family'] for r in cases if r['accepted']}),
        train_trajectories=len(data['splits']['train']),
        networks={role:dict(dp=training(candidate/role/'training',identities,records=records),
            rear=training(candidate/role/'backend'/role/'training',identities,rear=True,records=records)) for role in 'AB'},
        stages=stages,wall_seconds=wall,wall_semantics='Stage groups use unions of real intervals, not sums of concurrent worker times. End-to-end starts at collection worker and includes admission, cache, updates, recovery and evaluation; excludes earlier source-comparison preparation.',autonomy=autonomous,
        full_task_legal=sum(r['legal'] and not r['grasp_stage_censored'] for r in autonomous['rows']),
        full_task_insertions=sum(r['legal'] and not r['grasp_stage_censored'] and r.get('insertion',False) for r in autonomous['rows']),
        evaluation_wall_seconds=evaluation_result.get('elapsed_seconds'),
        limitations='Four attempts are two existing training families, not four independent new starts. Expert rescue, offline fit and student autonomy are separate. This report does not release unattended work or certify grasp reliability.')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,help='Explicit separate completed evaluation after a preserved software failure')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve original cycle readout')
    torch.set_num_threads(2)
    atomic_json(a.output,run(a.candidate.resolve(),a.collection.resolve(),evaluation=a.evaluation))
