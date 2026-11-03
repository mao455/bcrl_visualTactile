"""Preregistered, conservative Sample/Epsilon ranking without test access."""
import math


def choose(results):
    if set(results) != {'sample','epsilon'}:
        raise ValueError('Exactly two preregistered models required')
    cohorts={name:{row['identity']:row for row in value['loop']['trials']
        if row.get('legal') and row['cohort']=='validation'} for name,value in results.items()}
    paired=set(cohorts['sample']) & set(cohorts['epsilon'])
    metrics={}
    for name,value in results.items():
        summary=value['offline']['summary']['A0']
        def metric(key):
            number=summary[key]['mean']
            if number is None or not math.isfinite(number):raise ValueError('Missing/invalid ranking metric: '+key)
            return number
        rows=[cohorts[name][identity] for identity in sorted(paired)]
        latency=value['offline']['planning_cycle_latency_s']['A0']['p95']
        if latency is None or not math.isfinite(latency):raise ValueError('Missing/invalid inference timing')
        offline_rows=value['offline']['rows']
        count=sum(row['valid12_positions'] for row in offline_rows)
        if count<=0:raise ValueError('No valid prefix12 labels')
        arm,hand=[math.sqrt(sum(row['valid12_positions']*row[key]**2 for row in offline_rows)/count)
            for key in ('arm12_rmse_rad','hand12_rmse_rad')]
        if not all(math.isfinite(number) for number in (arm,hand)):
            raise ValueError('Nonfinite pooled prefix12 error')
        metrics[name]=dict(successes=sum(bool(row['success']) for row in rows),
            subgoals=sum(bool(row.get('native_subgoals')) for row in rows),
            lifts=sum(bool(row.get('native_lift')) for row in rows),
            ee_mm=metric('precontact_ee12_target_fk_rmse_m')*1000,
            arm=arm,hand=hand,balanced_mse=.5*(arm*arm+hand*hand),valid_positions=count,
            boundary=metric('arm_boundary_jump_rms_rad'),latency=latency)
    for name,other in [('sample','epsilon'),('epsilon','sample')]:
        row=metrics[name];reference=metrics[other]
        row['degraded']=row['arm']>1.2*reference['arm'] or row['hand']>1.2*reference['hand'] or row['boundary']>1.5*reference['boundary']
        row['guard_penalty']=int(row['degraded'] and row['successes']<reference['successes']+2)
    def rank(name):
        row=metrics[name]
        return (row['guard_penalty'],-row['successes'],-row['subgoals'],-row['lifts'],
            row['ee_mm'],row['balanced_mse'],row['boundary'],row['latency'],name!='sample')
    selected=min(results,key=rank)
    return dict(selected=selected,tentative=True,statistical_superiority_established=False,
        paired_validation_ids=sorted(paired),metrics=metrics,ranks={name:list(rank(name)) for name in results},
        reason='Preregistered multiple-metric exploratory ranking; one policy seed and incomplete/repeated-success evidence cannot establish statistical superiority',
        loss_values_not_compared=True)
