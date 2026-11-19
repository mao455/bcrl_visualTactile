"""Derive collection assignments from the completed first-round students.

Only original training-family starts may enter this plan. A replayed screen is
an opportunity-selection observation, never a claim of expert recoverability.
"""
import collections
from pathlib import Path

import h5py

from tools.tight_adaptive_dagger import ROOT, PUBLIC, CONFIG, read, write
from visuotactile.adaptive_takeover import profile
from visuotactile.takeover_trigger import make_trigger, observe_transition
from visuotactile.compact import sha256


def derive():
    assert read(ROOT/'round1_review_ready.json')['final_test_opened'] is False
    assert not (ROOT/'final_test').exists()
    settings=read(CONFIG/'experiment.json')
    training=set(settings['training_seeds'])
    permitted=set(range(28817000,28817030))
    assert training<=permitted
    assert not permitted.intersection(settings['development_seeds']+settings['final_test_seeds'])
    registration=read(ROOT/'round01/evaluation/registration.json')
    rows=read(PUBLIC/'round01/evaluation/per_case.json')
    observations={};assignments={};evidence=[]
    for wave_name in registration['waves']:
        wave=Path(wave_name);plan=read(wave/'plan.json')
        for case in plan['cases']:
            seed=case['spec']['seed'];name=case['role']
            if seed not in training or name not in ('A_new','B_new'):continue
            role=name[0];result=read(wave/case['name']/'result.json')
            assert result['legal'] and result['software_complete']
            hits={}
            with h5py.File(wave/case['name']/'numeric.h5') as stream:
                t=stream['episode/transition']
                for stage in ('contact','transport','alignment','instability','post_subgoal'):
                    trigger=make_trigger(profile(stage),10**9)
                    for tick in range(result['steps']):
                        observe_transition(trigger,tick,t,index=tick)
                        if trigger.trigger_completed_tick is not None:break
                    hits[stage]=trigger.trigger_completed_tick
            before=next(r for r in rows if r['seed']==seed and r['role']==role+'_parent')
            after=next(r for r in rows if r['seed']==seed and r['role']==name)
            available=[s for s,n in hits.items() if n is not None]
            if before['success'] and not after['success']:
                preference=('contact','transport','instability','alignment','post_subgoal')
                reason='Earlier reachable screen on a newly lost training success'
            elif after['native_subgoals'] and hits['post_subgoal'] is not None:
                preference=('post_subgoal','instability','alignment','transport','contact')
                reason='Full new student reached native later phase; sample its actual post-subgoal state'
            elif hits['instability'] is not None:
                preference=('instability','transport','contact','alignment','post_subgoal')
                reason='Actual new-student support instability before terminal failure'
            elif hits['contact'] is not None:
                preference=('contact','transport','alignment','instability','post_subgoal')
                reason='Actual new-student contact stall'
            else:
                preference=('transport','alignment','instability','contact','post_subgoal')
                reason='Stage actually reached by the new complete student'
            chosen=next((s for s in preference if s in available),None)
            if chosen is not None:assignments[role,seed]=chosen
            observations[role,seed]=dict(success=after['success'],parent_success=before['success'],
                hits=hits,chosen=chosen,reason=reason)
            evidence.append(dict(role=role,seed=seed,**observations[role,seed],
                source_wave=str(wave),result_sha256=sha256(wave/case['name']/'result.json')))
    assert len(observations)==2*len(training)
    # Fresh executions use these event screens, never the replayed tick number.
    # R1 late failed rescues get earlier stages, even when absent from eval8.
    assignments.setdefault(('A',28817010),'transport')
    assignments.setdefault(('B',28817023),'post_subgoal')
    if ('A',28817004) not in assignments:assignments['A',28817004]='transport'
    counts=collections.Counter()
    family_order=sorted(training,key=lambda seed:(
        not any(observations[r,seed]['parent_success'] and not observations[r,seed]['success'] for r in 'AB'),
        all(observations[r,seed]['success'] for r in 'AB'),seed))
    family_order += [seed for seed in (28817010,28817023) if seed not in family_order]
    family_order += [seed for seed in sorted(permitted) if seed not in family_order]
    family_order=family_order[:28]
    schedule=[]
    for offset in range(0,len(family_order),4):
        families=family_order[offset:offset+4];stages=[]
        for seed in families:
            default=('contact','transport','instability','alignment')[(offset+len(stages))%4]
            per_role={role:assignments.get((role,seed),default) for role in 'AB'}
            for role,stage in per_role.items():
                profile(stage);counts[role,stage]+=1
            stages.append(per_role)
        schedule.append(dict(families=families,stages=stages))
    assert len(set(family_order))==len(family_order) and set(family_order)<=permitted
    assert sum(len(item['families'])*2 for item in schedule)<=56
    document=dict(source=str(ROOT/'round01/learning'),schedule=schedule,evidence=evidence,
        target_accepted=24,minimum_accepted_per_student=12,maximum_opportunities=56,
        identity='Actual round1 selected full pairs; actual starts from original train30 only',
        source_pairs=registration['pairs'],
        stage_assignments=[dict(role=r,stage=s,attempts=n) for (r,s),n in sorted(counts.items())],
        limitation='Observed screen reachability does not guarantee recurrence or rescue; no fixed-time fallback, no state teleport, no development/test capture',
        adaptation='After four unsuccessful attempts at a stage within a role, replace remaining unobserved assignments with transport/contact; preserve all failures')
    (PUBLIC/'round02').mkdir(parents=True,exist_ok=True)
    write(PUBLIC/'round02/strategy.json',document)
    return document


if __name__=='__main__':derive()
