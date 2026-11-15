"""Bounded post-training grasp review: difficult train starts and fresh development."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import time

from tools.tight_fresh_grasp_evidence import behavior, read
from tools.tight_fresh_eval_recovery import wait_owned, stop_owned
from visuotactile.training_resources import atomic_json


def prepare_cohort(root, collection, output, config, kind):
    from tools.tight_fresh_batch_run import prepare
    plan = prepare(root, output, lanes=config['lanes'], count=config['development_count_per_role'],
                   seed=config['development_seed'], execution_backend=config['execution_backend'],
                   vision_microbatch=config['vision_microbatch'])
    data=read(root/'A/state_data/data.json')
    if kind == 'hard':
        source=read(collection/'plan.json')
        cases=[]
        for case in source['cases']:
            if case['spec']['family'] not in config['hard_families']: continue
            spec=copy.deepcopy(case['spec'])
            for key in ('takeover_tick','training_expansion','initial_configuration_sha256'): spec.pop(key,None)
            spec.update(identity='hard_'+case['name'],cohort='fresh',split='train')
            cases.append(dict(name=spec['identity'],role=case['role'],spec=spec,video=False))
        if len(cases) != 2*len(config['hard_families']): raise ValueError('Hard-family coverage differs')
        plan.update(cases=cases,cohort='Training-family autonomous diagnostics; new real trajectory, no expert, not held-out')
    else:
        from visuotactile.training_expansion import read_registry, configuration_digest
        train_seeds=set(); train_configs=set()
        for row in read(config['frozen_families']):
            train_seeds.add(row['seed']);train_configs.add(configuration_digest(row['narrow_reset']))
        for binding in data.get('training_expansion_registries',[]):
            for row in read_registry(binding)['families'].values():
                train_seeds.add(row['seed']);train_configs.add(row['configuration_sha256'])
        for case in plan['cases']:
            if case['spec']['seed'] in train_seeds or configuration_digest(case['spec']['narrow_reset']) in train_configs:
                raise ValueError('New development family overlaps training')
            case['video']=False
        plan['cohort']='New independent development reset families; excluded from training; original test22 unopened'
    plan['grasp_review_kind']=kind
    atomic_json(output/'plan.json',plan)
    return plan


def run(root, collection, output, configuration):
    config=read(configuration);output.mkdir(parents=True,exist_ok=False)
    deadline=read(root/'session.json')['deadline']-60
    state=dict(started=time.time(),deadline=deadline,complete=False,stages=[],behaviors={})
    tree=Path.cwd();python=str(tree/'.venv_isaacsim/bin/python')
    def execute(name, arguments):
        if time.time() >= deadline: raise TimeoutError('Original absolute deadline reached')
        stage=dict(name=name,started=time.time(),command=arguments);state['stages'].append(stage)
        atomic_json(output/'status.json',state)
        with (output/(name+'.log')).open('x') as log:
            child=subprocess.Popen(arguments,cwd=tree,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                reason=wait_owned(child,deadline)
                if reason or child.returncode: raise RuntimeError(f'{name}: {reason}, exit={child.returncode}')
            finally:
                if child.poll() is None: stop_owned(child)
        stage.update(ended=time.time(),complete=True)
        atomic_json(output/'status.json',state)
    try:
        state['behaviors']['fixed']=behavior(root/'development')
        for kind in ('hard','new_development'):
            directory=output/kind
            prepare_cohort(root,collection,directory,config,kind)
            execute(kind,[python,'-m','tools.tight_batch_owner','--output',str(directory),
                '--worker-tree',str(tree),
                '--startup-seconds','300'])
            state['behaviors'][kind]=behavior(directory)
        for role in ('A','B'):
            execute('frontend_fit_'+role,[python,'-m','tools.tight_fresh_grasp_fit','--root',str(root),
                '--role',role,'--output',str(output/('frontend_fit_'+role+'.json'))])
        prior=read(config['source_evidence']);decisions={}
        old_fail={(r['role'],r['family']) for r in prior['retrospective']['failed_training_prefixes']}
        for role in ('A','B'):
            fixed=state['behaviors']['fixed']['counts'][role]
            dev=state['behaviors']['new_development']['counts'][role]
            old=prior['sources'][role]['counts'][role]
            improved=[r['seed'] for r in state['behaviors']['hard']['rows'] if r['role']==role
                and (role,r['family']) in old_fail and r['legal'] and r['hold60']['sustained']]
            total=fixed['legal']+dev['legal'];held=fixed['hold60']+dev['hold60']
            preserves=(fixed['legal']==old['legal'] and fixed['hold60']>=old['hold60'])
            stage_target=(total>=config['minimum_legal_per_role_for_stage_target'] and
                held/max(total,1)>=config['minimum_sustained_hold_fraction'] and
                fixed['hold60']/max(fixed['legal'],1)>=.9 and dev['hold60']/max(dev['legal'],1)>=.9)
            decisions[role]=dict(previously_failed_now_hold60_seeds=improved,
                fixed_source_hold60=old['hold60'],fixed_candidate_hold60=fixed['hold60'],
                preserves_fixed_grasp=preserves,combined_hold60=held,combined_legal=total,
                observed_90pct_target=stage_target,
                adopt_for_next_grasp_collection=preserves and bool(improved),
                promote_to_late_focus=False,
                reason='A later independent cohort must confirm reliability; unchanged fallback remains until review')
        state.update(complete=True,decisions=decisions,test_used=False)
    except BaseException as error:
        state['error']=repr(error)
        raise
    finally:
        state['ended']=time.time();atomic_json(output/'result.json',state);atomic_json(output/'status.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__)
    for name in ('root','collection','output','configuration'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.root,a.collection,a.output,a.configuration)
