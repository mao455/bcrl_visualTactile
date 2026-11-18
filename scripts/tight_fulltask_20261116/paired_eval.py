"""Explicit native reset RNG pairing, with pre-action physical-state verification.

The native reset math/ranges are unchanged. Private streams restore the surrounding
CPU/CUDA/Python/NumPy state; native vectorized event draws remain shared, not exact
hidden-state replay. No lane reuse, no policy input or action changes.
"""
import json,sys
from pathlib import Path
import numpy as np,torch
from visuotactile.training_resources import atomic_json
from tools.tight_fresh_rollouts import read

from visuotactile.paired_execution import install


def upgrade_plan(output):
    import copy
    from scripts.tight_fulltask_20261116.run import ROOT,DIAG
    from visuotactile.compact import sha256
    assert not (output/'run_started.json').exists()
    plan=read(output/'plan.json');assert set(plan['pairs'])=={'A','B'}
    atomic_json(output/'unstarted_pre_pairing_plan.json',plan)
    pairs={r+'_source':p for r,p in read(DIAG/'candidates/plan.json')['pairs'].items()}
    pairs.update({r+'_updated':p for r,p in plan['pairs'].items()})
    templates={c['spec']['seed']:c for c in plan['cases']};seeds=list(templates)
    remaining=read(ROOT/'session.json')['deadline']-__import__('time').time()
    count=4 if remaining>=420 else 2 if remaining>=270 else 1 if remaining>=150 else 0
    if remaining<270:pairs={k:v for k,v in pairs.items() if k.endswith('_updated')}
    assert count,'Insufficient bounded time for a complete native task; do not launch another simulation'
    templates={seed:templates[seed] for seed in seeds[:count]};seeds=list(templates)
    plan['bounded_cohort_selection']=dict(remaining_seconds=remaining,selected_seeds=seeds,
        rule='All4 pairs: first4 seeds if >=420s, first2 if >=270s. With150..269s compare only updated A/B on first seed; otherwise stop. Registered before updated outcomes, no replacement.')
    plan['pairs']=pairs;plan['cases']=[]
    for seed,template in templates.items():
        for group in pairs:
            c=copy.deepcopy(template);c.update(name=group+'_'+c['spec']['identity'],role=group,video=seed in seeds[:2]);plan['cases'].append(c)
    plan['batch']['lanes']=len(plan['cases'])
    plan['paired_native_reset']=read(ROOT/'paired_request.json')
    plan['evaluation_scope'].update(protocol='All four complete pairs receive the same native per-family reset draws; every legal case runs to native termination',
        video_selection='First two preregistered seeds for all four pairs; other cases retain full numeric evidence')
    plan['source_sha256']={str(p):sha256(p) for folder in ['tools','visuotactile','scripts/tight_fulltask_20261116'] for p in Path(folder).rglob('*.py')}
    atomic_json(output/'plan.json',plan)

def main():
    output=Path(sys.argv[sys.argv.index('--output')+1]);install(output)
    from scripts.tight_fulltask_20261116.video_eval import main as video
    video()
if __name__=='__main__':main()
