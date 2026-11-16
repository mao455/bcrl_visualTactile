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

def install(output):
    from visuotactile import narrow_data
    from visuotactile.environment_rng import LaneRandomStreams
    from visuotactile.gpu_batch_controller import MultiPolicyController
    plan=read(output/'plan.json');assert len(plan['cases'])==plan['batch']['lanes']
    original=narrow_data.configure_direct_reset_lanes
    def configured(env,configurations):
        original(env,configurations)
        reset=env._reset_idx
        streams=LaneRandomStreams([c['spec']['seed']+500000 for c in plan['cases']],device=str(env.device))
        def paired_reset(env_ids):
            ids=list(range(env.num_envs)) if env_ids is None else env_ids.tolist()
            for lane in ids:
                with streams.scope(lane):reset(torch.tensor([lane],device=env.device,dtype=torch.long))
        env._reset_idx=paired_reset
        atomic_json(output/'paired_reset_installed.json',dict(seeds=[c['spec']['seed']+500000 for c in plan['cases']],
            contract='same per-family native reset stream; surrounding event/model RNG restored; no direct state forcing',utc=__import__('time').time()))
    narrow_data.configure_direct_reset_lanes=configured
    oldact=MultiPolicyController.act;checked=False
    def verified(self,packet,diagnostics,active):
        nonlocal checked
        if not checked:
            rows=[];groups={}
            for lane,c in enumerate(plan['cases']):groups.setdefault(c['spec']['seed'],[]).append(lane)
            for seed,ids in groups.items():
                base=ids[0]
                # Include illegal starts separately; they never receive control.
                for lane in ids[1:]:
                    q=float((diagnostics['joint_position_rad'][lane]-diagnostics['joint_position_rad'][base]).abs().max())
                    qd=float((diagnostics['joint_velocity_rad_s'][lane]-diagnostics['joint_velocity_rad_s'][base]).abs().max())
                    target=float((packet['actuator_target_rad'][lane]-packet['actuator_target_rad'][base]).abs().max())
                    rows.append(dict(seed=seed,reference=base,lane=lane,q_max_rad=q,qd_max_rad_s=qd,target_max_rad=target,
                        reference_active=active[base],active=active[lane]))
            receipt=dict(rows=rows,passed=all(max(r['q_max_rad'],r['qd_max_rad_s'],r['target_max_rad'])<=1e-5 and r['active']==r['reference_active'] for r in rows),
                scope='Pre-action q/qdot/actuator targets and legal masks; object/hole reset parameters identical. Hidden PhysX/sensor state not claimed exact.')
            atomic_json(output/'paired_initial_check.json',receipt)
            assert receipt['passed'],'Native paired reset failed; no action admitted'
            checked=True
        return oldact(self,packet,diagnostics,active)
    MultiPolicyController.act=verified

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
    count=4 if remaining>=420 else 2 if remaining>=270 else 1 if remaining>=180 else 0
    assert count,'Insufficient bounded time for a complete native task; do not launch another simulation'
    templates={seed:templates[seed] for seed in seeds[:count]};seeds=list(templates)
    plan['bounded_cohort_selection']=dict(remaining_seconds=remaining,selected_seeds=seeds,
        rule='First4 if >=7min; first2 if >=4.5min; first1 if >=3min; otherwise stop. Registered before updated outcomes, no replacement.')
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
