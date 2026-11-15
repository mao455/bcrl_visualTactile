"""CPU-only expert queries on recorded autonomous states and full LSTM history.

Validate against one closed real DAgger shadow stream before interpreting any
new query. Outputs are diagnostic suggestions, never executed expert labels or
new training data. No simulator, state restoration or student update is used.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch
import yaml

from visuotactile.compact import sha256
from visuotactile.position_control import processed_expert_target, ExpertPositionBridge
from visuotactile.training_resources import atomic_json


def load_model(checkpoint, size, tree):
    from rl_games.algos_torch.model_builder import ModelBuilder
    params=yaml.safe_load((tree/'isaacsimenvs/cfg/train/PreciseAssemblySAPG.yaml').read_text())['params']
    model=ModelBuilder().load(params).build(dict(actions_num=29,input_shape=(size,),num_seqs=1,
        value_size=1,normalize_value=True,normalize_input=True,type='extra_param',
        coef_ids=torch.linspace(50.,0.,6),coef_id_idx=size-1))
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if 0 in weights:weights=weights[0]
    model.load_state_dict(weights['model'],strict=True)
    if 'running_mean_std' in weights:model.running_mean_std.load_state_dict(weights['running_mean_std'])
    return model.eval()


@torch.no_grad()
def query(model, path, tree, *, ticks=150, decompose_student=False):
    with h5py.File(path) as f:
        t=f['episode/transition'];m=json.loads(f.attrs['metadata_json'])
        actor=torch.from_numpy(t['expert_actor_input_diagnostic'][:ticks])
        previous=torch.from_numpy(f['episode/obs/actuator_target_rad'][:ticks])
        actual=torch.from_numpy(t['target_after_rad'][:ticks])
        draw=torch.from_numpy(t['expert_bridge_delay_rng_draw' if 'dagger' in t else 'unused_delay_rng_draw'][:ticks]).long().reshape(-1)
        recorded_mean=torch.from_numpy(t['expert_raw_policy_mean_before_clip'][:ticks]) if 'dagger' in t else None
        recorded_mapped=torch.from_numpy(t['dagger/expert_mapped_target_rad'][:ticks]) if 'dagger' in t else None
    assert actor.ndim==2 and torch.isfinite(actor).all() and (actor[:,-1]==50).all()
    states=model.get_default_rnn_state();means=[]
    for row in actor:
        result=model(dict(is_train=False,prev_actions=None,obs=row[None],rnn_states=states))
        states=result['rnn_states'];means.append(result['mus'][0])
    means=torch.stack(means);clipped=means.clamp(-1,1)
    dr=m['config']['domain_randomization'];delay=bool(dr['use_action_delay'] and dr['action_delay_max']>0)
    index=draw if delay else torch.zeros_like(draw)
    assert not (index<0).any()
    delayed=clipped[(torch.arange(len(actor))-index).clamp_min(0)]
    canonical=m['joint_order'];lab=m['joint_order_lab'];perm=torch.tensor([canonical.index(j) for j in lab]);inverse=perm.argsort()
    limits=torch.tensor(m['joint_limits_rad']).reshape(29,2)
    spec=importlib.util.spec_from_file_location('native_counterfactual_action',tree/'isaacsimenvs/tasks/play/utils/action_utils.py')
    native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
    proxy=SimpleNamespace(device='cpu',num_envs=len(actor),step_dt=m['control_dt'],
        cfg=SimpleNamespace(action=SimpleNamespace(**m['config']['action']),
            domain_randomization=SimpleNamespace(use_action_delay=False,action_delay_max=0)),
        _perm_canon_to_lab=perm,_prev_targets=previous[:,perm].clone(),_cur_targets=previous[:,perm].clone(),
        _arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),
        _arm_lower=limits[perm][:7,0],_arm_upper=limits[perm][:7,1],
        _hand_lower=limits[perm][7:,0],_hand_upper=limits[perm][7:,1])
    native.apply_action_pipeline(proxy,delayed)
    _,mapped=processed_expert_target(previous,proxy._cur_targets[:,inverse],ExpertPositionBridge.speed_scheme,m['control_dt'])
    error=actual-mapped;increment=mapped-previous
    def metrics(a,b):
        e=error[a:b];inc=increment[a:b]
        return dict(ticks=len(e),arm_rmse_rad=float(e[:,:7].square().mean().sqrt()),
            hand_rmse_rad=float(e[:,7:].square().mean().sqrt()),
            arm_expert_increment_rmse_rad=float(inc[:,:7].square().mean().sqrt()),
            hand_expert_increment_rmse_rad=float(inc[:,7:].square().mean().sqrt()),
            arm_bias_rad=e[:,:7].mean(0).tolist(),hand_bias_rad=e[:,7:].mean(0).tolist())
    decomposition=None
    if decompose_student:
        with np.load(Path(path).with_name('trace.npz')) as trace:
            stages={name:torch.from_numpy(trace[key][:len(actor)]).float() for name,key in (
                ('nominal','nominal'),('rear_before_projection','execution_preprojection_target_rad'),
                ('final','target'))}
        assert torch.equal(stages['final'],actual), 'Trace/HDF5 command alignment differs'
        from visuotactile.position_control import project_processed_position_targets, REACHABLE_POSITION_EXECUTION
        reproduced=project_processed_position_targets(stages['rear_before_projection'],previous,limits,REACHABLE_POSITION_EXECUTION)
        projection_reconstruction_error=float((reproduced-actual).abs().max())
        assert projection_reconstruction_error<=2e-6, 'Recorded final execution projection did not reproduce'
        stages['projected_nominal_diagnostic']=project_processed_position_targets(stages['nominal'],previous,limits,REACHABLE_POSITION_EXECUTION)
        decomposition={}
        for a,b in [(0,48),(48,100),(100,150)]:
            b=min(b,len(actor))
            if a>=b:continue
            errors={name:value[a:b]-mapped[a:b] for name,value in stages.items()}
            def rmse(value,sl):return float(value[:,sl].square().mean().sqrt())
            row={name:dict(arm_rmse_rad=rmse(e,slice(0,7)),hand_rmse_rad=rmse(e,slice(7,None)))
                 for name,e in errors.items()}
            row['projection_change_max_abs_rad']=float((stages['final'][a:b]-stages['rear_before_projection'][a:b]).abs().max())
            row['projection_reconstruction_max_abs_rad']=projection_reconstruction_error
            decomposition[f'controls_{a}_{b}']=row
    return dict(path=str(path),ticks=len(actor),command_decomposition=decomposition,
        recorded_raw_mean_max_abs_difference=None if recorded_mean is None else float((means-recorded_mean).abs().max()),
        recorded_mapped_target_max_abs_difference=None if recorded_mapped is None else float((mapped-recorded_mapped).abs().max()),
        metrics={f'controls_{a}_{b}':metrics(a,min(b,len(actor))) for a,b in [(0,48),(48,100),(100,150)] if len(actor)>a},
        arm_command_error_rad=error[:,:7].square().mean(-1).sqrt().tolist(),
        hand_command_error_rad=error[:,7:].square().mean(-1).sqrt().tolist(),expert_sha256=m['expert_model_sha256'])


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--validation-numeric',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decompose-student',action='store_true',help='Compare saved nominal/rear/final commands against the same-state expert; A nominal has a different target domain')
    p.add_argument('--evaluation',type=Path,action='append',help='Completed autonomous directory; repeat to compare source and fitted policies with one expert validation')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Preserve prior diagnostic')
    torch.set_num_threads(2);tree=Path(__file__).resolve().parents[1]
    session=json.loads((a.root/'session.json').read_text());checkpoint=Path(session['expert_checkpoint'])
    assert sha256(checkpoint)==session['expert_sha256']
    with h5py.File(a.validation_numeric) as f:size=f['episode/transition/expert_actor_input_diagnostic'].shape[-1]
    model=load_model(checkpoint,size,tree);validation=query(model,a.validation_numeric,tree,ticks=150)
    assert validation['expert_sha256']==session['expert_sha256']
    if (validation['recorded_raw_mean_max_abs_difference'] is None
            or validation['recorded_raw_mean_max_abs_difference']>1e-4
            or validation['recorded_mapped_target_max_abs_difference']>2e-6):
        atomic_json(a.output.with_suffix('.rejected.json'),validation)
        raise ValueError('Recorded recurrent expert/action conversion did not reproduce; no autonomous query conclusion')
    directories=a.evaluation or [a.root/'original_training_probe'];rows=[]
    for directory in directories:
        plan=json.loads((directory/'plan.json').read_text())
        assert json.loads((directory/'result.json').read_text())['software_complete']
        for case in plan['cases']:
            result=json.loads((directory/case['name']/'result.json').read_text())
            if not result['legal']:continue
            row=query(model,directory/case['name']/'numeric.h5',tree,decompose_student=a.decompose_student)
            assert row['expert_sha256']==session['expert_sha256']
            errors=row['hand_command_error_rad'];onset=next((t for t in range(len(errors)-2)
                if all(e>.015 for e in errors[t:t+3])),None)
            row.update(evaluation=str(directory),case=case['name'],role=case['role'],family=case['spec']['family'],
                sustained_hand_error_onset=dict(threshold_rad=.015,minimum_consecutive_controls=3,
                    zero_based_control_tick=onset,simulation_seconds=None if onset is None else onset/60,
                    scope='Descriptive command deviation from diagnostic same-state expert; not a grasp failure criterion.'))
            rows.append(row)
    atomic_json(a.output,dict(complete=True,validation=validation,rows=rows,
        expert_sha256=session['expert_sha256'],source_sha256=sha256(Path(__file__)),simulator_starts=0,optimizer_updates=0,
        scope='Diagnostic expert suggestions on the actual stored student states with its recorded observation stream, full zero-initialized recurrent history and actual delay draws. Original native mapper, current actual target, unchanged expert progression. No suggestions are executed or admitted as expert labels. First150 controls only; no terminal/reset history crossing. A nominal forecasts next reached q, so its difference from an expert command is not a nominal prediction error or evidence authorizing bare-A deployment. B nominal and rear share the expert command target domain.'))


if __name__=='__main__':main()
