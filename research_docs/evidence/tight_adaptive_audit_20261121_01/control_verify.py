from pathlib import Path
from types import SimpleNamespace as NS
import sys,json,importlib.util
import h5py,numpy as np,torch
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(SRC))
from visuotactile.position_control import processed_expert_target,project_processed_position_targets
spec=importlib.util.spec_from_file_location('audit_actual_native_action',SRC/'isaacsimenvs/tasks/play/utils/action_utils.py');native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
def maximum(a,b):return float((a-b).abs().max())
def main():
 torch.set_num_threads(2);rows=[]
 for fact in json.load(open(OUT/'data_chain.json'))['episodes']:
  with h5py.File(fact['path']) as f:
   g=f['episodes'][fact['identity']];tr=g['transition'];d=tr['dagger'];meta=json.loads(f.attrs['metadata_json']);n=len(d['control_source']);v=lambda k:torch.from_numpy(d[k][:]);t=lambda k:torch.from_numpy(tr[k][:]);mask=v('expert_label_valid').bool();source=v('control_source').bool();indices=v('expert_delay_index').long();steps=v('expert_source_step').long();limits=torch.tensor(meta['joint_limits_rad'][0]);before=t('target_before_rad');target=t('target_after_rad');act=meta['config']['action']
   proxy=NS(device='cpu',num_envs=n,step_dt=meta['control_dt'],_perm_canon_to_lab=torch.arange(29),_arm_joint_ids=torch.arange(7),_hand_joint_ids=torch.arange(7,29),_arm_lower=limits[:7,0],_arm_upper=limits[:7,1],_hand_lower=limits[7:,0],_hand_upper=limits[7:,1],_prev_targets=before.clone(),_cur_targets=before.clone(),cfg=NS(action=NS(**{k:act[k] for k in ('dof_speed_scale','arm_moving_average','hand_moving_average')}),domain_randomization=NS(use_action_delay=False,action_delay_max=0)))
   native.apply_action_pipeline(proxy,v('expert_delayed_action'));progressed,mapped=processed_expert_target(before,proxy._cur_targets,dict(arm=.5,hand=1.,hand_target_cap_rad_s=2.),1/60);projected=project_processed_position_targets(target,before,limits,meta['execution_adapter'])
   expectedmask=source & (torch.arange(n)>=v('takeover_tick')+v('expert_queue_length')-1)&(steps>=v('takeover_tick'))
   errors=dict(raw_to_clipped=maximum(v('expert_raw_action').clamp(-1,1),v('expert_clipped_action')),delayed_source=maximum(v('expert_delayed_action'),v('expert_clipped_action')[steps]),delay_index=maximum(steps,(torch.arange(n)-indices).clamp_min(0)),native_mapped_rad=maximum(proxy._cur_targets,v('expert_native_target_rad')),progressed_rad=maximum(progressed,v('expert_progressed_target_rad')),hand_speed_once_rad=maximum(mapped,v('expert_mapped_target_rad')),expert_executed_rad=maximum(mapped[source],target[source]),student_executed_rad=maximum(v('student_proposed_target_rad')[~source],target[~source]),shared_reachable_projection_rad=maximum(projected,target),mask_formula_exact=bool(torch.equal(mask,expectedmask)),second_delay_enabled=bool(tr['control_trace/delay_enabled'][:].any()))
   rows.append(dict(identity=fact['identity'],round=fact['round'],errors=errors,queue_length=int(v('expert_queue_length')[0]),switch=fact['switch'],first_legal=fact['first_legal']))
 (OUT/'control_verification.json').write_text(json.dumps(rows,indent=2)+'\n');print('count',len(rows),'maxima',{k:max(r['errors'][k] for r in rows) for k in rows[0]['errors']})
if __name__=='__main__':main()
