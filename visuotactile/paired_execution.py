"""Native reset pairing and pre-action checks, lifted without math changes.

Install only inside an isolated simulation worker; all historical callers share
this implementation. It preserves native reset draws and verifies observed state,
not hidden PhysX state.
"""
import torch
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
                    physical={}
                    for key in ('object_pose_w','hole_pose_w','goal_pose_w','table_pose_w','palm_body_pose_w'):
                        a=diagnostics[key][base].clone();b=diagnostics[key][lane].clone()
                        a[:3]-=diagnostics['base_pose_w'][base,:3]
                        b[:3]-=diagnostics['base_pose_w'][lane,:3]
                        physical[key+'_position_m']=float((a[:3]-b[:3]).abs().max())
                        physical[key+'_quaternion']=float(torch.minimum((a[3:]-b[3:]).abs().max(),(a[3:]+b[3:]).abs().max()))
                    for key in ('object_velocity_w_m_s_rad_s','goal_velocity_w_m_s_rad_s'):
                        physical[key]=float((diagnostics[key][lane]-diagnostics[key][base]).abs().max())
                    rows.append(dict(seed=seed,reference=base,lane=lane,q_max_rad=q,qd_max_rad_s=qd,target_max_rad=target,
                        physical=physical,reference_active=bool(active[base]),active=bool(active[lane])))
            receipt=dict(rows=rows,passed=all(max(r['q_max_rad'],r['qd_max_rad_s'],r['target_max_rad'],*r['physical'].values())<=1e-5 and r['active']==r['reference_active'] for r in rows),
                scope='Pre-action q/qdot/actuator targets, base-relative physical poses, object/goal velocities and legal masks; tolerance 1e-5 in each reported unit. Hidden PhysX/sensor state and future random draws not claimed exact.')
            atomic_json(output/'paired_initial_check.json',receipt)
            assert receipt['passed'],'Native paired reset failed; no action admitted'
            checked=True
        return oldact(self,packet,diagnostics,active)
    MultiPolicyController.act=verified
