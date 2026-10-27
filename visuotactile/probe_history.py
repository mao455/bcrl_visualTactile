"""Audit real trace/memory at fixed weights; no optimizer and no simulation.

Requires an untrained perception-shadow trace (seeded model provenance). Stored
predictions are replay inputs, never inferred from later expert commands.
"""
import argparse
import io
import json
from pathlib import Path
import pickle

import h5py
import numpy as np
import torch

from .dataset import (EpisodeSequenceDataset, CorrectionSequenceDataset,
                      encode_episode_chunks, replay_correction_chunks)
from .history import recurrent_scan
from .kinematics import EncoderKinematics, rotate_force_to_link
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .models.correction import CorrectionPolicy
from .recording import EpisodeReader, flatten, model_fingerprint


def audit_trace(path, reference=None):
    reader = EpisodeReader(path)
    if reader.trace_version != 2 or reader.schema_version != 2:
        raise ValueError("Expected perception2/trace2")
    m = reader.metadata
    fk = EncoderKinematics(m['config']['assets']['robot_urdf'],m['joint_order'])
    assert fk.sha256 == m['fk_urdf_sha256']
    assert m.get('base_orientation_wxyz',[1.,0.,0.,0.]) == [1.,0.,0.,0.]
    rows, terminal, max_fk, max_force, delay_error = 0,0,0.,0.,0.
    reference_fields = ('raw_action','executed_action','target_before_rad','target_after_rad','reward',
        'diagnostics_after/joint_position_rad','diagnostics_after/joint_velocity_rad_s',
        'diagnostics_after/object_pose_w','diagnostics_after/hole_pose_w','diagnostics_after/table_pose_w',
        'diagnostics_after/goal_pose_w','diagnostics_after/successes','diagnostics_after/retract_succeeded')
    with h5py.File(path) as f:
        ref = h5py.File(reference) if reference else None
        try:
            for ep in reader.episodes:
                g = f['episodes'][ep]; length = int(g.attrs['length'])
                obs,tr = g['obs'],g['transition']
                step = obs['step_id'][:length]
                np.testing.assert_array_equal(step,np.arange(length))
                np.testing.assert_array_equal(tr['step_id'][:length],step)
                np.testing.assert_array_equal(tr['episode_id'][:length],obs['episode_id'][:length])
                time = obs['time_s'][:length]
                np.testing.assert_allclose(np.diff(time),m['control_dt'],atol=1e-12,rtol=0)
                np.testing.assert_array_equal(time,tr['time_before_s'][:length])
                np.testing.assert_allclose(tr['time_after_s'][:length]-time,m['control_dt'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(obs['camera_time_s'][:length],np.repeat(time[:,None],2,1),atol=1e-12,rtol=0)
                np.testing.assert_allclose(np.diff(obs['contact_time_s'][:length],axis=1),m['physics_dt'],atol=1e-12,rtol=0)
                command = tr['command_clipped_pre_delay'][:length]
                np.testing.assert_array_equal(command,np.clip(tr['submitted_action'][:length],-1,1))
                np.testing.assert_array_equal(command,tr['executed_action'][:length])
                trace = tr['control_trace']
                source = trace['selected_command_step'][:length]
                assert ((source>=0)&(source<=step)).all() and trace['valid'][:length].all()
                np.testing.assert_array_equal(command[source],trace['delayed_command'][:length])
                np.testing.assert_array_equal(step-source,trace['effective_lag_steps'][:length])
                np.testing.assert_array_equal(source,np.maximum(step-trace['delay_index'][:length],0))
                if 'selected_command_time_s' in trace:
                    delay_error = max(delay_error,float(np.max(np.abs(time[source]-trace['selected_command_time_s'][:length]))))
                    np.testing.assert_allclose(time[source],trace['selected_command_time_s'][:length],atol=3e-9,rtol=0)
                np.testing.assert_array_equal(tr['provenance/expert_command'][:length],command)
                assert tr['provenance/expert_computed'][:length].all()
                assert not tr['provenance/correction_executed'][:length].any()
                plan = tr['plan_trace']
                np.testing.assert_array_equal(plan['memory_updates'][:length],step+1)
                generation = (step//m['nominal_model_config']['replan_steps'])*m['nominal_model_config']['replan_steps']
                np.testing.assert_array_equal(plan['generation_step'][:length],generation)
                np.testing.assert_array_equal(plan['generation_time_s'][:length],time[generation])
                np.testing.assert_array_equal(plan['generated_now'][:length],step==generation)
                assert not plan['selected_for_execution'][:length].any()
                # Recompute FK from actual radians at EACH physics sample, using
                # canonical joint names to audit the online Lab-order conversion.
                for prefix in ('obs','transition/kinematics_after'):
                    q = torch.from_numpy(g[prefix+'/finger_joint_position_rad'][:length]).flatten(0,1)
                    pos,rot = fk(q)
                    pos,rot = pos.reshape(length,2,5,3),rot.reshape(length,2,5,3,3)
                    stored_pos = torch.from_numpy(g[prefix+'/finger_position_base_m'][:length])
                    stored_rot = torch.from_numpy(g[prefix+'/finger_rotation_base'][:length])
                    max_fk = max(max_fk,float((pos-stored_pos).abs().max()),float((rot-stored_rot).abs().max()))
                    torch.testing.assert_close(pos,stored_pos,atol=1e-6,rtol=1e-5)
                    torch.testing.assert_close(rot,stored_rot,atol=1e-6,rtol=1e-5)
                    force = torch.from_numpy(g['obs/contact_substeps_w_N' if prefix=='obs' else 'transition/contact_substeps_after_w_N'][:length])
                    local = torch.from_numpy(g[prefix+'/contact_substeps_local_N'][:length])
                    max_force = max(max_force,float((rotate_force_to_link(force,rot)-local).abs().max()))
                    torch.testing.assert_close(rotate_force_to_link(force,rot),local,atol=1e-5,rtol=1e-5)
                np.testing.assert_array_equal(obs['finger_joint_position_rad'][1:length,-1],tr['diagnostics_before/joint_position_rad'][1:length])
                np.testing.assert_array_equal(obs['actuator_target_rad'][:length],tr['target_before_rad'][:length])
                np.testing.assert_allclose(obs['tracking_error_rad'][:length],tr['target_before_rad'][:length]-tr['diagnostics_before/joint_position_rad'][:length],atol=0,rtol=0)
                if 'joint_limits_rad' in m:
                    env_index=int(ep[3:6])
                    limits=torch.tensor(m['joint_limits_rad'][env_index],dtype=torch.float32)
                    prev=torch.from_numpy(tr['target_before_rad'][:length])
                    delayed=torch.from_numpy(trace['delayed_command'][:length])
                    arm=torch.tensor([i for i,name in enumerate(m['joint_order']) if name.startswith('iiwa')])
                    hand=torch.tensor([i for i in range(29) if i not in arm.tolist()])
                    assert len(arm)==7 and len(hand)==22
                    expected=prev.clone();act=m['config']['action']
                    raw=(prev[:,arm]+act['dof_speed_scale']*m['control_dt']*delayed[:,arm]).clamp(limits[arm,0],limits[arm,1])
                    expected[:,arm]=(act['arm_moving_average']*raw+(1-act['arm_moving_average'])*prev[:,arm]).clamp(limits[arm,0],limits[arm,1])
                    raw=limits[hand,0]+.5*(delayed[:,hand]+1)*(limits[hand,1]-limits[hand,0])
                    expected[:,hand]=(act['hand_moving_average']*raw+(1-act['hand_moving_average'])*prev[:,hand]).clamp(limits[hand,0],limits[hand,1])
                    torch.testing.assert_close(expected,torch.from_numpy(tr['target_after_rad'][:length]),atol=1e-7,rtol=1e-6)
                if 'expert_actor_input_diagnostic' in tr:
                    assert np.isfinite(tr['expert_actor_input_diagnostic'][:length]).all()
                    assert len(m['expert_model_sha256'])==64
                # Read image pairs row-by-row: bounded memory, no terminal reset leakage.
                for t in range(length):
                    done = bool(g['terminated'][t] or g['truncated'][t])
                    after = tr['observation_after']
                    assert bool(after['available'][t]) == (not done)
                    if done:
                        assert not after['image'][t].any() and not after['camera_valid'][t].any()
                        terminal += 1
                    elif t+1<length:
                        for key in ('image','proprio','time_s','episode_id','step_id','camera_time_s','camera_frame','depth_valid'):
                            np.testing.assert_array_equal(after[key][t],obs[key][t+1])
                    for key in ('image','proprio','contact_substeps_w_N','finger_position_base_m'):
                        assert np.isfinite(obs[key][t]).all()
                if ref:
                    for key in reference_fields:
                        np.testing.assert_array_equal(tr[key][:length],ref['episodes'][ep]['transition/'+key][:length],err_msg=ep+'/'+key)
                    for key in ('terminated','truncated'):
                        np.testing.assert_array_equal(g[key][:length],ref['episodes'][ep][key][:length])
                rows += length
        finally:
            if ref: ref.close()
    return dict(rows=rows,episodes=len(reader.episodes),terminal_images_explicitly_missing=terminal,
                max_fk_reconstruction_error=max_fk,max_local_force_error_N=max_force,
                max_delay_timestamp_error_s=delay_error,reference_fields_exact=15 if reference else None,
                target_equations_checked='joint_limits_rad' in m,
                expert_checkpoint_hash_recorded=bool(m.get('expert_model_sha256')))


def audit_memory(path):
    reader = EpisodeReader(path); m = reader.metadata
    if m['nominal_plan_source'] != 'untrained_perception_probe' or m['memory_version'] != 2:
        raise ValueError('This reproducibility probe only reconstructs the seeded untrained Shadow models')
    cfg = PolicyConfig.from_dict(m['nominal_model_config'])
    torch.manual_seed(123); nominal = NominalDiffusionPolicy(cfg).eval()
    assert model_fingerprint(nominal) == m['nominal_model_sha256']
    options = dict(allow_incomplete=True,allow_unsuccessful=True)
    dataset = EpisodeSequenceDataset(path,**options)
    episode = max(reader.episodes,key=lambda ep:reader.attributes(ep)['length'])
    def encode(size):
        return torch.cat([value for value,_,_ in encode_episode_chunks(nominal.encoder,dataset.chunks(episode,size))],1)
    with torch.no_grad():
        # Full recurrence from frame features encoded in small independent batches;
        # never keep a full episode's CNN activations in memory.
        features = torch.cat([nominal.encoder.encode_frames(c['obs'])[0] for c in dataset.chunks(episode,8)],1)
        whole,_ = recurrent_scan(nominal.encoder.temporal,features,torch.ones(features.shape[:2],dtype=torch.bool))
        chunked,online = encode(7),encode(1)
        torch.testing.assert_close(whole,chunked,atol=3e-6,rtol=1e-5)
        torch.testing.assert_close(whole,online,atol=3e-6,rtol=1e-5)
        torch.manual_seed(123); correction = CorrectionPolicy(cfg).eval()
        if m.get('correction_model_sha256'):
            assert model_fingerprint(correction) == m['correction_model_sha256']
        data = CorrectionSequenceDataset(path,allow_untrained_probe=True,**options)
        def replay(size):
            return torch.cat([value for value,_,_ in replay_correction_chunks(correction,data.chunks(episode,size),detach_between_chunks=True)],1)
        a,b = replay(7),replay(1)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        with h5py.File(path) as f:
            original = torch.from_numpy(f['episodes'][episode]['transition/correction_shadow/untrained_candidate_diagnostic'][:])[None]
        torch.testing.assert_close(a,original,atol=3e-6,rtol=1e-5)
    return dict(episode=episode,steps=whole.shape[1],nominal_model_hash_verified=True,
        nominal_full_vs_chunk_max=float((whole-chunked).abs().max()),
        nominal_full_vs_online_max=float((whole-online).abs().max()),
        correction_chunk_vs_online_max=float((a-b).abs().max()),
        correction_vs_recorded_batch2_max=float((a-original).abs().max()),parameter_updates=0)


def audit_gradients(path):
    """Real contact prefix, fixed weights: Jacobian probe, not a learning objective."""
    reader = EpisodeReader(path)
    m = reader.metadata
    if m.get('memory_version') != 2 or m.get('nominal_plan_source') != 'untrained_perception_probe':
        raise ValueError('Expected seeded memory2 Shadow capture')
    cfg = PolicyConfig.from_dict(m['nominal_model_config'])
    torch.manual_seed(123); nominal = NominalDiffusionPolicy(cfg).eval()
    torch.manual_seed(123); correction = CorrectionPolicy(cfg).eval()
    before = (model_fingerprint(nominal), model_fingerprint(correction))
    assert before[0] == m['nominal_model_sha256']
    if m.get('correction_model_sha256'):
        assert before[1] == m['correction_model_sha256']
    dataset = CorrectionSequenceDataset(path,allow_incomplete=True,allow_unsuccessful=True,allow_untrained_probe=True)
    episode = max(reader.episodes,key=lambda ep:reader.attributes(ep)['length'])
    with h5py.File(path) as f:
        g = f['episodes'][episode]
        force = g['obs/contact_substeps_w_N'][:int(g.attrs['length'])]
        candidates = np.flatnonzero(np.linalg.norm(force,axis=-1).max(axis=(1,2)) > .1)
    if not len(candidates):
        raise ValueError('Real nonzero contact required for this branch check')
    target = int(candidates[0])
    nstate, cstate = None, None
    for chunk in dataset.chunks(episode,1):
        inputs = chunk['steps'][0]['inputs']
        if chunk['start'] == target:
            break
        with torch.no_grad():
            nstate = nominal.encoder(inputs['visual_history'], nstate)
            _, cstate = correction(**inputs,state=cstate,return_state=True)
    inputs['visual_history']['image'].requires_grad_()
    inputs['tactile_history']['force_w_N'].requires_grad_()
    condition = nominal.encoder(inputs['visual_history'], nstate)
    output = correction(**inputs,state=cstate)
    # A scalar projection tests derivatives, not a proposed Correction loss/label.
    (condition.square().sum()+output.square().sum()).backward()
    modules = {'nominal_rgb':nominal.encoder.vision.rgb,'nominal_depth':nominal.encoder.vision.depth,
               'nominal_temporal':nominal.encoder.temporal,'correction_rgb':correction.visual.vision.rgb,
               'correction_depth':correction.visual.vision.depth,'tactile':correction.tactile,
               'motion':correction.motion,'intent_plan':correction.intent,
               'intent_history':correction.intent_memory,'correction_head':correction.policy}
    gradient_norms = {}
    for name,module in modules.items():
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients), name
        gradient_norms[name] = sum(float(g.abs().sum()) for g in gradients)
        assert gradient_norms[name] > 0, name
    image_grad = inputs['visual_history']['image'].grad
    image_norms = [[float(image_grad[:,:,v,:3].abs().sum()),float(image_grad[:,:,v,3:].abs().sum())] for v in range(cfg.views)]
    assert all(rgb>0 and depth>0 for rgb,depth in image_norms)
    force_grad = inputs['tactile_history']['force_w_N'].grad
    assert torch.isfinite(force_grad).all() and force_grad.abs().sum()>0
    assert before == (model_fingerprint(nominal),model_fingerprint(correction))
    # Save/reload weights AND current recurrent values, then compare next evaluation.
    stream = io.BytesIO()
    torch.save(dict(nominal=nominal.state_dict(),correction=correction.state_dict(),nstate=nstate,cstate=cstate),stream)
    stream.seek(0); saved = torch.load(stream,weights_only=False)
    ncopy, ccopy = NominalDiffusionPolicy(cfg).eval(), CorrectionPolicy(cfg).eval()
    ncopy.load_state_dict(saved['nominal']); ccopy.load_state_dict(saved['correction'])
    with torch.no_grad():
        torch.testing.assert_close(condition,ncopy.encoder(inputs['visual_history'],saved['nstate']),atol=0,rtol=0)
        torch.testing.assert_close(output,ccopy(**inputs,state=saved['cstate']),atol=0,rtol=0)
    return dict(episode=episode,step=target,causal_prefix_steps=target,
                finite_nonzero_parameter_gradients_L1=gradient_norms,
                per_view_rgb_depth_input_gradients_L1=image_norms,
                force_input_gradient_L1=float(force_grad.abs().sum()),
                weights_and_carried_state_reload_exact=True,parameter_updates=0)


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--reference')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--trace-only',action='store_true')
    p.add_argument('--check-gradients',action='store_true',help='Real contact prefix Jacobian/reload check; no optimizer')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
    try:
        report={'trace':audit_trace(args.data,args.reference)}
        if not args.trace_only: report['memory']=audit_memory(args.data)
        if args.check_gradients: report['gradients']=audit_gradients(args.data)
        reader=EpisodeReader(args.data)
        small=min(reader.episodes,key=lambda ep:reader.attributes(ep)['length'])
        exported=reader.export_pickle(small,args.output/'episode.pkl')
        with exported.open('rb') as stream: restored=pickle.load(stream)  # created here, trusted
        values=flatten(restored['arrays'])
        for key,value in flatten(reader.read(small)).items():
            assert value.dtype==values[key].dtype and value.shape==values[key].shape
            np.testing.assert_array_equal(value,values[key])
        report['pickle_roundtrip']={'episode':small,'fields':len(values),'exact':True}
        (args.output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    except BaseException:
        import traceback
        (args.output/'FAILED.txt').write_text(traceback.format_exc());raise


if __name__=='__main__': main()
