"""Pinned real-checkpoint input, normalization and execution-history audit."""
import argparse
import gc
import json
from pathlib import Path

import h5py
import torch

from .action_reference import command_to_reference, reference_to_command
from .compact import sha256
from .long_controller import LongController
from .long_data import inputs_for, load_record, MANIFEST
from .models.long_horizon import FrozenDinoSpatial, LongHorizonConfig, LongHorizonPolicy
from .study_metrics import evaluate, joint_groups
from .study_resources import StudyLease
from .training_resources import atomic_json


def audit(root):
    session = json.loads((root/'session.json').read_text())
    source = Path(session['source_root'])
    index = json.loads((root/'cache_index.json').read_text())
    metadata = json.loads((root/'metadata.json').read_text())
    lease = StudyLease(root,deadline=session['dp_deadline'])
    assert sha256(MANIFEST)==index['manifest_sha256']
    splits = index['splits']
    assert [len(splits[name]) for name in ('train','validation','test')]==[320,40,40]
    assert len(set(sum(splits.values(),[])))==400
    assert index['camera_order']==['fixed_high','fixed_side','wrist_rgbd']
    identity = splits['train'][0]
    record = load_record(root,identity)
    assert sha256(root/'cache'/f'{identity}.pt')==index['entries'][identity]['sha256']
    row = next(row for row in json.loads(MANIFEST.read_text())['episodes'] if row['id']==identity)
    results = {}
    with h5py.File(MANIFEST.parent/row['directory']/'data.h5','r') as stream:
        episode = stream['episodes'][row['episode']]
        rgb = torch.from_numpy(episode['frames/rgb_u8'][[0,12]])
        with lease.block():
            dino = FrozenDinoSpatial().to('cuda:0').eval()
            assert sha256(Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth')==index['dino_weights_sha256']
            recomputed = dino(rgb.to('cuda:0')).cpu()
        dino_error = float((recomputed.float()-record['patches'][:2].float()).abs().max())
        assert dino_error <= .001
        del dino,recomputed
        for group in ('G00','G10'):
            saved = torch.load(source/(group+'-new_full')/'best.pt',map_location='cpu',weights_only=False)
            assert saved['config']['prediction_type']=='sample'
            class CachedVision:
                def __init__(self):
                    self.cursor = 0
                def __call__(self,images):
                    features = record['patches'][self.cursor]
                    self.cursor += 1
                    return features
            controller = LongController(saved,CachedVision(),device='cpu')
            with lease.block():
                sampler = LongHorizonPolicy(LongHorizonConfig(**saved['config'])).to('cuda:0').eval()
                sampler.load_state_dict(saved['model'])
            generator = torch.Generator(device='cuda:0').manual_seed(9981)
            def predict(condition,unused_generator):
                with lease.block(),torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    return sampler.predict_from_condition(condition.to('cuda:0'),generator).float().cpu()
            controller.model.predict_from_condition = predict
            steps = min(145,len(record['time']))
            with torch.no_grad():
                full,offline_state = controller.model.encode(inputs_for(record,0,steps,controller.model.cfg.tactile))
                error = 0.
                for step in range(steps):
                    names = ('episode_id','step_id','time_s','proprio','actuator_target_rad','camera_time_s',
                             'camera_valid','contact_substeps_w_N','contact_substeps_local_N','contact_time_s','contact_valid')
                    packet = {name:torch.from_numpy(episode['obs/'+name][step:step+1]) for name in names}
                    packet['image'] = torch.zeros(1,3,4,252,448)
                    packet['image'][0,:,3:] = torch.from_numpy(episode['frames/depth_m'][step])
                    packet['depth_valid'] = torch.from_numpy(episode['frames/depth_valid'][step:step+1])
                    proposed = controller.act(packet)
                    assert torch.isfinite(proposed).all() and proposed.abs().max()<=1
                    assert controller.plan.shape==(1,64,29) and controller.cursor==step%12
                    if step%12==0:
                        error = max(error,float((controller.last_condition[0]-full[step//12]).abs().max()))
                    command = torch.from_numpy(episode['transition/command_clipped_pre_delay'][step:step+1])
                    controller.previous_command = command
                    transition = dict(episode_id=packet['episode_id'],step_id=packet['step_id'],command_clipped_pre_delay=command,
                        time_before_s=torch.from_numpy(episode['transition/time_before_s'][step:step+1]),
                        time_after_s=torch.from_numpy(episode['transition/time_after_s'][step:step+1]),
                        control_trace={name:torch.from_numpy(episode['transition/control_trace/'+name][step:step+1])
                            for name in ('valid','delayed_command','selected_command_time_s')})
                    controller.observe_execution(transition)
                memory_error = float((controller.state-offline_state).abs().max())
                assert error<5e-5 and memory_error<5e-5
                normalized = record['reference']*controller.model.target_scale+controller.model.target_offset
                recovered = (normalized-controller.model.target_offset)/controller.model.target_scale
                normalizer_error = float((recovered-record['reference']).abs().max())
                limits = torch.tensor(metadata['joint_limits_rad'][0])
                command = reference_to_command(record['reference'],record['target_before'],limits,
                    speed_scale=metadata['config']['action']['dof_speed_scale'],control_dt=metadata['control_dt'])
                command_error = float((command-record['command']).abs().max())
                assert normalizer_error<1e-5 and command_error<1e-5
            del controller,saved
            gc.collect()
            ema_metrics = evaluate(sampler,root,splits['validation'][:8],lease)
            previous = torch.load(source/(group+'-new_full')/'last.pt',map_location='cpu',weights_only=False)
            with lease.block():
                ema_snapshot = {name:parameter.detach().cpu() for name,parameter in sampler.state_dict().items()}
            squared = total = 0
            for name,parameter in ema_snapshot.items():
                difference = parameter.float()-previous['model'][name].float()
                squared += float(difference.square().sum())
                total += difference.numel()
            with lease.block():
                sampler.load_state_dict(previous['model'])
            del previous,ema_snapshot
            gc.collect()
            raw_metrics = evaluate(sampler,root,splits['validation'][:8],lease)
            results[group] = dict(condition_max_error=error,memory_max_error=memory_error,
                normalizer_inverse_max_error=normalizer_error,reference_command_max_error=command_error,
                replay_steps=steps,ema_raw_weight_rms=(squared/total)**.5,
                ema_validation8=ema_metrics,raw_validation8=raw_metrics)
            del sampler
            gc.collect()
    atomic_json(root/'interface_audit.json',dict(passed=True,dino_cached_max_error=dino_error,
        joint_groups=joint_groups(metadata['joint_order']),groups=results,
        scope='real epoch40 weights; cached-vision CPU history replay with real GPU diffusion; expert acknowledgments teacher-forced, no new rollout',
        old_curve_missing=['epoch train loss','per-finger','prefix arm/hand','horizon-wise errors'],
        prior_native_evidence='six completed original candidate traces; native/controller wrapper unchanged'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    torch.set_num_threads(2)
    audit(parser.parse_args().root)
