"""Finite all-256-level CPU/CUDA RGB check; no Kit, models or old data reads."""
import argparse
import json
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from visuotactile.compact import EpisodeFile, rgb_u8, sha256
from visuotactile.recording import CompactEpisodeStream, verify_stream_fields
from visuotactile.sensor_rgb import RGB_NORMALIZATION, canonical_rgb_lookup, canonical_rgb_u8


def require(value,message):
    if not value:raise ValueError(message)


def compact_check(rgb,raw):
    """Exercise the unchanged block flush, terminal handling and source hashes."""
    # Three views, every intensity in each channel; a real nonterminal post/pre
    # pair and a terminal placeholder. These are synthetic encoding test bytes.
    images=np.concatenate((np.stack((rgb,rgb)),np.ones((2,3,1,16,16),dtype=np.float32)),axis=2)
    mask=np.ones((3,1,16,16),dtype=bool)
    with tempfile.TemporaryDirectory(prefix='tight-rgb-encoding-') as temp:
        stream=CompactEpisodeStream(Path(temp)/'capture',dict(perception_version=2,camera_rig=dict(version=4)),
            'encoding',block_size=2)
        for step in range(2):
            available=step==0
            row={'obs/image':images[step],'obs/depth_m':images[step,:,3:],'obs/depth_valid':mask,
                'obs/step_id':np.asarray(step),'obs/episode_id':np.asarray(1),
                'transition/step_id':np.asarray(step),'transition/episode_id':np.asarray(1),
                'transition/observation_after/available':np.asarray(available),
                'transition/observation_after/image':images[1] if available else np.zeros_like(images[0]),
                'transition/observation_after/depth_valid':mask if available else np.zeros_like(mask),
                'transition/observation_after/step_id':np.asarray(step+1),
                'transition/observation_after/episode_id':np.asarray(1),
                'terminated':np.asarray(not available),'truncated':np.asarray(False)}
            stream.append(row)
        output=stream.finish(dict(encoding_diagnostic=True,success=False))/'data.h5'
        verified=verify_stream_fields(output)
        with EpisodeFile(output) as file:
            decoded=file['episodes/encoding']['obs/image'][:]
            require(np.array_equal(decoded,images),'Compact source float bytes changed')
            require(np.array_equal(rgb_u8(decoded[:,:,:3]),np.stack((raw,raw))), 'Sensor uint8 bytes changed')
        return dict(verified_source_fields=verified,rows=2,source_float_exact=True,sensor_uint8_exact=True,
            terminal_placeholder_preserved=True,temporary_hdf5_removed=True)


def run(cuda_device):
    raw=np.arange(256,dtype=np.uint8).reshape(1,1,16,16)
    raw=np.broadcast_to(raw,(3,3,16,16)).copy()
    expected=raw.astype(np.float32)/np.float32(255)
    result=dict(version='sensor_rgb_lossless_check_v1',contract=RGB_NORMALIZATION,
        scope='All 256 uint8 levels; synthetic encoding only, not camera/Kit/task acceptance',
        input_values=256,input_shape=list(raw.shape),torch_version=torch.__version__,cuda_runtime=torch.version.cuda,
        devices={},started_utc=time.time())
    for device in ('cpu',cuda_device):
        byte_tensor=torch.from_numpy(raw).to(device)
        old=(byte_tensor.float()/255.).cpu().numpy()
        lookup=canonical_rgb_lookup(byte_tensor.device)
        new=canonical_rgb_u8(byte_tensor,lookup).cpu().numpy()
        require(np.array_equal(new,expected),'Canonical float32 differs from storage decoding')
        require(np.array_equal(rgb_u8(new),raw),'Canonical uint8 roundtrip differs')
        require(torch.equal(byte_tensor.cpu(),torch.from_numpy(raw)),'Normalization mutated sensor bytes')
        try:rgb_u8(old);old_rejected=False
        except ValueError:old_rejected=True
        if device=='cpu':require(np.array_equal(old,expected),'Historical CPU normalization changed')
        # Keep rejection exact: an arbitrary one-ULP perturbation is still refused.
        bad=new.copy();bad.reshape(-1)[1]=np.nextafter(bad.reshape(-1)[1],np.float32(1))
        try:rgb_u8(bad)
        except ValueError:pass
        else:raise ValueError('Recorder no longer refuses noncanonical floats')
        try:canonical_rgb_u8(byte_tensor.float(),lookup)
        except ValueError:pass
        else:raise ValueError('Non-byte camera pixels accepted')
        unequal=old!=expected
        result['devices'][device]=dict(device_name='CPU' if device=='cpu' else torch.cuda.get_device_name(byte_tensor.device),
            old_mismatched_values=int(unequal.sum()),old_mismatched_gray_levels=np.unique(raw[unequal]).tolist(),
            old_max_absolute_error=float(np.max(np.abs(old-expected))),old_compact_rejected=old_rejected,
            new_float32_bitwise_equal=True,new_uint8_exact=True,strict_rejection_preserved=True,
            compact=compact_check(new,raw))
    result.update(passed=True,finished_utc=time.time())
    return result


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--cuda-device',default='cuda:0')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(torch.device(args.cuda_device).type=='cuda','Explicit real CUDA device required')
    require(not args.output.exists(),'Preserve prior check evidence')
    result=run(args.cuda_device)
    tree=Path(__file__).resolve().parents[1]
    result['source_sha256']={name:sha256(tree/name) for name in (
        'tools/tight_rgb_storage_check.py','visuotactile/sensor_rgb.py',
        'visuotactile/sim/environment.py','visuotactile/compact.py','visuotactile/recording.py')}
    result['source_root']=str(tree)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
