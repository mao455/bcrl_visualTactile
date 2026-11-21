"""Verify existing R2 cached inputs against explicit microbatch2 raw RGB-D."""
import argparse
import json
from pathlib import Path
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def run(config):
    import h5py
    import torch
    from visuotactile.native_data import Episode, depth_grid
    from visuotactile.models.native_vision import NativeDinoSpatial
    from visuotactile.precision import state_fp32
    from visuotactile.visual_arithmetic import profile, digest
    from tools.tight_entry_phase import PROJECT, read
    c=read(config); history=PROJECT/c['history']; out=PROJECT/c['audit']
    state_fp32(); torch.set_num_threads(2)
    arithmetic=profile(2); dino=NativeDinoSpatial().cuda().eval(); started=time.time(); rows=[]
    data=read(history/'A/state_data/data.json'); index=read(history/'A/cache_index.json')
    assert read(history/'B/cache_index.json')['entries']==index['entries']
    for identity in data['splits']['train']+data['splits']['validation']:
        row=index['entries'][identity]; assert sha256(row['path'])==row['sha256']
        ep=Episode(history/'A/state_data',identity,'I-HistoryDP'); entry=data['entries'][identity]
        with h5py.File(entry['path'],'r') as f:
            raw=f['episodes'][entry['episode']]; length=len(raw['obs/time_s'])
            first=int(entry['takeover_tick'])+2 if entry.get('takeover_tick') is not None else 0
            ticks=sorted({0, min(first,length-1), max(0,first-1), length-1})
            for tick in ticks:
                rgb=torch.from_numpy(raw['frames/rgb_u8'][tick]).cuda()
                with torch.inference_mode():
                    patch=dino.forward_lanes(rgb[None],vision_microbatch=2)[0].cpu()
                depth=depth_grid(torch.from_numpy(raw['frames/depth_m'][tick]),torch.from_numpy(raw['frames/depth_valid'][tick]))
                error=float((patch.float()-ep.perception['patches'][tick].float()).abs().max())
                depth_error=float((depth-ep.perception['depth'][tick]).abs().max())
                assert error==0 and depth_error==0, (identity,tick,error,depth_error)
                rows.append(dict(identity=identity,tick=tick,patch_max_abs=error,depth_max_abs_m=depth_error,
                    cache_sha256=row['sha256'],kernel_images=6,tail='duplicate_current_lane'))
        del ep
    # Contract and actual terminal batch registrations are distinct evidence.
    final=PROJECT/'outputs/diagnostics/tight_adaptive_audit_20261121_01/actual_final_batches.json'
    result=dict(passed=True,episodes=len(index['entries']),frames=len(rows),rows=rows,
        visual_arithmetic=arithmetic,visual_arithmetic_sha256=digest(arithmetic),
        dino_checkpoint_sha256=read(history/'session.json')['dino_sha256'],
        dino_source_sha256=sha256(PROJECT/'visuotactile/models/native_vision.py'),
        historical_final_batches=str(final),historical_final_batches_sha256=sha256(final),
        limitation='Selected real frames per episode, not all-frame encoder recomputation',
        seconds=time.time()-started,optimizer_updates=0,physics_steps=0)
    atomic_json(out/'visual_cache_verification.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    run(p.parse_args().config)


if __name__=='__main__':main()
