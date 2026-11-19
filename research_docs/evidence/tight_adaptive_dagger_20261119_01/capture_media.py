"""Viewing derivatives of existing compact capture RGB; no simulation rerun."""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from tools.tight_frozen_frontend import read, write
from visuotactile.compact import sha256, write_video
from visuotactile.video import verify_video


def export(directory, output):
    result=read(directory/'result.json');output.mkdir(parents=True,exist_ok=True)
    source=Path(result['dataset_capture']['path'])
    assert sha256(source)==result['dataset_capture']['sha256']
    switch=result['dagger']['trigger_receipt']['trigger_completed_tick']
    count=result['steps'];indices=sorted({0,count-1,*range(60,count,60),
        *([max(0,switch-1),min(count-1,switch+24)] if switch is not None else [])})
    panels=[]
    with h5py.File(source) as stream:
        frames=stream['episodes'][result['identity']]['frames/rgb_u8']
        assert len(frames)==count
        def render(tick):
            rgb=frames[tick].transpose(0,2,3,1)
            cameras,height,width,_=rgb.shape
            image=Image.new('RGB',(cameras*width,height+32),(24,24,24))
            draw=ImageDraw.Draw(image)
            control='expert' if switch is not None and tick>=switch else 'student'
            draw.text((8,8),f'{directory.name} | pre-action tick {tick} | {control}',fill='white')
            for camera in range(cameras):image.paste(Image.fromarray(rgb[camera]),(camera*width,32))
            return image
        def video_frames():
            for tick in range(count):
                image=render(tick)
                if tick in indices:panels.append(image.copy())
                yield np.asarray(image)
        video=output/'capture.mp4'
        assert not video.exists(),'Preserve existing media derivative'
        write_video(video,video_frames())
    sheet=Image.new('RGB',(panels[0].width,panels[0].height*len(panels)))
    for row,panel in enumerate(panels):sheet.paste(panel,(0,row*panel.height))
    sheet.save(output/'contact_sheet.jpg',quality=90)
    receipt=dict(source=str(source),source_sha256=sha256(source),video=str(video),video_sha256=sha256(video),
        decoded=verify_video(video,expected_frames=count),panel_ticks=indices,takeover_tick=switch,
        scope='Original pre-action RGB, all three cameras; CPU viewing derivative, no new rollout, final post-action image unavailable.')
    write(output/'media.json',receipt);print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();export(args.case,args.output)
