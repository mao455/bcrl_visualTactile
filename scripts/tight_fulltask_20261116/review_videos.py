"""Read-only frame/contact review of the nine existing real simulation videos."""
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw

TREE = Path(__file__).resolve().parents[2]
ROOT = TREE/'outputs/visuotactile/run/tight_fulltask_20261116_01'
OUT = TREE/'outputs/visuotactile/run/tight_reliable_eval_20261116_01/video_review'


def sheet(video, ticks, destination):
    wanted = set(ticks)
    frames = {}
    decoded = 0
    with av.open(str(video)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            decoded += 1
            if index in wanted:
                frames[index] = frame.to_image().resize((806, 190))
    image = Image.new('RGB', (1612, ((len(frames)+1)//2)*215), 'white')
    draw = ImageDraw.Draw(image)
    for i, (tick, frame) in enumerate(frames.items()):
        x, y = (i%2)*806, (i//2)*215
        draw.text((x+4, y+3), f'{video.parent.name} | pre-action index {tick} | t={tick/60:.3f}s', fill='black')
        image.paste(frame, (x, y+24))
    image.save(destination)
    return decoded


def main():
    OUT.mkdir(exist_ok=True)
    index = []
    for video in sorted(ROOT.glob('baseline_evaluation/*/autonomous.mp4')) + sorted(ROOT.glob('collection/*/student_then_expert.mp4')):
        with av.open(str(video)) as stream:
            count = stream.streams.video[0].frames
        ticks = list(range(0, count, 20))
        if count-1 not in ticks:
            ticks.append(count-1)
        target = OUT/(video.parent.name+'.jpg')
        decoded = sheet(video, ticks, target)
        index.append(dict(video=str(video), sheet=str(target), decoded=decoded, sampled_preaction_indices=ticks))
    (OUT/'index.json').write_text(json.dumps(index, indent=2))


if __name__ == '__main__':
    main()
