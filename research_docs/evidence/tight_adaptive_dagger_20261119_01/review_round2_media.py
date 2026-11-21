"""Decode existing Round 2 videos; save bounded three-view contact sheets."""
import json
from pathlib import Path

import av
from PIL import Image, ImageDraw

PUBLIC = Path(__file__).resolve().parent
ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')
CASES = {
    'A_lost_parent': ('paired2_wave08_A', 'A_parent'),
    'A_lost_new': ('paired2_wave08_A', 'A_new'),
    'B_lost_parent': ('paired2_wave07_B', 'B_parent'),
    'B_lost_new': ('paired2_wave07_B', 'B_new'),
    'A_gained': ('paired2_wave01_A', 'A_new'),
    'B_gained': ('paired2_wave00_B', 'B_new'),
}


def export():
    output = ROOT / 'media/autonomous_round02'
    output.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for label, (wave, role) in CASES.items():
        videos = json.loads((PUBLIC / 'round02/evaluation' / wave / 'videos.json').read_text())
        metadata, = [r for r in videos['rows'] if r['case'].startswith(role + '_')]
        source = Path(metadata['result']).parent / 'autonomous.mp4'
        ticks = sorted({0, metadata['frames'] - 1} | {
            t for t in (60, 120, 180, 240, 360, 480, 600) if t < metadata['frames']})
        frames = []
        count = 0
        with av.open(str(source)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                count += 1
                if index not in ticks:
                    continue
                rgb = frame.to_image().convert('RGB')
                rgb.thumbnail((1008, 237))
                panel = Image.new('RGB', (1008, 263), (22, 22, 22))
                panel.paste(rgb, (0, 26))
                ImageDraw.Draw(panel).text((8, 7), f'{label} | tick {index} | {metadata["case"]}', fill='white')
                frames.append(panel)
        assert count == metadata['frames'] and len(frames) == len(ticks)
        sheet = Image.new('RGB', (1008, 263 * len(frames)))
        for index, panel in enumerate(frames):
            sheet.paste(panel, (0, index * 263))
        target = output / (label + '.jpg')
        assert not target.exists(), 'Keep prior review artifacts'
        sheet.save(target, quality=90)
        receipts[label] = dict(video=str(source), sheet=str(target), frames=count,
            view_ticks=ticks, success=metadata['success'], case=metadata['case'],
            scope='Existing actual pre-action RGB; representative outcome-selected viewing; zero physics reruns')
    (output / 'decoded.json').write_text(json.dumps(receipts, indent=2) + '\n')
    (PUBLIC / 'round02/autonomous_media_decode.json').write_text(json.dumps(receipts, indent=2) + '\n')
    print(json.dumps(dict(videos=len(receipts), decoded_frames=sum(r['frames'] for r in receipts.values()))))


if __name__ == '__main__':
    export()
