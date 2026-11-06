"""CPU-only wrist viewing derivatives; retain every frame and original videos."""
import argparse
from pathlib import Path

from tools.runtime import read, write, sha


def export(root):
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from visuotactile.compact import write_video
    from visuotactile.video import verify_video

    state = read(root/'status.json')
    if state['state'] != 'complete' or not state['owned_workers_exited']:
        raise RuntimeError('Export requires completed viewing owner and exited GPU workers')
    report = read(root/'report.json'); session = read(root/'session.json')
    demos = Path(session['demos']); entries = []
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    for row in report['rows']:
        if not row['legal']:
            continue
        identity = row['identity']; source = demos/(identity+'.mp4')
        if sha(source) != row['video_sha256']:
            raise RuntimeError('Original video changed')
        destination = demos/(identity+'_wrist.mp4')
        partial = demos/(identity+'_wrist.partial.mp4')
        if destination.exists() or partial.exists():
            raise FileExistsError('Preserve previous export; no silent overwrite')

        def frames():
            with av.open(str(source)) as container:
                if float(container.streams.video[0].average_rate) != 60.:
                    raise ValueError('Unexpected source rate')
                for index, frame in enumerate(container.decode(video=0)):
                    rgb = frame.to_ndarray(format='rgb24')
                    if rgb.shape != (316, 1344, 3):
                        raise ValueError('Unexpected source layout; refusing an inferred crop')
                    canvas = Image.new('RGB', (448, 316), (22, 22, 26))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((6, 2), f"StateDP u{report['updates']} | {identity}", font=font, fill='white')
                    draw.text((6, 22), f"seed {row['seed']} | viewing rerun | 1x", font=font, fill='white')
                    draw.text((6, 42), f"wrist | step {index} | t={index/60:.3f}s", font=font, fill='white')
                    canvas.paste(Image.fromarray(rgb[64:316, 896:1344]), (0, 64))
                    yield np.asarray(canvas)

        count = write_video(partial, frames(), fps=60)
        if count != row['steps']:
            raise ValueError('Frame count differs from completed native control steps')
        verification = verify_video(partial, row['steps'])
        partial.rename(destination)
        entries.append(dict(identity=identity, source_file=source.name,
            source_sha256=row['video_sha256'], file=destination.name,
            sha256=sha(destination), bytes=destination.stat().st_size,
            verification=verification))
    result = dict(version=1, role='wrist-only viewing derivative',
        source_crop_xyxy=[896, 64, 1344, 316], source_size_wh=[1344, 316],
        frames_added=0, frames_removed=0, playback_rate_hz=60,
        implementation_sha256=sha(__file__),
        limitation='Two fixed views are blank in the originals; preserve them as evidence. '
            'Only crop the visible wrist image and replace the caption; no new simulation or sensor data.',
        entries=entries)
    write(root/'wrist_exports.json', result)
    index = demos/'README.md'
    text = index.read_text()
    if '## Wrist viewing files' in text:
        raise RuntimeError('Index already contains wrist exports')
    text += '\n## Wrist viewing files\n\n'
    text += ('The fixed cameras are blank in these recordings. These CPU derivatives crop the '
        'visible wrist image, preserving all frames at 60 fps / 1x. Original three-view files '
        'above are retained. Both fixed RGB views are also constant at the first frame of '
        'all 200 new demonstrations (read-only check); their visual validity must be resolved '
        'before ImageDP. StateDP does not use RGB.\n\n')
    for entry in entries:
        text += f"- [{entry['identity']} — wrist]({entry['file']})\n"
    index.write_text(text)
    print(f'Exported and fully decoded {len(entries)} wrist videos')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    export(parser.parse_args().root.resolve())
