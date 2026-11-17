"""Publish compact receipts and inspect the completed cycle without new GPU work."""
import gzip
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from scripts.tight_cycle_20261118.cycle import ROOT, OUT, DATA, PARENT, TREE, read, write
from scripts.tight_layout_20261116.publication import table
from visuotactile.compact import sha256


def export():
    paths = [p for p in ROOT.glob('*.json') if not p.name.endswith('_private.json')]
    paths += list((ROOT/'learning/logs').glob('*.json'))
    for name in ['session.json', 'prepared.json', 'complete.json', 'owner_exit.json',
                 'training_sampling.json', 'bounded_extension.json', 'frozen_visual_hotset.json']:
        paths.append(ROOT/'learning'/name)
    for role in 'AB':
        for name in ['continuation.json', 'backend/base_identities.json', 'backend/cache_complete.json',
                     f'backend/{role}/training/selected.json', f'backend/{role}/nominal/index.json']:
            paths.append(ROOT/'learning'/role/name)
        src = ROOT/'learning'/role/'training/learning.jsonl'
        dest = OUT/'learning'/role/'DP/learning.jsonl.gz'
        dest.parent.mkdir(parents=True, exist_ok=True)
        with gzip.GzipFile(filename=str(dest), mode='wb', mtime=0) as f:
            for line in src.open():
                row = json.loads(line)
                for visit in row['visits']:
                    visit.pop('geometry', None)
                f.write((json.dumps(row, separators=(',', ':'))+'\n').encode())
    for wave in ['capability', 'collection', 'comparison00', 'comparison01', 'learning', 'learning/phase_review']:
        paths += list((ROOT/wave/'resources').glob('resources.json'))
        for name in ['owner_complete.json', 'owner_exit.json', 'kit_started.json', 'run_started.json']:
            paths.append(ROOT/wave/name)
    inventory = []
    for src in sorted(set(paths)):
        if not src.exists():
            continue
        dest = OUT/'receipts'/src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        inventory.append(dict(path=str(dest.relative_to(OUT)), sha256=sha256(dest), bytes=dest.stat().st_size))
    write(OUT/'receipt_inventory.json', inventory)
    timings = [dict(stage=p.stem.removesuffix('_complete'), **read(p))
               for p in sorted((ROOT/'learning/logs').glob('*_complete.json'))]
    timings += [dict(stage=p.stem.removesuffix('_complete'), **read(p))
                for p in sorted(ROOT.glob('*_complete.json')) if p.name != 'tail_complete.json']
    table(OUT/'stage_times.csv', timings)
    rows = []
    for wave in ['capability', 'collection', 'comparison00', 'comparison01']:
        d = ROOT/wave
        w = read(d/'wave_profiles.json')[0]
        rows.append(dict(wave=wave,lanes=read(d/'plan.json')['batch']['lanes'],
                         owner_seconds=read(d/'owner_complete.json')['seconds'],
                         kit_start_s=read(d/'kit_started.json')['utc']-read(d/'run_started.json')['utc'],
                         initialization_s=read(d/'initialization.json')['seconds'],
                         wave_s=w['seconds'], controller_s=w['controller_s'], environment_s=w['environment_s'],
                         render_s_nested_in_environment=w['render_reuse']['render_seconds'],
                         writer_submit_s=w['record_submit_s']))
    table(OUT/'simulation_times.csv', rows)


def media():
    import cv2
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    folder = ROOT/'comparison00'
    selected = [('A', 28826000), ('B', 28826000)]
    panels = []
    rows = []
    dest = OUT/'videos'
    dest.mkdir(exist_ok=True)
    for c in read(folder/'plan.json')['cases']:
        if (c['role'].split('_')[0], c['spec']['seed']) not in selected:
            continue
        source = folder/c['name']/'autonomous.mp4'
        target = dest/(c['role']+'_'+str(c['spec']['seed'])+'.mp4')
        shutil.copyfile(source, target)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-threads', '2',
                        '-i', str(target), '-f', 'null', '-'], check=True)
        cap = cv2.VideoCapture(str(target))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        result = read(folder/c['name']/'result.json')
        assert n == result['steps']
        ticks = sorted(set(min(n-1, t) for t in [72, 120, 180, 300, n-1]))
        thumbs = []
        for tick in ticks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
            ok, frame = cap.read()
            assert ok
            im = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            im.thumbnail((576, 324))
            panel = Image.new('RGB', (576, im.height+24), 'white')
            panel.paste(im, (0, 24))
            ImageDraw.Draw(panel).text((4, 4), f"{c['role']} {c['spec']['seed']} tick{tick}", fill='black')
            thumbs.append(panel)
        cap.release()
        panels.append(thumbs)
        rows.append(dict(case=c['name'], path=str(target.relative_to(OUT)), sha256=sha256(target),
                         frames=n, decoded_all=True, success=result['success'], ticks=ticks))
    canvas = Image.new('RGB', (max(sum(p.width for p in row) for row in panels),
                               sum(max(p.height for p in row) for row in panels)), 'white')
    y = 0
    for row in panels:
        x = 0
        for p in row:
            canvas.paste(p, (x,y)); x += p.width
        y += max(p.height for p in row)
    canvas.save(OUT/'evaluation/video_review.jpg', quality=88)
    write(OUT/'evaluation/media.json', dict(videos=rows, selection='First registered new-development family, both A/B baseline/candidate. Publication choice after outcomes; illustrative only, all64 executions determine results. No video reruns.'))


def closure():
    assert read(ROOT/'tail_complete.json')['complete'] and read(ROOT/'tail_exit.json')['children_reaped']
    assert not list(ROOT.rglob('*failure.json'))
    reg = read(ROOT/'registration.json')
    protected = {}
    for pair in reg['parent_pairs'].values():
        for comp in ['frontend', 'backend']:
            protected[pair[comp]] = sha256(pair[comp]) == pair[comp+'_sha256']
    for role in 'AB':
        for source in [PARENT, DATA]:
            cfg = read(source/role/'training/configuration.json')
            for name, key in [('data.json', 'data_sha256'), ('statistics.pt', 'statistics_sha256')]:
                old = source/role/'state_data'/name
                protected[str(old)] = sha256(old) == cfg[key]
    assert all(protected.values())
    checks = {p:sha256(TREE/p)==h for p,h in read(ROOT/'learning/prepared.json')['source_sha256'].items()}
    assert all(checks.values())
    for part in range(2):
        assert read(ROOT/f'comparison{part:02d}/paired_initial_check.json')['passed']
    data = [read(ROOT/'learning'/r/'state_data/data.json') for r in 'AB']
    assert set(data[0]['splits']['train']) == set(data[1]['splits']['train'])
    assert len(data[0]['splits']['train']) == 105
    dependencies = {e['path'] for d in data for e in d['entries'].values()}
    assert all(Path(p).is_file() for p in dependencies)
    active = []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():
            continue
        try:
            argv = (p/'cmdline').read_bytes().split(b'\0')
            if argv and b'python' in Path(argv[0].decode()).name.encode() and b'-m' in argv:
                module = argv[argv.index(b'-m')+1].decode()
                owned = str(ROOT).encode() in b' '.join(argv) or module=='scripts.tight_cycle_20261118.finish'
                if owned: active.append(dict(pid=int(p.name), module=module))
        except (OSError, IndexError):
            pass
    assert not active, active
    telemetry = []
    for p in sorted(ROOT.rglob('resources/resources.jsonl')):
        samples = [json.loads(line) for line in p.open()]
        dt = np.diff([s['utc_epoch'] for s in samples])
        gpu = [s for s in samples if s.get('gpu_free_mib') is not None]
        telemetry.append(dict(path=str(p.relative_to(ROOT)), samples=len(samples),
                              maximum_interval_s=float(dt.max()) if len(dt) else None,
                              min_ram_gib=min(s['ram_available_bytes'] for s in samples)/1024**3,
                              min_gpu_free_mib=min(s['gpu_free_mib'] for s in gpu),
                              peak_gpu_used_mib=max(s['gpu_used_mib'] for s in gpu)))
    end = read(ROOT/'tail_exit.json')['utc']
    session = read(ROOT/'session.json')
    assert end < session['deadline']
    write(OUT/'closure.json', dict(complete=True, completed_utc=end, elapsed_since_registration_s=end-session['started_utc'],
                                 protected=protected, source_files_checked=len(checks), source_hashes_pass=True,
                                 current_source_files_present=len(dependencies), owned_workers_active=active,
                                 telemetry=telemetry, disk_free_bytes=shutil.disk_usage(ROOT).free,
                                 deadline=session['deadline'], no_second_collection_batch=True,
                                 no_final_test=True, model_promotion=False, utc=time.time()))


if __name__ == '__main__':
    for name in __import__('sys').argv[1:]:
        globals()[name]()
