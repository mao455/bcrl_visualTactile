"""Register a finite rear-only experiment using existing unified job workers.

No model, sampler, loss or controller implementation is replaced here.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json as write

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    return {str(p.relative_to(PROJECT)): sha256(p)
            for folder in ('tools', 'visuotactile')
            for p in sorted((PROJECT / folder).rglob('*.py'))}


def training_job(root):
    backend = root/'B/backend'
    return dict(version='tight_project_job_v1', operation='train', resource_root=str(root), stages=[
        dict(entry='backend_cache', args=['--root',str(backend)]),
        dict(entry='backend_train', args=['--root',str(backend/'B'),'--job','image','--output',str(backend/'B/training')]),
        dict(entry='backend_offline', args=['--root',str(backend/'B'),'--job','image','--output',str(backend/'B/offline')])])


def prepare(config):
    from tools.tight_fresh_backend import prepare as prepare_backend
    import torch
    c = read(config)
    root, parent, history = (PROJECT / c[k] for k in ('root', 'parent', 'history'))
    assert not root.exists(), 'Preserve attempted experiment identities'
    selected = torch.load(history/'B/backend/B/training/best.pt', map_location='cpu', weights_only=False)
    assert selected['updates'] == c['historical_selected_updates']
    assert selected['epoch'] == c['epochs']
    old = read(history/'B/backend/configuration.json')['residual_pair']
    root.mkdir(parents=True)
    now = time.time()
    session = read(history/'session.json')
    session.update(started_utc=now, deadline=now+c['hours']*3600,
                   resource_root=str(root), owned_worker_roots=[str(root)],
                   authorization='Registered B frozen72 frontend, rear-only130, then same16-family three-way evaluation',
                   source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    for key in ('training_deadline', 'cache_deadline', 'offline_deadline'):
        session[key] = session['deadline']
    write(root/'session.json', session)
    write(root/'backend_episode_sampling.json', old['episode_sampling'])
    b = root/'B'; b.mkdir(); (b/'training').mkdir()
    for name in ('state_data', 'cache_index.json'):
        (b/name).symlink_to(history/'B'/name)
    (b/'training/best.pt').symlink_to(parent/'B/training/best.pt')
    continuation = read(history/'B/continuation.json')
    assert continuation['path'] == str(parent/'B/training/best.pt')
    assert sha256(continuation['path']) == continuation['sha256']
    assert sha256(continuation['previous_backend']) == continuation['previous_backend_sha256']
    write(b/'continuation.json', continuation)
    prepare_backend(root, 'B', c['epochs'], continue_rear=True,
                    offline_anchor_batch=old['cache_offline_planning_profile']['max_anchors'],
                    frozen_visual_io=old['frozen_visual_io'])
    # Keep the original plateau/LR rule. The outer experiment stops at the
    # historical selected epoch15, before its original maximum24.
    backend = b/'backend'
    v = read(backend/'configuration.json')['residual_pair']
    v.update(bounded_extension=old['bounded_extension'],
             minimum_control_tick_epochs=c['epochs'], comparison_control_tick_epochs=[])
    for p in (backend, backend/'B'):
        write(p/'configuration.json', dict(residual_pair=v))
        s = read(p/'session.json'); s['residual_pair'] = v; write(p/'session.json', s)
    job = training_job(root)
    dest = PROJECT/c['job_directory']; dest.mkdir(parents=True, exist_ok=True)
    write(dest/'train.json', job)
    write(root/'registration.json', dict(config=c, config_sha256=sha256(config),
        selected_history=dict(updates=selected['updates'],epoch=selected['epoch'],
            checkpoint_sha256=sha256(history/'B/backend/B/training/best.pt')),
        budget='15 identical replay decks, expected4839 updates; validation-selected checkpoint may inherit fewer. Same plateau rule, external15epoch cap versus original24.',
        initialization='72B selected rear parameters, fresh AdamW1e-4/weight_decay1e-4, no EMA, no optimizer/scheduler continuation; matches historical stage initialization',
        freeze='No frontend optimizer. Entire72DP including HistoryVision/GRUs/stats loaded eval/requires_gradFalse by nominal-cache worker; hash protected.',
        data_sha256=sha256(b/'state_data/data.json'),source_sha256=sources(),session=session))
    print(dest/'train.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare'])
    p.add_argument('--config', type=Path, required=True)
    a = p.parse_args(); prepare(a.config)


if __name__ == '__main__':
    main()
