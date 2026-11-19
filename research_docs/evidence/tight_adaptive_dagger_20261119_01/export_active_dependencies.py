"""Export protection paths from existing manifests without reading model/data bytes."""
import collections
import json
from pathlib import Path
import time

from tools.tight_adaptive_dagger import CANONICAL, ROOT, BASE, PROJECT


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def export():
    original = read(ROOT / 'ACTIVE_DEPENDENCIES.json')
    files = {}

    def add(path, category, digest=None):
        supplied = Path(path)
        resolved = supplied.resolve(strict=True)
        item = files.setdefault(str(resolved), dict(
            path=str(resolved), aliases=[], categories=[], expected_sha256=None,
            size_bytes=resolved.stat().st_size))
        if str(supplied) not in item['aliases']:
            item['aliases'].append(str(supplied))
        if category not in item['categories']:
            item['categories'].append(category)
        if digest:
            assert item['expected_sha256'] in (None, digest), str(resolved)
            item['expected_sha256'] = digest

    for path, digest in original['protected'].items():
        add(path, 'original_active_protection', digest)

    pools = [BASE, ROOT / 'round01/learning']
    second = ROOT / 'round02/learning'
    if (second / 'A/state_data/data.json').exists():
        pools.append(second)
    populations = []
    for pool in pools:
        for role in 'AB':
            data_path = pool / role / 'state_data/data.json'
            data = read(data_path)
            add(data_path, 'active_pool_index')
            populations.append(dict(root=str(pool), role=role,
                train=len(data['splits']['train']), validation=len(data['splits']['validation'])))
            for entry in data['entries'].values():
                add(entry['path'], 'active_raw', entry['source_sha256'])
            cache_path = pool / role / 'cache_index.json'
            if cache_path.exists():
                add(cache_path, 'active_visual_index')
                for entry in read(cache_path)['entries'].values():
                    add(entry['path'], 'active_frozen_visual', entry['sha256'])
            for name in ('statistics.pt', 'numeric.pt', 'loss_context.pt'):
                path = data_path.parent / name
                if path.exists():
                    add(path, 'active_prepared_data')

    manifest_path = CANONICAL / 'outputs/visuotactile/data/close_v4/manifest.json'
    add(manifest_path, 'original400_manifest')
    manifest = read(manifest_path)
    for episode in manifest['episodes']:
        for name, value in episode['files'].items():
            add(manifest_path.parent / episode['directory'] / name,
                'original400', value['sha256'])

    history = CANONICAL / 'outputs/visuotactile/run/tight_long_20261117_01/round05/learning'
    for role in 'AB':
        backend = history / role / f'backend/{role}'
        configuration = read(backend / 'configuration.json')['residual_pair']
        for base in configuration['bases'].values():
            add(base['checkpoint'], 'historical72_frontend', base['sha256'])
        add(backend / 'training/best.pt', 'historical72_backend',
            read(backend / 'offline/result.json')['checkpoint_sha256'])
        add(configuration['residual_statistics_path'], 'historical72_statistics',
            configuration['residual_statistics_sha256'])
        for path in (backend / 'configuration.json', backend / 'training/finished.json',
                     backend / 'offline/result.json'):
            add(path, 'historical72_evaluation_dependency')

    session = read(ROOT / 'session.json')
    for name in ('expert', 'dino'):
        add(session[name + '_checkpoint'], name + '_weights', session[name + '_sha256'])
    categories = collections.Counter(category for item in files.values()
                                     for category in item['categories'])
    path = ROOT / 'ACTIVE_DEPENDENCIES_EXPANDED.json'
    document = dict(version='expanded_active_protection_v1', utc=time.time(),
        protected_roots=[str(ROOT), str(PROJECT), str(BASE), str(history)],
        protected_root_policy='Preserve entire roots, including new files created after this snapshot',
        source_manifest=str(ROOT / 'ACTIVE_DEPENDENCIES.json'),
        populations=populations, unique_files=len(files), categories=dict(categories),
        verification='Existence and manifest digest consistency only; no new full-byte SHA audit',
        original400_episodes=len(manifest['episodes']), files=list(files.values()))
    assert categories['active_raw'] >= 165 and categories['active_frozen_visual'] >= 165
    assert document['original400_episodes'] == 400
    atomic(path, document)
    original['expanded_protection_manifest'] = str(path)
    original['protected_roots_include_future_files'] = document['protected_roots']
    atomic(ROOT / 'ACTIVE_DEPENDENCIES.json', original)
    request = read(ROOT / 'CLEANUP_COORDINATION_REQUEST.json')
    request['expanded_protection_manifest'] = str(path)
    atomic(ROOT / 'CLEANUP_COORDINATION_REQUEST.json', request)
    print(json.dumps({key: document[key] for key in
                     ('unique_files', 'categories', 'original400_episodes', 'verification')}))


if __name__ == '__main__':
    export()
