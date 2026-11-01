"""Retrieve small pinned source excerpts for the native DP design audit."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

from .training_resources import atomic_json


REPOSITORIES = {
    'dp': ('real-stanford/diffusion_policy', '5ba07ac6661db573af695b419a7947ecb704690f'),
    'robomimic': ('ARISE-Initiative/robomimic', None),
    'dp3': ('YanjieZe/3D-Diffusion-Policy', None),
    'octo': ('octo-models/octo', None),
    'rdt': ('thu-ml/RoboticsDiffusionTransformer', None),
    'rdp': ('xiaoxiaoxh/reactive_diffusion_policy', '824c5e8de1fd1811106907a04b5f0186e0138c0b'),
    'irdp': ('Chen-Wendi/ImplicitRDP', '112d6857d914c5dee4c4c172790088adc6bdc3a5'),
    'dexumi': ('real-stanford/DexUMI', None),
    'consistency': ('Aaditya-Prasad/Consistency-Policy', None),
    'vla_touch': ('clear-nus/vla-touch', None),
    'umi': ('real-stanford/universal_manipulation_interface', None),
}
PATTERNS = ('hybrid_image_policy.py', 'lowdim_policy.py', 'ema_model.py', 'normalizer.py',
    'sampler.py', 'train_diffusion_unet_hybrid_workspace.yaml', 'diffusion_policy.py',
    'obs_core.py', 'base_config.py', 'bc.py', 'dp3.py', 'simple_dp3.py', 'dp3.yaml',
    'octo_module.py', 'action_heads.py', 'configs.py', 'modeling_rdt.py', 'configs/base.yaml',
    'latent_diffusion_unet_image_policy.py', 'diffusion_transformer_image_policy.py',
    'train_latent_diffusion_unet_real_image_workspace.yaml',
    'train_reactive_diffusion_transformer_real_image_workspace.yaml',
    'consistency_unet_image_policy.py', 'README.md', 'LICENSE')


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'native-dp-source-audit'})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(4_000_001)
    if len(data) > 4_000_000:
        raise ValueError('Source response exceeds bounded download')
    return data


def collect(root):
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, (repository, revision) in REPOSITORIES.items():
        entry = dict(repository=repository, revision=revision, files=[], error=None)
        results[name] = entry
        try:
            if revision is None:
                revision = json.loads(fetch('https://api.github.com/repos/'+repository+'/commits/HEAD'))['sha']
                entry['revision'] = revision
            tree = json.loads(fetch('https://api.github.com/repos/'+repository+'/git/trees/'+revision+'?recursive=1'))
            if tree.get('truncated'):
                raise ValueError('Repository source tree truncated')
            paths = [item['path'] for item in tree['tree'] if item['type'] == 'blob']
            entry['tree'] = paths
            selected = [path for path in paths if any(path.endswith(pattern) for pattern in PATTERNS)]
            for path in selected[:35]:
                content = fetch('https://raw.githubusercontent.com/'+repository+'/'+revision+'/'+path)
                destination = root/name/path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
                entry['files'].append(dict(path=path, bytes=len(content), sha256=hashlib.sha256(content).hexdigest()))
        except Exception as error:
            entry['error'] = repr(error)
        atomic_json(root/'sources.json', results)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    collect(parser.parse_args().root)
