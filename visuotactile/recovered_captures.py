"""Explicit immutable inputs from closed successes in interrupted collections."""
import json
from pathlib import Path
from .compact import sha256

def load(binding,pairs):
    path=Path(binding['path'])
    if sha256(path)!=binding['sha256']:raise ValueError('Recovered-input registration changed')
    manifest=json.loads(path.read_text());entries={};registries=[]
    if manifest.get('version')!='closed_success_recovery_inputs_v1':raise ValueError('Unknown recovery input contract')
    for source in manifest['sources']:
        root=Path(source['root'])
        for name,digest in source['files'].items():
            if sha256(root/name)!=digest:raise ValueError('Closed capture evidence changed: '+name)
        plan=json.loads((root/'plan.json').read_text())
        if plan.get('test_used') is not False:raise ValueError('No held-out recovery data')
        for role,pair in pairs.items():
            if any(plan['pairs'][role][k]!=pair[k] for k in ('frontend_sha256','backend_sha256')):
                raise ValueError('Recovery student identity differs')
        for identity,entry in json.loads((root/'correction_entries.json').read_text()).items():
            if identity in entries:raise ValueError('Duplicate recovered example')
            result=json.loads((root/identity/'result.json').read_text())
            if not (result.get('software_complete') and result.get('accepted_for_training')
                    and result.get('success') and result['dagger']['expert_takeover_success']
                    and result['dagger']['expert_valid_ticks']>0):
                raise ValueError('Only naturally successful admitted closed captures may be recovered')
            if str((root/identity/'result.json').relative_to(root)) not in source['files']:
                raise ValueError('Unbound recovered result')
            if entry['split']!='train':raise ValueError('Recovered input must be train')
            entries[identity]=entry
        registry=plan['training_expansion_registry']
        if registry not in registries:registries.append(registry)
    return entries,registries
