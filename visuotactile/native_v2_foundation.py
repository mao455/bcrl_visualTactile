"""Bind scientific admission to the audited sources and completed expert cohort."""
import hashlib
import json
from pathlib import Path

from .compact import sha256


def verify_foundation_provenance(root):
    session = json.loads((root/'session.json').read_text())
    evidence_root = root
    if 'parent_root' in session:
        evidence_root = Path(session['parent_root'])
        original = json.loads((evidence_root/'session.json').read_text())
        if session['source_root'] != original['source_root'] or session['deadline'] != original['deadline']:
            raise ValueError('Supplement changed source or original deadline')
    else:
        original = session
    source = Path(original['source_root'])
    previous = Path(original['previous_root'])
    audit = json.loads((previous/'data_audit.json').read_text())
    control = json.loads((previous/'control_audit.json').read_text())
    data = json.loads((source/'data.json').read_text())
    if sha256(source/'data.json') != audit['source_data_sha256']:
        raise ValueError('Audited data index changed')
    if sha256(source/'statistics.pt') != audit['statistics_sha256'] or data['statistics_sha256'] != audit['statistics_sha256']:
        raise ValueError('Audited train-only statistics changed')
    if sha256('isaacsimenvs/tasks/play/utils/action_utils.py') != control['native_action_sha256']:
        raise ValueError('Audited native command implementation changed')
    if len(data['entries']) != 400:
        raise ValueError('Original400 cohort changed')
    for identity, entry in data['entries'].items():
        if sha256(source/'numeric'/(identity+'.pt')) != entry['numeric_sha256']:
            raise ValueError('Audited numeric episode changed: '+identity)
        if sha256(entry['path']) != entry['source_sha256']:
            raise ValueError('Audited original episode changed: '+identity)
    environment = json.loads((evidence_root/'expert1/environment.json').read_text())
    digest = hashlib.sha256(json.dumps(environment['config'], sort_keys=True).encode()).hexdigest()
    if digest != environment['sha256'] or sha256(original['expert_checkpoint']) != original['expert_sha256']:
        raise ValueError('Online expert configuration/checkpoint identity changed')
    result = json.loads((evidence_root/'expert1/result.json').read_text())
    if not result.get('software_complete'):
        raise ValueError('Online expert software cohort incomplete')
    from .native_v2_gates import data_gate, online_expert_gate
    numeric = data_gate(audit, control)
    online = online_expert_gate(result['trials'], [f's{seed}_lane0' for seed in original['single_seeds']],
        original['expert_sha256'], digest, original['expert_minimum_successes'])
    if not numeric.acceptance_passed or not online.acceptance_passed:
        raise ValueError('Source evidence no longer passes G0-A/B')
    for gate in (numeric, online):
        recorded = json.loads((root/'gates'/(gate.name+'.json')).read_text())
        if recorded != gate.to_dict():
            raise ValueError('Stored gate disagrees with actual evidence: '+gate.name)
    return dict(raw_episodes=400, numeric_episodes=400, data_sha256=audit['source_data_sha256'],
                statistics_sha256=audit['statistics_sha256'], expert_sha256=original['expert_sha256'])
