"""Freeze a small training-family diagnostic before its physical outcomes."""
import copy
import json
from pathlib import Path
import subprocess
import time

import h5py

from tools.tight_fresh_rollouts import completed_pair
from visuotactile.adaptive_takeover import profile
from visuotactile.compact import sha256
from visuotactile.position_control import ExpertPositionBridge
from visuotactile.training_resources import atomic_json

PROJECT = Path('/home/CNS2025331827/docs/bc+rl_RSS')
SOURCE = PROJECT/'outputs/visuotactile/run/tight_entry_phase_20261121_01'
ROOT = PROJECT/'outputs/visuotactile/run/tight_alignment_pilot_20261122_01'
EVIDENCE = PROJECT/'research_docs/evidence/tight_premature_scope_20261122_01/pilot'
JOB = PROJECT/'configs/experiments/premature_scope_pilot/evaluate.json'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert not ROOT.exists(), 'Preserve every attempted root'
    session = read(PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01/session.json')
    now = time.time()
    session.update(version='alignment_diagnostic_pilot_v1', started_utc=now, deadline=now+90*60,
        training_deadline=now+90*60, collection_deadline=now+90*60, resource_root=str(ROOT),
        owned_worker_roots=[str(ROOT)], authorization='Frozen diagnostic pilot only; 0 optimizer; no data admitted to training')
    ROOT.mkdir(parents=True); EVIDENCE.mkdir(parents=True, exist_ok=True); JOB.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT/'session.json', session)
    pairs = {r: completed_pair(SOURCE, r) for r in 'AB'}
    registered = read(PROJECT/'research_docs/evidence/tight_entry_phase_20261121_01/development/registration.json')
    assert all(pairs[r] == registered['pairs'][r+'_entry'] for r in 'AB')
    data = {r: read(SOURCE/r/'state_data/data.json') for r in 'AB'}
    identities = [f'layout10_train_{s}' for s in (28817006, 28817014, 28817018, 28817022)]
    for d in data.values():
        assert all(i in d['normalization_train_ids'] and i in d['splits']['train'] for i in identities)
        assert not ({d['entries'][i]['family'] for i in identities} &
                    {d['entries'][i]['family'] for split, ids in d['splits'].items() if split != 'train' for i in ids})
    protected = {}
    for r in 'AB':
        for name in ('data.json', 'statistics.pt', 'numeric.pt'):
            p = SOURCE/r/'state_data'/name
            protected[str(p)] = sha256(p)
        for key in ('frontend', 'backend'):
            protected[pairs[r][key]] = pairs[r][key+'_sha256']
    sources = {str(p.relative_to(PROJECT)): sha256(p) for folder in ('tools', 'visuotactile')
               for p in sorted((PROJECT/folder).rglob('*.py'))}
    template = read(PROJECT/'outputs/visuotactile/run/tight_entry_final24_20261122_01/wave00/plan.json')
    stages, hashes = [], {}
    for index, identity in enumerate(identities):
        entry = data['A']['entries'][identity]
        assert all(data['B']['entries'][identity][k] == entry[k] for k in ('path', 'family', 'source_sha256'))
        assert sha256(entry['path']) == entry['source_sha256']
        protected[entry['path']] = entry['source_sha256']
        with h5py.File(entry['path']) as f:
            original = json.loads(f.attrs['result_json'])
        assert original['success']
        folder = ROOT/f'wave{index:02d}'; folder.mkdir()
        cases = []
        for r in 'AB':
            for arm in ('autonomous', 'alignment_rescue'):
                name = f'{r}_{identity}_{arm}'
                spec = dict(identity=name, family=entry['family'], split='train', cohort='dagger_train',
                    source_identity=identity, seed=original['seed'], event_seed=original['seed']+100000,
                    narrow_reset=copy.deepcopy(original['narrow_reset']), collision_gate='live_collision_mesh_bounds_v1',
                    takeover_tick=10**9, layout=session['layout'], diagnostic_only=True)
                dagger = dict(version=ExpertPositionBridge.version, expert_speed=ExpertPositionBridge.speed_scheme,
                    trigger=profile('alignment') if arm == 'alignment_rescue' else None)
                cases.append(dict(name=name, role=r, spec=spec, dagger=dagger, diagnostic_arm=arm,
                    video=False, source_data=str(SOURCE/r/'state_data/data.json')))
        cases = cases[index:] + cases[:index]
        plan = copy.deepcopy(template)
        plan.update(kind='takeover', diagnostic_contract='frozen_alignment_rescue_no_admission_v1', pairs=pairs,
            cases=cases, resource_root=str(ROOT), source_root=str(SOURCE), round_index=0,
            stage_takeover_contract='diagnostic_alignment_no_admission_v1', never_fallback=10**9,
            source_data_sha256={r: sha256(SOURCE/r/'state_data/data.json') for r in 'AB'},
            source_session_sha256=sha256(ROOT/'session.json'), source_sha256=sources,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            test_used=False, cohort='training_family_diagnostic', utc=now,
            evaluation_scope=dict(native_success_unchanged=True, diagnostic_only=True,
                expert_rescue_is_autonomous=False, training_admission='blocked'))
        plan.pop('holdout_contract', None)
        plan['batch'].update(lanes=4, seed=original['seed'], writer_bytes=128*2**20,
            capture_audit_memo=dict(version='process_local_closed_capture_audit_v1', max_payloads=256, max_pairs=4, retains_tensors=False, accepts_external_digests=False),
            ordering='One train family per wave; A/B x autonomous/rescue; cyclic lane positions, no refills')
        atomic_json(folder/'session.json', session); atomic_json(folder/'plan.json', plan)
        hashes[str(folder)] = sha256(folder/'plan.json')
        stages.append(dict(entry='diagnostic_alignment_pilot', args=['--output', str(folder)]))
    job = dict(version='tight_project_job_v1', operation='evaluate', resource_root=str(ROOT), stages=stages)
    atomic_json(JOB, job)
    registration = dict(version='frozen_alignment_pilot_v1', registered_utc=now, deadline=session['deadline'],
        source=str(SOURCE), identities=identities, opportunities=16, wave_plan_sha256=hashes,
        protected_sha256=protected, pairs=pairs, job=str(JOB), job_sha256=sha256(JOB),
        wrapper_sha256={str(p):sha256(p) for p in sorted((PROJECT/'scripts/tight_premature_scope_20261122').glob('pilot*.py'))},
        selection='Original training families with historical alignment correction coverage; fixed before pilot outcomes; no terminal trajectories used',
        trigger=profile('alignment'), trigger_before_native_15mm_gate=True,
        controls='Same capture, full student and expert-shadow queue; never scheduled expert control; actual command chain audited',
        weights_frozen=True, optimizer_updates=0, training_pool_admitted_trajectories=0,
        analysis='All16 opportunities; per role autonomous native success, trigger count, expert rescue conditional trigger, trigger geometry/contact/orientation; diagnostic selection not a success-rate estimate',
        limitations='Observed starts paired; hidden physics and future environmental streams not exact; 4 selected train families do not estimate generalization',
        recovery='Only technical interruption; retain failed roots and all physical attempts, same frozen deadline and no outcome-driven retries')
    atomic_json(ROOT/'registration.json', registration); atomic_json(EVIDENCE/'registration.json', registration)
    atomic_json(EVIDENCE/'session.json', session)
    print(json.dumps(dict(root=str(ROOT), job=str(JOB), registration_sha256=sha256(ROOT/'registration.json'), deadline=session['deadline'])))


if __name__ == '__main__':
    main()
