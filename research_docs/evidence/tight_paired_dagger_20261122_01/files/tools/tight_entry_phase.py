"""Prepare one matched R2 native-entry experiment; never start workers implicitly."""
import argparse
import copy
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

PROJECT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text())


def preflight(config):
    import torch
    from visuotactile.native_data import visual_records
    from visuotactile.anchor_sampling import native_entry_sampling_profile, sample_training_anchors, supervision_rows
    from visuotactile.effective_exposure import Ledger, dp_ticks
    from visuotactile.entry_exposure import EntryLedger
    from visuotactile.latest_takeover_sampling import sampling_plan, episode_deck
    from visuotactile.bounded_extension import split_audit
    from visuotactile.visual_arithmetic import profile as visual_profile, digest

    c = read(config)
    history = PROJECT / c['history']
    out = PROJECT / c['audit']
    out.mkdir(parents=True, exist_ok=True)
    result = dict(config_sha256=sha256(config), roles={}, visual_arithmetic=visual_profile(2))
    result['visual_arithmetic_sha256'] = digest(result['visual_arithmetic'])
    rates = {}; backend_rates = {}
    for role in 'AB':
        child = history / role
        data = read(child / 'state_data/data.json')
        continuation = read(child / 'continuation.json')
        assert len(data['splits']['train']) == 180 and len(data['splits']['validation']) == 6
        assert data['supervision'] == ('q_reached' if role == 'A' else 'u_expert')
        protected = {str((child / 'state_data' / n).resolve()): sha256(child / 'state_data' / n)
                     for n in ('data.json', 'numeric.pt', 'statistics.pt')}
        for key, hk in [('path', 'sha256'), ('previous_backend', 'previous_backend_sha256')]:
            assert sha256(continuation[key]) == continuation[hk]
            protected[continuation[key]] = continuation[hk]
        for suffix in ('training/best.pt', f'backend/{role}/training/best.pt'):
            protected[str(child / suffix)] = sha256(child / suffix)
        # Read rates actually used, rather than infer a plateau from its final state.
        validation = [json.loads(x) for x in (child / 'training/validation.jsonl').read_text().splitlines()]
        rear_validation = read(child / f'backend/{role}/training/validation.json')
        assert [v['epoch'] for v in validation] == list(range(1, 61))
        assert [v['epoch'] for v in rear_validation] == list(range(1, 61))
        rates[role] = [float(v['learning_rates'][0]) for v in validation]
        backend_rates[role] = [float(v['learning_rates'][0]) for v in rear_validation]
        records = visual_records(child / 'state_data', data['splits']['train'], 'G00',
                                 target_window_contract=continuation['target_window_contract'])
        _, multiplicities = sampling_plan(data, continuation['sampling_contract'])
        old_profile = read(history / 'training_sampling.json')['anchors']
        fixed_profile = native_entry_sampling_profile(old_profile['seed'])
        absolute = {k: Ledger({i: supervision_rows(r)[0] for i, r in records.items()}) for k in ('original', 'entry')}
        entry = {k: EntryLedger(records) for k in absolute}
        g = torch.Generator().manual_seed(2026111402)
        updates = 0
        with gzip.open(out / f'{role}_planned_sampling.jsonl.gz', 'wt') as log:
            for epoch in range(1, 61):
                order = episode_deck(data['splits']['train'], multiplicities, g)
                assert order == read(child / f'training/deck_{epoch:03d}.json')['order']
                for offset in range(0, len(order), 2):
                    updates += 1
                    contributions = {k: [] for k in absolute}
                    selections = {k: [] for k in absolute}
                    for pos in range(offset, min(offset+2, len(order))):
                        identity = order[pos]; r = records[identity]
                        old, receipt = sample_training_anchors(r, old_profile, identity=identity, epoch=epoch, visit_index=pos+1)
                        new, fixed_receipt = sample_training_anchors(r, fixed_profile, identity=identity, epoch=epoch, visit_index=pos+1)
                        assert fixed_receipt['original_coverage_anchors'] == old.tolist()
                        assert fixed_receipt['original_random_anchors'] == receipt['original_random_anchors']
                        for kind, anchors in [('original', old), ('entry', new)]:
                            ticks = dp_ticks(r, anchors)
                            assert set(ticks) == set(supervision_rows(r)[0].nonzero().flatten().tolist())
                            contributions[kind].append((identity, ticks)); selections[kind].append((identity, anchors))
                        log.write(json.dumps(dict(epoch=epoch, update=updates, visit_index=pos+1, identity=identity,
                            original=receipt, entry=fixed_receipt))+'\n')
                    for kind in absolute:
                        absolute[kind].step(updates, contributions[kind])
                        entry[kind].step(updates, entry[kind].contributions(records, selections[kind]))
        assert updates == 7200 and entry['entry'].summary()['eligible']
        for kind in absolute:
            with (out / f'{role}_{kind}_planned_entry.csv').open('w') as f:
                rows = entry[kind].rows(); writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(rows)
        state = torch.load(child / 'training/last.pt', weights_only=False, map_location='cpu', mmap=True)
        result['roles'][role] = dict(protected=protected, split=split_audit(data), continuation=continuation,
            original_configuration=state['configuration'], inference_sampler_seed=state['contract']['sampler_seed'],
            planned_updates=updates, planned_decks=60, entry={k: v.summary() for k, v in entry.items()},
            absolute={k: v.summary({i: 60*multiplicities[i] for i in records}) for k, v in absolute.items()},
            control_reuse='pending source/visual equivalence gate; no assertion of a new trained control',
            actual_epoch_rates=rates[role], actual_backend_epoch_rates=backend_rates[role])
        del state, records
    atomic_json(out / 'matched_learning_rates.json', rates)
    atomic_json(out / 'matched_backend_learning_rates.json', backend_rates)
    atomic_json(out / 'sampling_preflight.json', result)
    print(json.dumps({r: result['roles'][r]['entry'] for r in 'AB'}), flush=True)


def prepare(config):
    from visuotactile.anchor_sampling import native_entry_sampling_profile
    from visuotactile.entry_exposure import VERSION
    c = read(config); out = PROJECT / c['audit']; history = PROJECT / c['history']; root = PROJECT / c['root']
    proof = read(out / 'sampling_preflight.json')
    assert proof['config_sha256'] == sha256(config)
    visual_proof = read(out / 'visual_cache_verification.json')
    assert visual_proof['passed'] and visual_proof['visual_arithmetic'] == proof['visual_arithmetic']
    control_proof = read(out / 'original_control_audit.json')
    assert control_proof['passed']
    if shutil.disk_usage(PROJECT).free < c['minimum_launch_free_gib'] * 2**30:
        raise RuntimeError('Insufficient storage for registered outputs plus the unchanged 12 GiB reserve')
    if root.exists():
        raise FileExistsError('Preserve attempted experiment identity')
    root.mkdir(parents=True)
    for name in ('metadata.json','gate.json','frozen_visual_io.json','frozen_visual_hotset.json',
                 'backend_episode_sampling.json','cuda_allocator_policy.json','full_trajectory_batch.json',
                 'effective_exposure.json','bounded_extension.json'):
        atomic_json(root / name, read(history / name))
    session = read(history / 'session.json'); now = time.time()
    session.update(started_utc=now, deadline=now+c['hours']*3600, resource_root=str(root),
        owned_worker_roots=[str(root)], authorization='Matched R2 native entry repair, existing180 only; no new DAgger',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip())
    for key in ('training_deadline','cache_deadline','offline_deadline'):
        session[key] = session['deadline']
    session['visual_arithmetic'] = proof['visual_arithmetic']
    atomic_json(root / 'session.json', session)
    atomic_json(root / 'visual_arithmetic.json', proof['visual_arithmetic'])
    sampling = read(history / 'training_sampling.json')
    sampling['anchors'] = native_entry_sampling_profile(sampling['anchors']['seed'])
    atomic_json(root / 'training_sampling.json', sampling)
    atomic_json(root / 'entry_exposure.json', dict(version=VERSION, minimum_steps=50))
    for name in ('matched_learning_rates.json', 'matched_backend_learning_rates.json'):
        atomic_json(root / name, read(out / name))
    stages = []
    for role in 'AB':
        for path, digest in proof['roles'][role]['protected'].items():
            assert sha256(path) == digest
        child = root / role; child.mkdir()
        (child / 'state_data').symlink_to(history / role / 'state_data', target_is_directory=True)
        atomic_json(child / 'continuation.json', read(history / role / 'continuation.json'))
        cache = read(history / role / 'cache_index.json')
        cache.update(visual_arithmetic=proof['visual_arithmetic'], visual_arithmetic_sha256=proof['visual_arithmetic_sha256'],
                     arithmetic_verification=str(out / 'visual_cache_verification.json'))
        atomic_json(child / 'cache_index.json', cache)
        stages.append(dict(entry='frontend', args=['--root', str(root), '--role', role, '--epochs', '60',
                          '--full-trajectory-batch','--offline-anchor-batch','64']))
    for role in 'AB':
        backend = root / role / 'backend'
        stages.extend([
            dict(entry='backend_prepare',args=['--root',str(root),'--role',role,'--epochs','60','--continue-rear',
                 '--offline-anchor-batch','64','--frozen-visual-io-json',str(root/'frozen_visual_io.json')]),
            dict(entry='backend_cache',args=['--root',str(backend)]),
            dict(entry='backend_train',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'training')]),
            dict(entry='backend_offline',args=['--root',str(backend/role),'--job','image','--output',str(backend/role/'offline')])])
    atomic_json(PROJECT / c['job'], dict(version='tight_project_job_v1', operation='train', resource_root=str(root), stages=stages))
    atomic_json(root / 'registration.json', dict(config=c, preflight_sha256=sha256(out/'sampling_preflight.json'),
        visual_proof_sha256=sha256(out/'visual_cache_verification.json'), entry_target='50 actual committed optimizer steps',
        control_proof_sha256=sha256(out/'original_control_audit.json'),
        planned_only=True, control='original R2 reuse only after equivalence audit',
        selection='same fixed validation6 minimum among eligible epochs50..60; no closed-loop/test selection'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['preflight','prepare']); p.add_argument('--config',type=Path,required=True)
    a = p.parse_args(); globals()[a.mode](a.config)


if __name__ == '__main__':
    main()
