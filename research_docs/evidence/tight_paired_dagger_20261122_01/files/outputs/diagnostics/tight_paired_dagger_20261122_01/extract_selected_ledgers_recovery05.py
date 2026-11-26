"""Inactive final-only CPU extraction of the eight selected exposure ledgers.

No model construction/forward, CUDA, raw histories, cache tensors or log scans.
No checkpoint is opened until the genuine finite main-round closure exists.
Only a new AUDIT attempt is written; failed/pending attempts contain status only.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'
import argparse
import hashlib
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
ROOT = PROJECT/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
SCOPES = ('common180_parents', 'learning')
COMPONENTS = ('frontend', 'backend')
MAX_FILE = 16*1024**2
MAX_TOTAL_TICKS = 2_000_000
VERSION = 'selected_successful_step_ledger_export_recovery05_v1'


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    path = Path(path)
    require(path.stat().st_size <= MAX_FILE, 'Oversized JSON metadata: '+str(path))
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    raw = (json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)+'\n').encode()
    require(len(raw) <= MAX_FILE, 'Selected ledger JSON exceeds16MiB')
    return raw


def component_directory(root, scope, role, component):
    child = root/scope/role
    return child/('training' if component == 'frontend' else f'backend/{role}/training')


def closure_gate(root, audit):
    """Metadata-only gate; missing final evidence never opens even parent weights."""
    status_path = audit/'driver_main_recovery05/status.json'
    required = [status_path, root/'actual_parent_ages.json', root/'parent_pairs.json',
                root/'evaluation/test/registration.json', root/'cycle_logs/summarize_test_exit.json']
    for scope in SCOPES:
        for role in 'AB':
            required.append(root/scope/role/'state_data/data.json')
            for component in COMPONENTS:
                directory = component_directory(root, scope, role, component)
                required.extend(directory/name for name in ('best.pt', 'deployment.json', 'finished.json'))
    owner_directories = [root/'owners/main_train', root/'evaluation/development', root/'evaluation/test']
    for owner in owner_directories:
        required.append(owner/'project_exit.json')
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return dict(status='pending', missing=missing, checkpoint_reads=0)
    require(read(status_path).get('stage') == 'main_stages_complete', 'Main driver has not genuinely closed')
    require(read(root/'cycle_logs/summarize_test_exit.json').get('returncode') == 0,
            'Final full-denominator summary not closed successfully')
    for owner in owner_directories:
        receipt = read(owner/'project_exit.json')
        require(receipt.get('children_reaped') is True, 'Unreaped final owner: '+str(owner))
    freeze = read(root/'evaluation/test/registration.json')
    require(freeze.get('cohort') == 'test' and freeze.get('n') == 64,
            'Actual independent64 evaluation registration required')
    require(set(freeze.get('pairs', {})) == {'A_new', 'B_new', 'A_parent', 'B_parent'},
            'Four actual evaluated policies required')
    return dict(status='closed', checkpoint_reads=0)


def load_checkpoint(path):
    import torch
    require(not torch.cuda.is_initialized(), 'CPU extractor cannot use an initialized CUDA context')
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def mask_hash(masks):
    """Canonical mask-only hash, explicitly distinct from native segment+mask SHA."""
    h = hashlib.sha256()
    for identity in sorted(masks):
        name = identity.encode()
        mask = masks[identity]
        h.update(len(name).to_bytes(8, 'little')); h.update(name)
        h.update(len(mask).to_bytes(8, 'little')); h.update(bytes(mask))
    return h.hexdigest()


def selected_component(root, scope, role, component, *, loader=load_checkpoint):
    import torch
    directory = component_directory(root, scope, role, component)
    checkpoint = directory/'best.pt'
    deployment = read(directory/'deployment.json')
    require(deployment.get('epoch') == 60 and deployment.get('selection') == 'preregistered_epoch60_only'
            and deployment.get('validation_used_for_selection') is False,
            'Actual fixedepoch60 deployment required')
    expected = deployment.get('checkpoint_sha256')
    before = checkpoint.stat()
    require(before.st_size <= 4*1024**3 and sha(checkpoint) == expected, 'Selected checkpoint SHA/size mismatch')
    state = loader(checkpoint)
    after = checkpoint.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
            'Selected checkpoint changed during CPU read')
    updates = state.get('updates')
    require(type(updates) is int and updates > 0 and state.get('epoch') == 60
            and updates == deployment.get('update'), 'Selected checkpoint age differs')
    require(read(directory/'finished.json').get('updates') == updates, 'Finished receipt age differs from selected checkpoint')
    if component == 'frontend':
        require(state.get('ema_optimization_step') == updates, 'Selected DP EMA and optimizer ages differ')
    data_path = root/scope/role/'state_data/data.json'
    data = read(data_path); data_sha = sha(data_path)
    configuration = state['configuration']
    require(configuration.get('data_sha256') == data_sha, 'Selected data manifest SHA differs')
    ids = data['splits']['train']
    require(len(ids) == len(set(ids)) and 0 < len(ids) <= 4096
            and state.get('train_ids') == ids, 'Selected training population/order differs')
    ledger = state['effective_exposure_state']
    require(ledger.get('version') == 'new_optimizer_step_legal_tick_exposure_v1'
            and ledger.get('updates') == updates, 'Actual ledger version/age differs')
    require(set(ledger['legal']) == set(ledger['counts']) == set(ids), 'Actual ledger population differs')
    visits = state['sampling_exposure' if component == 'frontend' else 'episode_counts']
    require(set(visits) == set(ids) and all(type(v) is int and v >= 50 for v in visits.values()),
            'Actual selected visits below50 or wrong population')
    policy = configuration.get('effective_exposure', {})
    require(policy.get('ancestors_counted') is False and policy.get('duplicate_tick_within_step') == 'count_once',
            'Selected exposure policy must exclude ancestors and duplicate slots')
    episodes = {}; masks = {}; all_counts = []; total_ticks = 0
    for identity in ids:
        entry = data['entries'][identity]
        legal, counts = ledger['legal'][identity], ledger['counts'][identity]
        require(isinstance(legal, torch.Tensor) and isinstance(counts, torch.Tensor)
                and legal.device.type == counts.device.type == 'cpu'
                and legal.dtype == torch.bool and counts.dtype == torch.int64
                and legal.ndim == counts.ndim == 1 and legal.shape == counts.shape,
                'Selected mask/count must be aligned CPU bool/int64 vectors')
        require(type(entry['length']) is int and len(legal) == entry['length']
                and 0 < len(legal) <= 20000, 'Actual complete-history length differs')
        total_ticks += len(legal)
        require(total_ticks <= MAX_TOTAL_TICKS, 'Bounded selected complete-history population exceeded')
        require(bool(legal.any()) and not bool((counts < 0).any()) and not bool((counts > updates).any())
                and not bool(counts[~legal].any()), 'Illegal/negative/out-of-age exposure count')
        values = counts[legal].tolist(); ticks = legal.nonzero().flatten().tolist()
        require(min(values) >= 50, 'Selected legal expert tick below50')
        require(isinstance(entry.get('source_sha256'), str) and len(entry['source_sha256']) == 64,
                'Dataset source identity missing')
        masks[identity] = legal.tolist(); all_counts.extend(values)
        episodes[identity] = dict(family=entry['family'], source_sha256=entry['source_sha256'],
            complete_history_ticks=len(legal), legal_ticks=ticks, successful_optimizer_steps=values,
            visits=visits[identity], mask_sha256=hashlib.sha256(bytes(masks[identity])).hexdigest())
    entries = None
    if component == 'frontend':
        entry_state = state['entry_exposure_state']
        require(entry_state.get('version') == 'native_entry_commit_optimizer_exposure_v1'
                and entry_state.get('updates') == updates
                and set(entry_state['entries']) == set(entry_state['counts']), 'Native-entry ledger identity/age differs')
        entries = []
        expected_ids = {i for i in ids if not masks[i][0]}
        require(set(entry_state['entries']) == expected_ids, 'Native-entry correction population differs from actual legal masks')
        for identity, entry in entry_state['entries'].items():
            legal_indices = episodes[identity]['legal_ticks']
            first = min(legal_indices); anchor = (first//12)*12
            slots = [s for s in range(12) if anchor+s < len(masks[identity]) and masks[identity][anchor+s]]
            ticks = [anchor+s for s in slots]
            require(entry == dict(anchor=anchor, slots=slots, ticks=ticks), 'Native entry population/phase differs')
            counts = {int(t):n for t,n in entry_state['counts'][identity].items()}
            require(set(counts) == set(ticks) and all(type(n) is int and 50 <= n <= updates for n in counts.values()),
                    'Native-entry selected count below50 or invalid')
            for slot, tick in zip(slots, ticks):
                require(counts[tick] <= ledger['counts'][identity][tick].item(), 'Entry participation exceeds all legal tick participation')
                entries.append(dict(identity=identity, native_anchor=anchor, slot=slot, absolute_tick=tick,
                                    successful_optimizer_steps=counts[tick]))
    result = dict(version=VERSION, scope=scope, role=role, component=component, epoch=60, updates=updates,
        checkpoint=str(checkpoint), checkpoint_sha256=expected, data_manifest=str(data_path), data_sha256=data_sha,
        deployment_sha256=sha(directory/'deployment.json'), ancestors_counted=False,
        duplicate_within_step='count_once', count_scope='Actual new successful positive-weight optimizer updates in selected checkpoint',
        train_mask_sha256=mask_hash(masks), native_all_record_mask_and_segment_sha256=configuration.get('supervision_mask_sha256'),
        raw_mask_revalidation=False, mask_binding='Exact selected DP/rear legal masks must agree and selected data manifest SHA is checked; native all-record segment+mask SHA is recorded only, not recomputed from raw data',
        trajectories=len(ids), legal_ticks=len(all_counts), minimum_tick_steps=min(all_counts), maximum_tick_steps=max(all_counts),
        minimum_visits=min(visits.values()), below50_ticks=0, base_sha256=configuration.get('base_sha256'),
        episodes=episodes, native_entry_rows=entries)
    del state
    encoded(result)
    return result


def collect(root, audit, *, loader=load_checkpoint):
    gate = closure_gate(root, audit)
    if gate['status'] != 'closed':
        return gate, {}
    import torch
    require(not torch.cuda.is_initialized(), 'CPU-only extraction required')
    parents = read(root/'parent_pairs.json'); ages = read(root/'actual_parent_ages.json')['roles']
    evaluated = read(root/'evaluation/test/registration.json')['pairs']
    payloads = {}; actual = {}; reads = 0
    for scope in SCOPES:
        actual[scope] = {}
        for role in 'AB':
            actual[scope][role] = {}
            for component in COMPONENTS:
                row = selected_component(root, scope, role, component, loader=loader); reads += 1
                actual[scope][role][component] = row
                eval_role = role+('_new' if scope == 'learning' else '_parent')
                require(evaluated[eval_role][component+'_sha256'] == row['checkpoint_sha256'],
                        'Selected extraction differs from actually evaluated model')
                if scope == 'common180_parents':
                    require(parents[role][component+'_sha256'] == row['checkpoint_sha256']
                            and ages[role][component+'_epochs'] == 60 and ages[role][component+'_updates'] == row['updates'],
                            'Actual collection parent identity/age differs')
            front, rear = (actual[scope][role][c] for c in COMPONENTS)
            require(rear['base_sha256'] == front['checkpoint_sha256'], 'Rear not bound to its own selected DP')
            require(front['train_mask_sha256'] == rear['train_mask_sha256']
                    and front['data_sha256'] == rear['data_sha256'], 'Selected DP/rear legal mask/data binding differs')
        for component in COMPONENTS:
            require(actual[scope]['A'][component]['updates'] == actual[scope]['B'][component]['updates'],
                    'A/B actual selected successful update budgets differ')
    for scope in SCOPES:
        for role in 'AB':
            for component in COMPONENTS:
                name = f'{scope}_{role}_{component}.json'
                payloads[name] = encoded(actual[scope][role][component])
    require(reads == 8 and not torch.cuda.is_initialized(), 'Exactly8 CPU-only selected checkpoint reads required')
    summary = dict(status='passed', version=VERSION, checkpoint_reads=reads, cuda_initialized=False,
        all_selected_epoch=60, all_legal_ticks_at_least50=True, all_native_entry_slots_at_least50=True,
        paired_actual_updates_equal=True, own_frontend_rear_mask_and_base_binding=True,
        files={name:dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) for name,raw in payloads.items()},
        limitations='Actual selected counters are exported, not a new reconstruction of individual update IDs. Historical raw mask/segment provenance is preserved by checkpoint/data SHA and existing audits; no raw HDF5 is opened. Exposure does not prove nonzero gradient or autonomous capability.')
    payloads['selected_ledger_summary.json'] = encoded(summary)
    return summary, payloads


def extract(root, audit, output, *, loader=load_checkpoint):
    root, audit, output = (Path(p).resolve() for p in (root, audit, output))
    require(output.is_relative_to(audit) and output != audit and not output.exists(), 'Only a NEW AUDIT child output is permitted')
    try:
        status, payloads = collect(root, audit, loader=loader)
    except Exception as exc:
        status, payloads = dict(status='failed', error=repr(exc), checkpoint_payloads_published=0), {}
    status = dict(status, observed_utc=time.time(), extractor_sha256=sha(Path(__file__)),
                  root=str(root), output=str(output), raw_data_opened=False, model_forward_calls=0)
    status_raw = encoded(status)
    # No component evidence is written until ALL8 bindings/exposure gates pass.
    output.mkdir(parents=False, exist_ok=False)
    for name, raw in payloads.items():
        with (output/name).open('xb') as stream:
            stream.write(raw)
    with (output/'extraction_status.json').open('xb') as stream:
        stream.write(status_raw)
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--audit', type=Path, default=HERE)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    status = extract(args.root, args.audit, args.output)
    print(json.dumps(dict(status=status['status'], output=str(args.output), checkpoint_reads=status.get('checkpoint_reads')), ensure_ascii=False))
    raise SystemExit(0 if status['status'] == 'passed' else 2)


if __name__ == '__main__':
    main()
