"""CPU-only own-student aggregates for the preregistered paired DAgger round.

This module never modifies the original pool, recomputes an affine, or reads a
held-out trajectory. New demonstrations are shared; new takeovers remain owned.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

from tools.tight_paired_dagger import ROOT, BASE, AUDIT, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json

VERSION = 'paired_own_student_pool_v1'
SPEED = dict(kind='closed_loop', arm=.5, hand=1., phase='all', hand_target_cap_rad_s=2.)


def validate_partitions(old, full, corrections, role, registry):
    """Metadata-only split and ownership check, before opening source payloads."""
    from visuotactile.narrow_data import aggregate_family_partitions
    if role not in ('A', 'B') or old['supervision'] != ('q_reached' if role == 'A' else 'u_expert'):
        raise ValueError('Explicit A/q_reached or B/u_expert source required')
    families = aggregate_family_partitions(old)
    original_families = set(families)
    ids = old['splits']['train']
    affine = old['normalization_train_ids']
    if len(ids) != 180 or len(old['splits']['validation']) != 6 or len(affine) != 30 or not set(affine) <= set(ids):
        raise ValueError('Preserve the original180, validation6 and original30 affine')
    if set(full) & set(corrections) or (set(full) | set(corrections)) & set(old['entries']):
        raise ValueError('New episode identity collision')
    expected = registry['families']
    if len(full) != 24 or len({e['family'] for e in full.values()}) != 24 or {e['family'] for e in full.values()} != set(expected):
        raise ValueError('One accepted shared full demonstration for each registered24 family required')
    for identity, entry in {**full, **corrections}.items():
        family = entry.get('family')
        if family not in expected or family in original_families or entry.get('split') != 'train' or entry.get('seed') != expected[family]['seed']:
            raise ValueError('Only new registered train families may enter this round: '+identity)
        if identity in full:
            if entry.get('trajectory_kind') != 'full_expert_demonstration_v1' or entry.get('expert_speed') != SPEED:
                raise ValueError('Shared demonstrations must preserve the original expert speed contract')
        elif entry.get('trajectory_kind') != 'expert_takeover_v1' or entry.get('source_group') != role:
            raise ValueError('Each student may supervise only its OWN new expert takeovers')
        families[family] = 'train'
    return families


def full_record(entry, registered):
    """Audit full native expert execution; geometry is a separate loss context."""
    import torch
    from visuotactile.narrow_data import record, actual_configuration_labels
    if sha256(entry['path']) != entry['source_sha256']:
        raise ValueError('Full demonstration source bytes changed')
    key, metadata, row = record(entry['path'], episode=entry['episode'], privileged=False)
    if key != entry['episode'] or len(row['time']) != entry['length'] or metadata.get('initial_family') != entry['family'] or metadata.get('dataset_split') != 'train':
        raise ValueError('Full demonstration metadata/length differs')
    if metadata.get('expert_speed') != SPEED:
        raise ValueError('Actual full demonstration speed changed')
    for name, value in registered['narrow_reset'].items():
        config = metadata['config']['precise_assembly' if name.startswith('hole_') else 'reset']
        if config.get(name) != value:
            raise ValueError('Actual demonstrated reset differs from registration: '+name)
    reached, report = actual_configuration_labels(entry)
    n = len(row['time'])
    row.update(q_reached=reached, u_expert=row['command'], supervision_mask=torch.ones(n, dtype=torch.bool),
               control_segment=torch.zeros(n, dtype=torch.long))
    return row, report


def _materialize(source, out, old, full, corrections, role, registry, binding, *, full_loader=None, correction_loader=None, context_loader=None):
    """Write one immutable derived pool; loaders are injectable for CPU fixtures."""
    import torch
    from visuotactile.checkpointing import save
    from visuotactile.narrow_data import dagger_record, actual_configuration_labels
    from visuotactile.state_image_loss import geometry_loss_context
    from visuotactile.anchor_sampling import supervision_rows
    full_loader = full_record if full_loader is None else full_loader
    correction_loader = dagger_record if correction_loader is None else correction_loader
    context_loader = geometry_loss_context if context_loader is None else context_loader
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or out.exists():
        raise FileExistsError('Use a fresh output identity and preserve every failed attempt')
    families = validate_partitions(old, full, corrections, role, registry)
    raw = torch.load(source/'numeric.pt', map_location='cpu', weights_only=False)
    if raw['data_sha256'] != sha256(source/'data.json'):
        raise ValueError('Original numeric/data identity differs')
    out.mkdir(parents=True)
    provenance = dict(version=VERSION, role=role, source_data=str(source/'data.json'), source_data_sha256=sha256(source/'data.json'),
        registry_binding=binding, shared_full_ids=list(full), own_correction_ids=list(corrections),
        original180_ids=list(old['splits']['train']), original30_affine_ids=list(old['normalization_train_ids']),
        heldout_trajectories_opened=0, student_prefix_supervised=False)
    atomic_json(out/'build_started.json', provenance)
    records, contexts = {}, {}
    old_context = torch.load(source/'loss_context.pt', map_location='cpu', weights_only=False) if role == 'A' and (source/'loss_context.pt').exists() else {}
    retained = old['splits']['train'] + old['splits']['validation']
    for identity in retained:
        original = raw['records'][identity]
        row = {k: original[k] for k in ('robot', 'command', 'time', 'joint_limits')}
        row['q_reached'] = original['q_reached'] if 'q_reached' in original else actual_configuration_labels(old['entries'][identity])[0]
        n = len(row['time'])
        row.update(u_expert=original.get('u_expert', row['command']),
            supervision_mask=original.get('supervision_mask', torch.ones(n, dtype=torch.bool)),
            control_segment=original.get('control_segment', torch.zeros(n, dtype=torch.long)))
        supervision_rows(row)
        records[identity] = row
        if role == 'A':
            contexts[identity] = old_context[identity] if identity in old_context else context_loader(old['entries'][identity])
    full_audits = {}
    for identity, entry in full.items():
        records[identity], full_audits[identity] = full_loader(entry, registry['families'][entry['family']])
        supervision_rows(records[identity])
        if role == 'A':
            contexts[identity] = context_loader(entry)
    correction_audits = {}
    for identity, entry in corrections.items():
        row, touch, context = correction_loader(entry, include_geometry=role == 'A')
        supervision_rows(row)
        records[identity] = row
        correction_audits[identity] = dict(total_ticks=len(row['time']), legal_ticks=int(row['supervision_mask'].sum()),
            source_group=entry['source_group'], prefix_ticks=int((~row['supervision_mask']).sum()))
        if role == 'A':
            contexts[identity] = context
    splits = copy.deepcopy(old['splits'])
    splits['train'] += list(full) + list(corrections)
    bindings = copy.deepcopy(old.get('training_expansion_registries', []))
    if binding not in bindings:
        bindings.append(binding)
    data = dict(copy.deepcopy(old), entries={**old['entries'], **full, **corrections}, splits=splits, families=families,
        source_data_sha256=sha256(source/'data.json'), source_statistics_sha256=sha256(source/'statistics.pt'),
        normalization_scope='unchanged original30 training affine; no aggregate refit',
        training_expansion_registries=bindings, paired_pool_provenance=provenance,
        paired_pool=dict(version='paired_own_training_pool_v1', role=role, original_ids=list(old['splits']['train']),shared_full_ids=list(full),own_correction_ids=list(corrections)),
        pool_partitions=dict(old_full=list(old['normalization_train_ids']),
            old_correction=[i for i in old['splits']['train'] if i not in set(old['normalization_train_ids'])],
            new_full=list(full), own_new_correction=list(corrections)),
        aggregation=dict(version='expert_takeover_aggregate_v1', paired_pool_version=VERSION,
            selection='shared_full_expert_demonstrations_and_own_successful_expert_executed_corrections',
            student_prefix_supervised=False, new_training_ids=list(corrections),
            shared_full_training_ids=list(full), original_training_ids=list(old['splits']['train'])))
    data.pop('sampling_strata', None)
    for name in ('statistics.pt',):
        shutil.copyfile(source/name, out/name)
        if sha256(source/name) != sha256(out/name):
            raise ValueError('Original affine bytes changed')
    atomic_json(out/'data.json', data)
    save(out/'numeric.pt', dict(contract=raw['contract'], data_sha256=sha256(out/'data.json'), records=records))
    if role == 'A':
        save(out/'loss_context.pt', contexts)
    # Frozen visual caches are built from entries' real source HDF5 paths later;
    # their index binds this new manifest, never the original registry manifest.
    legal = {i: int(records[i]['supervision_mask'].sum()) for i in splits['train']}
    artifacts = ['data.json', 'numeric.pt', 'statistics.pt'] + (['loss_context.pt'] if role == 'A' else [])
    atomic_json(out/'aggregation_complete.json', dict(complete=True, **provenance,
        partitions={k:len(v) for k,v in data['pool_partitions'].items()}, legal_training_ticks=sum(legal.values()),
        legal_ticks_by_episode=legal, full_demonstration_audits=full_audits, correction_audits=correction_audits,
        artifacts={name:sha256(out/name) for name in artifacts}))
    return data


def build_pool(role, output, collection, *, source=None, experts=None, registry=None):
    """Build after closed24-demo and96-opportunity collection receipts exist."""
    from visuotactile.training_expansion import registry_binding, read_registry, validate_expansion_entry
    source = BASE/role/'state_data' if source is None else Path(source)
    experts = ROOT/'experts' if experts is None else Path(experts)
    from tools.tight_paired_dagger import active_registry
    registry = active_registry() if registry is None else Path(registry)
    collection = Path(collection)
    old = read(source/'data.json')
    binding = registry_binding(registry)
    registered = read_registry(binding, old)  # Validate ORIGINAL source only.
    expert_receipt = read(experts/'result.json')
    if not expert_receipt.get('software_complete') or expert_receipt.get('accepted') != 24:
        raise ValueError('All24 shared full demonstrations must be closed before aggregation')
    full = read(experts/'expert_entries.json')
    expert_rows = read(experts/'results.json')
    accepted = {r['case'] for r in expert_rows if r['accepted']}
    if expert_receipt.get('attempts') != len(expert_rows):
        raise ValueError('Expert attempt denominator differs')
    for row in expert_rows:
        if row['accepted']:
            result = row['result']
            if not all(result.get(k) for k in ('legal', 'software_complete', 'success')) or full.get(row['case'], {}).get('family') != row['family']:
                raise ValueError('Accepted full demonstration lacks a legal native expert success')
    if accepted != set(full):
        raise ValueError('Full demonstration acceptance/results differ')
    from tools.tight_paired_collection import summarize
    summary = summarize(collection)
    if summary['opportunities'] != 96 or summary['roles'][role]['opportunities'] != 48:
        raise ValueError('Preserve both full opportunity denominators')
    all_corrections = read(collection/'correction_entries.json')
    if any(e.get('source_group') not in ('A', 'B') for e in all_corrections.values()):
        raise ValueError('Unknown correction owner')
    results = {r['case']:r for r in read(collection/'results.json')}
    accepted_ids = {i for i,r in results.items() if r['result'].get('accepted_for_training')}
    if accepted_ids != set(all_corrections):
        raise ValueError('Correction manifest and actual admission results differ')
    for identity, entry in all_corrections.items():
        row = results[identity]
        result = row['result']
        if row['role'] != entry['source_group'] or not all(result.get(k) for k in ('legal', 'software_complete', 'success')) or not result.get('dagger', {}).get('expert_takeover_success'):
            raise ValueError('Correction lacks a legal native own-student expert recovery')
    corrections = {i:e for i,e in all_corrections.items() if e['source_group'] == role}
    for entry in corrections.values():
        validate_expansion_entry(entry, old, binding)
    data = _materialize(source, output, old, full, corrections, role, registered, binding)
    atomic_json(Path(output)/'collection_sources.json', dict(experts=str(experts.resolve()), collection=str(collection.resolve()),
        expert_entries_sha256=sha256(experts/'expert_entries.json'), expert_results_sha256=sha256(experts/'results.json'),
        correction_entries_sha256=sha256(collection/'correction_entries.json'), collection_results_sha256=sha256(collection/'results.json')))
    return data


def self_check(output):
    """Exercise real serialization with synthetic causal histories, no renderer."""
    import tempfile
    import torch
    from visuotactile.checkpointing import save
    checks = []
    with tempfile.TemporaryDirectory(prefix='paired_pool_fixture_') as temporary:
        folder = Path(temporary)
        full = {f'full{i}':dict(family=f'new{i}', seed=i, split='train', trajectory_kind='full_expert_demonstration_v1', expert_speed=SPEED) for i in range(24)}
        registry = dict(families={f'new{i}':dict(seed=i) for i in range(24)})
        correction = dict(own=dict(family='new0', seed=0, split='train', source_group='A', trajectory_kind='expert_takeover_v1'))
        def row(mask):
            n=len(mask); command=torch.arange(n*29).reshape(n,29).float()
            return dict(robot=torch.arange(n*178).reshape(n,178).float(), command=command, time=torch.arange(n).double()/60,
                joint_limits=torch.tensor([[-1.,1.]]*29), q_reached=command+.1, u_expert=command,
                supervision_mask=torch.tensor(mask), control_segment=torch.tensor([0,1,1]))
        original_row=row([False,True,True]); fullrow=row([True,True,True]); newrow=row([False,False,True])
        for role in 'AB':
            source=folder/role;source.mkdir()
            ids=[f'old{i}' for i in range(180)];validation=[f'val{i}' for i in range(6)]
            old=dict(supervision='q_reached' if role=='A' else 'u_expert', normalization_train_ids=ids[:30],
                normalization_identity='unchanged_fixture_affine', splits=dict(train=ids, validation=validation, test=[]),
                entries={i:dict(family=i,split='train' if i in ids else 'validation') for i in ids+validation})
            atomic_json(source/'data.json',old)
            save(source/'statistics.pt',dict(fixed=torch.tensor([7.,9.])))
            save(source/'numeric.pt',dict(data_sha256=sha256(source/'data.json'), contract={}, records={i:original_row for i in ids+validation}))
            if role=='A':save(source/'loss_context.pt',{i:dict(geometry='loss_only') for i in ids+validation})
            own=copy.deepcopy(correction);own['own']['source_group']=role
            result=_materialize(source,folder/(role+'_out'),old,full,own,role,registry,{},
                full_loader=lambda e,r:(fullrow,{}), correction_loader=lambda e,include_geometry:(newrow,None,{'geometry':'loss_only'}),
                context_loader=lambda e:dict(geometry='loss_only'))
            out=folder/(role+'_out');cache=torch.load(out/'numeric.pt',weights_only=False)
            assert sha256(source/'statistics.pt')==sha256(out/'statistics.pt')
            assert result['splits']['validation']==validation and result['normalization_train_ids']==ids[:30]
            assert result['pool_partitions']['own_new_correction']==['own']
            assert torch.equal(cache['records']['old0']['supervision_mask'],original_row['supervision_mask'])
            assert torch.equal(cache['records']['own']['robot'],newrow['robot'])
            assert not cache['records']['own']['supervision_mask'][:2].any()
            assert cache['records']['full0']['supervision_mask'].all()
            assert not torch.equal(cache['records']['full0']['q_reached'],cache['records']['full0']['u_expert'])
            assert (out/'loss_context.pt').exists()==(role=='A')
            checks.extend([role+':fixed_affine',role+':preserved_splits',role+':retained_history_and_masks',role+':all_full_labels',role+':distinct_targets',role+':geometry_boundary'])
            wrong=copy.deepcopy(own);wrong['own']['source_group']='B' if role=='A' else 'A'
            try:validate_partitions(old,full,wrong,role,registry)
            except ValueError:checks.append(role+':cross_owner_rejected')
            else:raise AssertionError('Cross-owner correction was admitted')
            leak=copy.deepcopy(full);leak['full0']['family']='val0'
            try:validate_partitions(old,leak,own,role,registry)
            except ValueError:checks.append(role+':heldout_family_rejected')
            else:raise AssertionError('Heldout family admitted')
            if role=='B':
                empty=_materialize(source,folder/'zero_correction_out',old,full,{},role,registry,{},
                    full_loader=lambda e,r:(fullrow,{}), context_loader=lambda e:dict(geometry='loss_only'))
                assert len(empty['splits']['train'])==204 and empty['aggregation']['new_training_ids']==[]
                checks.append('B:zero_new_corrections_accepted')
    result=dict(passed=True,checks=checks,check_count=len(checks),gpu_steps=0,physical_steps=0,
        api='build_pool(role, output, collection, source=BASE/role/state_data, experts=ROOT/experts, registry=ROOT/training_expansion_registry.json)',
        format='data.json + numeric.pt + original statistics.pt; A-only loss_context.pt; source HDF5 entries for later frozen visual cache',
        zero_corrections_allowed=True, no_source_affine_refit=True)
    atomic_json(output,result)
    print(json.dumps(result),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['build','self-check'])
    p.add_argument('--role',choices=['A','B'])
    p.add_argument('--output',type=Path)
    p.add_argument('--collection',type=Path,default=ROOT/'collection')
    args=p.parse_args()
    if args.mode=='self-check':return self_check(args.output or AUDIT/'pool_impl_status.json')
    if args.role is None or args.output is None:p.error('build requires --role and --output')
    return build_pool(args.role,args.output,args.collection)


if __name__=='__main__':main()
