"""Preregister new training families and mixed real-student takeover opportunities.

Preparation only: execution uses the ordinary bounded batch owner/collector.
No held-out trajectory, optimizer, simulator, or expert outcome is read here.
"""
import argparse
import copy
from pathlib import Path
import shutil
import subprocess
import time

from tools.tight_fresh_rollouts import completed_pair, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json
from visuotactile.training_expansion import (COLLECTION_CONTRACT, NEVER_FALLBACK,
    canonical_digest, make_registry, read_registry, registry_binding,
    validate_collection_plan, validate_schedule,schedule_trigger)


def prepare(source, output, *, session_root, registry_id, seeds, partition_manifest,
            frozen_families, schedules, lanes, event_seed, round_index,
            existing_identities=(), execution_backend=None, writer_bytes=512 << 20, vision_microbatch=2,
            verification_workers=0,verification_max_pending=4,vision_tail_profile=None,inactive_render_profile=None):
    from visuotactile.execution_backend import normalize_backend
    from visuotactile.models.native_vision import validate_vision_microbatch,validate_vision_tail_profile
    from visuotactile.async_recording import validate_verification_parameters
    from visuotactile.position_control import ExpertPositionBridge
    from visuotactile.state_benchmark import dagger_configuration
    vision_microbatch = validate_vision_microbatch(vision_microbatch)
    tail = validate_vision_tail_profile(vision_tail_profile)
    from tools.tight_fresh_batch_run import validate_inactive_render_request
    inactive = validate_inactive_render_request(inactive_render_profile,execution_backend)
    validate_verification_parameters(verification_workers,verification_max_pending)
    source = Path(source).resolve()
    output = Path(output).resolve()
    session_root = Path(session_root).resolve()
    if output.exists():
        raise FileExistsError('Preserve existing expansion registration/collection')
    if (type(lanes) is not int or lanes < 1 or type(round_index) is not int or round_index < 1
            or type(event_seed) is not int or event_seed < 0):
        raise ValueError('Explicit positive lanes/round and nonnegative event seed required')
    if type(writer_bytes) is not int or writer_bytes < 32 << 20:
        raise ValueError('Explicit bounded capture memory required')
    if not schedules:
        raise ValueError('Explicit schedule cycle required before collection')
    for schedule in schedules:
        validate_schedule(schedule)
    if len(set(existing_identities)) != len(existing_identities):
        raise ValueError('Existing source identities must be unique')
    session = read(session_root / 'session.json')
    if session['deadline'] <= time.time():
        raise ValueError('Session deadline has already passed')
    source_session = read(source / 'session.json')
    if canonical_digest(session['domain']) != canonical_digest(source_session['domain']):
        raise ValueError('Expansion uses the unchanged source narrow reset domain')
    manifests = {role: source / role / 'state_data/data.json' for role in ('A', 'B')}
    data = {role: read(path) for role, path in manifests.items()}
    pairs = {role: completed_pair(source, role) for role in ('A', 'B')}
    registry = make_registry(registry_id, session['domain'], list(seeds), manifests,
        partition_manifest, frozen_families, created_utc=time.time())
    old_family_rows = {row['family']: row for row in read(frozen_families)}
    for prior in data['A'].get('training_expansion_registries', []):
        old_family_rows.update(read_registry(prior)['families'])
    opportunities = []
    seen = set()
    for identity in existing_identities:
        if any(identity not in value['splits']['train'] for value in data.values()):
            raise ValueError('Existing opportunities must belong to both source training splits')
        entry, other = data['A']['entries'][identity], data['B']['entries'][identity]
        if any(entry.get(key) != other.get(key) for key in ('path', 'family', 'source_sha256')):
            raise ValueError('Existing A/B trajectory identities differ')
        family = entry['family']
        if family in seen or family not in old_family_rows:
            raise ValueError('One existing training opportunity per registered family required')
        seen.add(family)
        opportunities.append(dict(old_family_rows[family], source_identity=identity, new_family=False))
    opportunities.extend(dict(row, new_family=True) for row in registry['families'].values())
    backend = normalize_backend(execution_backend)
    # The registry precedes the plan and any run_started/result file. Failed
    # preparation is retained in its own directory, never silently rewritten.
    output.mkdir(parents=True)
    shutil.copyfile(session_root / 'session.json', output / 'session.json')
    atomic_json(output / 'training_expansion_registry.json', registry)
    binding = registry_binding(output / 'training_expansion_registry.json')
    cases = []
    for index, row in enumerate(opportunities):
        schedule = schedules[index % len(schedules)]
        tick = int(schedule[5:]) if schedule.startswith('fixed') else NEVER_FALLBACK
        for role in (('A', 'B') if index % 2 == 0 else ('B', 'A')):
            name = f'r{round_index:02d}_{role}_{row["family"]}_{schedule}'
            spec = dict(identity=name, family=row['family'], split='train', cohort='dagger_train',
                source_identity=row['source_identity'], seed=row['seed'], event_seed=event_seed+index,
                narrow_reset=copy.deepcopy(row['narrow_reset']),
                collision_gate='live_collision_mesh_bounds_v1', takeover_tick=tick)
            if row['new_family']:
                spec.update(training_expansion=copy.deepcopy(binding),
                    initial_configuration_sha256=row['configuration_sha256'])
            bridge = dict(version=ExpertPositionBridge.version, expert_speed=ExpertPositionBridge.speed_scheme)
            trigger=schedule_trigger(schedule)
            if trigger is not None:bridge['trigger']=trigger
            dagger_configuration(bridge, spec, data[role])
            cases.append(dict(name=name, role=role, spec=spec, dagger=bridge, video=False,
                source_data=str(manifests[role]), expansion_schedule=schedule))
    plan = dict(kind='takeover', stage_takeover_contract=COLLECTION_CONTRACT,
        training_expansion_registry=binding, pairs=pairs, cases=cases, round_index=round_index,
        resource_root=str(session_root), source_root=str(source),
        source_data_sha256={role: sha256(path) for role, path in manifests.items()},
        source_session_sha256=sha256(session_root / 'session.json'),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(path): sha256(path) for folder in ('tools', 'visuotactile')
            for path in sorted(Path(folder).rglob('*.py'))},
        batch=dict(version='whole_wave_complete_student_v1', lanes=lanes, seed=event_seed,
            neural_batch=True, writer_bytes=writer_bytes, vision_microbatch=vision_microbatch,
            verification_workers=verification_workers,verification_max_pending=verification_max_pending,
            ordering='Role-grouped whole waves; no dynamic recycling; all opportunities retained'),
        execution_backend=backend, expert_sha256=session['expert_sha256'],
        inference_scheduler_index_mode='host', first_plan_forensics=False,
        schedule_cycle=list(schedules), registered_new_training_families=len(registry['families']),
        registered_existing_training_families=len(existing_identities), registered_opportunities=len(cases),
        selection='Only native-successful real expert suffixes enter training; all opportunities and failures retained',
        interpretation='Mixed schedules target different failure states; no causal schedule comparison or broad physics generalization',
        test_used=False, test_trajectories_opened=0, utc=time.time())
    if tail is not None:plan['batch']['vision_tail_profile']=tail
    if inactive is not None:plan['batch']['inactive_render_profile']=inactive
    validate_collection_plan(plan)
    atomic_json(output / 'plan.json', plan)
    return plan


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--session-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--registry-id', required=True)
    parser.add_argument('--initial-seeds', type=int, nargs='+', required=True)
    parser.add_argument('--partition-manifest', type=Path, required=True)
    parser.add_argument('--frozen-families', type=Path, required=True)
    parser.add_argument('--schedule-cycle', choices=['fixed40', 'fixed80', 'contact0', 'contact12','near60'], nargs='+', required=True)
    parser.add_argument('--existing-identities', nargs='*', default=[])
    parser.add_argument('--lanes', type=int, required=True)
    parser.add_argument('--event-seed', type=int, required=True)
    parser.add_argument('--round', type=int, required=True)
    parser.add_argument('--backend-json', type=Path)
    parser.add_argument('--writer-mib', type=int, default=512)
    parser.add_argument('--vision-microbatch', type=int, default=2)
    parser.add_argument('--vision-exact-tail',action='store_true',help='Explicit candidate: no padding of the final DINO lane group')
    parser.add_argument('--inactive-render-products',action='store_true',help='Explicit CPU-standard inactive render product candidate')
    parser.add_argument('--verification-workers',type=int,default=0)
    parser.add_argument('--verification-max-pending',type=int,default=4)
    args = parser.parse_args()
    from visuotactile.models.native_vision import EXACT_TAIL_PROFILE
    inactive=None
    if args.inactive_render_products:
        from visuotactile.inactive_render_products import PROFILE
        inactive=PROFILE
    prepare(args.source, args.output, session_root=args.session_root, registry_id=args.registry_id,
        seeds=args.initial_seeds, partition_manifest=args.partition_manifest, frozen_families=args.frozen_families,
        schedules=args.schedule_cycle, lanes=args.lanes, event_seed=args.event_seed, round_index=args.round,
        existing_identities=args.existing_identities,
        execution_backend=read(args.backend_json) if args.backend_json else None, writer_bytes=args.writer_mib << 20,
        vision_microbatch=args.vision_microbatch,verification_workers=args.verification_workers,
        verification_max_pending=args.verification_max_pending,
        vision_tail_profile=EXACT_TAIL_PROFILE if args.vision_exact_tail else None,inactive_render_profile=inactive)


if __name__ == '__main__':
    main()
