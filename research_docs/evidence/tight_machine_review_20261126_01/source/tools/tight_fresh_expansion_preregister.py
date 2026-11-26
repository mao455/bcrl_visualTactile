"""Metadata-only train/development reset separation; creates no research session.

This evidence is not the collection admission registry: later preparation must
bind its actual session source manifests and reproduce these requested resets.
"""
import argparse
import json
from pathlib import Path

from visuotactile.compact import sha256
from visuotactile.narrow_data import direct_configuration
from visuotactile.training_expansion import (configuration_digest, make_registry, read_registry)
from visuotactile.training_resources import atomic_json


def read(path):
    return json.loads(Path(path).read_text())


def preregister(source_pool, partition_manifest, frozen_families, registry_id, train_seeds, development_seeds):
    pool=Path(source_pool).resolve()
    domain=read(pool/'session.json')['domain']
    manifests={role:pool/role/'state_data/data.json' for role in ('A','B')}
    registry=make_registry(registry_id,domain,train_seeds,manifests,partition_manifest,
        frozen_families,created_utc=0)
    data=read(manifests['A'])
    if (not development_seeds or any(type(seed) is not int or seed<0 for seed in development_seeds)
            or len(set(development_seeds))!=len(development_seeds)):
        raise ValueError('Unique nonnegative development seeds required')
    excluded=list(read(frozen_families))+list(registry['families'].values())
    for binding in registry['prior_training_registries']:
        excluded.extend(read_registry(binding)['families'].values())
    seeds={row['seed'] for row in excluded}
    hashes={configuration_digest(row['narrow_reset']) for row in excluded}
    development=[]
    for seed in development_seeds:
        reset=direct_configuration(domain,seed);digest=configuration_digest(reset)
        if seed in seeds or digest in hashes:
            raise ValueError('Development overlaps an existing or newly registered seed/configuration')
        development.append(dict(seed=seed,family=f'fresh_development_{seed}',
            split='development_only',narrow_reset=reset,configuration_sha256=digest))
        seeds.add(seed);hashes.add(digest)
    validation={role:{identity:read(path)['entries'][identity]['family']
        for identity in read(path)['splits']['validation']} for role,path in manifests.items()}
    return dict(version='training_expansion_development_preregistration_v1',
        intended_registry_id=registry_id,source_pool=str(pool),domain=domain,
        domain_sha256=registry['domain_sha256'],source_manifests=registry['source_manifests'],
        original_partition_manifest=registry['original_partition_manifest'],
        original_frozen_families=registry['original_frozen_families'],
        source_pool_counts={split:len(ids) for split,ids in data['splits'].items()},
        source_train_families=sorted({data['entries'][identity]['family'] for identity in data['splits']['train']}),
        train=list(registry['families'].values()),development=development,
        existing_validation_identities=validation,existing_validation_families=registry['source_validation_families'],
        protected_original_families=registry['protected_families'],
        checks=dict(unique_new_train_configurations=len(train_seeds),unique_new_development_configurations=len(development),
            train_development_seed_overlap=[],train_development_configuration_overlap=[],
            development_existing_seed_overlap=[],development_existing_configuration_overlap=[],
            six_validation_families_unchanged=True),
        interpretation='Requested native float32 reset fields only; no hidden-state independence or broad mass/friction generalization claim.',
        registration='Metadata preregistration only; future collection admission must bind actual session source manifests and match these resets. No session is created.',
        created_session=False,simulator_starts=0,optimizer_updates=0,outcomes_used=False,
        test_trajectories_opened=0,code_sha256=sha256(__file__))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-pool',type=Path,required=True)
    parser.add_argument('--partition-manifest',type=Path,required=True)
    parser.add_argument('--frozen-families',type=Path,required=True)
    parser.add_argument('--registry-id',required=True)
    parser.add_argument('--train-seeds',nargs='+',type=int,required=True)
    parser.add_argument('--development-seeds',nargs='+',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior preregistration evidence')
    result=preregister(args.source_pool,args.partition_manifest,args.frozen_families,
        args.registry_id,args.train_seeds,args.development_seeds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)


if __name__=='__main__':
    main()
