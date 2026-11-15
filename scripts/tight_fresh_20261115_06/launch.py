#!/usr/bin/env python3
"""Reuse the completed single-cycle owner with this explicit new source bundle."""
import argparse,importlib.util,json
from pathlib import Path

BUNDLE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('single_cycle_launch',BUNDLE.parent/'tight_fresh_20261115_02/launch.py')
launcher=importlib.util.module_from_spec(spec);spec.loader.exec_module(launcher)
launcher.BUNDLE=BUNDLE;launcher.TREE=BUNDLE.parents[1]

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--worker-tree',type=Path,required=True)
    parser.add_argument('--expected-commit',required=True);parser.add_argument('--launch',action='store_true')
    args=parser.parse_args();tree=args.worker_tree.resolve(strict=True)
    config,loop=launcher.prepare(tree,args.expected_commit)
    print(json.dumps(launcher.launch(tree,args.expected_commit,config,loop) if args.launch else
        dict(configuration=config,loop=loop,started=False),indent=2))
