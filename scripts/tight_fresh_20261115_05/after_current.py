#!/usr/bin/env python3
"""Observe the current cycle boundary once, then launch within its original clock.

Ordinary bounded supervisor; no model polling, new deadline, retry or promotion.
The first closed collection wave is observed and recorded before this exits.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import time

BUNDLE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('grasp_launch',BUNDLE/'launch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
launcher=module.launcher
read,write,require=launcher.read,launcher.write,launcher.require


def run(tree, expected, ledger):
    config,loop=launcher.prepare(tree,expected)
    require(config.get('inherit_absolute_deadline') is True,'Preserve this resource window')
    old=Path(config['old_session']);deadline=read(old/'session.json')['deadline']
    old_hash=launcher.sha(old/'session.json');ledger.mkdir(parents=True,exist_ok=False)
    state=dict(stage='waiting_current_cycle',started=time.time(),deadline=deadline,
        expected_commit=expected,source_session_sha256=old_hash,signals_sent_to_current_job=0)
    def record():state['observed_utc']=time.time();write(ledger/'status.json',state)
    try:
        record()
        while True:
            if (ledger/'STOP').exists():raise InterruptedError('Requested stop before boundary')
            if time.time()>deadline-loop['bounds']['minimum_round_remaining_seconds']-120:
                raise TimeoutError('Insufficient time for another full cycle in the original window')
            finished=old/'research_loop/owner_finished.json'
            if finished.exists():
                result=read(finished)
                require(result.get('software_complete') is True,'Current cycle did not complete; preserve its failure')
                if (old/'monitor_exit.json').exists() and all(
                    launcher.common.unit_state(config[key])['properties'].get('ActiveState') in ('inactive','failed')
                    for key in ('old_owner','old_monitor')):break
            record();time.sleep(10)
        require(launcher.sha(old/'session.json')==old_hash,'Current session bytes changed')
        from tools.tight_fresh_grasp_evidence import behavior
        state['previous_cycle_grasp']=behavior(Path(config['source'])/'development')
        state['stage']='launching_early_grasp';record()
        launched=launcher.launch(tree,expected,config,loop)
        state.update(stage='observing_first_closed_wave',launch=launched);record()
        root=Path(config['output']);collection=root/'new_family_15'
        while time.time()<deadline-60:
            if (ledger/'STOP').exists():raise InterruptedError('Requested observer stop')
            owner=root/'research_loop/owner_finished.json'
            if owner.exists() and not read(owner).get('software_complete'):
                raise RuntimeError('New owner stopped: '+str(read(owner).get('reason')))
            results=collection/'results.json'
            if results.exists():
                rows=read(results)
                accepted=[r for r in rows if r['result'].get('accepted_for_training')]
                if len(rows)>=2:
                    state.update(stage='first_closed_results_observed',resolved=len(rows),accepted=len(accepted),
                        actual_takeovers=[dict(case=r['case'],accepted=r['result'].get('accepted_for_training',False),
                            trigger=r['result'].get('dagger',{}).get('trigger_receipt'),
                            valid_expert_ticks=r['result'].get('dagger',{}).get('expert_valid_ticks')) for r in rows])
                    record();return state
            record();time.sleep(10)
        raise TimeoutError('Original deadline before first closed results')
    except BaseException as error:
        state.update(stage='stopped',error=repr(error));record();raise
    finally:
        state['ended']=time.time();write(ledger/'finished.json',state)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--worker-tree',type=Path,required=True)
    p.add_argument('--expected-commit',required=True);p.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args();run(a.worker_tree.resolve(strict=True),a.expected_commit,a.ledger)
