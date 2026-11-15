"""Expose actual backend cache progress to an already frozen supervised owner.

Temporary compatibility for finish26 source01ecfe4: its watchdog only observes
the role-level cache_progress_index, not backend/cache_progress. No fake progress,
threshold change, model mutation or modification to the frozen running code.
"""
import argparse
import fcntl
import json
from pathlib import Path
import time

from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


def run(root):
    out=root/'round_15';ledger=root/'backend_progress_relay';ledger.mkdir(exist_ok=False)
    deadline=json.loads((root/'session.json').read_text())['deadline']
    previous={};counts=dict(A=0,B=0)
    with (ledger/'owner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(ledger/'registration.json',dict(started=time.time(),deadline=deadline,source_sha256=sha256(__file__),
            reason='Frozen watchdog omitted actual backend cache progress path; copy only changed real worker progress into its otherwise unused role-level progress index'))
        for role in 'AB':
            assert not (out/role/'cache_progress_index.json').exists()
        while time.time()<deadline and not (root/'owner_finished.json').exists():
            for role in 'AB':
                source=out/role/'backend/cache_progress.json'
                if not source.exists():continue
                payload=source.read_bytes()
                if payload==previous.get(role):continue
                value=json.loads(payload);previous[role]=payload;counts[role]+=1
                atomic_json(out/role/'cache_progress_index.json',dict(version='backend_cache_progress_relay_v1',
                    source_path=str(source),actual_worker_progress=value,complete=False,
                    meaning='Live progress only; not a frozen-DINO cache index or a completion receipt'))
                atomic_json(ledger/'status.json',dict(stage='relaying_real_cache_progress',heartbeat=time.time(),counts=counts))
            time.sleep(5)
        atomic_json(ledger/'finished.json',dict(ended=time.time(),counts=counts,owner_closed=(root/'owner_finished.json').exists()))


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.root.resolve())
