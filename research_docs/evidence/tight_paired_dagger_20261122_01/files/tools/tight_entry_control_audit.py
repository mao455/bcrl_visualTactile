"""Compare the old sampler preflight to every durable R2 optimizer receipt."""
import argparse
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
from pathlib import Path

from tools.tight_entry_phase import PROJECT, read
from visuotactile.compact import sha256
from visuotactile.training_resources import atomic_json


@contextmanager
def learning_rows(directory):
    path=Path(directory)/'learning.jsonl'
    if path.exists():
        with path.open() as stream:
            yield (json.loads(line) for line in stream)
    else:
        import zstandard
        receipt=read(path.with_suffix('.jsonl.archive.json'))
        archive=path.with_suffix('.jsonl.zst')
        assert (receipt.get('archive_verified') or receipt.get('byte_roundtrip_verified')) and receipt['original_removed']
        assert sha256(archive)==receipt['archive_sha256']
        with archive.open('rb') as source, zstandard.ZstdDecompressor().stream_reader(source) as decoded, io.BufferedReader(decoded) as stream:
            def checked_rows():
                h=hashlib.sha256()
                for line in stream:
                    h.update(line)
                    yield json.loads(line)
                assert h.hexdigest()==receipt['sha256']
            yield checked_rows()


def run(config):
    from visuotactile.run_facts import sampler_facts
    c=read(config); out=PROJECT/c['audit']; history=PROJECT/c['history']; proof=read(out/'sampling_preflight.json')
    rows=[]
    for role in 'AB':
        directory=history/role/'training'; visits=0; updates=0
        with gzip.open(out/f'{role}_planned_sampling.jsonl.gz','rt') as planned, learning_rows(directory) as actual:
            for update, receipt in enumerate(actual,1):
                assert receipt['updates']==update
                for visit in receipt['visits']:
                    expected=json.loads(next(planned))
                    assert expected['update']==update and expected['identity']==visit['episode']
                    assert expected['epoch']==receipt['epoch']
                    assert expected['original']==visit['anchor_sampling']
                    visits+=1
                updates=update
            assert next(planned,None) is None
        assert updates==7200 and visits==14400
        cfg=proof['roles'][role]['original_configuration']
        unchanged={}
        for folder in ('visuotactile/models','visuotactile/dp_objectives.py','visuotactile/trajectory_batch.py',
                       'visuotactile/trajectory_training.py','visuotactile/precision.py','visuotactile/target_windows.py',
                       'visuotactile/offline_planning.py','visuotactile/position_control.py'):
            p=PROJECT/folder
            for file in (p.rglob('*.py') if p.is_dir() else [p]):
                key=str(file.relative_to(PROJECT))
                expected=cfg['source_sha256'].get(key)
                if expected is None:
                    expected=next((v for k,v in cfg['source_sha256'].items() if k.endswith('/'+key)),None)
                assert expected==sha256(file),key
                unchanged[key]=expected
        rows.append(dict(role=role,actual_optimizer_receipts=updates,actual_episode_visits=visits,
            exact_sampler_receipts=True,exact_actual_decks=True,unchanged_numeric_sources=unchanged,
            source_sampler_facts=sampler_facts(history,role),
            old_configuration_sha256=sha256(directory/'configuration.json'),
            learning_rates=proof['roles'][role]['actual_epoch_rates'],
            backend_learning_rates=proof['roles'][role]['actual_backend_epoch_rates'],
            reuse_status='sampler/model/math/LR verified; visual proof and matching backend identity still required'))
    result=dict(passed=True,rows=rows,scope='Original frozen R2 is the original-sampler arm; no new control optimizer updates',
        selected_checkpoint_hashes={r:proof['roles'][r]['protected'] for r in 'AB'},
        caveat='New anchor count changes training noise consumption; equal seeds do not imply equal per-visit noise tensors')
    atomic_json(out/'original_control_audit.json',result)
    print(json.dumps(dict(passed=True,roles=2,optimizer_receipts=sum(r['actual_optimizer_receipts'] for r in rows))),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    run(p.parse_args().config)


if __name__=='__main__':main()
