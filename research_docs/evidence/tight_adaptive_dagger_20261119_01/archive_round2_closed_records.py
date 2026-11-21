"""Losslessly retire only closed R2 DP receipt files after independent acceptance."""
import argparse
import importlib.util
import json
from pathlib import Path
import time

PUBLIC=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('closed_r2_archive',PUBLIC/'archive_closed_records.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
a.LEDGER=a.ROOT/'resource_cleanup_20261120/attempt13_round2_records'
RELATIVES=tuple(f'round02/learning/{role}/training/learning.jsonl' for role in 'AB')


def safe(path):
    path=Path(path)
    assert path.resolve(strict=True)==path and not path.is_symlink() and path.is_file()
    assert path.is_relative_to(a.ROOT) and str(path.relative_to(a.ROOT)) in RELATIVES
    assert path.stat().st_nlink==1
    assert json.loads((path.parent/'finished.json').read_text())['complete'] is True
    return path


def gate():
    accepted=json.loads((a.ROOT/'round02/learning/exposure_acceptance.json').read_text())
    assert accepted['passed'] and len(accepted['components'])==4
    assert all(row['eligible'] and row['receipt_replay_exact'] and row['minimum_tick_steps']>=50
               and row['minimum_visits']>=50 for row in accepted['components'])
    assert (a.ROOT/'round02/complete.json').is_file() and (a.ROOT/'final_freeze.json').is_file()
    return accepted


def plan():
    gate();a.LEDGER.mkdir(parents=True,exist_ok=False)
    protected=a.helper.protection()
    for role in 'AB':
        for relative in ('training/best.pt','training/last.pt',f'backend/{role}/training/best.pt',
                         f'backend/{role}/training/last.pt','state_data/data.json',
                         'state_data/statistics.pt','cache_index.json'):
            p=a.ROOT/'round02/learning'/role/relative
            protected[str(p)]=a.helper.identity(p)
    for p in (a.ROOT/'final_freeze.json',a.ROOT/'round02/learning/exposure_acceptance.json'):
        protected[str(p)]=a.helper.identity(p)
    _,opened,census=a.helper.process_references();candidates=[]
    for relative in RELATIVES:
        p=safe(a.ROOT/relative);info=a.helper.identity(p)
        assert (info['dev'],info['ino']) not in opened and str(p) not in protected
        assert time.time()-info['mtime_ns']/1e9>1800
        candidates.append(dict(path=str(p),sha256=a.helper.digest(p),identity=info))
    manifest=dict(files=candidates,protected_before=protected,census=census,utc=time.time(),
                  scope='Only two completed R2 DP receipt files after all four independent exposure audits and final freeze; models/raw/current caches preserved')
    a.helper.dump(a.LEDGER/'manifest.json',manifest)
    a.helper.dump(a.LEDGER/'restore_manifest.json',dict(files=[dict(path=x['path']) for x in candidates]))
    summary=dict(files=len(candidates),original_bytes=sum(x['identity']['size'] for x in candidates),
                 protected_paths=len(protected),manifest_sha256=a.helper.digest(a.LEDGER/'manifest.json'))
    a.helper.dump(a.LEDGER/'plan.json',summary);print(json.dumps(summary),flush=True)


a.safe=safe
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply-sha256');args=parser.parse_args()
    if args.apply_sha256:gate();a.apply(args.apply_sha256)
    else:plan()
