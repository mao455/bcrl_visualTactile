"""Remove explicitly scoped, inactive historical nominal derivatives only."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

PUBLIC=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('scratch_cleanup',PUBLIC/'cleanup_stale_assets.py')
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
ROOT=helper.ROOT
BASE=ROOT.parent
LEDGER=ROOT/'resource_cleanup_20261120/attempt04_nominal'
ALLOWED=('tight_fulltask_20261116_01','tight_fresh_20261116_04','tight_budget_20261116_01',
         'tight_fresh_20261116_02','tight_fresh_20261116_00')

def read(path):return json.loads(Path(path).read_text())
def sha(path):return helper.digest(Path(path))
def dump(path,value):helper.dump(Path(path),value)
def safe(path):
    path=Path(path)
    assert path.resolve(strict=True)==path and path.is_file() and not path.is_symlink()
    rel=path.relative_to(BASE)
    assert rel.parts[0] in ALLOWED and path.parent.name=='nominal' and path.suffix=='.pt'
    assert path.stem not in ('index','scales','statistics','numeric','best','last')
    assert path.stat().st_nlink==1
    return path

def processes():
    active,opened,receipt=helper.process_references()
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==os.getpid():continue
        try:
            if p.stat().st_uid!=os.getuid():continue
            args=(p/'cmdline').read_bytes().split(b'\0')
            assert not any(a.decode(errors='replace').startswith(str(BASE/name)) for name in ALLOWED for a in args),p
        except (FileNotFoundError,ProcessLookupError):pass
    return opened,receipt

def plan():
    LEDGER.mkdir(parents=True,exist_ok=False)
    protection=helper.protection();protected={Path(p).resolve() for p in protection}
    opened,process_receipt=processes();files=[];groups=[];skipped=[];retained={}
    for run in ALLOWED:
        for idx in sorted((BASE/run).rglob('nominal/index.json')):
            if idx.resolve()!=idx:continue
            candidates=[p for p in idx.parent.glob('*.pt') if not p.is_symlink() and p.stat().st_nlink==1]
            if not candidates:continue
            backend=idx.parent.parent
            cfg=backend/'configuration.json'
            if not cfg.exists():cfg=backend/'session.json'
            try:
                configuration=read(cfg)['residual_pair'];index=read(idx)
                source=Path(configuration['source_root']);data_path=source/'state_data/data.json'
                data=read(data_path)
                assert sha(data_path)==index['data_sha256']
                refs=list(configuration['bases'].values());assert len(refs)==1
                front=Path(refs[0]['checkpoint']);assert front.is_file()
                assert sha(front)==refs[0]['sha256']==index['base_sha256']
                assert all(Path(e['path']).is_file() for e in data['entries'].values())
                assert (backend/'training/finished.json').is_file()
                identities=[safe(p) for p in candidates]
                assert all(p.stem in index['episodes'] for p in identities)
                assert all(p.resolve() not in protected and (p.stat().st_dev,p.stat().st_ino) not in opened for p in identities)
                for p in [idx,cfg,front,data_path,source/'state_data/statistics.pt',backend/'training/finished.json']:
                    retained[str(p)]=helper.identity(p)
                for p in identities:
                    expected=index['episodes'][p.stem]['sha256'];assert sha(p)==expected
                    files.append(dict(path=str(p),sha256=expected,**helper.identity(p)))
                groups.append(dict(directory=str(idx.parent),files=len(identities),index_sha256=sha(idx),
                    source_frontend=str(front),source_frontend_sha256=index['base_sha256'],data_manifest=str(data_path),
                    source_raw_files=len(data['entries']),all_source_raw_present=True,configuration=str(cfg)))
            except (AssertionError,KeyError,FileNotFoundError) as error:
                skipped.append(dict(directory=str(idx.parent),reason=repr(error)))
    document=dict(version='inactive_historical_nominal_retirement_v1',utc=time.time(),
        authorization='User explicitly requests cleanup of this project only; obsolete reproducible derivatives',
        scope=list(ALLOWED),files=files,groups=groups,skipped=skipped,protected_before=protection,
        retained_before=retained,processes=process_receipt,
        limitation='Historical nominal payloads must be regenerated with retained frontend, source data and configuration before historical offline/training replay; old index hashes retained, no bitwise regeneration claim')
    with gzip.open(LEDGER/'manifest.json.gz','wt') as stream:json.dump(document,stream)
    result=dict(files=len(files),allocated_bytes=sum(r['blocks']*512 for r in files),groups=groups,skipped=skipped,
        protected_paths=len(protection),manifest_sha256=sha(LEDGER/'manifest.json.gz'))
    dump(LEDGER/'plan.json',result);print(json.dumps(result),flush=True)

def apply(expected):
    assert sha(LEDGER/'manifest.json.gz')==expected
    with gzip.open(LEDGER/'manifest.json.gz','rt') as stream:document=json.load(stream)
    assert not (LEDGER/'deleted.jsonl').exists()
    opened,process_receipt=processes()
    for name,info in {**document['protected_before'],**document['retained_before']}.items():
        assert helper.identity(Path(name))==info,name
    for row in document['files']:
        p=safe(row['path']);assert helper.identity(p)=={k:row[k] for k in helper.identity(p)}
        assert (row['dev'],row['ino']) not in opened
    before=shutil.disk_usage(ROOT).free;count=0;allocated=0
    with (LEDGER/'deleted.jsonl').open('x') as journal:
        for row in document['files']:
            p=safe(row['path']);assert helper.identity(p)=={k:row[k] for k in helper.identity(p)}
            journal.write(json.dumps(dict(event='intent',path=str(p),sha256=row['sha256'],utc=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
            fd=os.open(p.parent,os.O_DIRECTORY|os.O_RDONLY|os.O_NOFOLLOW)
            try:
                s=os.stat(p.name,dir_fd=fd,follow_symlinks=False)
                assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(row['dev'],row['ino'],row['size'],row['mtime_ns'],row['ctime_ns'])
                os.unlink(p.name,dir_fd=fd)
            finally:os.close(fd)
            count+=1;allocated+=row['blocks']*512
            journal.write(json.dumps(dict(event='deleted',path=str(p),utc=time.time()))+'\n');journal.flush();os.fsync(journal.fileno())
    unchanged=all(helper.identity(Path(p))==s for p,s in {**document['protected_before'],**document['retained_before']}.items())
    result=dict(complete=True,files_deleted=count,attributed_allocated_bytes=allocated,free_before=before,
        free_after=shutil.disk_usage(ROOT).free,protected_and_retained_stat_unchanged=unchanged,
        protected_paths=len(document['protected_before']),retained_metadata_model_paths=len(document['retained_before']),
        raw_files_deleted=0,model_files_deleted=0,current_visual_caches_deleted=0,other_project_mutations=0,
        utc=time.time(),processes=process_receipt,limitation=document['limitation'])
    dump(LEDGER/'result.json',result);assert unchanged;print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--apply-sha256');args=parser.parse_args()
    if args.apply_sha256:apply(args.apply_sha256)
    else:plan()
