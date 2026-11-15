"""Apply one reviewed closed-run payload list; preserve raw, bests and metadata."""
import argparse,hashlib,json,os,pathlib,shutil,time

RUN=pathlib.Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run')
EXPECTED='1fda61256392efebc6245148762944b923d85e5cf1126e470e1de4132665bbae'
MANIFEST=RUN/'tight_fresh_20261115_02/completed_review/retirement_candidates.json'


def identity(path):
    s=path.stat()
    return dict(dev=s.st_dev,ino=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,blocks=s.st_blocks)


def main(output,apply):
    raw=MANIFEST.read_bytes();assert hashlib.sha256(raw).hexdigest()==EXPECTED
    manifest=json.loads(raw);rows=manifest['files'];assert len(rows)==570
    paths={pathlib.Path(r['path']) for r in rows};assert len(paths)==len(rows)
    assert all(p.resolve()==p and p.is_file() and not p.is_symlink() and p.stat().st_nlink==1 for p in paths)
    for r in rows:
        p=pathlib.Path(r['path']);assert identity(p)=={k:r[k] for k in identity(p)}
        if r['kind']=='completed_nonbest_last':
            assert p.name=='last.pt' and p.parent.name=='training'
            assert pathlib.Path(r['retained_best'])==p.with_name('best.pt') and p.with_name('best.pt').is_file()
            done=json.loads(pathlib.Path(r['finished']).read_text());assert done.get('complete') or done.get('target_reached')
        else:
            assert r['kind']=='retired_frozen_visual' and p.parent.name=='perception'
    source=pathlib.Path(next(r['retained_source_manifest'] for r in rows if r['kind']=='retired_frozen_visual'))
    data=json.loads(source.read_text());assert all(pathlib.Path(e['path']).is_file() for e in data['entries'].values())
    current=RUN/'tight_fresh_20261115_02/round_12'
    for role in ('A','B'):
        assert not paths & {p.resolve() for p in (current/role/'state_data/perception').glob('*.pt')}
    inodes={(r['dev'],r['ino']) for r in rows}
    for proc in pathlib.Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            for fd in (proc/'fd').iterdir():
                try:s=fd.stat()
                except (FileNotFoundError,PermissionError):continue
                assert (s.st_dev,s.st_ino) not in inodes,'Candidate still open by PID '+proc.name
        except (FileNotFoundError,PermissionError):continue
    protected={p:identity(pathlib.Path(p)) for p in manifest['protected_bests']}
    output.mkdir(parents=True,exist_ok=False);before=shutil.disk_usage(RUN).free
    report=dict(manifest=str(MANIFEST),manifest_sha256=EXPECTED,apply=apply,started=time.time(),
        selected_files=len(rows),selected_allocated_bytes=sum(r['blocks']*512 for r in rows),
        retained_raw_paths=len(data['entries']),protected_bests=protected,free_before=before,deleted=0)
    (output/'started.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        if apply:
            with (output/'deleted.jsonl').open('x') as log:
                for r in rows:
                    p=pathlib.Path(r['path']);assert identity(p)=={k:r[k] for k in identity(p)}
                    p.unlink();report['deleted']+=1;log.write(json.dumps(r)+'\n');log.flush();os.fsync(log.fileno())
        assert all(identity(pathlib.Path(p))==s for p,s in protected.items())
        report['complete']=True
    finally:
        report.update(ended=time.time(),free_after=shutil.disk_usage(RUN).free)
        report['free_delta']=report['free_after']-before
        (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='protected_bests'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=pathlib.Path,required=True);parser.add_argument('--apply',action='store_true')
    args=parser.parse_args();main(args.output,args.apply)
