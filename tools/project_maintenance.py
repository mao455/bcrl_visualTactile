"""Dependency inventory and reviewed, identity-checked artifact removal.

No implicit age/size deletion. A request names every candidate subtree, its reason,
references and protected paths. Plan is durable before apply; symlinks are refused.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

CODE = Path(__file__).resolve().parents[1]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def source_inventory(entries):
    paths=[CODE/p for p in subprocess.check_output(['git','ls-files','*.py'],cwd=CODE,text=True).splitlines()]
    modules={'.'.join(p.relative_to(CODE).with_suffix('').parts).removesuffix('.__init__'):p for p in paths if p.exists()}
    graph={};errors=[]
    for name,path in modules.items():
        try: tree=ast.parse(path.read_text())
        except (SyntaxError,UnicodeError) as error:
            errors.append(dict(path=str(path.relative_to(CODE)),error=str(error)));continue
        deps=set();package=name if path.name=='__init__.py' else name.rpartition('.')[0]
        for node in ast.walk(tree):
            candidates=[]
            if isinstance(node,ast.Import):candidates=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                prefix=package.split('.')[:len(package.split('.'))-node.level+1] if node.level else []
                base='.'.join(prefix+([node.module] if node.module else []))
                candidates=[base]+[base+'.'+a.name for a in node.names]
            elif isinstance(node,ast.Constant) and isinstance(node.value,str):
                # Includes configured '-m package.module' entries and importlib strings.
                candidates=[s.strip('"\'(),[]') for s in node.value.split()]
            deps.update(c for c in candidates if c in modules and c!=name)
        graph[name]=sorted(deps)
    live=set();pending=list(entries)
    while pending:
        name=pending.pop()
        if name in live:continue
        if name not in graph:raise ValueError('Unknown entrypoint: '+name)
        live.add(name);pending.extend(graph[name])
    return dict(entries=entries,edges=graph,reachable=sorted(live),parse_errors=errors,
        unreachable=[dict(module=n,path=str(p.relative_to(CODE))) for n,p in modules.items() if n not in live],
        scope='Python AST imports and literal module entrypoints; shell/config/runtime references require separate review')


def identity(path):
    st=path.lstat()
    return dict(device=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,
        links=st.st_nlink,allocated_bytes=st.st_blocks*512)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def open_inodes():
    found=set();inaccessible=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            for fd in (proc/'fd').iterdir():
                try:st=fd.stat();found.add((st.st_dev,st.st_ino))
                except FileNotFoundError:pass
        except (PermissionError,ProcessLookupError):inaccessible.append(int(proc.name))
        except FileNotFoundError:pass
    return found,inaccessible


def within(path, root):return path==root or root in path.parents


def restricted_system_process(pid):
    """These nondumpable session daemons cannot be experiment workers; document them."""
    p=Path('/proc')/str(pid)
    try:
        command=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        group=(p/'cgroup').read_text().strip();name=(p/'comm').read_text().strip()
        accepted=(group.endswith('/init.scope') and name in ('systemd','(sd-pam)') and
            (command=='/usr/lib/systemd/systemd --user' or command=='(sd-pam)')) or (
            name=='ssh-agent' and group.endswith('/gnome-keyring-daemon.service') and
            command.startswith('/usr/bin/ssh-agent -D -a /run/user/'))
        return dict(pid=pid,command=command,cgroup=group,reason='Protected session daemon, not an experiment worker') if accepted else None
    except FileNotFoundError:return dict(pid=pid,reason='Process exited during inventory')


def plan(request_path, ledger):
    assert not ledger.exists(), 'Never replace a previous deletion ledger'
    request=json.loads(request_path.read_text());allowed=Path(request['allowed_root']).resolve()
    assert within(allowed,(CODE/'outputs').resolve()) or within(allowed,CODE), 'Outside this project'
    protected={Path(p).resolve():why for p,why in request['protected'].items()}
    opened,inaccessible=open_inodes();rows=[];kept=[];seen=set()
    for candidate in request['candidates']:
        original=Path(candidate['path'])
        assert not original.is_symlink(), 'No symlink candidate'
        base=original.resolve();assert within(base,allowed) and base!=allowed
        paths=sorted(base.rglob('*')) if base.is_dir() else [base]
        for path in paths:
            if path.is_symlink():kept.append(dict(path=str(path),reason='symlink: unresolved ownership'));continue
            if not path.is_file() or path in seen:continue
            seen.add(path);st=identity(path)
            why=next((reason for p,reason in protected.items() if within(path,p)),None)
            if (st['device'],st['inode']) in opened:why='open in a current user process'
            if why:kept.append(dict(path=str(path),reason=why));continue
            rows.append(dict(path=str(path),**st,reason=candidate['reason'],
                references=candidate.get('references',[]),sha256=digest(path) if candidate.get('hash',False) else None))
    write(ledger,dict(state='planned',created=time.time(),request=str(request_path.resolve()),
        request_sha256=digest(request_path),allowed_root=str(allowed),protected=request['protected'],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CODE,text=True).strip(),
        files=rows,retained=kept,inaccessible_user_processes=inaccessible,
        logical_candidate_bytes=sum(x['bytes'] for x in rows)))
    return dict(files=len(rows),retained=len(kept),logical_bytes=sum(x['bytes'] for x in rows))


def apply(ledger):
    data=json.loads(ledger.read_text());assert data['state'] in ('planned','applying')
    allowed=Path(data['allowed_root']);protected=[Path(p).resolve() for p in data['protected']]
    journal=ledger.with_suffix('.deletions.jsonl')
    done={r['path']:r for r in (json.loads(s) for s in journal.read_text().splitlines())} if journal.exists() else {}
    opened,inaccessible=open_inodes()
    exclusions=[restricted_system_process(pid) for pid in inaccessible]
    assert all(exclusions), ('Unreadable unclassified user fd inventories',inaccessible)
    data['restricted_process_exclusions']=exclusions
    # Preflight the entire plan before the first irreversible removal.
    for row in data['files']:
        if row['path'] in done:continue
        path=Path(row['path']);assert within(path.resolve(),allowed) and not path.is_symlink()
        assert not any(within(path.resolve(),p) for p in protected)
        now=identity(path)
        assert all(now[k]==row[k] for k in ('device','inode','bytes','mtime_ns')), 'Candidate changed: '+str(path)
        assert (now['device'],now['inode']) not in opened, 'Candidate opened: '+str(path)
        if row['sha256']:assert digest(path)==row['sha256']
    data.update(state='applying',free_bytes_before=os.statvfs(allowed).f_bavail*os.statvfs(allowed).f_frsize)
    write(ledger,data)
    with journal.open('a') as stream:
        for row in data['files']:
            if row['path'] in done:continue
            path=Path(row['path']);now=identity(path)
            assert all(now[k]==row[k] for k in ('device','inode','bytes','mtime_ns'))
            path.unlink()
            result=dict(path=row['path'],utc=time.time(),released_blocks_bytes=now['allocated_bytes'] if now['links']==1 else 0)
            stream.write(json.dumps(result)+'\n');stream.flush();os.fsync(stream.fileno());done[row['path']]=result
    directories=sorted({p for x in data['files'] for p in Path(x['path']).parents if within(p,allowed) and p!=allowed},key=lambda p:len(p.parts),reverse=True)
    removed=[]
    for path in directories:
        try:path.rmdir();removed.append(str(path))
        except (OSError,FileNotFoundError):pass
    data.update(state='completed',finished=time.time(),deleted_files=len(done),removed_empty_directories=removed,
        deleted_checkpoints=sum(Path(x['path']).suffix in ('.pt','.pth','.ckpt') for x in data['files']),
        released_blocks_bytes=sum(x['released_blocks_bytes'] for x in done.values()),
        free_bytes_after=os.statvfs(allowed).f_bavail*os.statvfs(allowed).f_frsize)
    write(ledger,data)
    return {k:data[k] for k in ('deleted_files','deleted_checkpoints','released_blocks_bytes','free_bytes_before','free_bytes_after')}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['sources','plan','apply'])
    p.add_argument('--entries',nargs='+');p.add_argument('--request',type=Path);p.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='sources':r=source_inventory(a.entries);write(a.ledger,r);print(json.dumps(dict(reachable=len(r['reachable']),unreachable=len(r['unreachable']),errors=r['parse_errors'])))
    else:print(json.dumps(plan(a.request,a.ledger) if a.mode=='plan' else apply(a.ledger)))
