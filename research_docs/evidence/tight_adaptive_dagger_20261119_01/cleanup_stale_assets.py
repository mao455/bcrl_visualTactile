"""Inventory/apply exact stale generated USD scratch files; never project data."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time

PROJECT = Path(__file__).resolve().parents[3]
ROOT = Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')
LEDGER = ROOT / 'resource_cleanup_20261120' / 'attempt02'
PATTERN = re.compile(r'/tmp/precise_assembly_assets_[a-zA-Z0-9_]+')


def dump(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''): h.update(block)
    return h.hexdigest()


def identity(path):
    s = path.lstat()
    return dict(dev=s.st_dev, ino=s.st_ino, size=s.st_size, blocks=s.st_blocks,
        mode=s.st_mode, uid=s.st_uid, nlink=s.st_nlink, mtime_ns=s.st_mtime_ns,
        ctime_ns=s.st_ctime_ns)


def protection():
    data = json.loads((ROOT/'ACTIVE_DEPENDENCIES_EXPANDED.json').read_text())
    paths = {Path(item['path']) for item in data['files']}
    for role in 'AB':
        for suffix in ('training/best.pt', 'training/last.pt',
                       f'backend/{role}/training/best.pt', f'backend/{role}/training/last.pt'):
            paths.add(ROOT / 'round01/learning' / role / suffix)
    for path in (ROOT/'round02').rglob('*'):
        if path.is_file() and path.name in ('data.h5','numeric.h5','trace.npz','plans.npz',
                'result.json','capture_complete.json','plan.json','session.json'):
            paths.add(path)
    return {str(p): identity(p) for p in sorted(paths)}


def process_references():
    roots = set(); inodes = set(); scanned = 0; errors = []; processes = []
    known_services = {'/usr/lib/systemd/systemd --user', '(sd-pam)',
                      '/usr/bin/ssh-agent -D -a /run/user/1001/keyring/.ssh'}
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        try:
            if process.stat().st_uid != os.getuid(): continue
            scanned += 1
            command = (process/'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            values = [command, os.readlink(process/'cwd')]
            try: values.append((process/'maps').read_text())
            except FileNotFoundError: continue
            for fd in (process/'fd').iterdir():
                try:
                    values.append(os.readlink(fd))
                    s = fd.stat(); inodes.add((s.st_dev, s.st_ino))
                except (FileNotFoundError, ProcessLookupError): pass
            hits = set(PATTERN.findall('\n'.join(values)))
            roots.update(hits)
            if hits: processes.append(dict(pid=int(process.name), roots=sorted(hits), command=command[:500]))
        except (FileNotFoundError, ProcessLookupError): pass
        except PermissionError as error:
            if command.strip() not in known_services:
                raise RuntimeError('Unknown same-user process cannot be inspected: '+command) from error
            errors.append(dict(pid=process.name, command=command.strip(), error=str(error),
                classification='Separately inspected ordinary systemd/session authentication service; not a simulation or asset consumer'))
    return roots, inodes, dict(scanned_same_user_processes=scanned,
        active_asset_roots=sorted(roots), matching_processes=processes,
        permission_limitations=errors,
        scope='Accessible same-user cmdline/cwd/maps and all open fd paths/inodes; established session services have recorded permission limitations; other users are not inspected', utc=time.time())


def tree(path):
    rows = []
    for base, directories, files in os.walk(path, followlinks=False):
        for name in [None] + sorted(files):
            current = Path(base) if name is None else Path(base)/name
            info = identity(current)
            if info['uid'] != os.getuid() or current.is_symlink():
                raise ValueError('Foreign owner/symlink: ' + str(current))
            if stat.S_ISDIR(info['mode']): kind = 'directory'
            elif stat.S_ISREG(info['mode']) and info['nlink'] == 1:
                kind = 'file'
                if current.suffix not in ('.usd', '.yaml') and current.name != '.asset_hash':
                    raise ValueError('Unknown scratch payload: ' + str(current))
            else: raise ValueError('Special/hardlinked scratch payload: ' + str(current))
            rows.append(dict(path=str(current), kind=kind, **info))
        for name in directories:
            if (Path(base)/name).is_symlink(): raise ValueError('Directory symlink')
    return sorted(rows, key=lambda r:r['path'])


def plan(minimum_age_hours=24):
    if minimum_age_hours < .5:
        raise ValueError('Generated scratch must be idle for at least thirty minutes')
    LEDGER.mkdir(parents=True, exist_ok=False)
    before = protection(); active, opened, references = process_references()
    protected_inodes = {(s['dev'],s['ino']) for s in before.values()}
    candidates = []; skipped = []; now = time.time()
    for path in sorted(Path('/tmp').glob('precise_assembly_assets_*')):
        if not PATTERN.fullmatch(str(path)) or path.is_symlink() or not path.is_dir(): continue
        try:
            if set(p.name for p in path.iterdir()) != {'usd', 'baked_usd'}:
                raise ValueError('Unknown root contents')
            rows = tree(path)
            age = now - max(max(r['mtime_ns'],r['ctime_ns'])/1e9 for r in rows)
            if age < minimum_age_hours*3600: raise ValueError(f'Younger than {minimum_age_hours} hours')
            if str(path) in active: raise ValueError('Active path reference')
            if any((r['dev'],r['ino']) in opened | protected_inodes for r in rows):
                raise ValueError('Active/protected inode reference')
            candidates.append(dict(root=str(path), age_seconds=age, rows=rows))
        except ValueError as error: skipped.append(dict(path=str(path), reason=str(error)))
    document = dict(version='generated_stale_usd_cleanup_v1', authorization='User: 你自己清理资源',
        scope=f'Only exact owned /tmp/precise_assembly_assets_* older than {minimum_age_hours}h with generated USD/YAML/hash files',
        minimum_age_hours=minimum_age_hours,
        utc=now, free_bytes=shutil.disk_usage(ROOT).free, references=references,
        generator=str(PROJECT/'isaacsimenvs/tasks/precise_assembly/scene_utils.py'),
        generator_sha256=digest(PROJECT/'isaacsimenvs/tasks/precise_assembly/scene_utils.py'),
        generation_contract='setup_scene lines 57 onward creates mkdtemp; URDF conversion and bake create these derivatives; source assets and project outputs are preserved',
        protected_before=before, candidates=candidates, skipped=skipped)
    with gzip.open(LEDGER/'manifest.json.gz','wt') as stream: json.dump(document,stream)
    summary = dict(directories=len(candidates), entries=sum(len(c['rows']) for c in candidates),
        allocated_bytes=sum(r['blocks']*512 for c in candidates for r in c['rows']),
        protected_paths=len(before), skipped=skipped, references=references,
        manifest_sha256=digest(LEDGER/'manifest.json.gz'), free_bytes=document['free_bytes'])
    dump(LEDGER/'plan.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='skipped'}))


def apply(expected):
    manifest = LEDGER/'manifest.json.gz'
    assert digest(manifest) == expected
    assert not (LEDGER/'operations.jsonl').exists(), 'No automatic repeat'
    with gzip.open(manifest,'rt') as stream: document=json.load(stream)
    assert protection() == document['protected_before'], 'Protected identity changed before apply'
    active, opened, references = process_references()
    for candidate in document['candidates']:
        assert candidate['root'] not in active
        assert tree(Path(candidate['root'])) == candidate['rows'], candidate['root']
        assert not any((r['dev'],r['ino']) in opened for r in candidate['rows'])
    free_before=shutil.disk_usage(ROOT).free; count=0; allocated=0
    with (LEDGER/'operations.jsonl').open('x') as journal:
        for candidate in document['candidates']:
            root=Path(candidate['root'])
            assert root.parent == Path('/tmp') and PATTERN.fullmatch(str(root))
            journal.write(json.dumps(dict(event='intent',root=str(root),utc=time.time()))+'\n')
            journal.flush(); os.fsync(journal.fileno())
            files=[r for r in candidate['rows'] if r['kind']=='file']
            directories=sorted((r for r in candidate['rows'] if r['kind']=='directory'),
                               key=lambda r:len(Path(r['path']).parts),reverse=True)
            for row in files:
                path=Path(row['path']); assert identity(path)=={k:row[k] for k in identity(path)}
                fd=os.open(path.parent, os.O_DIRECTORY|os.O_RDONLY|os.O_NOFOLLOW)
                try:
                    s=os.stat(path.name,dir_fd=fd,follow_symlinks=False)
                    assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(
                        row['dev'],row['ino'],row['size'],row['mtime_ns'],row['ctime_ns'])
                    os.unlink(path.name,dir_fd=fd)
                finally: os.close(fd)
            for row in directories:
                path=Path(row['path']); s=path.lstat()
                assert stat.S_ISDIR(s.st_mode) and (s.st_dev,s.st_ino)==(row['dev'],row['ino'])
                path.rmdir()
            count+=1; allocated+=sum(r['blocks']*512 for r in candidate['rows'])
            journal.write(json.dumps(dict(event='deleted',root=str(root),utc=time.time()))+'\n')
            journal.flush(); os.fsync(journal.fileno())
    after=protection()
    result=dict(complete=True,directories_deleted=count,attributed_allocated_bytes=allocated,
        free_bytes_before=free_before,free_bytes_after=shutil.disk_usage(ROOT).free,
        protected_identities_unchanged=after==document['protected_before'],protected_paths=len(after),
        verification='All protected file stat identities compared before/after; no new complete-byte SHA audit',
        original_deadline_unchanged=True,utc=time.time(),references_before_apply=references)
    dump(LEDGER/'result.json',result)
    assert result['protected_identities_unchanged']
    dump(ROOT/'CLEANUP_COORDINATION_COMPLETED.json',dict(owner='Experiment session under new explicit user authorization',
        cleanup_result=str(LEDGER/'result.json'),**result))
    print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--apply-sha256')
    parser.add_argument('--minimum-age-hours',type=float,default=24)
    parser.add_argument('--ledger',type=Path,default=LEDGER)
    args=parser.parse_args()
    LEDGER=args.ledger.resolve()
    if not LEDGER.is_relative_to(ROOT/'resource_cleanup_20261120'):
        raise ValueError('Cleanup ledger must stay in this experiment cleanup root')
    if args.apply_sha256: apply(args.apply_sha256)
    else: plan(args.minimum_age_hours)
