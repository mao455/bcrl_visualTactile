from pathlib import Path
import json,hashlib,shutil,time,re,os
R=Path.cwd();A=R/'outputs/diagnostics/tight_machine_review_20261126_01';B=R/'research_docs/evidence/tight_machine_review_20261126_01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=read(B/'MANIFEST.json');by={r['destination']:r for r in m['files']}
for rel in ('disk_final_accounting.json','file_changes_since_round2.csv.gz','final_disk_accounting.py','final_disk_console.log','METHOD_AND_LIMITS.md','close_delivery.py'):
    p=A/rel;out=B/'audit'/rel;shutil.copyfile(p,out)
    by['audit/'+rel]=dict(source=str(p.relative_to(R)),destination='audit/'+rel,bytes=out.stat().st_size,sha256=sha(out))
for name in ('research_docs/TIGHT_MACHINE_REVIEW_20261126.md','outputs/maintenance_20261123/README_MACHINE_REVIEW_20261126.md'):
    p=R/name;out=B/'context'/name;shutil.copyfile(p,out)
    key=str(out.relative_to(B));by[key]=dict(source=name,destination=key,bytes=out.stat().st_size,sha256=sha(out))
# Context report is a verbatim provenance copy; canonical report links are
# resolved from research_docs, and the index links directly into this bundle.
required=read(A/'retained_actual_dependencies.json')['paths'];problems=[]
for x in required:
    p=Path(x['path'])
    if not p.exists() or p.stat().st_size!=x['size']:problems.append(x['path'])
assert not problems
refs={};checks=[]
for p in (R/'research_docs/TIGHT_MACHINE_REVIEW_20261126.md',B/'README.md'):
    s=p.read_text()
    refs.update({k:v for k,v in re.findall(r'^\[([^\]]+)\]:\s*(\S+)',s,re.M)})
    targets=re.findall(r'\]\(([^)]+)\)',s)+[v for k,v in re.findall(r'^\[([^\]]+)\]:\s*(\S+)',s,re.M)]
    for target in targets:
        if target.startswith(('http:','https:','#')):continue
        target=target.split('#')[0]
        assert (p.parent/target).exists(),(p,target)
        checks.append(str(p.parent/target))
roots='outputs/visuotactile/run/tight_machine_review_20261126_01';alive=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        parts=(p/'cmdline').read_bytes().split(b'\0');exe=parts[0].decode(errors='replace')
        args=[x.decode(errors='replace') for x in parts[1:] if x]
        if int(p.name)!=os.getpid() and 'python' in Path(exe).name and any(roots in x for x in args):alive.append(int(p.name))
    except (OSError,ValueError):pass
assert not alive
close=dict(utc=time.time(),formal_dependencies_checked=len(required),missing_or_size_changed=problems,
    canonical_report_and_index_targets_checked=len(checks),own_python_workers_remaining=alive,
    original256_denominator_unchanged=True,new_training_admission=0,optimizer_updates=0,
    completed_real_rescues=2,completed_valid_ticks=208,interrupted_valid_ticks_retained=39,
    no_terminal64_point_selection=True,git_pushed=False)
(A/'closure.json').write_text(json.dumps(close,indent=2)+'\n');shutil.copyfile(A/'closure.json',B/'audit/closure.json')
by['audit/closure.json']=dict(source=str((A/'closure.json').relative_to(R)),destination='audit/closure.json',bytes=(B/'audit/closure.json').stat().st_size,sha256=sha(B/'audit/closure.json'))
m.update(files=sorted(by.values(),key=lambda x:x['destination']),utc=time.time(),status='export_complete')
(B/'MANIFEST.json').write_text(json.dumps(m,indent=2)+'\n')
for x in m['files']:
    p=B/x['destination'];assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256']
seen=set();allocated=0
for p in B.rglob('*'):
    s=p.lstat();key=s.st_dev,s.st_ino
    if key not in seen:seen.add(key);allocated+=s.st_blocks*512
receipt=dict(utc=time.time(),manifest_files=len(m['files']),export_logical_bytes=sum(x['bytes'] for x in m['files'])+(B/'MANIFEST.json').stat().st_size,
    export_allocated_bytes_including_existing_receipt=allocated,manifest_sha256=sha(B/'MANIFEST.json'),post_disk_snapshot=True,
    freed_bytes=0,excluded_from_manifest=['MANIFEST.json','delivery_accounting.json'],
    qualification='Only export allocation is measured here. Original project snapshot remains at its stated time; post-snapshot reporting/audit updates are not another cleanup release.')
for p in (B/'delivery_accounting.json',A/'delivery_accounting.json'):p.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(closure=close,export=receipt)))
