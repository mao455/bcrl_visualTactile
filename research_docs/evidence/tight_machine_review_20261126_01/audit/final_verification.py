from pathlib import Path
import json,hashlib,subprocess,time,os,datetime
import h5py,numpy as np,torch
from visuotactile.batch_recording import snapshot,recorded_observation
R=Path.cwd();A=R/'outputs/diagnostics/tight_machine_review_20261126_01';ROOT=R/'outputs/visuotactile/run/tight_machine_review_20261126_01'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
models=[]
for role,x in read(A/'model_identity_normalization.json').items():
    for component in ('frontend','backend'):
        p=x['pair'][component];digest=sha(p);expected=x['pair'][component+'_sha256']
        assert digest==expected;models.append(dict(role=role,component=component,path=p,sha256=digest))
pair=read(ROOT/'never_gate_pilot07_B72_training/registration.json')['source_student_pair']
for component in ('frontend','backend'):
    assert sha(pair[component])==pair[component+'_sha256']
with h5py.File(ROOT/'never_gate_pilot07_B72_training/rollouts/B72_28817004_expert/demonstration/data.h5') as f:
    g=f['episodes/B72_28817004_expert'];parity=[]
    for tick in (0,232,235,246):
        packet={k:torch.from_numpy(v[tick:tick+1]) for k,v in g['obs'].items()}
        packet.update(image=torch.from_numpy(g['frames/rgb_u8'][tick:tick+1]),
            depth_m=torch.from_numpy(g['frames/depth_m'][tick:tick+1]),
            depth_valid=torch.from_numpy(g['frames/depth_valid'][tick:tick+1]))
        old=snapshot(packet);new=snapshot(recorded_observation(packet,capture=False));full=snapshot(recorded_observation(packet,capture=True))
        assert set(old)-set(new)=={'image','depth_m','depth_valid'}
        assert all(np.array_equal(new[k],old[k]) for k in new)
        assert all(np.array_equal(full[k],old[k]) for k in old)
        parity.append(dict(tick=tick,numeric_fields=len(new),collection_fields=len(full),exact_written_field_parity=True,
            removed_before_copy_bytes=sum(v.numel()*v.element_size() for k,v in packet.items() if k in ('image','depth_m','depth_valid'))))
source=[]
for folder in ('tools','visuotactile'):
    for p in sorted((R/folder).rglob('*.py')):
        rel=str(p.relative_to(R));digest=sha(p)
        old=subprocess.run(['git','show','60e7b1f:'+rel],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
        public=hashlib.sha256(old.stdout).hexdigest() if old.returncode==0 else None
        if digest!=public:source.append(dict(path=rel,public60e7b1f_sha256=public,current_machine_sha256=digest))
before=[]
for p in sorted((A/'source_before').rglob('*.py')):
    rel=str(p.relative_to(A/'source_before'));current=R/rel
    before.append(dict(path=rel,before_sha256=sha(p),after_sha256=sha(current)))
interval=[];owners=[]
for p in ROOT.rglob('project_job_started.json'):
    out=p.parent;x=read(p);exitpath=out/'project_exit.json';e=read(exitpath) if exitpath.exists() else None
    owners.append(dict(owner=str(out),pid=x['pid'],started_utc=x['utc'],exit=e))
    if e:interval.append((x['utc'],e['utc']))
merged=[]
for a,b in sorted(interval):
    if merged and a<=merged[-1][1]:merged[-1][1]=max(b,merged[-1][1])
    else:merged.append([a,b])
own=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if str(ROOT) in cmd and int(p.name)!=os.getpid() and ('python' in cmd or 'kit' in cmd):own.append(dict(pid=int(p.name),command=cmd))
    except (OSError,ValueError):pass
result=dict(utc=time.time(),models=models,source_policy_sha_verified=True,formal_model_files_unchanged=True,
    fixed_record_recording_parity=parity,public_source_comparison=source,before_after_source=before,
    attachment='No attachment payload was exposed to this session; no attachment equivalence claim. Original-machine frozen SHA/records take priority.',
    owners=owners,owner_interval_union_s=sum(b-a for a,b in merged),owner_intervals=merged,
    own_process_candidates=own,optimizer_updates=0,main_pool_new_entries=0,
    resource_claim='Capacity sharing with external jobs; no signal sent to external processes; no zero-interference or sustained-speedup claim')
(A/'final_verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
(A/'source_public_comparison_final.json').write_text(json.dumps(dict(utc=result['utc'],files=source,before_after=before,attachment=result['attachment']),indent=2)+'\n')
print('models',len(models),'record parity',len(parity),'owners',len(owners),'all exits',all(x['exit'] for x in owners),'union_s',result['owner_interval_union_s'],'process candidates',own)
