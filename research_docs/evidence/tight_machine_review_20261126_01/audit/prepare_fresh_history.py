"""Prepare isolated numeric/vision inputs from the two actual diagnostic rescues.

No source is admitted to any training pool; no model or affine is altered.
Existing canonical record, actual-configuration and visual-cache readers are used.
"""
from pathlib import Path
import json,copy,shutil,time
import torch,h5py
from tools.tight_machine_review import ROOT,PAIRED,CONFIG,session_at,sources,job,read
from visuotactile.compact import sha256
from visuotactile.narrow_data import record,actual_configuration_labels
from visuotactile.training_resources import atomic_json

owner=ROOT/'fresh_rescue_history02';session=session_at(owner,duration=3600)
session['cache_deadline']=session['deadline'];session['resources']['min_ram_gib']=6
atomic_json(owner/'session.json',session)
roll=ROOT/'never_gate_pilot07_B72_training/rollouts'
entries={};records={};cases=[];audits=[]
for seed in (28817004,28817008):
    identity=f'B72_{seed}_expert';d=roll/identity
    result=read(d/'result.json');audit=read(d/'capture_audit.json')
    assert result['dagger']['expert_takeover_success'] and audit['passed']
    path=d/'demonstration/data.h5';key,meta,row=record(path,episode=identity,privileged=False)
    entry=dict(path=str(path),episode=key,length=len(row['time']),source_sha256=sha256(path),
        family=result['family'],seed=seed,split='diagnostic_train',trajectory_kind='expert_takeover_v1')
    reached,q_audit=actual_configuration_labels(entry)
    with h5py.File(path) as f:
        tr=f['episodes'][key]['transition'];dag=tr['dagger']
        load=lambda n:torch.from_numpy(dag[n][:])
        mask=load('expert_label_valid');control=load('control_source');clean=load('switch_clean')
        u=load('u_expert');q=load('q_reached');segment=load('segment_id')
        assert not (mask&((control!=1)|~clean)).any()
        assert torch.equal(row['command'][mask],u[mask]) and torch.equal(reached[mask],q[mask])
    row.update(u_expert=torch.where(mask[:,None],u,0),q_reached=torch.where(mask[:,None],q,0),
        supervision_mask=mask,control_segment=segment)
    first=int(mask.nonzero()[0]);anchor=first//12*12;stop=anchor+12
    cases.append(dict(identity=identity,new50=False,first_legal=first,anchor=anchor,stop=stop,
        ticks=list(range(first,stop)),raw_path=str(path),visual_cache_sha256='pending cache; committed index verified by preparation stage'))
    entries[identity]=entry;records[identity]=row
    audits.append(dict(identity=identity,legal_ticks=int(mask.sum()),q_audit=q_audit,capture_audit=str(d/'capture_audit.json'),
        full_prefix_preserved=True,prefix_supervised=False,main_pool_admission=False))
for role in 'AB':
    child=owner/role;out=child/'state_data';out.mkdir(parents=True)
    old=read(PAIRED/f'learning/{role}/state_data/data.json')
    data=dict(old,entries=entries,splits=dict(train=[],validation=[],test=[],correction_dev=list(entries)),
        diagnostic_only=True,training_forbidden=True,aggregation=dict(new_training_ids=[],diagnostic_ids=list(entries)))
    atomic_json(out/'data.json',data)
    torch.save(dict(data_sha256=sha256(out/'data.json'),records=records),out/'numeric.pt')
    atomic_json(child/'session.json',session)
    atomic_json(child/'continuation.json',dict(reuse_index=str(PAIRED/'learning/A/cache_index.json')))
    for name in ('visual_arithmetic.json',):
        p=PAIRED/'learning'/name
        if not p.exists():p=PAIRED/'learning'/role/name
        shutil.copyfile(p,child/name)
plan=read(ROOT/'fixed_history_fit01/entry_same_history_plan.json')
plan.update(cases=cases,interpretation='Fixed-history transfer/fit of unchanged parent/new models on two new diagnostic real expert rescues; neither case was used for their training. No optimization or autonomous recovery claim.')
atomic_json(owner/'entry_same_history_plan.json',plan)
atomic_json(owner/'plan.json',dict(source_sha256=sources(),diagnostic_data_root=str(owner),interpretation=plan['interpretation']))
atomic_json(owner/'data_preparation.json',dict(utc=time.time(),rows=audits,training_admission=False,optimizer_updates=0))
job(owner,[dict(entry='visual_cache',args=['--root',str(owner),'--output',str(owner),'--payload-audit-memo']),
    dict(entry='machine_review',args=['worker','--output',str(owner),'--mode','history'])],'fresh_rescue_history')
print(owner)
