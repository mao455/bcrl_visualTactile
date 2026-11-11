"""Persistent shared pool: source aliases, immutable rounds and held-out isolation."""
import json
import shutil

import h5py
import pytest
import torch

from test_dagger_data import takeover
from visuotactile.compact import sha256
from visuotactile.narrow_data import (dagger_record,ingest_dagger_pool,freeze_dagger_snapshot,
    read_dagger_snapshot,build_dagger_aggregate)
from visuotactile.native_data import CAMERAS,visual_records


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))


def core(tmp_path,record_source=None,supervision='u_expert'):
    out=tmp_path;out.mkdir(exist_ok=True)
    splits={name:[name+str(i) for i in range(count)] for name,count in (('train',166),('validation',30),('test',22))}
    entries={i:dict(path='/MUST_NOT_OPEN_CORE_OR_TEST/'+i,family=name,source_sha256=(name[0]*64),length=40,episode=i)
        for name,ids in splits.items() for i in ids}
    data=dict(splits=splits,entries=entries,families={k:k for k in splits},supervision=supervision,
        normalization_identity=supervision+'_train166_affine_v1')
    write(out/'data.json',data)
    if record_source:
        row,touch,_=dagger_record(record_source,include_geometry=False)
        base={k:row[k] for k in ('robot','command','time','joint_limits','q_reached')}
        ids=splits['train']+splits['validation']
        torch.save(dict(contract={},data_sha256=sha256(out/'data.json'),records={i:base for i in ids}),out/'numeric.pt')
        torch.save({i:touch for i in ids},out/'tactile.pt')
        torch.save(dict(robot_mean=torch.zeros(178),robot_std=torch.ones(178),action_scale=torch.ones(29),action_offset=torch.zeros(29)),out/'statistics.pt')
        torch.save(dict(tactile_mean=torch.zeros(32),tactile_std=torch.ones(32)),out/'tactile_statistics.pt')
    return out


def collection(path,*,value=0.,development=False,existing=None,missing_touch=False):
    path.mkdir(parents=True)
    if existing:
        entry=dict(existing);copy=path/'renamed.h5';shutil.copyfile(existing['path'],copy);entry['path']=str(copy)
    else:
        entry=takeover(path)
        with h5py.File(entry['path'],'r+') as f:
            metadata=json.loads(f.attrs['metadata_json']);metadata.update(camera_ids=CAMERAS,policy_checkpoint_sha256='b'*64)
            f.attrs['metadata_json']=json.dumps(metadata)
            g=f['episodes/0'];g['transition/diagnostics_before/joint_velocity_rad_s'][0,0]=value
            g['obs'].create_dataset('contact_w_N',data=torch.zeros(40,5,3).numpy())
            if not missing_touch:
                g['obs'].create_dataset('finger_position_base_m',data=torch.zeros(40,5,3).numpy())
                g['obs'].create_dataset('finger_rotation_base',data=torch.zeros(40,5,3,3).numpy())
        entry['source_sha256']=sha256(entry['path'])
    entry['family']='train'
    result=dict(legal=True,software_complete=True,success=True,reasons={'max_successes':True},checkpoint_sha256='b'*64,
        dagger=dict(expert_takeover_success=True,expert_valid_ticks=16,provenance_passed=True,position_executor_continuous=True))
    row=dict(identity='local0',accepted=True,label_contract_checked=True,seed=5,takeover_tick=14,result=result)
    write(path/'collection_plan.json',[dict(identity='local0',family='train',seed=5,cohort='dagger_train',pool_partition='correction_dev' if development else 'train')])
    write(path/'correction_entries.json',{'local0':entry});write(path/'collection_results.json',[row])
    write(path/'initial_pair.json',dict(frontend_sha256='f'*64,backend_sha256='b'*64))
    (path/'attempts.jsonl').write_text(json.dumps({'identity':'local0'})+'\n')
    return entry


def ingest(pool,path,source,*,group='G11',run='new',round_index=1):
    return ingest_dagger_pool(pool,path,run_id=run,source_group=group,round_index=round_index,core_data_root=source)


def test_dino_resume_commits_each_episode_and_preserves_unknown_payload(tmp_path,monkeypatch):
    from visuotactile.native_data import cache_visual_pair,DINO_REVISION
    from visuotactile import training_resources
    root=tmp_path/'job';data=root/'state_data';data.mkdir(parents=True)
    dino=tmp_path/'dino';dino.write_bytes(b'frozen backbone')
    payload=tmp_path/'features.pt';torch.save({'features':torch.ones(1)},payload)
    reusable=tmp_path/'index.json'
    contract=dict(dino_sha256=sha256(dino),dino_revision=DINO_REVISION,stride=12,cameras=CAMERAS,
        resolution=[448,252],patches=[384,18,32],depth=[2,18,32],random_augmentation=False,trainable_projection_cached=False)
    row=dict(path=str(payload),sha256=sha256(payload),source_sha256='a'*64,frames=2)
    write(reusable,dict(contract,complete=True,entries={'legacy_local':row}))
    write(data/'data.json',dict(splits=dict(train=['pool_train'],validation=[],correction_dev=['pool_dev']),
        entries={i:dict(path='/MUST_NOT_OPEN_RGBD',length=20,source_sha256='a'*64) for i in ('pool_train','pool_dev')}))
    write(root/'session.json',dict(dino_checkpoint=str(dino),dino_sha256=sha256(dino),cache_deadline=1))
    def no_gpu(*a,**kw):raise AssertionError('A committed/reused feature must not run DINO')
    monkeypatch.setattr('visuotactile.models.native_vision.NativeDinoSpatial',no_gpu)
    monkeypatch.setattr('visuotactile.state_resources.lease_for',lambda *a,**kw:None)
    atomic=training_resources.atomic_json
    def interrupt(path,value):
        if path.name=='cache_progress_index.json' and len(value['entries'])==2:raise RuntimeError('worker interruption')
        atomic(path,value)
    monkeypatch.setattr(training_resources,'atomic_json',interrupt)
    with pytest.raises(RuntimeError,match='interruption'):cache_visual_pair(root,reuse_index=reusable)
    assert set(json.loads((root/'cache_progress_index.json').read_text())['entries'])=={'pool_train'}
    monkeypatch.setattr(training_resources,'atomic_json',atomic)
    cache_visual_pair(root,reuse_index=reusable)
    (root/'cache_index.json').unlink()
    cache_visual_pair(root)  # Resume only from complete per-episode progress.
    assert set(json.loads((root/'cache_index.json').read_text())['entries'])=={'pool_train','pool_dev'}
    # A payload alone is not acceptance. Preserve it before trying reconstruction.
    (root/'cache_index.json').unlink();(root/'cache_progress_index.json').unlink()
    with pytest.raises(FileNotFoundError):cache_visual_pair(root)
    assert list((data/'perception').glob('pool_train.pt.unindexed.*'))
    assert json.loads((root/'cache_progress_index.json').read_text())['preserved_orphans']


def test_pool_keeps_both_group_sources_deduplicates_aliases_and_freezes_snapshot(tmp_path):
    source=core(tmp_path/'core');pool=tmp_path/'pool'
    one=collection(tmp_path/'one',value=1);two=collection(tmp_path/'two',value=2)
    a=ingest(pool,tmp_path/'one',source,run='old');b=ingest(pool,tmp_path/'two',source,group='QG01')
    assert a['integrity_passed'] and b['integrity_passed'] and b['pool_entries']==2
    frozen=freeze_dagger_snapshot(pool,tmp_path/'snap.json',run_id='new',round_index=1)
    assert len(frozen['train_ids'])==2 and len(set(frozen['train_ids']))==2
    assert {e['pool_source']['group'] for e in frozen['entries'].values()}=={'G11','QG01'}
    first=next(iter(frozen['entries'].values()))
    assert first['supervision_boundary']['first_backend_supervised_tick']==24
    assert first['supervision_boundary']['first_dp_supervised_anchor']==24
    assert first['compatibility']['G11']['compatible'] and not first['compatibility']['QG01']['compatible']
    assert not list(pool.rglob('*.h5'))
    duplicate=ingest(pool,tmp_path/'one',source,run='old')
    assert duplicate['inserted']==[] and duplicate['pool_entries']==2
    collection(tmp_path/'alias',existing=one)
    copied=ingest(pool,tmp_path/'alias',source,run='alias',group='QG01')
    assert copied['pool_entries']==2 and copied['duplicates'][0]['reason']=='same_file_or_numeric_execution_content'
    collection(tmp_path/'third',value=3,development=True)
    assert ingest(pool,tmp_path/'third',source,round_index=2)['pool_entries']==3
    assert freeze_dagger_snapshot(pool,tmp_path/'snap.json',run_id='new',round_index=1)==frozen
    later=freeze_dagger_snapshot(pool,tmp_path/'later.json',run_id='new',round_index=2)
    assert len(later['correction_dev_ids'])==1 and len(later['train_ids'])==2
    # Snapshot source membership is protected by its own digest.
    edited=dict(frozen,train_ids=[]);write(tmp_path/'snap.json',edited)
    with pytest.raises(ValueError,match='Snapshot identity'):read_dagger_snapshot(tmp_path/'snap.json')


def test_development_cannot_reenter_training_and_invalid_labels_stop_admission(tmp_path):
    source=core(tmp_path/'core');pool=tmp_path/'pool';entry=collection(tmp_path/'dev',development=True)
    assert ingest(pool,tmp_path/'dev',source)['development_entries']==1
    collection(tmp_path/'train',existing=entry)
    refused=ingest(pool,tmp_path/'train',source,run='other')
    assert not refused['integrity_passed'] and 'train/development' in refused['data_errors'][0]['reason']
    assert refused['training_entries']==0
    broken=collection(tmp_path/'broken',value=4)
    with h5py.File(broken['path'],'r+') as f:f['episodes/0/transition/dagger/q_reached'][25,0]+=1
    broken['source_sha256']=sha256(broken['path']);write(tmp_path/'broken/correction_entries.json',{'local0':broken})
    receipt=ingest(pool,tmp_path/'broken',source,run='broken')
    assert not receipt['integrity_passed'] and receipt['pool_entries']==1
    assert 'actual issued' in receipt['data_errors'][0]['reason']


def test_shared_snapshot_aggregate_is_idempotent_separates_dev_and_preserves_affine(tmp_path):
    first=collection(tmp_path/'old',value=1);source=core(tmp_path/'core',first);pool=tmp_path/'pool'
    ingest(pool,tmp_path/'old',source,run='historical')
    collection(tmp_path/'current',value=2);ingest(pool,tmp_path/'current',source,group='QG01')
    collection(tmp_path/'dev',value=3,development=True);ingest(pool,tmp_path/'dev',source)
    snapshot=tmp_path/'snapshot.json';frozen=freeze_dagger_snapshot(pool,snapshot,run_id='new',round_index=1)
    out=tmp_path/'aggregate'
    result=build_dagger_aggregate(source,{},out,'u_expert',include_geometry=False,snapshot=snapshot,consumer_group='G11')
    assert [len(result['splits'][k]) for k in ('train','validation','test','correction_dev')]==[168,30,22,1]
    assert {k:len(v) for k,v in result['sampling_strata'].items()}==dict(core=166,historical_correction=1,current_correction=1)
    assert not set(result['splits']['correction_dev'])&set(result['splits']['train'])
    assert sha256(source/'statistics.pt')==sha256(out/'statistics.pt')
    assert (out/'tactile.pt').is_symlink()
    loaded=visual_records(out,result['splits']['correction_dev'],'G00')
    assert next(iter(loaded.values()))['supervision_mask'].sum()==16
    assert build_dagger_aggregate(source,{},out,'u_expert',include_geometry=False,snapshot=snapshot,consumer_group='G11')==result
    with pytest.raises(ValueError,match='Completed immutable'):
        build_dagger_aggregate(source,{},out,'u_expert',include_geometry=False,snapshot=snapshot,consumer_group='QG01')
    # A later round sourced from an aggregate strips its already-present aliases
    # and rebuilds from Core plus the single canonical frozen pool membership.
    second=build_dagger_aggregate(out,{},tmp_path/'second','u_expert',include_geometry=False,snapshot=snapshot,consumer_group='G11')
    assert second['splits']==result['splits'] and second['normalization_train_ids']==result['normalization_train_ids']
    assert (out/'tactile.pt').resolve()==(tmp_path/'second/tactile.pt').resolve()


def test_family_and_plan_partition_are_not_inferred_from_results(tmp_path):
    source=core(tmp_path/'core');pool=tmp_path/'pool';collection(tmp_path/'wrong')
    with pytest.raises(ValueError,match='preregistered'):
        ingest_dagger_pool(pool,tmp_path/'wrong',run_id='new',source_group='G11',round_index=1,core_data_root=source,development_ids=['local0'])
    entries=json.loads((tmp_path/'wrong/correction_entries.json').read_text());entries['local0']['family']='test'
    write(tmp_path/'wrong/correction_entries.json',entries)
    receipt=ingest(pool,tmp_path/'wrong',source)
    assert not receipt['integrity_passed'] and 'training families' in receipt['data_errors'][0]['reason']
