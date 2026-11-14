import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from test_selected_frontend_fit import artifacts,write
from test_batch_collection_review import numeric as write_numeric
from tools import tight_fresh_stage_fit as fit


@pytest.fixture
def union_artifacts(artifacts,tmp_path):
    root,frozen,selected=artifacts;role=root/'B';data=fit.read(role/'state_data/data.json')
    torch.save(dict(fixture=torch.zeros(1)),role/'state_data/statistics.pt')
    selected['configuration']['statistics_sha256']=fit.sha256(role/'state_data/statistics.pt')
    data['splits']['train']=['old1','mid1','new1']
    data['entries']['mid1']=dict(split='train',family='train_mid',trajectory_kind='expert_takeover_v1',
        source_sha256='source-mid1',length=80)
    collections=[];records={};cache=fit.read(role/'cache_index.json')
    for round_index,identity,case_name in [(6,'old1','renamed_capture'),(7,'mid1','mid1'),(8,'new1','new1')]:
        collection=tmp_path/f'collection{round_index}';case=collection/case_name;case.mkdir(parents=True)
        result=write_numeric(case,40)
        result.update(accepted_for_training=True,success=True)
        result['dataset_capture']['sha256']=data['entries'][identity]['source_sha256']
        entry=data['entries'][identity];entry.update(length=80,takeover_tick=40,episode=case_name,
            path='/NEVER_OPEN_RAW/'+case_name,round_index=round_index)
        with h5py.File(case/'numeric.h5') as stream:
            t=stream['episode/transition']
            robot=torch.arange(80,dtype=torch.float32)[:,None].expand(80,87).clone()/100
            robot[:,58:87]+=.2
            records[identity]=dict(time=torch.from_numpy(t['time_before_s'][:]),
                supervision_mask=torch.from_numpy(t['dagger/expert_label_valid'][:]),
                control_segment=torch.from_numpy(t['dagger/segment_id'][:]),
                u_expert=torch.from_numpy(t['dagger/u_expert'][:]),q_reached=torch.from_numpy(t['dagger/q_reached'][:]),
                command=torch.zeros(80,29),joint_limits=torch.zeros(29,2),robot=robot)
        visual=role/'state_data/perception'/(identity+'.pt');write(visual,{'frozen_train':identity})
        cache['entries'][identity]=dict(path=str(visual),sha256=fit.sha256(visual),source_sha256=entry['source_sha256'],frames=80)
        write(collection/'plan.json',dict(kind='takeover',round_index=round_index,
            cases=[dict(name=case_name,role='B',spec=dict(identity=case_name,split='train',family=entry['family']))]))
        write(collection/'result.json',dict(software_complete=True))
        write(collection/'results.json',[dict(case=case_name,result=result)])
        write(collection/'correction_entries.json',{case_name:entry})
        collections.append(collection)
    write(role/'state_data/data.json',data)
    torch.save(dict(data_sha256=fit.sha256(role/'state_data/data.json'),records=records),role/'state_data/numeric.pt')
    cache['data_sha256']=fit.sha256(role/'state_data/data.json');write(role/'cache_index.json',cache)
    source=tmp_path/'source/B';source.mkdir(parents=True)
    source_data=dict(splits=dict(train=['source_alias'],validation=[],test=[]),
        entries=dict(source_alias=copy.deepcopy(data['entries']['old1'])))
    write(source/'state_data/data.json',source_data)
    old=copy.deepcopy(selected);old.update(epoch=2,updates=2,train_ids=['source_alias'],validation_ids=[],ema={'scale':torch.tensor(1.)})
    old.pop('sampling_exposure')
    old['configuration'].update(version='fresh30_continuous_episode_v1',data_sha256=fit.sha256(source/'state_data/data.json'))
    (source/'training').mkdir();torch.save(old,source/'training/best.pt')
    write(source/'training/finished.json',dict(checkpoint_sha256=fit.sha256(source/'training/best.pt')))
    continuation=fit.read(role/'continuation.json')
    continuation.update(path=str(source/'training/best.pt'),sha256=fit.sha256(source/'training/best.pt'))
    write(role/'continuation.json',continuation)
    selected.update(train_ids=data['splits']['train'],sampling_exposure=dict(old1=3,mid1=3,new1=6),ema={'scale':torch.tensor(2.)})
    selected['configuration'].update(continuation=continuation,data_sha256=fit.sha256(role/'state_data/data.json'),
        cache_index_sha256=fit.sha256(role/'cache_index.json'))
    write(role/'training/configuration.json',selected['configuration']);torch.save(selected,role/'training/best.pt')
    write(role/'training/finished.json',dict(checkpoint_sha256=fit.sha256(role/'training/best.pt')))
    return SimpleNamespace(root=root,frozen=frozen,collections=collections,data=data,source=source,records=records)


def prepare(x,out):
    return fit.prepare(x.root,'B',x.collections[2],x.frozen,x.frozen,out,include_collection=x.collections[:2])


def test_union_admission_matches_renamed_capture_and_source_ids_by_content(union_artifacts,tmp_path,monkeypatch):
    x=union_artifacts;opened=[];original=h5py.File
    def only_numeric(path,*args,**kwargs):
        assert Path(path).name=='numeric.h5','Raw/visual H5 must not be opened by stage admission'
        opened.append(str(path));return original(path,*args,**kwargs)
    monkeypatch.setattr(h5py,'File',only_numeric)
    plan=prepare(x,tmp_path/'prepared')
    assert plan['version']==fit.UNION_VERSION
    assert plan['identities']==['old1','mid1','new1']
    assert plan['strata']==dict(all=['old1','mid1','new1'],latest=['new1'],source_seen=['old1'],
        source_unseen=['mid1','new1'],collection_06=['old1'],collection_07=['mid1'],collection_08=['new1'])
    assert plan['episodes']['old1']['source_training_id']=='source_alias'
    assert plan['episodes']['old1']['collection_case']=='renamed_capture'
    assert plan['episodes']['old1']['source_prior_full_history_visits']==2
    assert plan['episodes']['mid1']['source_prior_full_history_visits']==0
    assert plan['stratum_counts']['all']['legal_expert_actions']==114
    assert plan['stratum_counts']['latest']['selected_full_history_visits']==6
    assert len(opened)==3 and 'renamed_capture' in opened[0]
    projected=fit.read(Path(plan['train_projection']['root'])/'data.json')
    assert projected['splits']==dict(train=['old1','mid1','new1'],validation=[],test=[])
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('fault,match',[
    ('duplicate_path','Duplicate collection'),('duplicate_content','Duplicate capture'),
    ('not_train','training-family'),('heldout_family','training-family'),
    ('capture_sha','Accepted union recovery'),('case_family','training identity'),
    ('incomplete','Completed collection'),('primary_not_latest','Primary collection'),
    ('missing_current','current training content')])
def test_union_rejects_ambiguous_or_unadmitted_data(union_artifacts,fault,match):
    x=union_artifacts;latest=x.collections[2];extra=list(x.collections[:2]);data=copy.deepcopy(x.data)
    if fault=='duplicate_path':extra.append(latest)
    elif fault=='duplicate_content':
        duplicate=x.collections[0].with_name('duplicate')
        import shutil
        shutil.copytree(x.collections[0],duplicate);extra.append(duplicate)
    elif fault in ('not_train','heldout_family'):
        entry=data['entries']['old1']
        if fault=='not_train':entry['split']='validation'
        else:entry['family']=data['entries']['val1']['family']
        write(x.collections[0]/'correction_entries.json',{'renamed_capture':entry})
    elif fault=='capture_sha':
        rows=fit.read(x.collections[0]/'results.json');rows[0]['result']['dataset_capture']['sha256']='wrong'
        write(x.collections[0]/'results.json',rows)
    elif fault=='case_family':
        plan=fit.read(x.collections[0]/'plan.json');plan['cases'][0]['spec']['family']='different'
        write(x.collections[0]/'plan.json',plan)
    elif fault=='incomplete':write(x.collections[0]/'result.json',dict(software_complete=False))
    elif fault=='primary_not_latest':latest,extra=x.collections[0],[x.collections[1],x.collections[2]]
    elif fault=='missing_current':data['entries']['old1']['source_sha256']='absent'
    with pytest.raises(ValueError,match=match):fit.bind_collection_union(data,latest,extra)


def test_additional_visual_and_selected_exposure_are_verified(union_artifacts,tmp_path):
    x=union_artifacts
    (x.root/'B/state_data/perception/old1.pt').write_bytes(b'changed frozen shard')
    with pytest.raises(ValueError,match='SHA256 differs'):prepare(x,tmp_path/'bad_visual')


def test_source_exposure_never_uses_selected_id_spelling_or_unproven_epochs(union_artifacts):
    x=union_artifacts;path=x.source/'training/best.pt';saved=torch.load(path,weights_only=False)
    rows,_=fit.source_training_exposure(dict(path=str(path)),saved,x.data,['old1','mid1'])
    assert rows['old1']['source_training_id']=='source_alias' and rows['old1']['full_history_visits']==2
    saved['updates']=3
    with pytest.raises(ValueError,match='Cannot infer'):fit.source_training_exposure(dict(path=str(path)),saved,x.data,['old1'])
    saved['sampling_exposure']={'source_alias':3}
    rows,_=fit.source_training_exposure(dict(path=str(path)),saved,x.data,['old1'])
    assert rows['old1']['full_history_visits']==3


def test_strata_weight_actual_slots_and_keep_empty_group_unknown():
    def rows(count,value):return {key:dict(evaluated_commit_slots=count,arm_squared_sum=count*7*value**2,
        hand_squared_sum=count*22*value**2) for key in (*fit.STAGES,'legal_expert')}
    episodes=dict(short=rows(1,1),long=rows(9,3));baseline=dict(hold_anchor_q=episodes)
    grouped=fit.stratified_results(episodes,baseline,dict(all=['short','long'],empty=[]))
    assert grouped['all']['stages']['holding']['arm_rmse_rad']==pytest.approx(np.sqrt(8.2))
    assert grouped['empty']['stages']['holding']['arm_rmse_rad'] is None
    assert grouped['empty']['stages']['holding']['evaluated_commit_slots']==0


def test_union_arms_recompute_full_history_once_each_and_compare_identical_hold_slots(union_artifacts,tmp_path,monkeypatch):
    from visuotactile.anchor_sampling import planning_anchors,labels
    x=union_artifacts;out=tmp_path/'prepared';plan=prepare(x,out);calls=[]
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(0.))
            self.contract=dict(supervision='u_expert',visual_stride=1)
    def visual_records(path,ids,modality,device):
        assert Path(path)==Path(plan['train_projection']['root']) and ids==['old1','mid1','new1']
        actual=torch.load(Path(path)/'numeric.pt',weights_only=False)['records']
        return {i:dict(actual[i],clean_sample=actual[i]['u_expert']) for i in ids}
    def encode(model,record):
        assert len(record['time'])==80 and not record['supervision_mask'][:42].any()
        calls.append((float(model.scale),len(record['time'])))
        anchors=planning_anchors(record)
        # Entire actual prefix contributes afresh for each model/episode.
        return record['robot'][:,0].sum().expand(len(anchors),1)*model.scale,anchors
    trainer=SimpleNamespace(visual_records=visual_records,new_model=lambda *a:Model(),supervised_conditions=encode,
        predict_on_native_clock=lambda m,c,a:c[:,:,None].expand(-1,24,29),bounded_targets=lambda p,l:p,labels=labels)
    monkeypatch.setattr(fit,'import_frozen_trainer',lambda receipt:trainer)
    monkeypatch.setattr('visuotactile.precision.state_fp32',lambda:None)
    monkeypatch.setattr('visuotactile.state_resources.lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    write(x.root/'session.json',dict(deadline=1))
    source=fit.run_arm(out/'registration.json','source');selected=fit.run_arm(out/'registration.json','selected')
    assert calls==[(1.,80)]*3+[(2.,80)]*3
    assert source['full_history_recomputed_each_episode'] and not source['learned_hidden_state_cached']
    result=fit.compare(out/'registration.json')
    assert result['strata']['all']['stages']['legal_expert']['evaluated_commit_slots']==114
    assert result['strata']['latest']['stages']['legal_expert']['evaluated_commit_slots']==38
    for name in fit.baseline_names('B'):
        assert source['baselines'][name]==selected['baselines'][name]
    # A result cannot relabel a seen episode as unseen while retaining the same total.
    (out/'comparison.json').unlink()
    wrong=fit.read(out/'selected/result.json');wrong['strata']['source_unseen']['identities']=['old1','new1']
    write(out/'selected/result.json',wrong)
    with pytest.raises(ValueError,match='bound episode totals'):fit.compare(out/'registration.json')
    assert not torch.cuda.is_initialized()


def test_cli_keeps_legacy_none_and_registers_explicit_repeated_collections(monkeypatch,tmp_path):
    import sys
    calls=[]
    monkeypatch.setattr(fit,'prepare',lambda **kwargs:(calls.append(kwargs) or {}))
    args=['stage_fit','prepare','--root','r','--role','B','--collection','c8','--selected-frozen','selected',
        '--source-frozen','source','--output','out']
    monkeypatch.setattr(sys,'argv',args);fit.main();assert calls[-1]['include_collection'] is None
    monkeypatch.setattr(sys,'argv',args+['--include-collection','c6','--include-collection','c7'])
    fit.main();assert calls[-1]['include_collection']==[Path('c6'),Path('c7')]
