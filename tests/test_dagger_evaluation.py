"""CPU bounded evaluation ownership, restart evidence, and cache recovery."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tools import tight_finish as owner


RULES=dict(initial_trials=4,max_trials=10,expand_if_success=1,expand_if_subgoal=1,expand_if_native_lift=2)


def evaluation(tmp_path,groups=('G11','QG01')):
    plan=[dict(identity=f'case{i}',spec=dict(identity=f'case{i}',seed=i,cohort='development',split='development')) for i in range(10)]
    owner.write(tmp_path/'evaluation_plan.json',plan)
    for g in groups:
        p=tmp_path/g/'training/best.pt';p.parent.mkdir(parents=True);p.write_bytes(g.encode())
    return list(groups),plan


def callback(root,calls,progress=None,fail=None):
    def run(group,case,out,video):
        calls.append((group,case['identity'],video));out.mkdir(parents=True)
        checkpoint=root/group/'training/best.pt'
        owner.write(out/'started.json',dict(spec=case['spec'],checkpoint=str(checkpoint)))
        if fail==(group,case['identity']):raise RuntimeError('simulated interrupted worker')
        row=dict(**case['spec'],software_complete=True,legal=True,success=False,native_subgoals=0,native_lift=False,
            native_alignment=False,steps=3,checkpoint_sha256=owner.sha(checkpoint))
        row.update((progress or {}).get((group,case['identity']),{}))
        owner.write(out/'result.json',row);(out/'trace.npz').write_bytes(b'closed numeric evidence')
        if video:(out/'rollout.mp4').write_bytes(b'video verified by injected CPU seam')
        return row
    return run


def verify(path,expected_frames):return dict(decoded_frames=expected_frames)


@pytest.mark.parametrize('metric,value,second',[('success',True,False),('native_subgoals',1,False),('native_lift',True,True)])
def test_each_group_expands_only_on_its_registered_native_threshold_and_resumes_without_trials(tmp_path,metric,value,second):
    groups,plan=evaluation(tmp_path);calls=[];progress={('QG01','case0'):{metric:value}}
    if second:progress[('QG01','case1')]={metric:value}
    report=owner.dagger_evaluation(tmp_path,groups,plan,RULES,callback(tmp_path,calls,progress),verify)
    assert report['complete'] and report['completed']==14 and report['unrun']==6
    assert calls[:8]==[(g,f'case{i}',i==0) for i in range(4) for g in groups]
    assert sum(g=='G11' for g,_,_ in calls)==4 and sum(g=='QG01' for g,_,_ in calls)==10
    assert sum(video for _,_,video in calls)==2
    before=len(calls);again=owner.dagger_evaluation(tmp_path,groups,plan,RULES,callback(tmp_path,calls),verify)
    assert again['complete'] and len(calls)==before
    g=owner.read(tmp_path/'G11/evaluation_summary.json')
    assert g['planned']==10 and g['completed']==4 and g['unrun_identities']==[f'case{i}' for i in range(4,10)]
    assert not g['formal_test22_used'] and g['expert_takeover'] is False


def test_one_lift_does_not_expand_and_rejections_keep_denominator(tmp_path):
    groups,plan=evaluation(tmp_path,('G11',));calls=[]
    progress={('G11','case0'):dict(legal=False),('G11','case1'):dict(native_lift=True)}
    owner.dagger_evaluation(tmp_path,groups,plan,RULES,callback(tmp_path,calls,progress),verify)
    summary=owner.read(tmp_path/'G11/evaluation_summary.json')
    assert summary['attempts']==summary['completed']==4 and summary['valid']==3 and summary['rejected']==1
    assert summary['native_lifts']==1 and summary['unrun']==6 and not summary['expansion']['expanded']


@pytest.mark.parametrize('existing',['started_only','marker_only','changed_trace','new_failure'])
def test_incomplete_attempt_is_preserved_and_never_repeated_while_other_group_continues(tmp_path,existing):
    groups,plan=evaluation(tmp_path);calls=[];run=callback(tmp_path,calls,fail=('G11','case1') if existing=='new_failure' else None)
    if existing=='started_only':
        out=tmp_path/'G11/evaluation/case1';owner.write(out/'started.json',dict(spec=plan[1]['spec'],checkpoint=str(tmp_path/'G11/training/best.pt')))
    elif existing=='marker_only':owner.write(tmp_path/'G11/evaluation_started/case1.json',{})
    elif existing=='changed_trace':
        owner.dagger_evaluation(tmp_path,groups,plan,RULES,run,verify);calls.clear()
        (tmp_path/'G11/evaluation/case1/trace.npz').write_bytes(b'changed')
    report=owner.dagger_evaluation(tmp_path,groups,plan,RULES,run,verify)
    assert not report['complete'] and report['software_failures']==1
    assert owner.read(tmp_path/'QG01/evaluation_summary.json')['complete']
    failure=tmp_path/'G11/evaluation_failures/case1.json';original=failure.read_bytes();count=len(calls)
    again=owner.dagger_evaluation(tmp_path,groups,plan,RULES,run,verify)
    assert not again['complete'] and len(calls)==count and failure.read_bytes()==original
    assert sum(g=='G11' and identity=='case1' for g,identity,_ in calls)==(1 if existing=='new_failure' else 0)


def test_plan_checkpoint_and_test_cohort_guards(tmp_path):
    groups,plan=evaluation(tmp_path);calls=[]
    assert owner.dagger_evaluation_rules(tmp_path,{},dict(cohort='development')) is None
    owner.write(tmp_path/'configuration.json',dict(residual_pair=dict(dagger_evaluation=RULES)))
    assert owner.dagger_evaluation_rules(tmp_path,{},dict(cohort='development'))==RULES
    with pytest.raises(ValueError,match='development-only'):owner.dagger_evaluation_rules(tmp_path,{},dict(cohort='formal_test'))
    owner.dagger_evaluation(tmp_path,groups,plan,RULES,callback(tmp_path,calls),verify)
    (tmp_path/'G11/training/best.pt').write_bytes(b'changed')
    with pytest.raises(ValueError,match='Frozen evaluation'):owner.dagger_evaluation(tmp_path,groups,plan,RULES,callback(tmp_path,calls),verify)
    plan[0]['spec']['split']='test'
    with pytest.raises(ValueError,match='Held-out'):owner.dagger_evaluation(tmp_path,groups,plan,RULES,callback(tmp_path,calls),verify)


def cache_fixture(tmp_path,monkeypatch,corrupt=False):
    from visuotactile import native_data,state_resources
    root=tmp_path;source=root/'source';source.mkdir();state=root/'state_data';state.mkdir();dest=root/'G11/nominal';dest.mkdir(parents=True)
    ids=['core','old','new','val','dev'];splits=dict(train=ids[:3],validation=['val'],correction_dev=['dev'],test=['TEST_NEVER_READ'])
    data=dict(splits=splits,entries={i:{} for i in ids},sampling_strata=dict(core=['core'],historical_correction=['old'],current_correction=['new']))
    owner.write(state/'data.json',data);torch.save(dict(action_scale=torch.ones(29)),state/'statistics.pt')
    base=root/'base.pt';base.write_bytes(b'no GPU load needed');statsha=owner.sha(state/'statistics.pt')
    settings=dict(groups=['G11'],bases={'G11':dict(checkpoint=str(base),sha256=owner.sha(base),group='G10')},dagger_round=2,
        version='target_residual_transformer60_v1',residual_statistics_sha256=statsha,residual_statistics_path=str(state/'statistics.pt'),
        residual_normalization_identity='u_expert_train166_affine_v1',fixed_residual_scales=[.1]*29)
    owner.write(root/'session.json',dict(residual_pair=settings,source_root=str(source),cache_deadline=1e12));owner.write(root/'base_identities.json',dict(protected={}))
    owner.write(source/'cache_index.json',dict(entries={}))
    n=25;records={};actuals={}
    binding=dict(base_checkpoint_sha256=owner.sha(base),statistics_sha256=statsha,data_sha256=owner.sha(state/'data.json'),residual_statistics_sha256=statsha)
    for j,i in enumerate(ids):
        robot=torch.zeros(n,178);robot[:,0]=j;robot[:,116:145]=.234  # Actual issued history distinct from nominal zero.
        actual=dict(robot=robot,tactile={'force_w_N':torch.zeros(n*2,5,3)},tactile_valid=torch.ones(n*2,dtype=torch.bool),gate=torch.zeros(n,dtype=torch.bool))
        record=dict(identity=i,robot=robot,command=torch.ones(n,29)*.4,expert_command=torch.ones(n,29)*.4,time=torch.arange(n)/60,
            joint_limits=torch.tensor([[-2.,2.]]).repeat(29,1),supervision_mask=torch.arange(n)>=12,control_segment=(torch.arange(n)>=12).long())
        cursor=torch.arange(n)%12
        cache=dict(actual,identity=i,cache_identity=binding,target=record['expert_command'],time=record['time'],limits=record['joint_limits'],
            supervision_mask=record['supervision_mask'],control_segment=record['control_segment'],plans=torch.zeros(n,24,29),
            plan_valid=torch.arange(24)[None]+cursor[:,None]<24,base=torch.zeros(n,29),visual=torch.zeros(n,512),plan_cursor=cursor,
            previous_nominal=torch.zeros(n,29),previous_nominal_valid=torch.arange(n)>0,nominal_supervision='u_expert',
            history_source='strictly_past_actual_executed_targets_including_student_prefix; never regenerated nominal labels')
        records[i]=record;actuals[j]=actual
        if corrupt and i=='core':cache['robot']=robot.clone();cache['robot'][:,116:145]=0
        torch.save(cache,dest/(i+'.pt'))
    requested=[]
    monkeypatch.setattr(native_data,'visual_records',lambda path,ids,*a:(requested.append(list(ids)) or {i:records[i] for i in ids}))
    monkeypatch.setattr(native_data,'residual_observations',lambda entry,robot,time:actuals[int(robot[0,0])])
    monkeypatch.setattr(state_resources,'lease_for',lambda *a,**kw:SimpleNamespace(block=nullcontext))
    return records,requested,binding


def test_cache_orphans_reindex_without_gpu_include_correction_dev_and_preserve_actual_issued(tmp_path,monkeypatch):
    from visuotactile.models import native_dp
    records,requested,binding=cache_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(native_dp,'load_policy',lambda *a,**kw:pytest.fail('All verified orphan payloads already exist; no policy load'))
    owner.residual_cache(tmp_path,acceptance=True)
    first=owner.read(tmp_path/'G11/nominal/index.json')['supervision_mask_sha256']
    owner.residual_cache(tmp_path)
    index=owner.read(tmp_path/'G11/nominal/index.json')
    assert set(index['episodes'])=={'core','old','new','val','dev'} and index['supervision_mask_sha256']==first
    assert requested==[['core','old','new','val','dev']]*2
    cached=torch.load(tmp_path/'G11/nominal/core.pt',weights_only=False)
    assert torch.equal(cached['robot'],records['core']['robot']) and cached['robot'][:,116:145].abs().sum()>0
    scales=owner.read(tmp_path/'G11/nominal/scales.json');strata=scales['required_correction_overflow_by_stratum']
    assert set(strata)=={'core','historical_correction','current_correction'}
    assert all(v['supervised_ticks']==13 and v['joint_fraction']==1 for v in strata.values())
    assert scales['control_tick_population']==39  # dev never trains or enters amplitude statistics.


def test_orphan_with_regenerated_issued_history_fails_source_validation(tmp_path,monkeypatch):
    records,_,binding=cache_fixture(tmp_path,monkeypatch,corrupt=True)
    with pytest.raises(ValueError,match='actual feedback'):
        owner.recover_residual_cache(tmp_path/'G11/nominal/core.pt','core',records['core'],{},binding)


def test_unverified_orphan_is_retained_before_bounded_rebuild(tmp_path,monkeypatch):
    from visuotactile import native_data
    from visuotactile.models import native_dp
    records,_,binding=cache_fixture(tmp_path,monkeypatch,corrupt=True)
    path=tmp_path/'G11/nominal/core.pt';original=path.read_bytes();old_sha=owner.sha(path)
    real_load=torch.load;rebuilt=real_load(path,weights_only=False);rebuilt['robot']=records['core']['robot'].clone()
    visual=tmp_path/'state_data/perception/core.pt';visual.parent.mkdir();visual.write_bytes(b'pinned visual payload')
    owner.write(tmp_path/'source/cache_index.json',dict(entries={'core':dict(sha256=owner.sha(visual))}))
    monkeypatch.setattr(torch,'load',lambda path,*a,**kw:{} if str(path)==str(tmp_path/'base.pt') else real_load(path,*a,**kw))
    monkeypatch.setattr(native_dp,'load_policy',lambda *a,**kw:torch.nn.Linear(1,1))
    def rebuild(model,record,entry,lease,**kwargs):
        assert torch.equal(record['robot'],records['core']['robot']) and kwargs['cache_identity']==binding
        return rebuilt
    monkeypatch.setattr(native_data,'cache_residual_episode',rebuild)
    owner.residual_cache(tmp_path,acceptance=True)
    retained=path.with_name('core.orphan_unverified_'+old_sha[:16]+'.pt')
    assert retained.read_bytes()==original
    actual=real_load(path,weights_only=False)
    assert torch.equal(actual['robot'],records['core']['robot'])
    assert owner.read(tmp_path/'G11/nominal/core.cache_recovery.json')['sha256']==old_sha
    assert owner.read(tmp_path/'G11/nominal/index.json')['episodes']['core']['sha256']==owner.sha(path)
