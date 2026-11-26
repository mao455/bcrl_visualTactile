import copy
from collections import Counter

import pytest
import torch

from visuotactile.latest_takeover_sampling import sampling_plan
from visuotactile.paired_training_decks import (POOL_VERSION, build_pair, completed_counts, digest,
    group_updates, group_windows, validate_deck, validate_pool_pair, validate_progress)
from visuotactile.residual_tick_window import ResidualTickWindow, TICK_WINDOW_CONTRACT
from visuotactile.effective_exposure import Ledger


def fixtures():
    original = [f'full{i}' for i in range(30)]+['old0', 'old1']
    shared = ['newfull0', 'newfull1']
    data, records, specs = {}, {}, {}
    for role in 'AB':
        own = [role+'corr'+str(j) for j in range(1 if role=='A' else 2)]
        train = original+shared+own
        entries = {i:dict(path='/actual/'+i,episode=i,source_sha256=digest(i),trajectory_kind='expert_demonstration_v1') for i in train}
        for i in ['old0', 'old1']+own:
            entries[i].update(trajectory_kind='expert_takeover_v1',source_group=role)
        data[role] = dict(splits=dict(train=train,validation=['v'+str(j) for j in range(6)],test=[]),
            entries=entries,normalization_train_ids=original[:30],normalization_identity=dict(teacher='role-specific'),
            aggregation=dict(version='expert_takeover_aggregate_v1',student_prefix_supervised=False,new_training_ids=own),
            paired_pool=dict(version=POOL_VERSION,role=role,original_ids=original,shared_full_ids=shared,own_correction_ids=own))
        records[role] = {}
        for j,i in enumerate(train):
            length = 81+j%4*32
            mask = torch.arange(length)%3!=0
            if i in own:
                mask.zero_();mask[-7:]=True
            records[role][i] = dict(robot=torch.zeros(length,1),supervision_mask=mask)
        specs[role]=dict(version='paired_dagger_replay_v1',extra_replays=dict.fromkeys(train,0))
    return data,records,specs


@pytest.fixture(scope='module')
def prepared():
    data,records,specs=fixtures()
    decks=build_pair(data,records,dict(A='dataA',B='dataB'),sampling_by_role=specs)
    return data,records,specs,decks


def test_paired_sampling_actual_partitions_weights_and_empty_own_allowed():
    data,records,specs=fixtures()
    validated,counts=sampling_plan(data['A'],specs['A'])
    assert counts['full0']==3 and counts['old0']==1 and counts['newfull0']==3 and counts['Acorr0']==1
    assert validated==specs['A'] and validated is not specs['A']
    without=copy.deepcopy(data['A']);own=without['paired_pool']['own_correction_ids'];without['splits']['train']=[i for i in without['splits']['train'] if i not in own]
    without['paired_pool']['own_correction_ids']=[];without['aggregation']['new_training_ids']=[]
    assert sampling_plan(without,dict(version='paired_dagger_replay_v1',extra_replays=dict.fromkeys(without['splits']['train'],0)))[1]['old0']==1
    specs['A']['extra_replays']['old0']=31
    assert sampling_plan(data['A'],specs['A'])[1]['old0']==32
    specs['A']['extra_replays']['old0']=32
    with pytest.raises(ValueError):sampling_plan(data['A'],specs['A'])


@pytest.mark.parametrize('bad',['source_role','heldout','overlap','unpartitioned','student_prefix','full_is_takeover','missing_extra'])
def test_invalid_own_sampling_rejected(bad):
    data,_,specs=fixtures();d=data['A'];s=specs['A']
    if bad=='source_role':d['entries']['Acorr0']['source_group']='B'
    if bad=='heldout':d['splits']['validation'].append('Acorr0')
    if bad=='overlap':d['paired_pool']['shared_full_ids'].append('old0')
    if bad=='unpartitioned':d['paired_pool']['own_correction_ids']=[]
    if bad=='student_prefix':d['aggregation']['student_prefix_supervised']=True
    if bad=='full_is_takeover':d['entries']['newfull0']['trajectory_kind']='expert_takeover_v1'
    if bad=='missing_extra':s['extra_replays'].pop('old0')
    with pytest.raises(ValueError):sampling_plan(d,s)


def test_different_own_pools_require_explicit_gate_and_identical_shared_identity():
    data,_,_=fixtures()
    with pytest.raises(ValueError,match='cohort'):validate_pool_pair(data['A'],data['B'],role='A')
    validate_pool_pair(data['A'],data['B'],role='A',enabled=True)
    data['B']['entries']['newfull0']['source_sha256']='changed'
    with pytest.raises(ValueError,match='identities'):validate_pool_pair(data['A'],data['B'],role='A',enabled=True)


@pytest.mark.parametrize('group',[['a'],['a','b'],['a','a','b','c'],['zero','c']])
def test_exact_update_preflight_matches_real_window_and_unique_successful_step_ledger(group):
    masks={'a':torch.arange(161)%3!=0,'b':torch.arange(100)%5==0,'c':torch.ones(67,dtype=torch.bool),'zero':torch.zeros(71,dtype=torch.bool)}
    windows=group_windows(group,masks)
    model=torch.nn.Linear(1,1,bias=False)
    opt=torch.optim.SGD(model.parameters(),lr=.01)
    window=ResidualTickWindow(TICK_WINDOW_CONTRACT)
    # Ledger receives only episodes with an actual legal population.
    ledger=Ledger({i:m for i,m in masks.items() if m.any()})
    contributions=[];updates=0;reports=[]
    for start in range(0,max(len(masks[i]) for i in group),32):
        n=0
        for i in group:
            ticks=(masks[i][start:start+32].nonzero().flatten()+start).tolist()
            n+=len(ticks)
            if ticks:contributions.append((i,ticks))
        if n:
            loss=(model.weight.square()).sum()
            window.backward(loss,opt,weight_mass=n,supervised_ticks=n,contact_ticks=0,history_ticks=0,weight_report={})
        if window.pending and (window.ready or start+32>=max(len(masks[i]) for i in group)):
            reports.append(window.step(model,opt));updates+=1
            ledger.step(updates,contributions);contributions=[]
    assert updates==group_updates(group,masks)==len(windows)
    assert [r['supervised_ticks'] for r in reports]==[w['supervised_ticks'] for w in windows]
    for i,m in ledger.legal.items():
        expected=torch.zeros(len(m),dtype=torch.int64)
        for w in windows:
            if i in w['contributions']:expected[w['contributions'][i]]+=1
        assert torch.equal(ledger.counts[i],expected)
        assert not ledger.counts[i][~m].any()
        assert (ledger.counts[i]<=updates).all()


def test_fixed60_decks_match_exact_updates_and_primary_plus_real_replays(prepared):
    data,records,specs,decks=prepared
    assert decks['A']['updates_per_epoch']==decks['B']['updates_per_epoch']
    for role,d in decks.items():
        assert validate_deck(d,data[role],records[role],data_sha256='data'+role)==d
        assert len(d['epoch_groups'])==60
        assert all(sum(xs)==d['updates_per_epoch'] for xs in d['epoch_group_updates'])
        actual=Counter(i for gs in d['epoch_groups'] for group in gs for i in group)
        assert actual==Counter({i:60*n+d['extra_replay_visits'].get(i,0) for i,n in d['base_multiplicities'].items()})
        assert all(actual[i]>=60 for i in data[role]['splits']['train'])
        assert any(d['extra_replay_visits'].values())


@pytest.mark.parametrize('bad',['hash','order','mask','source','group','cursor','visit_count','update_count'])
def test_deck_and_restore_tampering_rejected(prepared,bad):
    data,records,_,decks=prepared;d=copy.deepcopy(decks['A']);r=copy.deepcopy(records['A'])
    if bad=='hash':d['updates_per_epoch']+=1
    if bad=='order':d['epoch_groups'][0][0].reverse();d['sha256']=digest({k:v for k,v in d.items() if k!='sha256'})
    if bad=='mask':r['old0']['supervision_mask'][0]=True
    if bad=='source':d['binding']['sources']['old0']='changed';d['sha256']=digest({k:v for k,v in d.items() if k!='sha256'})
    if bad=='group':d['epoch_groups'][0][0]=['v0'];d['sha256']=digest({k:v for k,v in d.items() if k!='sha256'})
    if bad in ('cursor','visit_count','update_count'):
        cursor=2;state=dict(epoch=2,next_episode_group=cursor,episode_counts=completed_counts(d,2,cursor),
            episode_order=sum(d['epoch_groups'][1],[]),updates=d['updates_per_epoch']+sum(d['epoch_group_updates'][1][:cursor]))
        validate_progress(d,state)
        if bad=='cursor':state['next_episode_group']+=1
        if bad=='visit_count':state['episode_counts']['old0']+=1
        if bad=='update_count':state['updates']+=1
        with pytest.raises(ValueError):validate_progress(d,state)
    else:
        with pytest.raises(ValueError):validate_deck(d,data['A'],r,data_sha256='dataA')


def test_actual_residual_restore_uses_group_cursor_and_refuses_changed_order(prepared):
    from visuotactile.anchor_continue import restore_residual_training
    data,_,_,decks=prepared;deck=decks['A'];ids=data['A']['splits']['train']
    model=torch.nn.Linear(1,1)
    model.loss_contract=model.history_contract=model.output_contract=None
    optimizer=torch.optim.AdamW(model.parameters())
    generator=torch.Generator().manual_seed(11)
    config=dict(paired_group_deck=deck,episode_sampling=deck['sampling'])
    epoch,cursor=2,3;counts=completed_counts(deck,epoch,cursor)
    saved=dict(model=copy.deepcopy(model.state_dict()),optimizer=optimizer.state_dict(),schedule_rng=generator.get_state(),
        torch_rng=torch.get_rng_state(),cuda_rng=[],updates=deck['updates_per_epoch']+sum(deck['epoch_group_updates'][1][:cursor]),
        epoch=epoch,next_episode_group=cursor,episode_order=sum(deck['epoch_groups'][1],[]),episode_counts=counts,
        supervised_ticks=sum(counts[i]*deck['binding']['masks'][i]['legal_ticks'] for i in ids),
        control_ticks=sum(counts[i]*deck['binding']['masks'][i]['length'] for i in ids),contact_anchors=0,best=1.,history=[],
        configuration=config,train_ids=ids)
    with torch.no_grad():model.weight.add_(1)
    report=restore_residual_training(saved,model,optimizer,generator,config,ids,episode_multiplicities=deck['base_multiplicities'])
    assert report['passed'] and report['next_episode_group']==cursor
    assert torch.equal(model.weight,saved['model']['weight'])
    saved['episode_order']=list(reversed(saved['episode_order']))
    with pytest.raises(ValueError,match='group/order'):restore_residual_training(saved,model,optimizer,generator,config,ids)


def test_distinct_successful_step_lower_bound_for_all_legal_ticks(prepared):
    data,records,_,decks=prepared
    for role,deck in decks.items():
        # Every epoch contains a full visit to every own episode. Deduplication
        # within one optimizer step cannot reduce the per-tick lower bound below60.
        masks={i:r['supervision_mask'] for i,r in records[role].items()}
        minimum=999999
        for gs in deck['epoch_groups'][:1]:
            ledger=Ledger(masks);step=0
            for g in gs:
                for w in group_windows(g,masks):
                    step+=1;ledger.step(step,list(w['contributions'].items()))
            assert step==deck['updates_per_epoch']
            minimum=min(int(ledger.counts[i][m].min()) for i,m in masks.items())
        assert minimum>=1 and 60*minimum>=50


def test_explicit_finite_deck_preserves_default60_contract():
    data,records,specs=fixtures()
    with pytest.raises(ValueError):
        build_pair(data,records,dict(A='dataA',B='dataB'),sampling_by_role=specs,epochs=1)
    decks=build_pair(data,records,dict(A='dataA',B='dataB'),sampling_by_role=specs,epochs=1,finite=True)
    for role,deck in decks.items():
        assert deck['epochs']==1 and len(deck['epoch_groups'])==1
        assert validate_deck(deck,data[role],records[role],data_sha256='data'+role)==deck
        assert sum(deck['epoch_group_updates'][0])==deck['updates_per_epoch']
        assert set(completed_counts(deck,1,len(deck['epoch_groups'][0])))==set(data[role]['splits']['train'])
        legacy=copy.deepcopy(deck);legacy['version']='paired_complete_group_decks_v1'
        legacy['sha256']=digest({k:v for k,v in legacy.items() if k!='sha256'})
        with pytest.raises(ValueError):validate_deck(legacy,data[role])
