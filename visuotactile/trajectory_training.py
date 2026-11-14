"""Explicit B2 training semantics; ordinary one-trajectory updates stay separate."""
VERSION = 'full_trajectory_batch_v1'


def profile():
    return dict(version=VERSION, batch_size=2, grouping='adjacent_registered_deck_visits',
        tail_policy='keep_singleton', loss_reduction='mean_per_trajectory_in_actual_batch',
        optimizer_steps='one_per_batch', ema_steps='one_per_optimizer_step',
        learning_rate='unchanged_no_automatic_scaling', geometry='original_selection_per_visit',
        noise='original_randn_then_randint_per_visit', history='independent_complete_real_prefix')


def validate_profile(value):
    if value is None:return None
    if not isinstance(value,dict) or value!=profile():
        raise ValueError('Exact explicit full_trajectory_batch_v1 B2 profile required')
    return dict(value)


def budget(visits_per_epoch, epochs):
    if any(type(n) is not int or n<1 for n in (visits_per_epoch,epochs)):
        raise ValueError('Positive visit and epoch budgets required')
    updates=(visits_per_epoch+1)//2
    return dict(episodes_per_epoch=visits_per_epoch,optimizer_updates_per_epoch=updates,
        planned_episode_visits=visits_per_epoch*epochs,planned_optimizer_updates=updates*epochs,
        planned_ema_updates=updates*epochs,tail_singletons_per_epoch=visits_per_epoch%2)


def batches(order):
    if not order:raise ValueError('Nonempty registered visit deck required')
    visits=list(enumerate(order,start=1))
    return [visits[i:i+2] for i in range(0,len(order),2)]


def validate_exposure(exposure,multiplicities,*,completed_epochs,updates,visits_seen):
    if (not isinstance(multiplicities,dict) or not multiplicities or
            any(not isinstance(i,str) or not i or type(n) is not int or n<1 for i,n in multiplicities.items()) or
            type(completed_epochs) is not int or completed_epochs<0):
        raise ValueError('B2 complete-deck multiplicities and epoch cursor required')
    expected=completed_epochs*sum(multiplicities.values())
    if (type(completed_epochs) is not int or completed_epochs<0 or type(updates) is not int or
            type(visits_seen) is not int or visits_seen!=expected or
            updates!=completed_epochs*((sum(multiplicities.values())+1)//2) or
            not isinstance(exposure,dict) or set(exposure)!=set(multiplicities) or
            any(type(exposure[i]) is not int or exposure[i]!=completed_epochs*n for i,n in multiplicities.items())):
        raise ValueError('B2 optimizer/visit/complete-deck exposure mismatch')


def validate_registration(root,requested):
    """A registered B2 root cannot silently launch the old single-visit method."""
    import json
    from pathlib import Path
    requested=validate_profile(requested)
    path=Path(root)/'full_trajectory_batch.json'
    if path.exists():
        value=json.loads(path.read_text())
        if validate_profile(value['profile']) is None or value['profile']!=requested:
            raise ValueError('Training profile differs from registered B2 method')
    return requested


def validate_deck_receipts(directory,rows):
    """Bind durable visit identity/order to the original immutable epoch deck."""
    import hashlib,json
    from pathlib import Path
    decks={}
    for row in rows:
        epoch=row['epoch']
        if type(epoch) is not int or epoch<1:raise ValueError('Invalid B2 epoch deck cursor')
        if epoch not in decks:
            path=Path(directory)/('deck_%03d.json'%epoch)
            if not path.exists():raise ValueError('Missing durable B2 epoch deck')
            deck=json.loads(path.read_text());order=deck.get('order')
            if (deck.get('epoch')!=epoch or not isinstance(order,list) or not order or
                    any(not isinstance(i,str) or not i for i in order)):
                raise ValueError('Invalid durable B2 epoch deck identity')
            digest=hashlib.sha256(json.dumps(order,separators=(',',':')).encode()).hexdigest()
            if deck.get('sha256')!=digest:raise ValueError('B2 epoch deck content hash differs')
            decks[epoch]=deck
        deck=decks[epoch];order=deck['order'];index=row['batch_index']
        if (type(index) is not int or index<1 or index>(len(order)+1)//2 or
                row['deck_sha256']!=deck['sha256'] or row['episodes_per_epoch']!=len(order)):
            raise ValueError('B2 learning receipt deck binding differs')
        expected=list(enumerate(order,start=1))[(index-1)*2:index*2]
        actual=[(v['episode_index'],v['episode']) for v in row['visits']]
        if actual!=expected:raise ValueError('B2 durable visits differ from original epoch deck order')


def reject_incompatible(root,role,*,pairing=None,io_config=None,auxiliary=None):
    from pathlib import Path
    root=Path(root)
    if pairing is not None or (root/'paired_episode_io.json').exists():
        raise ValueError('B2 cannot use paired_episode_io_v1 visit==update synchronization')
    if io_config is not None or (root/role/'episode_io.json').exists():
        raise ValueError('B2 cannot nest the single-visit EpisodePrefetch allocation scope')
    if auxiliary is not None:raise ValueError('B2 visual auxiliary exposure is not implemented')


def execute_update(model,optimizer,ema,objective,identities,records,episodes,noise,*,check=lambda:None,
        selected_anchors=None,anchor_receipts=None,geometry_generators=None,geometry_sampling_profile=None,diagnostic_gradients=False):
    """Exactly one clipped AdamW/EMA transaction for one/two independent visits."""
    import torch
    from .trajectory_batch import supervised_trajectories,trajectory_objective
    from .anchor_continue import projection_batch
    if len(identities) not in (1,2) or len(records)!=len(identities) or len(episodes)!=len(identities):
        raise ValueError('B2 requires one/two aligned complete visits')
    if (selected_anchors is None)!=(anchor_receipts is None):raise ValueError('Actual sampled starts and receipts must be supplied together')
    check();model.train();optimizer.zero_grad(set_to_none=True)
    if selected_anchors is not None and (anchor_receipts is None or geometry_generators is None or
            len(anchor_receipts)!=len(identities) or len(geometry_generators)!=len(identities)):
        raise ValueError('Explicit per-visit selections, receipts and geometry generators must align')
    values=supervised_trajectories(model,records,episodes,**(dict(selected_anchors=selected_anchors) if selected_anchors is not None else {}))
    if anchor_receipts is not None:
        for identity,value,receipt in zip(identities,values,anchor_receipts):
            if receipt['identity']!=identity or receipt['selected_anchors']!=value['anchors'].detach().cpu().tolist():
                raise ValueError('Sampling receipt differs from actual encoded trajectory times')
    device=next(model.parameters()).device
    contexts=[projection_batch(objective,[(identity,int(t)) for t in value['anchors']],device,
        **(dict(geometry_generator=geometry_generators[index],sampling_profile=geometry_sampling_profile)
        if geometry_sampling_profile is not None and objective.hand_geometry is not None else {}))
        for index,(identity,value) in enumerate(zip(identities,values))]
    loss,reports,_=trajectory_objective(model,objective,values,noise,contexts)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite B2 loss')
    influence=None
    if diagnostic_gradients and objective.hand_geometry is not None:
        hand=torch.stack([r['parts']['hand_geometry'] for r in reports]).mean()*objective.gain*objective.weights['hand_geometry']
        parameters=[p for p in model.parameters() if p.requires_grad]
        hg=torch.autograd.grad(hand,parameters,retain_graph=True,allow_unused=True)
        other=torch.autograd.grad(loss-hand,parameters,retain_graph=True,allow_unused=True)
        hn=sum(float(g.detach().double().square().sum()) for g in hg if g is not None)**.5
        on=sum(float(g.detach().double().square().sum()) for g in other if g is not None)**.5
        dot=sum(float((a.detach().double()*b.detach().double()).sum()) for a,b in zip(hg,other) if a is not None and b is not None)
        influence=dict(weighted_geometry_gradient_norm=hn,other_loss_gradient_norm=on,ratio=hn/max(on,1e-30),
            cosine=dot/max(hn*on,1e-30),geometry_loss=float(hand.detach()),
            scope='Actual full-model gradients before clipping; extra reverse passes only in the unified diagnostic')
        del hg,other
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite B2 gradient')
    check();optimizer.step();ema.step(model)
    visits=[]
    for identity,record,report in zip(identities,records,reports):
        parts={key:float(value.detach()) for key,value in report['parts'].items()}
        visits.append(dict(episode=identity,anchors=report['anchors'],target_ticks=report['target_ticks'],
            visual_frames=len(record['time']),parts=parts,loss=parts['base']+objective.gain*sum(
                objective.weights[key]*parts[key] for key in objective.weights),
            loss_weight=1/len(identities),geometry=report['geometry']))
    if anchor_receipts is not None:
        for row,receipt,context in zip(visits,anchor_receipts,contexts):
            row.update(anchor_sampling=receipt,geometry_sampling=context.get('selection_report') if context else None)
    result=dict(loss=float(loss.detach()),gradient_norm=float(norm.detach()),visits=visits)
    if influence is not None:result['gradient_influence']=influence
    return result
