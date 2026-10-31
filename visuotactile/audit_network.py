"""Actual-checkpoint CPU hooks, parameter census and paired force interventions."""
import argparse
import csv
import gc
import json
from pathlib import Path

import h5py
import torch
from torch.nn.utils.rnn import PackedSequence

from .compact import sha256
from .long_data import MANIFEST, inputs_for, load_record, supervision
from .manifest import ManifestSequenceDataset
from .models.long_horizon import FrozenDinoSpatial, LongHorizonConfig, LongHorizonPolicy
from .models.residual import ResidualCorrection
from .models.transformer_residual import TransformerResidual, TransformerResidualConfig, combine, residual_loss
from .residual_data import load_base
from .study_metrics import denoising_objectives, joint_groups
from .train_residual import chunk_record
from .train_transformer_residual import ResidualMetrics
from .training_resources import atomic_json
from .transformer_data import load_residual_record, residual_inputs, residual_labels


def describe(value):
    if isinstance(value,PackedSequence):
        return dict(packed_data=describe(value.data),batch_sizes=value.batch_sizes.tolist())
    if isinstance(value,torch.Tensor):
        return dict(shape=list(value.shape),dtype=str(value.dtype),requires_grad=value.requires_grad)
    if isinstance(value,(tuple,list)):
        return [describe(item) for item in value]
    if isinstance(value,dict):
        return {key:describe(item) for key,item in value.items()}
    return str(value)


class Hooks:
    def __init__(self, model, label, checkpoint):
        self.model,self.label,self.checkpoint=model,label,checkpoint
        self.calls,self.handles={},[]
        for name,module in model.named_modules():
            if not list(module.children()) or isinstance(module,torch.nn.MultiheadAttention):
                def capture(layer, inputs, output, name=name):
                    row=self.calls.setdefault(name,dict(calls=0,inputs=[],outputs=[]))
                    row['calls']+=1
                    for field,value in (('inputs',inputs),('outputs',output)):
                        description=describe(value)
                        if description not in row[field]:
                            row[field].append(description)
                self.handles.append(module.register_forward_hook(capture))

    def finish(self):
        for handle in self.handles:
            handle.remove()
        rows=[]
        for name,module in self.model.named_modules():
            if list(module.children()) and not isinstance(module,torch.nn.MultiheadAttention):
                continue
            parameters=list(module.parameters(recurse=False))
            captured=self.calls.get(name,dict(calls=0,inputs=[],outputs=[]))
            rows.append(dict(model=self.label,checkpoint_sha256=self.checkpoint,module=name,
                kind=type(module).__name__,configuration=module.extra_repr(),
                direct_parameters=sum(parameter.numel() for parameter in parameters),
                trainable=sum(parameter.numel() for parameter in parameters if parameter.requires_grad),
                gradient_l2=sum(float(parameter.grad.double().square().sum()) for parameter in parameters if parameter.grad is not None)**.5,
                gradient_present=sum(parameter.grad is not None for parameter in parameters),
                calls=captured['calls'],input=json.dumps(captured['inputs']),output=json.dumps(captured['outputs']),
                scope='CPU FP32 diagnostic; training uses original precision; no optimizer update',
                uncalled_reason='' if captured['calls'] else 'functional weight access or inactive path; not an observed module call'))
        return rows


def parameter_rows(model,label,initial=None,optimizer=None):
    states={} if optimizer is None else optimizer['state']
    indices=[] if optimizer is None else [index for group in optimizer['param_groups'] for index in group['params']]
    eligible=[(name,value) for name,value in model.named_parameters() if value.requires_grad]
    if optimizer is not None and len(indices)!=len(eligible):
        raise ValueError('Optimizer tensor count differs from trainable parameter order')
    mapped=dict(zip((name for name,_ in eligible),indices))
    rows=[]
    for name,parameter in model.named_parameters():
        state=states.get(mapped.get(name),{})
        if 'exp_avg' in state and state['exp_avg'].shape!=parameter.shape:
            raise ValueError('Optimizer moment shape differs from parameter')
        delta=None if initial is None else parameter.detach().cpu()-initial[name]
        rows.append(dict(model=label,parameter=name,shape=str(list(parameter.shape)),count=parameter.numel(),
            requires_grad=parameter.requires_grad,in_optimizer=name in mapped,
            optimizer_step=int(state['step']) if 'step' in state else None,
            optimizer_exp_avg_l2=float(state['exp_avg'].double().norm()) if 'exp_avg' in state else None,
            gradient_l2=float(parameter.grad.double().norm()) if parameter.grad is not None else None,
            delta_l2=float(delta.double().norm()) if delta is not None else None,
            delta_max=float(delta.abs().max()) if delta is not None else None))
    return rows


def write_csv(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def intervene(inputs, mode):
    changed={key:value.clone() for key,value in inputs.items()}
    if mode=='zero':
        for key in ('tactile','recent_tactile'):
            changed[key][...,:6]=0
    elif mode=='reverse_past':
        changed['tactile'][...,:6]=inputs['tactile'].flip(1)[...,:6]
        recent=changed['recent_tactile']
        for anchor in range(len(recent)):
            for finger in range(5):
                valid=recent[anchor,:,finger,-1].bool()
                recent[anchor,valid,finger,:6]=inputs['recent_tactile'][anchor,valid,finger,:6].flip(0)
    elif mode!='real':
        raise ValueError(mode)
    return changed


def residual_audit(source,output,label,identities,layers,parameters):
    directory=source/'transformer_training'/label
    saved=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    latest=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(latest['configuration']['seed'])
        model=TransformerResidual(TransformerResidualConfig(**saved['config'])).eval()
    initial={name:value.clone() for name,value in model.state_dict().items()}
    model.load_state_dict(saved['model'])
    hook=Hooks(model,label,sha256(directory/'best.pt'))
    record=load_residual_record(source,identities[0])
    inputs=residual_inputs(record,0,min(96,len(record['time'])))
    labels=residual_labels(record,torch.arange(0,min(96,len(record['time'])),6))
    prediction,_=model(inputs)
    loss=residual_loss(prediction,labels['delta'],labels['valid']&labels['gate'])['loss']
    loss.backward()
    layers.extend(hook.finish())
    rows=parameter_rows(model,label,initial,latest['optimizer'])
    assert all(torch.equal(saved['model'][name],latest['model'][name]) for name in saved['model'])
    assert sum(row['count'] for row in rows if row['in_optimizer'])==sum(parameter.numel() for parameter in model.parameters())
    parameters.extend(rows)
    groups=joint_groups(json.loads((source/'metadata.json').read_text())['joint_order'])
    metrics={mode:ResidualMetrics(groups) for mode in ('real','zero','reverse_past')}
    split_metrics={split:{mode:ResidualMetrics(groups) for mode in metrics} for split in ('train_selected','validation_selected')}
    current_metrics={mode:ResidualMetrics(groups) for mode in metrics}
    changes={mode:dict(square=torch.zeros(29,dtype=torch.float64),count=0,maximum=0.) for mode in ('zero','reverse_past')}
    current_changes={mode:dict(square=torch.zeros(29,dtype=torch.float64),count=0) for mode in changes}
    trajectory=[]
    with torch.no_grad():
        for ordinal,identity in enumerate(identities):
            record=load_residual_record(source,identity)
            state=None
            for start in range(0,len(record['time']),96):
                stop=min(start+96,len(record['time']))
                inputs=residual_inputs(record,start,stop)
                anchors=torch.arange(start,stop,6)
                labels=residual_labels(record,anchors)
                predictions={}
                for mode in metrics:
                    predictions[mode],following=model(intervene(inputs,mode),state)
                    metrics[mode].add(predictions[mode],labels)
                    split_metrics['train_selected' if ordinal<4 else 'validation_selected'][mode].add(predictions[mode],labels)
                    current_metrics[mode].add(predictions[mode][:,:1],{key:value[:,:1] for key,value in labels.items()})
                    if mode=='real':
                        real_state=following
                state=real_state
                contact=labels['valid']&labels['gate']
                for mode,totals in changes.items():
                    difference=predictions[mode]-predictions['real']
                    totals['square']+=torch.where(contact[...,None],difference.square(),0.).sum((0,1)).double()
                    totals['count']+=int(contact.sum())
                    totals['maximum']=max(totals['maximum'],float(difference.abs().max()))
                    current_changes[mode]['square']+=torch.where(contact[:,:1,None],difference[:,:1].square(),0.).sum((0,1)).double()
                    current_changes[mode]['count']+=int(contact[:,:1].sum())
                if ordinal==0:
                    corrected=combine(labels['base'],predictions['real'],labels['gate'])
                    for anchor_index,anchor in enumerate(anchors.tolist()):
                        for offset in range(6):
                            step=anchor+offset
                            if step>=len(record['time']):
                                continue
                            row=dict(variant=label,identity=identity,split='train_selected',step=step,time_s=step/60.,
                                gate=bool(record['gate'][step]),checkpoint_sha256=sha256(directory/'best.pt'))
                            for joint in (0,3,7,12,16,20,24):
                                row[f'expert_{joint}']=float(labels['expert'][anchor_index,offset,joint])
                                row[f'base_{joint}']=float(labels['base'][anchor_index,offset,joint])
                                row[f'corrected_{joint}']=float(corrected[anchor_index,offset,joint])
                            trajectory.append(row)
    write_csv(output/(label+'_fixed_trajectory.csv'),trajectory)
    effect={mode:dict(contact_steps=value['count'],maximum_delta_change=value['maximum'],
        delta_change_rmse={part:float((value['square'][indices].mean()/max(1,value['count'])).sqrt()) for part,indices in groups.items()})
        for mode,value in changes.items()}
    updated={prefix:dict(parameter_delta_l2=sum(row['delta_l2']**2 for row in rows if row['parameter'].split('.')[0]==prefix)**.5,
        probe_gradient_l2=sum((row['gradient_l2'] or 0)**2 for row in rows if row['parameter'].split('.')[0]==prefix)**.5)
        for prefix in sorted({row['parameter'].split('.')[0] for row in rows})}
    return dict(checkpoint_sha256=sha256(directory/'best.pt'),updates=latest['updates'],
        next_epoch=latest['next_epoch'],next_episode=latest['next_episode'],epoch=saved['epoch'],
        optimizer_groups=[{key:value for key,value in group.items() if key!='params'}|{'parameter_tensors':len(group['params'])}
                          for group in latest['optimizer']['param_groups']],
        optimizer_step_range=[min(row['optimizer_step'] for row in rows),max(row['optimizer_step'] for row in rows)],
        scheduler='stateless learning_rate(updates,total,50) assigned before each AdamW step; no scheduler state object',
        initialization='reconstructed with recorded seed20261103 and unchanged model source; normalizer buffers excluded',
        parameters=sum(row['count'] for row in rows),module_changes=updated,
        training_history=[dict(epoch=row['epoch'],updates=row['updates'],train=row.get('train')) for row in latest['history']],
        intervention_metrics={mode:value.result() for mode,value in metrics.items()},intervention_effect=effect,
        intervention_split_metrics={split:{mode:value.result() for mode,value in modes.items()} for split,modes in split_metrics.items()},
        current_query_metrics={mode:value.result() for mode,value in current_metrics.items()},
        current_query_effect={mode:dict(contact_anchors=value['count'],
            delta_change_rmse={part:float((value['square'][indices].mean()/max(1,value['count'])).sqrt()) for part,indices in groups.items()})
            for mode,value in current_changes.items()},
        intervention_scope='same weights/base/plan/gate/input boundary state; real state carried every96ticks; current block force and recent window intervened; no future force; CPU FP32 recalculation',
        episodes=identities,probe_loss=float(loss),no_optimizer_updates=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    args.output.mkdir(parents=True,exist_ok=True)
    distribution=json.loads((args.output/'dataset_distribution.json').read_text())
    identities=distribution['selected_train']
    splits=json.loads((args.source/'splits.json').read_text())
    layers,parameters,report=[],[],dict(models={},scope='actual checkpoints and real samples; CPU FP32; no parameter updates')
    manifest=json.loads(MANIFEST.read_text())
    row=next(row for row in manifest['episodes'] if row['id']==identities[0])
    with h5py.File(MANIFEST.parent/row['directory']/'data.h5','r') as stream:
        rgb=torch.from_numpy(stream['episodes'][row['episode']]['frames/rgb_u8'][0])
    dino=FrozenDinoSpatial().eval()
    dino_path=Path(torch.hub.get_dir())/'checkpoints/dinov2_vits14_pretrain.pth'
    hook=Hooks(dino,'FrozenDINO',sha256(dino_path))
    patches=dino(rgb)
    layers.extend(hook.finish());parameters.extend(parameter_rows(dino,'FrozenDINO'))
    report['models']['FrozenDINO']=dict(parameters=sum(parameter.numel() for parameter in dino.parameters()),
        trainable=0,output_shape=list(patches.shape),checkpoint_sha256=sha256(dino_path))
    del dino;gc.collect()
    record=load_record(args.source,identities[0])
    for label in ('G00-B','G10-B'):
        path=args.source/label/'best.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=LongHorizonPolicy(LongHorizonConfig(**saved['config'])).eval()
        model.load_state_dict(saved['model'])
        hook=Hooks(model,label,sha256(path))
        inputs=inputs_for(record,0,13,model.cfg.tactile,torch.tensor([12]))
        condition,state=model.encode(inputs)
        targets=supervision(record,torch.tensor([12]))
        objective=label.rsplit('-',1)[1]
        loss=denoising_objectives(model,condition,targets['reference'],targets['valid'],torch.Generator().manual_seed(20261031))[objective]
        loss.backward()
        plan=model.predict_from_condition(condition.detach(),torch.Generator().manual_seed(20261031))
        last=torch.load(args.source/label/'last.pt',map_location='cpu',weights_only=False)
        layers.extend(hook.finish());parameters.extend(parameter_rows(model,label,optimizer=last['optimizer']))
        report['models'][label]=dict(checkpoint_sha256=sha256(path),config=saved['config'],epoch=saved['epoch'],
            updates=saved['updates'],parameters=sum(parameter.numel() for parameter in model.parameters()),
            condition_shape=list(condition.shape),memory_shape=list(state.shape),plan_shape=list(plan.shape),probe_loss=float(loss),probe_objective=objective)
        report['models'][label]['optimizer_groups']=[{key:value for key,value in group.items() if key!='params'}|{'parameter_tensors':len(group['params'])}
                                                   for group in last['optimizer']['param_groups']]
        del model,saved,last,condition,state,plan,loss;gc.collect()
    for label in ('force','masked'):
        report['models'][label]=residual_audit(args.source,args.output,label,identities+splits['validation'][:4],layers,parameters)
        atomic_json(args.output/'network_audit.json',report)
    coverage=dict(episodes=len(splits['train']),native_ticks=0,anchors=0,padded_tokens=0,valid_tokens=0,contact_steps=0,chunks=0)
    for identity in splits['train']:
        cached=load_residual_record(args.source,identity)
        inputs=residual_inputs(cached,0,len(cached['time']))
        anchors=len(inputs['anchors'])
        coverage['native_ticks']+=len(cached['time'])
        coverage['anchors']+=anchors
        coverage['padded_tokens']+=anchors*204
        coverage['valid_tokens']+=anchors*10+int(inputs['plan_valid'].sum())+int(inputs['recent_tactile'][...,-1].sum())
        coverage['contact_steps']+=int(cached['gate'].sum())
        coverage['chunks']+=(len(cached['time'])+95)//96
    report['residual_coverage_per_epoch']=coverage
    report['base_freezing']=dict(checkpoint_sha256=sha256(args.source/'G00-B/best.pt'),
        matches_cache=json.loads((args.source/'transformer_cache/index.json').read_text())['base_sha256']==sha256(args.source/'G00-B/best.pt'),
        optimizer_scope='exact828189 residual parameters only; base never included in residual optimizer',
        offline='base precomputed cache detached; no computational graph to base',deployment='TransformerController freezes base model and residual for inference')
    base=load_base()
    old_root=Path('outputs/visuotactile/run/residual_20261030_01')
    old_path=old_root/'training_full/b_best.pt'
    saved=torch.load(old_path,map_location='cpu',weights_only=False)
    model=ResidualCorrection(base.model.encoder,base.model.cfg,saved['scales']).eval()
    model.load_state_dict(saved['model'])
    old_record=torch.load(old_root/'cache'/(identities[0]+'.pt'),map_location='cpu',weights_only=False)
    hook=Hooks(model,'HistoricalGRU',sha256(old_path))
    dataset=ManifestSequenceDataset(MANIFEST,ids=[identities[0]],stage='production')
    chunk=next(dataset.chunks(identities[0],2))
    with torch.no_grad():
        frame_features,frame_valid=model.visual.encode_frames(chunk['obs'])
    visual_error=float((frame_features[0]-old_record['features'][:2]).abs().max())
    inputs=chunk_record(old_record,0,13)
    prediction,state=model.forward_cached(**inputs)
    prediction.square().mean().backward()
    layers.extend(hook.finish());parameters.extend(parameter_rows(model,'HistoricalGRU',optimizer=saved['optimizer']))
    report['models']['HistoricalGRU']=dict(checkpoint_sha256=sha256(old_path),epoch=saved['epoch'],
        updates=saved['optimizer_updates'],parameters=sum(parameter.numel() for parameter in model.parameters()),
        trainable=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        frozen_frame_recalculation_max_error=visual_error,
        scope='actual RGB-D visual encoder2frames plus cached13tick recurrent forward; not a matched Transformer comparator')
    write_csv(args.output.parent/'network_layers.csv',layers)
    write_csv(args.output/'network_parameters.csv',parameters)
    report['leaf_rows']=len(layers)
    report['uncalled_leaf_modules']=[dict(model=row['model'],module=row['module'],reason=row['uncalled_reason']) for row in layers if row['calls']==0]
    atomic_json(args.output/'network_audit.json',report)
    print(json.dumps(dict(models=list(report['models']),leaf_rows=len(layers),parameter_rows=len(parameters))))


if __name__=='__main__':
    main()
