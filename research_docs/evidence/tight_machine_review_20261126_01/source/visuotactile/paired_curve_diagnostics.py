"""Opt-in descriptive family/recorded-phase curves; no extra model forwards.

All tensors consumed here are detached copies of existing predictions. Geometry
labels stay offline and never become model inputs, loss weights or selection.
"""
import csv
import json
import math
from pathlib import Path

VERSION = 'paired_family_phase_diagnostics_v1'


def profile():
    return dict(version=VERSION, phase_labels='recorded_geometry_sampling_phases_v1',
        training='existing_forward_detached_legal_error_vectors', validation='existing_evaluation_detached_legal_error_vectors',
        extra_model_forwards=False, affects_sampling=False, affects_selection=False)


def validate(value):
    if value is None:return None
    if value != profile():raise ValueError('Exact opt-in paired family/phase diagnostics profile required')
    return dict(value)


def load(root):
    path=Path(root)/'family_phase_diagnostics.json'
    return validate(json.loads(path.read_text())) if path.exists() else None


def read_labels(data):
    """Only train/validation numeric diagnostics; never RGB, test or actor input."""
    import h5py
    import torch
    from .geometry_sampling import diagnostic_phases
    result={}
    for identity in data['splits']['train']+data['splits']['validation']:
        entry=data['entries'][identity]
        with h5py.File(entry['path']) as stream:
            group=stream['episodes'][entry['episode']];tr=group['transition']
            before,after=tr['diagnostics_before'],tr['diagnostics_after']
            labels=diagnostic_phases(after['object_pose_w'][:],after['palm_body_pose_w'][:],tr['object_contact_force_w'][:],
                before['goal_pose_w'][:],after['goal_pose_w'][:],before['successes'][:],after['successes'][:],
                tr['native_keypoints_max_dist'][:])
        if labels.shape!=(entry['length'],) or labels.dtype!=torch.long:
            raise ValueError('Offline phase labels lost actual episode alignment')
        result[identity]=labels
    return result


def _write_csv(path, rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    temporary=path.with_suffix('.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    temporary.replace(path)


class Recorder:
    def __init__(self, directory, data, *, component, role, action_scale, labels=None):
        import torch
        from .geometry_sampling import STAGES
        from .training_resources import atomic_json
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.data=data;self.component=component;self.role=role;self.scale=action_scale.detach().cpu().double()
        self.labels=read_labels(data) if labels is None else labels
        self.stages=STAGES;self.pending={};self.epoch=0
        if self.scale.shape!=(29,) or any(i not in self.labels for i in data['splits']['train']+data['splits']['validation']):
            raise ValueError('Canonical29D affine and complete train/validation labels required')
        self.target=('q_reached' if role=='A' else 'u_expert') if component=='DP' else 'u_expert'
        self.log=self.directory/'family_phase_events.jsonl'
        atomic_json(self.directory/'family_phase_definition.json',dict(profile=profile(),component=component,role=role,
            target=self.target,stages=list(STAGES),phase_source='visuotactile.geometry_sampling.diagnostic_phases',
            labels_are_offline_only=True,test_trajectories_opened=0,
            limitations=['Recorded diagnostic phase is descriptive; held-near60mm is not the15mm gate or native success.',
                'DP training uses sample-prediction diffusion errors over H24 supervised windows; repeated windows count repeatedly.',
                'DP validation uses existing DDIM16 native commit12 predictions. These differ from training noise errors.',
                'Rear training uses actual composed final-command errors before its successful optimizer step; validation uses epoch-final weights.',
                'Squared errors are diagnostics; geometry/FY/regularization components are not decomposed by phase.',
                'A and B DP target spaces differ; these nominal error curves do not rank groups.',
                'Train-evaluation is the trainer existing fixed first2 episodes, not the whole training pool.']))

    def _items(self, identity, ticks, errors):
        import torch
        ticks=ticks.detach().cpu().long().reshape(-1);errors=errors.detach().cpu().double().reshape(-1,29)
        if len(ticks)!=len(errors) or (ticks<0).any() or (ticks>=len(self.labels[identity])).any() or not torch.isfinite(errors).all():
            raise ValueError('Diagnostics require aligned real legal error vectors')
        phases=self.labels[identity][ticks];z=errors*self.scale
        result=[]
        for phase in phases.unique().tolist():
            selected=phases==phase;e=errors[selected];n=z[selected]
            result.append(dict(identity=identity,family=self.data['entries'][identity]['family'],phase=self.stages[phase],
                samples=len(e),arm_squared_sum=float(e[:,:7].square().sum()),hand_squared_sum=float(e[:,7:].square().sum()),
                normalized_arm_squared_sum=float(n[:,:7].square().sum()),normalized_hand_squared_sum=float(n[:,7:].square().sum())))
        return result

    def observe(self, identity, ticks, errors):
        """Stage CPU summaries only. Commit only after the real update succeeds."""
        for row in self._items(identity,ticks,errors):
            key=(row['identity'],row['phase'])
            if key not in self.pending:self.pending[key]=row
            else:
                for name in ('samples','arm_squared_sum','hand_squared_sum','normalized_arm_squared_sum','normalized_hand_squared_sum'):
                    self.pending[key][name]+=row[name]

    def _append(self, epoch, update, split, items):
        with self.log.open('a') as stream:
            stream.write(json.dumps(dict(epoch=int(epoch),update=int(update),split=split,component=self.component,role=self.role,target=self.target,items=items))+'\n')

    def commit(self, epoch, update):
        if self.pending:self._append(epoch,update,'train_updates',list(self.pending.values()))
        self.pending={}

    def evaluation(self, epoch, update, split, identity, ticks, errors):
        if split not in ('validation','train_evaluation','correction_evaluation'):raise ValueError('Explicit existing evaluation population required')
        # One row per actual evaluated episode; identity is part of replacement key.
        self._append(epoch,update,split+':'+identity,self._items(identity,ticks,errors))

    def residual_chunk(self, identities, start, action, target, legal):
        import torch
        for lane,identity in enumerate(identities):
            mask=legal[lane].detach();ticks=torch.arange(start,start+len(mask),device=mask.device)[mask]
            self.observe(identity,ticks,(action[lane]-target[lane])[mask])

    def residual_evaluation(self, epoch, update, split, ids, records, predictions):
        import torch
        for identity,prediction in zip(ids,predictions):
            record=records[identity];mask=record.get('supervision_mask',torch.ones(len(prediction),dtype=torch.bool)).cpu()
            self.evaluation(epoch,update,split,identity,mask.nonzero().flatten(),(prediction-record['target'].cpu())[mask])

    def dp_training(self, epoch, update, identities, values, prediction, model):
        import torch
        offset=0
        for identity,value in zip(identities,values):
            end=offset+len(value['anchors']);mask=value['valid']
            ticks=value['anchors'][:,None]+torch.arange(model.horizon,device=mask.device)[None]
            physical=(prediction[offset:end].detach()-model.action_offset.detach())/model.action_scale.detach()
            self.observe(identity,ticks[mask],(physical-value['truth'])[mask]);offset=end
        self.commit(epoch,update)

    def dp_evaluation(self, epoch, update, split, identity, anchors, errors, mask):
        import torch
        ticks=anchors[:,None]+torch.arange(mask.shape[1],device=mask.device)[None]
        self.evaluation(epoch,update,split,identity,ticks[mask],errors[mask])

    def export(self, maximum_update, *, plot=False):
        """Keep only committed updates, last event wins after a durable resume."""
        events={}
        if self.log.exists():
            for line in self.log.read_text().splitlines():
                row=json.loads(line)
                if row['update']<=maximum_update:events[(row['epoch'],row['update'],row['split'])]=row
        groups={}
        for event in events.values():
            split=event['split'].split(':',1)[0]
            for item in event['items']:
                for family,phase in ((item['family'],item['phase']),('__all__',item['phase']), (item['family'],'__all__'),('__all__','__all__')):
                    key=(event['epoch'],split,family,phase)
                    if key not in groups:groups[key]=dict(samples=0,arm_squared_sum=0.,hand_squared_sum=0.,normalized_arm_squared_sum=0.,normalized_hand_squared_sum=0.)
                    for name in groups[key]:groups[key][name]+=item[name]
        rows=[]
        for (epoch,split,family,phase),values in sorted(groups.items()):
            n=values['samples']
            rows.append(dict(component=self.component,role=self.role,target=self.target,epoch=epoch,split=split,family=family,phase=phase,samples=n,
                arm_rmse_rad=math.sqrt(values['arm_squared_sum']/(7*n)),hand_rmse_rad=math.sqrt(values['hand_squared_sum']/(22*n)),
                normalized_equal_mse=.5*(values['normalized_arm_squared_sum']/(7*n)+values['normalized_hand_squared_sum']/(22*n))))
        _write_csv(self.directory/'family_phase_curves.csv',rows)
        if plot and rows:self.plot(rows)
        return rows

    def plot(self, rows):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(11,3.5))
        for split in ('train_updates','train_evaluation','correction_evaluation','validation'):
            selected=[r for r in rows if r['split']==split and r['family']=='__all__' and r['phase']=='__all__']
            if selected:axes[0].plot([r['epoch'] for r in selected],[r['normalized_equal_mse'] for r in selected],label=split)
        for phase in self.stages:
            selected=[r for r in rows if r['split']=='validation' and r['family']=='__all__' and r['phase']==phase]
            if selected:axes[1].plot([r['epoch'] for r in selected],[r['normalized_equal_mse'] for r in selected],label=phase)
        for axis in axes:
            axis.set_xlabel('epoch');axis.set_ylabel('diagnostic normalized equal MSE');axis.legend(fontsize=6)
        fig.suptitle(f'{self.role} {self.component}: {self.target} (offline labels; own target only)');fig.tight_layout()
        fig.savefig(self.directory/'family_phase_curves.png',dpi=160);plt.close(fig)


def export_dp_visit_losses(directory, data):
    """Existing actual visit losses, family curves; no invented phase allocation."""
    from collections import defaultdict
    directory=Path(directory);groups=defaultdict(list)
    path=directory/'learning.jsonl'
    if not path.exists():return []
    for line in path.read_text().splitlines():
        batch=json.loads(line)
        for visit in batch.get('visits',[batch]):
            identity=visit['episode'];groups[(batch['epoch'],data['entries'][identity]['family'])].append(visit['loss'])
    rows=[dict(epoch=e,family=f,actual_visits=len(losses),mean_actual_objective_loss=sum(losses)/len(losses),
        phase='whole_episode_mixed_phases',scope='actual per-visit total training objective; no phase decomposition') for (e,f),losses in sorted(groups.items())]
    _write_csv(directory/'family_objective_loss.csv',rows)
    return rows
