"""CPU-only real training masks/batches; no simulation, parameter update or test data."""
import argparse
import collections
import json
from pathlib import Path

import h5py
import torch

from visuotactile.anchor_sampling import anchor_batch, labels, planning_anchors
from visuotactile.native_data import visual_records, residual_batch, residual_supervision_mask
from visuotactile.state_history import PersistentImage
from visuotactile.target_windows import RECOVERY_ONSET_WINDOW, with_target_window


def read(path):
    return json.loads(Path(path).read_text())


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def target_hits(record, anchors, counts=None):
    hit=torch.zeros(len(record['time']),dtype=torch.long)
    mask=labels(record,anchors,24)[1]
    for row,anchor in enumerate(anchors.tolist()):
        offsets=mask[row].nonzero().flatten()
        hit[anchor+offsets]+=1 if counts is None else counts[row]
    return hit


def audit(root):
    report=dict(scope='CPU real training records, saved sampler counts and actual anchor_batch/residual_batch; no optimizer or simulator',test22=False,groups={})
    for group in ('QG01','G01'):
        backend=root/'formal'/group/'backend'
        config=read(backend/'configuration.json')['residual_pair']
        data=read(backend/'state_data/data.json')
        ids=[i for i in data['splits']['train'] if data['entries'][i].get('trajectory_kind')=='expert_takeover_v1']
        assert set(ids)<=set(data['splits']['train'])
        # This existing mixed numeric container is train/validation only; never
        # request a test identity or load the separate test trajectories.
        records=visual_records(backend/'state_data',ids,'G00')
        new=with_target_window(records,RECOVERY_ONSET_WINDOW)
        selected_checkpoint=Path(config['bases'][group]['checkpoint'])
        saved=load(selected_checkpoint)
        training=load(selected_checkpoint.parent/'last.pt')
        sampler=training['anchor_sampler']
        rear=load(backend/group/'training/last.pt')
        episode_counts=rear['episode_counts']
        population=collections.Counter();actual=collections.Counter();phase=collections.Counter();rows=[]
        for identity,record in records.items():
            entry=data['entries'][identity];mask=record['supervision_mask'];legal=mask.nonzero().flatten()
            first=int(legal[0]);anchors=planning_anchors(record);fixed=planning_anchors(new[identity])
            hits=target_hits(record,anchors);fixed_hits=target_hits(new[identity],fixed)
            used=target_hits(record,torch.tensor(sampler['anchors'][identity]),sampler['counts'][identity])
            takeover=entry['supervision_boundary']['takeover_tick'];phase[str(takeover%12)]+=1
            population['episodes']+=1;population['legal_ticks']+=len(legal)
            population['missing_target_ticks']+=int((hits[legal]==0).sum())
            population['fixed_missing_target_ticks']+=int((fixed_hits[legal]==0).sum())
            for width in (12,24,48):
                early=legal[:width]
                population[f'initial{width}_population']+=len(early)
                population[f'initial{width}_covered']+=int((hits[early]>0).sum())
                population[f'initial{width}_fixed_covered']+=int((fixed_hits[early]>0).sum())
                actual[f'dp_initial{width}_unique_covered']+=int((used[early]>0).sum())
                actual[f'dp_initial{width}_target_draws']+=int(used[early].sum())
                actual[f'rear_initial{width}_unique_covered']+=len(early)*int(episode_counts[identity]>0)
                actual[f'rear_initial{width}_draws']+=len(early)*episode_counts[identity]
            actual['rear_episodes_seen']+=int(episode_counts[identity]>0)
            rows.append(dict(identity=identity,stratum=next(k for k,v in data['sampling_strata'].items() if identity in v),
                takeover=takeover,first_legal=first,old_first_anchor=int(anchors[0]),new_first_anchor=int(fixed[0]),
                dp_first_anchor_draws=sampler['counts'][identity][0],rear_episode_draws=episode_counts[identity],
                missing=legal[hits[legal]==0].tolist()))
        chosen=[next(row for row in rows if row['takeover']==t) for t in (60,58)]
        chosen.append(next(row for row in rows if row['stratum']=='current_correction' and data['entries'][row['identity']].get('source_group')==group))
        # Real frozen EMA encoder and the trainer's actual batch constructor.
        model=PersistentImage(saved['group'],saved['contract']).eval().requires_grad_(False)
        model.load_state_dict(saved['ema'])
        dp_root=selected_checkpoint.parent.parent/'state_data'
        sampled=visual_records(dp_root,[row['identity'] for row in chosen],'G00')
        fixed_sampled=with_target_window(sampled,RECOVERY_ONSET_WINDOW)
        for row in chosen:
            identity=row['identity'];entry=data['entries'][identity];record=sampled[identity]
            first=row['first_legal'];anchor=row['old_first_anchor'];onset=row['new_first_anchor']
            observed=[];original=model.observe
            def observe(robot, stamp, tick, frame, memory, **kw):
                observed.append(tick)
                return original(robot,stamp,tick,frame,memory,**kw)
            model.observe=observe
            with torch.no_grad():
                condition,target,valid=anchor_batch(model,sampled,[(identity,anchor)])
            assert observed==list(range(anchor+1))
            model.observe=original
            with torch.no_grad():
                fixed_condition,fixed_target,fixed_valid=anchor_batch(model,fixed_sampled,[(identity,onset)])
            assert torch.isfinite(condition).all() and torch.isfinite(fixed_condition).all()
            nominal=load(backend/group/'nominal'/(identity+'.pt'))
            chunk_start=first//32*32
            batch=residual_batch([nominal],chunk_start,chunk_start+32,'cpu')
            rear_mask=residual_supervision_mask([nominal],chunk_start,chunk_start+32,'cpu')
            with h5py.File(entry['path']) as stream:
                raw=stream['episodes'][entry['episode']];dagger=raw['transition/dagger']
                raw_mask=torch.from_numpy(dagger['expert_label_valid'][:]);raw_control=torch.from_numpy(dagger['control_source'][:])
                raw_clean=torch.from_numpy(dagger['switch_clean'][:]);raw_target=torch.from_numpy(dagger['q_reached' if group=='QG01' else 'u_expert'][:])
                assert torch.equal(raw_mask,record['supervision_mask'])
                assert int(((raw_control==1)&raw_clean).nonzero()[0])==first
                assert torch.equal(raw_target[first:first+12],record['clean_sample'][first:first+12])
                assert torch.equal(torch.from_numpy(raw['obs/time_s'][:]),record['time'])
            old_offsets=valid[0].nonzero().flatten();new_offsets=fixed_valid[0].nonzero().flatten()
            assert torch.equal(target[0,old_offsets],raw_target[anchor+old_offsets])
            assert torch.equal(fixed_target[0,new_offsets],raw_target[onset+new_offsets])
            assert bool(rear_mask[0,first-chunk_start])
            row.update(actual_batch=dict(condition_shape=list(condition.shape),prefix_observed_ticks=len(observed),
                old_target_ticks=(anchor+old_offsets).tolist(),new_target_ticks=(onset+new_offsets).tolist(),
                first_legal_observation_in_old_condition=first<=anchor,
                first_legal_not_future_input_in_onset_condition=first>onset,
                backend_chunk_start=chunk_start,backend_offset=first-chunk_start,
                backend_first_legal_mask=bool(rear_mask[0,first-chunk_start]),
                backend_first_cursor=int(batch['plan_cursor'][0,first-chunk_start]),
                raw_mask_time_and_target_equal=True,finite_real_encoder=True))
        report['groups'][group]=dict(population=dict(population),takeover_phase=dict(phase),actual=dict(actual),
            dp_training_updates=training['updates'],dp_selected_updates=saved['updates'],rear_updates=rear['updates'],
            samples=chosen,omitted_episodes=[r for r in rows if r['missing']],
            source_dp=str(selected_checkpoint),source_backend=str(backend),
            phase_caveat='Only phases0 and10 are represented; legal onset phases2 and0. Diverse absolute takeover times are not phase diversity.')
        del model,saved,training,rear,records,new
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.root);args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({g:{k:v for k,v in r.items() if k in ('population','actual','dp_training_updates','dp_selected_updates','rear_updates')} for g,r in result['groups'].items()},indent=2))

if __name__=='__main__':main()
