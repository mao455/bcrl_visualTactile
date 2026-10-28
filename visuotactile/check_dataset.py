"""Real data zero-update checks; reads expert labels, never creates an optimizer."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from .dataset import EpisodeSequenceDataset, CorrectionEpisodeDataset, encode_episode_chunks
from .models.nominal import NominalDiffusionPolicy, PolicyConfig
from .trial_utils import write_json
from .manifest import ManifestSequenceDataset


def fingerprint(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def check(original,compact):
    flags=dict(allow_incomplete=True,allow_unsuccessful=True)
    old,new=EpisodeSequenceDataset(original,**flags),EpisodeSequenceDataset(compact,**flags)
    episode=old.episodes[0];rows=0
    for a,b in zip(old.chunks(episode,7),new.chunks(episode,7),strict=True):
        for key in a['obs']:assert torch.equal(a['obs'][key],b['obs'][key]),key
        for key in ('action_target','action_valid'):assert torch.equal(a[key],b[key]),key
        assert not b['obs']['prior_plan_valid'].any() and not b['obs']['previous_nominal_valid'].any()
        rows+=a['stop']-a['start']
    torch.manual_seed(88)
    model=NominalDiffusionPolicy(PolicyConfig(inference_steps=1,views=len(new.metadata['camera_ids']))).eval()
    before=fingerprint(model)
    with torch.no_grad():
        def encode(size):
            return torch.cat([c for c,_,_ in encode_episode_chunks(model.encoder,new.chunks(episode,size))],1)
        chunks=encode(7);online=encode(1)
        repeated=encode(7)  # fresh episode state, unchanged parameters/buffers
        assert torch.equal(chunks,repeated)
        torch.testing.assert_close(chunks,online,atol=2e-6,rtol=1e-5)
        assert torch.isfinite(chunks).all()
        # Finite diffusion output at selected true conditions, no optimizer/loss.
        prediction=model.predict_from_condition(chunks[:, -1],generator=torch.Generator().manual_seed(88))
        assert torch.isfinite(prediction).all() and prediction.shape==(1,12,29)
    assert fingerprint(model)==before
    try:CorrectionEpisodeDataset(compact,**flags)
    except ValueError as exc:rejected=str(exc)
    else:raise AssertionError('Pure expert data unexpectedly accepted for Correction')
    return dict(rows=rows,old_new_inputs_labels_exact=True,parameter_updates=0,
        parameter_and_buffer_sha256_before=before,parameter_and_buffer_sha256_after=fingerprint(model),
        chunk7_step1_max_error=float((chunks-online).abs().max()),reset_repeat_exact=True,
        correction_rejection=rejected,prediction_shape=list(prediction.shape),
        unresolved='Expert captures lack own-plan history; future trainer needs explicit no-plan baseline or identified causal replay. memory2 validation loss still absent.')


def check_manifest(path, ids):
    """Two real full episodes through a single carried-state iterator, no updates."""
    if len(ids)!=2:
        raise ValueError('Select exactly two complete episodes for this bounded check')
    dataset=ManifestSequenceDataset(path,ids=ids)
    try: dataset.require_self_plan_training()
    except ValueError as exc: plan_guard=str(exc)
    else: raise AssertionError('Expected pure expert records in this check')
    torch.manual_seed(88)
    first=dataset.datasets[ids[0]]
    model=NominalDiffusionPolicy(PolicyConfig(inference_steps=1,views=len(first.metadata['camera_ids']))).eval()
    before=fingerprint(model)
    def sequence(size):
        for episode in ids:
            yield from dataset.chunks(episode,size)
    def encode(chunks):
        return torch.cat([c for c,_,_ in encode_episode_chunks(model.encoder,chunks)],1)
    rows=0
    for key in ids:
        inner=dataset.datasets[key]
        local=dataset.reader.entries[key]['episode']
        for a,b in zip(dataset.chunks(key,7),inner.chunks(local,7),strict=True):
            for name in a['obs']: assert torch.equal(a['obs'][name],b['obs'][name]),name
            for name in ('action_target','action_valid'): assert torch.equal(a[name],b[name]),name
            rows+=a['stop']-a['start']
    with torch.no_grad():
        combined=encode(sequence(7))
        online=encode(sequence(1))
        independent=torch.cat([encode(dataset.chunks(k,7)) for k in ids],1)
        assert torch.equal(combined,independent), 'Episode reset did not clear state'
        torch.testing.assert_close(combined,online,atol=2e-6,rtol=1e-5)
        assert torch.isfinite(combined).all()
        prediction=model.predict_from_condition(combined[:,-1],generator=torch.Generator().manual_seed(88))
        assert torch.isfinite(prediction).all()
    assert fingerprint(model)==before
    return dict(episodes=ids,rows=rows,parameter_updates=0,manifest_single_file_inputs_labels_exact=True,
        combined_vs_independent_reset_exact=True,chunk7_step1_max_error=float((combined-online).abs().max()),
        parameter_and_buffer_sha256_before=before,parameter_and_buffer_sha256_after=fingerprint(model),
        self_plan_guard=plan_guard,prediction_shape=list(prediction.shape))


def main():
    p=argparse.ArgumentParser();p.add_argument('--original');p.add_argument('--compact');p.add_argument('--output',required=True)
    p.add_argument('--manifest');p.add_argument('--ids',nargs=2)
    a=p.parse_args();torch.set_num_threads(2)
    if a.manifest and a.ids: r=check_manifest(a.manifest,a.ids)
    elif a.original and a.compact: r=check(a.original,a.compact)
    else: p.error('Use --original/--compact or --manifest/--ids')
    write_json(a.output,r);print(json.dumps(r,indent=2))


if __name__=='__main__':main()
