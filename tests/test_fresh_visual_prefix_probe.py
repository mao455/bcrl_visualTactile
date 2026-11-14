import copy
import json
import math

import h5py
import numpy as np
import pytest
import torch

from tools import tight_fresh_visual_prefix_probe as probe


def population():
    original = [f'train{i}' for i in range(30)]
    validation = [f'val{i}' for i in range(6)]
    test = ['test_unopened']
    entries = {i:dict(path='/never/open/'+i, episode=i, family=i, length=4, source_sha256='a'*64)
               for i in original+validation+test}
    entries['correction'] = dict(entries[original[0]], episode='correction',
                                 split='train', trajectory_kind='expert_takeover_v1')
    return dict(normalization_train_ids=original, entries=entries,
                splits=dict(train=original+['correction'], validation=validation, test=test))


def test_population_excludes_test_payload_and_matches_roles():
    data = population()
    selected = probe.admit_population(data, copy.deepcopy(data))
    assert selected == dict(original=data['normalization_train_ids'],
                            validation=data['splits']['validation'], corrections=['correction'])
    other = copy.deepcopy(data)
    other['entries']['train0']['source_sha256'] = 'b'*64
    with pytest.raises(ValueError, match='A/B'):
        probe.admit_population(data, other)


@pytest.mark.parametrize('change', ['test_family', 'unknown_family', 'validation_overlap', 'duplicate_original'])
def test_metadata_admission_rejects_family_and_population_errors(change):
    data = population()
    if change == 'test_family':
        data['entries']['correction']['family'] = 'test_unopened'
    elif change == 'unknown_family':
        data['entries']['correction']['family'] = 'unknown'
    elif change == 'validation_overlap':
        data['splits']['validation'][0] = 'train0'
    else:
        data['normalization_train_ids'][1] = 'train0'
    with pytest.raises(ValueError):
        probe.admit_population(data, copy.deepcopy(data))


def test_bad_population_rejected_before_source_payloads(tmp_path, monkeypatch):
    data = population()
    data['entries']['correction']['family'] = 'test_unopened'
    for role in 'AB':
        target = tmp_path/role/'state_data/data.json'
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps(data))
    monkeypatch.setattr(probe, 'source_labels', lambda *a:pytest.fail('Opened trajectory before admission'))
    with pytest.raises(ValueError, match='family overlap'):
        probe.run(tmp_path, tmp_path/'report.json')
    assert not (tmp_path/'report.json').exists()


def authority():
    return [torch.tensor([0, 0, 1, 1]), torch.tensor([False, False, False, True]),
            torch.tensor([False, False, True, True]), torch.tensor([0, 0, 1, 2]),
            torch.tensor([False, True, True, True])]


def test_masks_keep_transition_unlabelled_and_contact_causal():
    masks = probe.authority_masks(*authority())
    assert masks['student_prefix'].tolist() == [True, True, False, False]
    assert masks['student_after_contact'].tolist() == [False, True, False, False]
    # Expert authority + clean switch still cannot manufacture a missing label.
    assert masks['legal_expert'].tolist() == [False, False, False, True]


@pytest.mark.parametrize('change', ['zero_labels', 'student_label', 'dirty_label', 'switchback', 'segment', 'dtype'])
def test_invalid_authority_rejected(change):
    values = authority()
    if change == 'zero_labels': values[1][:] = False
    elif change == 'student_label': values[1][1] = True
    elif change == 'dirty_label': values[2][-1] = False
    elif change == 'switchback': values[0][-1] = 0
    elif change == 'segment': values[3][-1] = values[3][-2]
    else: values[1] = values[1].long()
    with pytest.raises(ValueError):
        probe.authority_masks(*values)


def test_fit_excludes_correction_validation_and_uses_original_statistics():
    records = {i:dict(visual=torch.tensor([[x, 0.], [x+1., 1.]]))
               for i,x in [('original', 0.), ('correction', 1000.), ('validation', -1000.)]}
    truth = {i:torch.tensor([[1., 0., 0.], [2., 0., 0.]]) for i in records}
    params, report = probe.fit_original(records, truth, ['original'], 'visual_only')
    records['correction']['visual'].fill_(float('nan'))
    records['validation']['visual'].fill_(float('nan'))
    truth['validation'].fill_(float('nan'))
    repeated, again = probe.fit_original(records, truth, ['original'], 'visual_only')
    assert all(torch.equal(a, b) for a,b in zip(params, repeated))
    assert torch.equal(params[0], torch.tensor([.5, .5], dtype=torch.float64))
    assert report == again and report['training_frames'] == 2


def cache():
    stamp = torch.arange(4, dtype=torch.float64)/60
    masks = probe.authority_masks(*authority())
    masks['all'] = torch.ones(4, dtype=torch.bool)
    binding = dict(base_checkpoint_sha256='a'*64, data_sha256='b'*64,
                   statistics_sha256='c'*64, residual_statistics_sha256='d'*64)
    record = dict(identity='case', cache_identity=binding.copy(), time=stamp.clone(),
                  visual=torch.zeros(4, 512), robot=torch.zeros(4, 178),
                  supervision_mask=masks['legal_expert'].clone(), control_segment=authority()[3])
    return record, stamp, binding, masks


@pytest.mark.parametrize('change', ['base', 'stats', 'extra_binding', 'time', 'shape', 'finite', 'label', 'segment'])
def test_cache_rejects_stale_or_misaligned_features(change):
    record, stamp, binding, masks = cache()
    segment = record['control_segment'].clone()
    probe.validate_cache(record, 'case', stamp, binding, masks, segment)
    if change == 'base': record['cache_identity']['base_checkpoint_sha256'] = 'e'*64
    elif change == 'stats': record['cache_identity']['residual_statistics_sha256'] = 'e'*64
    elif change == 'extra_binding': record['cache_identity']['extra'] = 'e'*64
    elif change == 'time': record['time'] += 1/60
    elif change == 'shape': record['visual'] = torch.zeros(4, 511)
    elif change == 'finite': record['robot'][0, 0] = float('nan')
    elif change == 'label': record['supervision_mask'][2] = True
    else: record['control_segment'][2] = 9
    with pytest.raises(ValueError):
        probe.validate_cache(record, 'case', stamp, binding, masks, segment)


def test_frame_episode_metrics_and_empty_phase():
    errors = dict(short=torch.tensor([[.001, 0., 0.]]), long=torch.tensor([[.003, 0., 0.]]*3))
    masks = {i:dict(all=torch.ones(len(v), dtype=torch.bool), empty=torch.zeros(len(v), dtype=torch.bool))
             for i,v in errors.items()}
    result = probe.summarize(errors, masks, list(errors), ['all', 'empty'])
    assert result['all']['frame_weighted_vector_rmse_mm'] == pytest.approx(math.sqrt(7))
    assert result['all']['episode_equal_mean_vector_rmse_mm'] == pytest.approx(2.)
    assert result['empty'] == dict(frames=0, episodes=0, empty_episodes=2,
        frame_weighted_vector_rmse_mm=None, episode_equal_mean_vector_rmse_mm=None)


def source(tmp_path):
    urdf = tmp_path/'object.urdf'
    urdf.write_text('<robot><link name="object"><collision><origin xyz="0 .025 0"/>'
                    '<geometry><box size=".02 .05 .02"/></geometry></collision></link></robot>')
    path = tmp_path/'source.h5'
    q = [math.sqrt(.5), 0., 0., math.sqrt(.5)]
    stamp = np.arange(4, dtype=np.float64)/60
    force = np.zeros((4, 2, 5, 3), dtype=np.float32)
    force[1:, 0, 0, 2] = .2
    with h5py.File(path, 'w') as stream:
        stream.attrs['metadata_json'] = json.dumps(dict(initial_family='family', dataset_split='train',
            config=dict(assets=dict(object_urdf=str(urdf)))))
        group = stream.create_group('episodes/case')
        group.attrs['complete'] = True
        values = {'obs/time_s':stamp, 'transition/time_before_s':stamp,
            'transition/diagnostics_before/palm_body_pose_w':np.tile([1., 2., 3., *q], (4, 1)),
            'transition/diagnostics_before/object_pose_w':np.tile([2., 2., 3., *q], (4, 1)),
            'obs/contact_substeps_w_N':force, 'obs/contact_valid':np.array([False, True, True, True]),
            'obs/contact_time_s':np.tile(stamp[:, None], (1, 2)),
            'transition/contact_substeps_after_w_N':np.concatenate((force[1:], force[-1:])),
            'terminated':np.array([False]*3+[True]), 'truncated':np.zeros(4, dtype=bool)}
        for key,value in zip(('control_source', 'expert_label_valid', 'switch_clean', 'segment_id'), authority()):
            values['transition/dagger/'+key] = value.numpy()
        for key,value in values.items(): group.create_dataset(key, data=value)
    return dict(path=str(path), episode='case', family='family', length=4, source_sha256='a'*64)


def test_current_pose_rotations_and_actual_authority(tmp_path):
    entry = source(tmp_path)
    target, masks, stamp, segment, report = probe.source_labels(entry, 'train', True)
    assert torch.allclose(target, torch.tensor([[0., -.975, 0.]]*4, dtype=torch.float64), atol=1e-12)
    assert masks['student_after_contact'].tolist() == [False, True, False, False]
    assert report['first_observed_contact_step_id'] == 1
    assert report['phase_frames']['legal_expert'] == 1
    assert report['source_manifest_sha256'] == 'a'*64
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize('change', ['future_label', 'future_contact', 'wrong_family', 'nan_position'])
def test_source_clock_family_and_values_rejected(tmp_path, change):
    entry = source(tmp_path)
    with h5py.File(entry['path'], 'r+') as stream:
        group = stream['episodes/case']
        if change == 'future_label': group['transition/time_before_s'][:] += 1/60
        elif change == 'future_contact': group['obs/contact_time_s'][1, 0] += 1/60
        elif change == 'wrong_family': entry['family'] = 'wrong'
        else: group['transition/diagnostics_before/object_pose_w'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        probe.source_labels(entry, 'train', True)


def test_output_cannot_overwrite_or_follow_dangling_symlink(tmp_path):
    output = tmp_path/'evidence.json'
    probe.write_new(output, dict(value=1))
    with pytest.raises(FileExistsError): probe.write_new(output, dict(value=2))
    assert json.loads(output.read_text()) == dict(value=1)
    dangling = tmp_path/'dangling.json'
    dangling.symlink_to(tmp_path/'absent.json')
    with pytest.raises(FileExistsError): probe.run(tmp_path/'not_read', dangling)
    with pytest.raises(FileExistsError): probe.write_new(dangling, {})
    assert not (tmp_path/'absent.json').exists()
