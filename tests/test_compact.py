import json
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
import pytest

from visuotactile.compact import EpisodeFile, convert_episode, rgb_u8
from visuotactile.recording import EpisodeReader, flatten
from visuotactile.run_rl_trials import validate_completion, prior_attempt


def fixture(path, terminal, views=2):
    rng=np.random.default_rng(9)
    rgb=rng.integers(0,256,(4,views,3,6,8),dtype=np.uint8).astype('f4')/np.float32(255)
    depth=rng.random((4,views,1,6,8),dtype=np.float32)
    images=np.concatenate((rgb,depth),2);valid=np.ones_like(depth,dtype=bool)
    with h5py.File(path,'x') as f:
        f.attrs.update(schema_version=2,metadata_json=json.dumps({'perception_version':2, 'camera_rig':{'version':3 if views==3 else 2},'camera_ids':[f'view_{i}' for i in range(views)]}))
        g=f.create_group('episodes/test');g.attrs.update(length=3,complete=terminal)
        for key,arr in {'image':images,'depth_m':depth,'depth_valid':valid,
                        'camera_time_s':np.arange(4)[:,None].repeat(views,1)/60,
                        'camera_frame':np.arange(4)[:,None].repeat(views,1),
                        'episode_id':np.zeros(4,dtype='i8'),'step_id':np.arange(4)}.items():
            g['obs/'+key]=arr[:3]
            if key!='depth_m':
                post=arr[1:].copy()
                if terminal:post[-1]=0
                g['transition/observation_after/'+key]=post
        g['transition/observation_after/available']=np.array([True,True,not terminal])
        g['transition/action']=rng.random((3,29),dtype=np.float32)


@pytest.mark.parametrize('views',[2,3])
@pytest.mark.parametrize('terminal',[True,False])
def test_lossless_storage_and_terminal_boundary(tmp_path,terminal,views):
    source=tmp_path/'old.h5';fixture(source,terminal,views)
    output=tmp_path/'compact';info=convert_episode(source,output)
    assert info['frame_count']==3+int(not terminal)
    assert sorted(p.name for p in output.iterdir())==[*(f'camera_{v}.mp4' for v in range(views)),'data.h5']
    a=flatten(EpisodeReader(source).read('test'));b=flatten(EpisodeReader(output/'data.h5').read('test'))
    assert set(a)==set(b)
    for key in a:
        assert a[key].dtype==b[key].dtype
        np.testing.assert_array_equal(a[key],b[key])
    with EpisodeFile(output/'data.h5') as f:
        np.testing.assert_array_equal(f['episodes/test']['obs/image'][1,:,:3],a['obs/image'][1,:,:3])
    with pytest.raises(FileExistsError):convert_episode(source,output)


def test_refuse_noninvertible_rgb_and_misaligned_frames(tmp_path):
    with pytest.raises(ValueError,match='reversible'):rgb_u8(np.array([.12345],dtype='f4'))
    source=tmp_path/'old.h5';fixture(source,True)
    with h5py.File(source,'r+') as f:f['episodes/test/transition/observation_after/camera_frame'][0]=90
    with pytest.raises(ValueError,match='misalignment'):convert_episode(source,tmp_path/'bad')
    assert source.exists() and (tmp_path/'bad/data.h5.partial').exists()


def test_fresh_and_resume_share_result_identity_gate(tmp_path):
    path=tmp_path/'trial';path.mkdir()
    result={'steps':3,'started_episodes':1,'successful_episodes':0,'externally_censored':False}
    (path/'result.json').write_text(json.dumps(result))
    trial=dict(summary=dict(audit_errors=[],control_steps=3,outcome='failure'),metadata=dict(seed=10,condition='original'))
    with patch('visuotactile.run_rl_trials.load_trial',return_value=trial):
        assert validate_completion(path,{'seed':10}) is trial
        with pytest.raises(RuntimeError,match='identity'):validate_completion(path,{'seed':11})
        (tmp_path/'trial.status.json').write_text(json.dumps(dict(returncode=0,episode_started=True,audit_errors=[])))
        assert prior_attempt(tmp_path,'trial',{'seed':10})[1] is trial
        (path/'FAILED.txt').write_text('Kit returned zero after exception')
        with pytest.raises(RuntimeError):validate_completion(path)
        with pytest.raises(RuntimeError):prior_attempt(tmp_path,'trial',{'seed':10})
