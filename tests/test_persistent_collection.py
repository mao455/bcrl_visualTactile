import json
import hashlib
import numpy as np
import pytest
from visuotactile.recording import CompactEpisodeStream, EpisodeReader, flatten, verify_stream_fields


def rows(n=11, identity=3):
    rng=np.random.default_rng(identity)
    images=np.concatenate((rng.integers(0,256,(n+1,3,3,4,6),dtype='u1').astype('f4')/np.float32(255),
                           rng.random((n+1,3,1,4,6),dtype='f4')),axis=2)
    valid=np.ones((3,1,4,6),dtype=bool)
    for i in range(n):
        row={'obs/image':images[i],'obs/depth_m':images[i,:,3:],'obs/depth_valid':valid,
             'obs/episode_id':np.asarray(identity,dtype='i8'),'obs/step_id':np.asarray(i,dtype='i8'),
             'transition/episode_id':np.asarray(identity,dtype='i8'),'transition/step_id':np.asarray(i,dtype='i8'),
             'transition/action':rng.random(29,dtype='f4'),'terminated':np.asarray(i==n-1),'truncated':np.asarray(False)}
        for key in ('image','depth_valid','episode_id','step_id'):
            next_value={'image':images[i+1],'depth_valid':valid,'episode_id':np.asarray(identity,dtype='i8'),
                        'step_id':np.asarray(i+1,dtype='i8')}[key]
            row['transition/observation_after/'+key]=next_value if i<n-1 else np.zeros_like(next_value)
        row['transition/observation_after/available']=np.asarray(i<n-1)
        yield row


def test_stream_exact_fields_and_independent_episodes(tmp_path):
    streams=[CompactEpisodeStream(tmp_path/str(i),{'perception_version':2},f'env{i}') for i in range(2)]
    originals=[list(rows(identity=i+1)) for i in range(2)]
    for pair in zip(*originals):
        for stream,row in zip(streams,pair):stream.append(row)
    for i,s in enumerate(streams):
        s.finish({'steps':11})
        assert verify_stream_fields(tmp_path/str(i)/'data.h5')==len(originals[i][0])
        arrays=flatten(EpisodeReader(tmp_path/str(i)/'data.h5').read(f'env{i}'))
        for key in originals[i][0]:np.testing.assert_array_equal(arrays[key],np.stack([r[key] for r in originals[i]]))
    assert not np.array_equal(originals[0][0]['obs/image'],originals[1][0]['obs/image'])


def test_stream_rejects_misalignment(tmp_path):
    s=CompactEpisodeStream(tmp_path/'bad',{},'bad');r=list(rows())
    s.append(r[0]);r[1]['obs/image']=r[1]['obs/image'].copy();r[1]['obs/image'][0,0,0,0]=.5
    with pytest.raises(ValueError,match='misalignment'):s.append(r[1])
    s.file.close()
    assert (tmp_path/'bad/data.h5.partial').exists() and not (tmp_path/'bad/data.h5').exists()


def test_stream_graceful_censor_preserves_real_last_post_frame(tmp_path):
    source=list(rows())[:3]
    stream=CompactEpisodeStream(tmp_path/'censored',{},'ep')
    for row in source:stream.append(row)
    stream.finish({'steps':3,'externally_censored':True})
    assert verify_stream_fields(tmp_path/'censored/data.h5')==len(source[0])
    reader=EpisodeReader(tmp_path/'censored/data.h5')
    assert not reader.attributes('ep')['complete']
    actual=flatten(reader.read('ep'))
    for key in source[0]:np.testing.assert_array_equal(actual[key],np.stack([r[key] for r in source]))


def test_mount_env_instance_is_not_a_new_calibration():
    from visuotactile.contracts import assert_policy_compatible
    from visuotactile.camera_rigs import rig_contract
    import copy
    fields=('proprio_list image_modality image_width image_height image_input_width image_input_height '
            'crop_enabled crop_top_left crop_bottom_right depth_preprocess_mode depth_min_m depth_max_m '
            'camera_backend camera_update_period_s use_camera_delay camera_delay_max use_student_obs_delay '
            'student_obs_delay_max use_depth_aug use_camera_pose_rand hide_goal_viz').split()
    m=dict(perception_version=2,student_config={k:0 for k in fields},problem='tight_insertion',camera_mode='oblique',
        joint_order=[],physics_dt=1/120,control_dt=1/60,nominal_proprio='encoders',
        camera_rig=rig_contract('wrist_bracket_close'),camera_ids=['fixed_high','fixed_side','wrist_rgbd'],
        cameras=[dict(id=k,pos_env=[[0,0,1]],quat_ros_wxyz=[[1,0,0,0]],intrinsics=[np.eye(3).tolist()],resolution=[252,448])
                 for k in ('fixed_high','fixed_side','wrist_rgbd')],depth_encoding='metric',
        task_command={'vocabulary':['tight_insertion'],'frame':'robot_base'},fk_urdf_sha256='test',config={'sim':{'render':{}}})
    m['camera_rig']['wrist']['parent_prim']='/World/envs/env_0/Robot/iiwa14_link_7'
    other=copy.deepcopy(m);other['camera_rig']['wrist']['parent_prim']='/World/envs/env_3/Robot/iiwa14_link_7'
    assert_policy_compatible(m,other)
    other['camera_rig']['wrist']['pos_parent_m']=[.12,.17,.07]
    with pytest.raises(ValueError):assert_policy_compatible(m,other)
