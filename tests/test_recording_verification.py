"""Exact source verification after sharing compact-frame decompression."""
import h5py
import numpy as np
import pytest

from visuotactile.recording import (CompactEpisodeStream, verify_stream_fields,
                                   append_numeric, BufferedNumericWriter)


def capture(tmp_path,codec,terminal):
    rng=np.random.default_rng(51);n=11
    rgb=rng.integers(0,256,(n+1,3,3,4,5),dtype=np.uint8).astype(np.float32)/np.float32(255)
    depth=rng.random((n+1,3,1,4,5),dtype=np.float32)
    image=np.concatenate((rgb,depth),axis=2)
    valid=rng.random((n+1,3,1,4,5))>.2
    writer=CompactEpisodeStream(tmp_path/'capture',{},'episode',compression=codec)
    for i in range(n):
        available=not (terminal and i==n-1)
        writer.append({'obs/image':image[i],'obs/depth_m':depth[i],'obs/depth_valid':valid[i],
            'obs/step_id':np.asarray(i),'obs/episode_id':np.asarray(7),
            'transition/step_id':np.asarray(i),'transition/episode_id':np.asarray(7),
            'transition/observation_after/available':np.asarray(available),
            'transition/observation_after/image':image[i+1] if available else np.zeros_like(image[i]),
            'transition/observation_after/depth_valid':valid[i+1] if available else np.zeros_like(valid[i]),
            'transition/observation_after/step_id':np.asarray(i+1),
            'transition/observation_after/episode_id':np.asarray(7),
            'terminated':np.asarray(not available),'truncated':np.asarray(False)})
    writer.finish({'benchmark':'unit fixture'})
    return tmp_path/'capture/data.h5'


@pytest.mark.parametrize('codec',['lzf','gzip4_lossless_v1'])
@pytest.mark.parametrize('terminal',[True,False])
def test_all_source_digests_cover_partial_block_and_terminal(tmp_path,codec,terminal):
    path=capture(tmp_path,codec,terminal)
    assert verify_stream_fields(path)==14


@pytest.mark.parametrize('field',['frames/rgb_u8','frames/depth_m','frames/depth_valid','frames/post_index','obs/step_id'])
def test_corruption_still_rejected(tmp_path,field):
    path=capture(tmp_path,'lzf',True)
    with h5py.File(path,'r+') as f:
        ds=f['episodes/episode/'+field]
        value=ds[0]
        if field=='frames/post_index':value=np.asarray(0,dtype=value.dtype)
        elif value.dtype==bool:value=~value
        else:value=value+1
        ds[0]=value
    with pytest.raises(ValueError,match='content differs'):
        verify_stream_fields(path)


def test_buffered_numeric_preserves_snapshot_and_final_partial_block(tmp_path):
    with h5py.File(tmp_path/'numeric.h5', 'w') as f:
        legacy = f.create_group('legacy')
        buffered = f.create_group('buffered')
        writer = BufferedNumericWriter(buffered, buffer_steps=3)
        values = {'obs': {'state': np.zeros((1, 2), dtype=np.float32),
                          'valid': np.zeros((1,), dtype=bool)}}
        for step in range(5):
            values['obs']['state'][:] = [step, np.nan if step == 4 else -0.]
            values['obs']['valid'][:] = step % 2
            append_numeric(legacy, values, step)
            writer.append(values, step)
        values['obs']['state'][:] = -999  # snapshots must survive caller reuse
        writer.flush()
        writer.flush()  # exception/finalization cleanup may flush twice
        for field in ('obs/state', 'obs/valid'):
            actual, expected = buffered[field], legacy[field]
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.chunks == expected.chunks
            assert actual.compression == expected.compression
            assert actual[:].tobytes() == expected[:].tobytes()


def test_buffered_numeric_rejects_drift_without_losing_accepted_rows(tmp_path):
    with h5py.File(tmp_path/'numeric.h5', 'w') as f:
        writer = BufferedNumericWriter(f.create_group('episode'))
        writer.append({'state': np.asarray([[1.]], dtype=np.float32)}, 0)
        with pytest.raises(ValueError, match='consecutive'):
            writer.append({'state': np.asarray([[2.]], dtype=np.float32)}, 2)
        with pytest.raises(ValueError, match='fields changed'):
            writer.append({'other': np.asarray([[2.]], dtype=np.float32)}, 1)
        with pytest.raises(ValueError, match='shape/dtype'):
            writer.append({'state': np.asarray([[2.]], dtype=np.float64)}, 1)
        writer.flush()
        assert f['episode/state'][:].tolist() == [[1.]]
