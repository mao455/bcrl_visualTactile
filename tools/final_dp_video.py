"""Complete CPU video decode using the already installed PyAV binding."""
from pathlib import Path
import av


def verify_video(path, expected_frames=None):
    path=Path(path)
    frames=0;last_pts=None;first_shape=None
    with av.open(str(path)) as stream:
        assert len(stream.streams.video)==1
        rate=float(stream.streams.video[0].average_rate)
        for frame in stream.decode(video=0):
            assert frame.pts is not None and (last_pts is None or frame.pts>last_pts)
            array=frame.to_ndarray(format='rgb24')
            if first_shape is None:first_shape=list(array.shape)
            assert list(array.shape)==first_shape
            last_pts=frame.pts;frames+=1
    assert frames>0
    if expected_frames is not None:assert frames==expected_frames,(frames,expected_frames)
    return dict(decoded_frames=frames,frame_rate=rate,shape=first_shape,backend='PyAV full RGB decode')
