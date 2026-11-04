"""Native observation video recording and complete CPU decode validation."""
from pathlib import Path
import av

class RolloutVideo:
    """Bounded, CPU-encoded viewing derivative of every pre-action RGB packet."""
    def __init__(self, path, variant, seed, epoch, camera_ids):
        import av
        from PIL import ImageFont
        self.path=Path(path)
        self.partial=self.path.with_suffix('.mp4.partial')
        self.container=av.open(str(self.partial),'w',format='mp4')
        self.stream=None;self.frames=0
        self.title=f'{variant.upper()} | epoch {epoch} | seed {seed} | rerun for viewing | 1x simulation time'
        self.camera_ids=camera_ids
        self.font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)

    def append(self, packet, step, *, status=None):
        import av
        import numpy as np
        from PIL import Image,ImageDraw
        rgb=np.rint(packet['image'][0,:,:3].cpu().numpy().clip(0,1)*255).astype('uint8').transpose(0,2,3,1)
        n,h,w,_=rgb.shape
        if n!=len(self.camera_ids):raise ValueError('Video camera identity/count mismatch')
        canvas=Image.new('RGB',(n*w,h+64),(22,22,26));draw=ImageDraw.Draw(canvas)
        draw.text((10,3),self.title if status is None else status,font=self.font,fill='white')
        for i,name in enumerate(self.camera_ids):
            draw.text((i*w+10,32),f'{name} | step {step} | t={step/60:.3f}s',font=self.font,fill='white')
            canvas.paste(Image.fromarray(rgb[i]),(i*w,64))
        if self.stream is None:
            self.stream=self.container.add_stream('libx264',rate=60)
            self.stream.width,self.stream.height=canvas.size
            self.stream.pix_fmt='yuv420p';self.stream.codec_context.thread_count=2
            self.stream.options={'crf':'18','preset':'fast'}
        frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');frame.pts=self.frames
        for encoded in self.stream.encode(frame):self.container.mux(encoded)
        self.frames+=1

    def close(self, complete=False):
        if self.container is None:return
        if self.stream is not None:
            for encoded in self.stream.encode():self.container.mux(encoded)
        self.container.close();self.container=None
        if complete:
            self.partial.replace(self.path)
            return dict(file=self.path.name,frames=self.frames,fps=60,duration_s=self.frames/60,
                        camera_ids=self.camera_ids,frame_source='RGB observation before each native action',
                        terminal_image_available=False,reset_frames_included=False,
                        encoding='CPU libx264 CRF18 yuv420p; viewing only, not lossless training data')
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
