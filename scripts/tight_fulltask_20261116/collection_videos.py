"""Viewing derivatives of validated complete real RGB records, owner label per tick."""
import json,time
from pathlib import Path
import h5py,numpy as np,torch
from visuotactile.video import RolloutVideo,verify_video
from visuotactile.training_resources import atomic_json
from scripts.tight_fulltask_20261116.run import ROOT,read

def main():
    rows=[];started=time.time()
    for path in sorted((ROOT/'collection').glob('*/result.json')):
        result=read(path);capture=result.get('dataset_capture')
        if not capture or not result.get('success'):continue
        output=path.parent/'student_then_expert.mp4';assert not output.exists()
        video=RolloutVideo(output,result.get('source_group','takeover'),result['seed'],'source',['fixed0','fixed1','wrist'])
        with h5py.File(capture['path']) as f:
            g=f['episodes'][result['identity']];source=g['transition/dagger/control_source'][:].reshape(-1)
            for t in range(result['steps']):
                im=torch.from_numpy(g['frames/rgb_u8'][t].astype('float32')/255)[None]
                label='EXPERT TAKEOVER' if source[t] else 'AUTONOMOUS STUDENT'
                video.append({'image':im},t,status=f'{label} | {path.parent.name}')
        row=video.close(True);row.update(case=path.parent.name,decode=verify_video(output,result['steps']),
            switch_action_index=int(np.flatnonzero(source)[0]),success=result['success'],
            interpretation='Expert completes from real student grasp; not autonomous student insertion')
        rows.append(row)
    atomic_json(ROOT/'collection/videos.json',dict(complete=True,rows=rows,seconds=time.time()-started))
if __name__=='__main__':main()
