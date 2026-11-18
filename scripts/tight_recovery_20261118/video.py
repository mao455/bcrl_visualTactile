"""Viewing-only H264 from preserved real before-action camera frames."""
import subprocess
import imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
import h5py,numpy as np,cv2
from scripts.tight_recovery_20261118.run import ROOT,OUT,read,write

def main():
 out=OUT/'videos';out.mkdir(exist_ok=True);rows=[];sheets=[]
 for role in 'AB':
  name=f'r08_{role}_layout10_train_28817000_regression';r=read(ROOT/'pilot'/name/'result.json');path=out/f'pilot_{role}_28817000.mp4';take=r['dagger']['trigger_receipt']['trigger_completed_tick']
  with h5py.File(r['dataset_capture']['path']) as f:
   frames=f['episodes'][name]['frames/rgb_u8'];n=len(frames)
   cmd=[FFMPEG,'-v','error','-y','-f','rawvideo','-pixel_format','rgb24','-video_size','1344x280','-framerate','60','-i','-','-c:v','libx264','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
   proc=subprocess.Popen(cmd,stdin=subprocess.PIPE);selected={max(0,take-24),take-2,take+2,min(n-1,take+26),n-1}
   for i in range(n):
    raw=np.asarray(frames[i]).transpose(0,2,3,1);im=np.concatenate(list(raw),axis=1);im=np.pad(im,((28,0),(0,0),(0,0)));source='STUDENT history only' if i<take else 'HANDOFF masked' if i<take+2 else 'EXPERT lawful labels';cv2.putText(im,f'{role} / tick {i} / {source}',(10,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1);proc.stdin.write(im.tobytes())
    if i in selected:sheets.append(cv2.resize(im,(1008,210)))
   proc.stdin.close();assert proc.wait()==0
  subprocess.run([FFMPEG,'-v','error','-i',str(path),'-f','null','-'],check=True);rows.append(dict(role=role,file=str(path.relative_to(OUT)),frames=n,fps=60,takeover_tick=take,source=r['dataset_capture'],purpose='Full real captured before-action frames, viewing only; no simulation rerun or altered control'))
 cv2.imwrite(str(out/'pilot_review.jpg'),cv2.cvtColor(np.concatenate(sheets),cv2.COLOR_RGB2BGR));write(out/'index.json',rows)
if __name__=='__main__':main()
