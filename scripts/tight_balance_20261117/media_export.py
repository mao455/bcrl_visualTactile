"""Publish two measured candidate/baseline video pairs without rerunning physics."""
import json,shutil,subprocess
from pathlib import Path
import cv2,imageio_ffmpeg
from PIL import Image,ImageDraw
from scripts.tight_balance_20261117.offline_phases import ROOT,OUT
from scripts.tight_layout_20261116.publication import write,sha

def main():
 candidates=[]
 for part in range(4):
  folder=ROOT/f'comparison{part:02d}'/'development';plan=json.loads((folder/'plan.json').read_text())
  for role in 'AB':
   cases={c['role']:c for c in plan['cases'] if c['spec']['seed']==28824000+2*part}
   pair=[]
   for r in [role,role+'_baseline']:
    c=cases[r];p=folder/c['name'];v=p/'autonomous.mp4'
    if not v.exists():break
    pair.append(dict(role=r,seed=c['spec']['seed'],path=v,result=json.loads((p/'result.json').read_text())))
   if len(pair)==2:candidates.append((role,pair))
 media=OUT/'media';media.mkdir(exist_ok=True);rows=[];panels=[]
 for role in 'AB':
  choices=[p for r,p in candidates if r==role];assert choices
  # Prefer an observed outcome difference, then a candidate success, then first recorded pair.
  pair=max(choices,key=lambda p:(p[0]['result']['success']!=p[1]['result']['success'],p[0]['result']['success'],-p[0]['seed']))
  for item in pair:
   src=item['path'];dest=media/f"{item['role']}_{item['seed']}.mp4";shutil.copyfile(src,dest)
   subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-threads','2','-i',str(dest),'-f','null','-'],check=True,stdout=subprocess.DEVNULL)
   cap=cv2.VideoCapture(str(dest));frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));fps=cap.get(cv2.CAP_PROP_FPS);assert frames==item['result']['steps'],(frames,item['result']['steps'])
   ticks=sorted(set(min(frames-1,n) for n in [0,90,150,240,360,frames-1]));thumbs=[]
   for tick in ticks:
    cap.set(cv2.CAP_PROP_POS_FRAMES,tick);ok,frame=cap.read();assert ok
    im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));im.thumbnail((672,378));panel=Image.new('RGB',(672,im.height+24),'white');panel.paste(im,(0,24));ImageDraw.Draw(panel).text((4,4),f"{item['role']} seed{item['seed']} tick{tick} success={item['result']['success']}",fill='black');thumbs.append(panel)
   cap.release();rows.append(dict(role=item['role'],seed=item['seed'],success=item['result']['success'],source=str(src),published=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),decoded_all=True,frames=frames,fps=fps,ticks=ticks));panels.append(thumbs)
 width=max(sum(p.width for p in row) for row in panels);height=sum(max(p.height for p in row) for row in panels);canvas=Image.new('RGB',(width,height),'white');y=0
 for row in panels:
  x=0
  for p in row:canvas.paste(p,(x,y));x+=p.width
  y+=max(p.height for p in row)
 canvas.save(OUT/'development_video_review.jpg',quality=88)
 write(OUT/'media.json',dict(videos=rows,selection='One recorded candidate/baseline pair per role; outcome difference first, candidate success second, earliest seed last. Illustrative viewing only, all32 counted in evaluation.',format='Original full natural-termination60fps files copied byte-for-byte.'))
 print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
