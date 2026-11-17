"""View existing natural-termination videos, without new task rollouts."""
import cv2
from PIL import Image,ImageDraw
from .common import *
from scripts.tight_layout_20261116.publication import write

def main():
 ticks=[50,60,72,84,96,110];canvas=Image.new('RGB',(6*448,6*156),(245,245,245));draw=ImageDraw.Draw(canvas);receipts=[]
 for ci,(role,seed,part) in enumerate(CASES):
  for vi,variant in enumerate(['baseline','candidate']):
   p=path(role+('_baseline' if variant=='baseline' else ''),seed,part)/'autonomous.mp4';cap=cv2.VideoCapture(str(p));count=0;selected={}
   while True:
    ok,frame=cap.read()
    if not ok:break
    if count in ticks:selected[count]=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    count+=1
   cap.release();assert set(selected)==set(ticks)
   for ti,t in enumerate(ticks):
    img=selected[t];img.thumbnail((448,126));x=ti*448;y=(ci*2+vi)*156;canvas.paste(img,(x,y+26));draw.text((x+3,y+4),f'{role} {seed} {variant} frame {t}',fill=(0,0,0))
   receipts.append(dict(path=str(p),decoded_frames=count,selected_frames=ticks,frame_mapping='Recorder appends observation camera before action at same tick; inspect numeric after-action separately'))
 canvas.save(OUT/'original_video_ticks.jpg',quality=90);write(OUT/'video_review.json',receipts)
if __name__=='__main__':main()
