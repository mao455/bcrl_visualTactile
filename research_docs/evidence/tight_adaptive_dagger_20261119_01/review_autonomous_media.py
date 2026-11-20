"""Decode representative existing autonomous videos and extract labelled three-view sheets."""
import json
from pathlib import Path
import av
from PIL import Image,ImageDraw

PUBLIC=Path(__file__).resolve().parent
ROOT=Path('/home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_adaptive_dagger_20261119_01')
CASES={
 'A_lost_parent':('paired2_wave00_A','A_parent_capability_layout10_train_28817000',[0,60,120,180,240,305]),
 'A_lost_new':('paired2_wave00_A','A_new_capability_layout10_train_28817000',[0,60,120,180,240,360,600,825]),
 'B_lost_parent':('paired2_wave01_B','B_parent_capability_cycle_dev_28826000',[0,60,120,180,240,327]),
 'B_lost_new':('paired2_wave01_B','B_new_capability_cycle_dev_28826000',[0,60,120,180,240,360,456]),
 'A_gained':('paired2_wave14_A','A_new_capability_layout10_train_28817028',[0,60,120,180,240,299]),
 'B_gained':('paired2_wave22_B','B_new_adaptive_final_28838006',[0,60,120,180,240,275]),
}


def export():
 output=ROOT/'media/autonomous_round01';output.mkdir(parents=True,exist_ok=True)
 receipts={}
 for label,(wave,case,ticks) in CASES.items():
  receipt=json.loads((PUBLIC/'round01/evaluation'/wave/'videos.json').read_text())
  metadata=next(r for r in receipt['rows'] if r['case']==case)
  source=Path(metadata['result']).parent/'autonomous.mp4'
  frames=[];count=0
  with av.open(str(source)) as container:
   for index,frame in enumerate(container.decode(video=0)):
    count+=1
    if index in ticks:
     rgb=frame.to_image().convert('RGB');rgb.thumbnail((1008,237))
     panel=Image.new('RGB',(1008,263),(22,22,22));panel.paste(rgb,(0,26))
     ImageDraw.Draw(panel).text((8,7),f'{label} | tick {index} | {case}',fill='white');frames.append(panel)
  assert count==metadata['frames'] and len(frames)==len(ticks)
  sheet=Image.new('RGB',(1008,263*len(frames)))
  for index,panel in enumerate(frames):sheet.paste(panel,(0,index*263))
  target=output/(label+'.jpg');assert not target.exists();sheet.save(target,quality=92)
  receipts[label]=dict(video=str(source),sheet=str(target),frames=count,view_ticks=ticks,
                       success=metadata['success'],case=case,scope='Existing actual pre-action RGB, no physics rerun; representative outcome-selected viewing only')
 (output/'decoded.json').write_text(json.dumps(receipts,indent=2)+'\n')
 print(json.dumps(receipts))


if __name__=='__main__':export()
