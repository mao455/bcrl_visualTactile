"""Small human-readable camera delivery; original training pixels untouched."""
import argparse
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
from .compact import EpisodeFile, write_video
from .camera_review import data_path
from .trial_utils import write_json

FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

def rotation(pose):return Rotation.from_quat(np.asarray(pose)[...,[4,5,6,3]])

def source_audit(directory,compact):
    """Decode original delivered video; compare pixels to ALL possible views."""
    with EpisodeFile(data_path(directory)) as f:
        m=json.loads(f.attrs['metadata_json']);g=next(iter(f['episodes'].items()))[1]
        n=int(g.attrs['length']);h,w=m['cameras'][0]['resolution'];result={}
        files=[(Path(directory)/'review/synchronized.mp4',None)]
        files += [(Path(compact)/f'camera_{i}.mp4',i) for i in range(3)]
        for path,view in files:
            errors=np.zeros((3,3) if view is None else (1,3));count=0
            with av.open(str(path)) as c:
                for t,frame in enumerate(c.decode(video=0)):
                    if t>=n:raise ValueError('Extra video frame')
                    decoded=frame.to_ndarray(format='rgb24')
                    raw=np.rint(g['obs/image'][t,:,:3]*255).astype('u1').transpose(0,2,3,1)
                    for j in range(3 if view is None else 1):
                        column=decoded[:h,j*w:(j+1)*w] if view is None else decoded
                        errors[j]+=np.abs(column.astype(float)[None]-raw.astype(float)).mean((1,2,3))
                    count+=1
            if count!=n:raise ValueError('Missing video frame')
            means=errors/n;expected=np.arange(3) if view is None else [view]
            if not np.array_equal(means.argmin(1),expected):raise ValueError('Video source/order mismatch')
            result[str(path)]=dict(frames=count,column_to_raw_MAE=means.tolist(),best_raw_view=means.argmin(1).tolist())
        poses=g['obs/camera_pose_w'][:];parent=g['obs/camera_parent_pose_w'][:]
        local_pos=rotation(parent).inv().apply(poses[:,2,:3]-parent[:,:3])
        local_rot=rotation(parent).inv()*rotation(poses[:,2])
        mount=m['camera_rig']['wrist'];q=np.array(mount['quat_parent_ros_wxyz'])
        expected=Rotation.from_quat(q[[1,2,3,0]])
        result['dynamic_mount']=dict(max_translation_error_m=float(np.abs(local_pos-mount['pos_parent_m']).max()),
            max_rotation_error_rad=float((expected.inv()*local_rot).magnitude().max()),
            wrist_start_end_m=float(np.linalg.norm(poses[-1,2,:3]-poses[0,2,:3])),
            fixed_max_position_change_m=float(np.abs(poses[:,:2,:3]-poses[:1,:2,:3]).max()))
        return result


def deliver(directory,out):
    out=Path(out);out.mkdir(exist_ok=False)
    font=ImageFont.truetype(FONT,24);small=ImageFont.truetype(FONT,19)
    with EpisodeFile(data_path(directory)) as f:
        m=json.loads(f.attrs['metadata_json']);g=next(iter(f['episodes'].items()))[1];n=int(g.attrs['length'])
        h,w=m['cameras'][0]['resolution'];labels=['固定高位（左）','固定侧视（中）','腕部随动（右）']
        def canvas(t,depth=False):
            im=g['obs/image'][t];valid=g['obs/depth_valid'][t]
            rgb=np.rint(im[:,:3]*255).astype('u1').transpose(0,2,3,1)
            image=Image.new('RGB',(3*w,h+88+(h if depth else 0)),(25,25,30));draw=ImageDraw.Draw(image)
            for v in range(3):
                draw.text((v*w+12,5),labels[v],font=font,fill='white')
                draw.text((v*w+12,38),f"{m['camera_ids'][v]}  step {t}  {g['obs/camera_time_s'][t,v]:.3f}s",font=small,fill='white')
                image.paste(Image.fromarray(rgb[v]),(v*w,88))
                if depth:
                    d=(im[v,3]/1.5*255).clip(0,255).astype('u1');d=np.stack([d]*3,-1)
                    d[~valid[v,0]]=[255,0,255]
                    image.paste(Image.fromarray(d),(v*w,88+h))
            return image
        write_video(out/'three_view_demo_cn.mp4',(np.asarray(canvas(t)) for t in range(n)),fps=60)
        # Every 15th native step, plus last, avoids cherry-picking only good views.
        indices=sorted(set([*range(0,n,15),n-1]))
        sheet=Image.new('RGB',(3*w,(h+88)*len(indices)))
        for row,t in enumerate(indices):sheet.paste(canvas(t),(0,row*(h+88)))
        sheet.save(out/'all_stages_rgb.png')
        indices_depth=sorted(set([0,n//3,2*n//3,n-1]))
        sheet=Image.new('RGB',(3*w,(2*h+88)*len(indices_depth)))
        for row,t in enumerate(indices_depth):sheet.paste(canvas(t,True),(0,row*(2*h+88)))
        sheet.save(out/'depth_validity.png')
        canvas(n//3).save(out/'demo_still.png')
        # Camera frusta computed from recorded ROS poses/K, projected in XY/XZ.
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.family']='DejaVu Sans'
        t=n//3;poses=g['obs/camera_pose_w'][t];fig,axes=plt.subplots(1,2,figsize=(13,5))
        for ax,dims,title in zip(axes,[(0,1),(0,2)],['XY top view','XZ side view']):
            for v,color in enumerate(['tab:blue','tab:green','tab:orange']):
                p=poses[v,:3];R=rotation(poses[v]);K=np.asarray(m['cameras'][v]['intrinsics'])[0]
                corners=np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]])@np.linalg.inv(K).T
                corners=p+R.apply(corners)*(.55 if v<2 else .3)
                for corner in corners:ax.plot([p[dims[0]],corner[dims[0]]],[p[dims[1]],corner[dims[1]]],color=color,alpha=.45)
                loop=np.vstack([corners,corners[0]])
                ax.plot(loop[:,dims[0]],loop[:,dims[1]],color=color,alpha=.5)
                tip=p+R.apply([0,0,.22]);ax.annotate('',tip[list(dims)],p[list(dims)],arrowprops=dict(arrowstyle='->',color=color))
                ax.scatter(*p[list(dims)],color=color,label=m['camera_ids'][v])
            for key,marker,color in [('object_pose_w','o','red'),('hole_pose_w','s','black'),('palm_body_pose_w','x','purple')]:
                p=g['transition/diagnostics_before/'+key][t,:3];ax.scatter(*p[list(dims)],marker=marker,color=color,label=key)
            ax.set(title=title+' (metres)',xlabel='world X',ylabel='world '+('Y' if dims[1]==1 else 'Z'))
            ax.axis('equal');ax.grid(alpha=.2)
        axes[1].legend(fontsize=8);fig.suptitle(f'Recorded camera centres, optical axes and frusta; step {t}\nFrustum depth chosen for display; occlusion and visible coverage are NOT inferred')
        fig.tight_layout();fig.savefig(out/'scene_camera_overview.png',dpi=150);plt.close(fig)
        # Real pixels enlarged, no fabricated CAD or gap length.
        rgb=np.rint(g['obs/image'][0,0,:3]*255).astype('u1').transpose(1,2,0)
        crop=Image.fromarray(rgb).resize((896,504))
        image=Image.new('RGB',(896,650),'white');image.paste(crop,(0,0));draw=ImageDraw.Draw(image)
        for i,line in enumerate(['原始高位 RGB：腕部连接区域见臂末端与手背之间',
            '固定链：link_7 → link_ee（空）→ sharpa_mount（无可见网格）→ 手掌',
            '固定偏移 0.045 m + 0.05 m；不是实测可见缝隙长度。',
            '相机没有外壳网格；本轮未修改机器人连接、质量或碰撞。']):
            draw.text((10,510+33*i),line,font=small,fill='black')
        image.save(out/'real_mount_region.png')
        write_json(out/'index.json',dict(source=str(directory),ids=m['camera_ids'],steps=n,
            sampled_steps=indices,depth_invalid_color='magenta',depth_display_max_m=1.5,
            annotation='separate header; RGB pixels untouched before viewing-only MP4 encoding'))


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();deliver(a.directory,a.output)
if __name__=='__main__':main()
