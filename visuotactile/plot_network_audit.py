"""Reproducible architecture diagrams and plots from audited statistics."""
import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

from .compact import sha256
from .training_resources import atomic_json


def curve_sources(source, output):
    def write(name, rows):
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    horizon_rows,trace_rows,original_rows=[],[],[]
    for line in ('G00-A','G00-B','G10-A','G10-B'):
        history=json.loads((source/line/'metrics.json').read_text())
        selected=min(history,key=lambda row:row['validation']['selection_mse'])
        for space in ('command','reference'):
            for part in ('arm','hand'):
                for offset,value in enumerate(selected['validation']['horizon'][space][part]):
                    horizon_rows.append(dict(line=line,split='validation40',additional_updates=selected['additional_updates'],
                        checkpoint_sha256=sha256(source/line/'best.pt'),space=space,part=part,offset=offset,time_s=offset/60.,rmse=value))
        sample=selected['validation']['representative']
        expert=sum([block['expert'] for block in sample['blocks']],[])
        predicted=sum([block['predicted'] for block in sample['blocks']],[])
        for step,(expected,actual) in enumerate(zip(expert,predicted)):
            for joint in (0,3,7,12):
                trace_rows.append(dict(line=line,split='validation_representative',identity=sample['episode'],
                    additional_updates=selected['additional_updates'],step=step,time_s=step/60.,joint=joint,
                    expert=expected[joint],predicted=actual[joint]))
    session=json.loads((source/'session.json').read_text())
    for group in ('G00-new','G10-new'):
        history=json.loads((Path(session['source_root'])/(group+'_full')/'metrics.json').read_text())
        for row in history:
            original_rows.append(dict(group=group,updates=row['updates'],split='validation40 EMA',
                **{key:row['validation'][key] for key in ('command_arm_rmse','command_hand_rmse','first12_command_rmse')}))
    write('dp_horizon.csv',horizon_rows)
    write('dp_validation_examples.csv',trace_rows)
    write('original40_validation.csv',original_rows)
    progress,residual=[],[]
    for category in ('dp','residual'):
        directory=source/(category+'_closed_loop')
        trials=json.loads((directory/'trials.json').read_text())
        for trial in trials:
            with np.load(directory/trial['trace']) as trace:
                time_values=(trace['time_before']-trace['time_before'][0]).reshape(-1)
                if category=='dp':
                    with np.load(directory/(trial['attempt']+'_initial.npz')) as initial:
                        height=trace['object_pose'][:,2]-float(initial['object_initial_z_m'])
                    contact=np.cumsum(trace['gate'])/60.
                    for step in range(len(time_values)):
                        progress.append(dict(line=trial['line'],seed=trial['seed'],step=step,time_s=float(time_values[step]),
                            height_relative_reset_m=float(height[step]),finger_distance_m=float(trace['finger_distance'][step]),
                            cumulative_contact_s=float(contact[step])))
                else:
                    height=trace['object_pose'][:,2]-trace['object_pose'][0,2]
                    force=np.linalg.norm(trace['force'],axis=-1).max(axis=(1,2))
                    correction=np.abs(trace['command']-trace['base']).mean(-1)*trace['student']
                    applied=np.linalg.norm(trace['applied_force_w_N'],axis=-1).max(axis=(1,2))
                    for step in range(len(time_values)):
                        residual.append(dict(variant=trial['variant'],seed=trial['seed'],step=step,time_s=float(time_values[step]),
                            height_vs_first_after_step_m=float(height[step]),peak_finger_force_N=float(force[step]),
                            applied_correction_mean_abs=float(correction[step]),applied_world_force_N=float(applied[step])))
    write('dp_closed_loop_curves.csv',progress)
    write('transformer_closed_loop_curves.csv',residual)
    atomic_json(output/'curve_sources.json',dict(scope='existing plotted scalar data exported without model reevaluation',
        files={name:sha256(output/name) for name in ('dp_horizon.csv','dp_validation_examples.csv','original40_validation.csv',
            'dp_closed_loop_curves.csv','transformer_closed_loop_curves.csv')},
        checkpoint_sha256={line:sha256(source/line/'best.pt') for line in ('G00-A','G00-B','G10-A','G10-B')},
        source_run=source.name))


def diagram(output,name,title,nodes,edges,columns=3):
    rows=max(position[1] for _,position,_ in nodes)+1
    figure,axis=plt.subplots(figsize=(columns*5,rows*1.9+1))
    axis.set_xlim(-.1,columns*5)
    axis.set_ylim(-.2,rows*1.9+.4)
    axis.axis('off')
    positions={}
    grid={identity:position for identity,position,_ in nodes}
    for identity,(column,row),label in nodes:
        left,bottom=column*5, (rows-1-row)*1.9
        positions[identity]=(left+2.3,bottom+.72)
        axis.add_patch(FancyBboxPatch((left,bottom),4.6,1.45,boxstyle='round,pad=0.08',
            facecolor='#edf5fb',edgecolor='#276180',linewidth=1.2))
        axis.text(left+2.3,bottom+1.31,identity.upper(),ha='center',va='center',fontsize=8,color='#276180',weight='bold')
        axis.text(left+2.3,bottom+.75,label,ha='center',va='center',fontsize=9)
        incoming=[before for before,after in edges if after==identity]
        axis.text(left+2.3,bottom+.12,'From: '+', '.join(incoming) if incoming else 'Source / specification',
                  ha='center',va='center',fontsize=7,color='#475c69')
    for before,after in edges:
        if sum(abs(first-second) for first,second in zip(grid[before],grid[after]))!=1:
            continue
        first,second=positions[before],positions[after]
        delta=np.asarray(second)-np.asarray(first)
        if abs(delta[0])<.01:
            direction=np.sign(delta[1])
            start=(first[0],first[1]+.8*direction)
            end=(second[0],second[1]-.8*direction)
        else:
            direction=np.sign(delta[0])
            start=(first[0]+2.4*direction,first[1])
            end=(second[0]-2.4*direction,second[1])
        axis.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=12,color='#35434f',linewidth=1))
    figure.suptitle(title+'\nActual checkpoint; all connections indexed by From; arrows show adjacent connections',fontsize=13)
    figure.tight_layout(rect=(0,0,1,.97))
    for extension in ('png','svg'):
        if extension=='svg':
            with matplotlib.rc_context({'svg.fonttype':'none','svg.hashsalt':'network-audit-20261031'}):
                buffer=io.StringIO()
                figure.savefig(buffer,format='svg',metadata={'Date':None})
            (output/(name+'.svg')).write_text('\n'.join(line.rstrip() for line in buffer.getvalue().splitlines())+'\n')
        else:
            figure.savefig(output/(name+'.'+extension),dpi=140)
    plt.close(figure)


def architecture(output):
    nodes=[
        ('rgb',(0,0),'3 RGB views, uint8 [3,3,252,448]\n/255; ImageNet normalization\nNo crop/resize; shared frozen DINO'),
        ('dino',(0,1),'ViT-S/14: Conv14 stride14, 3->384\n576 patches + CLS; positional interpolation\n12 pre-LN blocks; 6 heads x64; FFN1536 GELU'),
        ('patch',(0,2),'Final LN384; discard CLS\n[3,384,18,32], stored FP16\nRGB: Conv1x1 384->32, GN4, SiLU'),
        ('depth',(1,0),'3 metric depth + validity\nvalid-weighted pool18x32; depth/2m\nConv3x3 2->16, GN4, SiLU'),
        ('spatial',(1,1),'Concat RGB32 + depth16\nAdaptive avgpool3x5 ->720 / view\nLinear720->128; LN128; SiLU'),
        ('held',(1,2),'Concat3 views ->384, invalid views zero\nNew visual frame every12 ticks (5Hz)\nHold last frame for native60Hz memory'),
        ('visual',(0,3),'11-frame visual history (2s)\n384 +3 camera ages +3valid =390\nPacked GRU390->128, one layer'),
        ('prop',(1,3),'121tick q/qd history:58 +age =59\nTrain-only mean/std; GRU59->64\n121tick past commands61 ->GRU61->64'),
        ('touch',(2,0),'G10 only: 5x(world3+local3)\nsigned log1p(N) +age+valid =32\n2 real120Hz samples/native tick'),
        ('touchhist',(2,1),'G10 tactile history:242 samples\nPacked GRU32->64, one layer\nCurrent2x32 flattened64; no FK here'),
        ('memory',(2,3),'Every native tick: visual384+6\n+q/qd58+past action61 [+touch64]\nGRU509/573->128, one persistent layer'),
        ('cond',(1,4),'Concat history128+64+64+memory128\n[+G10 touch64] =384 /448\nLinear->512; LN512; SiLU'),
        ('time',(2,4),'Diffusion index: sinusoidal128\nLinear128->512; Mish; Linear512->128\nConcat condition512 ->FiLM640'),
        ('down',(0,5),'Noisy normalized sample [B,29,64]\n2 ResBlocks29->256 @64; down to32\n2 ResBlocks256->512 @32; down to16'),
        ('mid',(1,5),'2 ResBlocks512->1024 @16\nNo further downsample; save skips\nMiddle:2 ResBlocks1024->1024 @16'),
        ('up',(2,5),'Concat1024 skip:2048->512 @16\n2 ResBlocks; upsample to32\nConcat512 skip:1024->256; 2 blocks; up64'),
        ('block',(0,6),'Each ResBlock: Conv5/GN8/Mish\nFiLM scale*x+bias from640\nConv5/GN8/Mish + identity/Conv1 skip'),
        ('out',(1,6),'Conv5 256->256/GN8/Mish\nConv1 256->29; transpose [B,64,29]\nUnused earliest256-channel down skip'),
        ('sample',(2,6),'100 training noise levels; sample/x0\nDDIM16 eta0, clip_sample=False\nInverse normalizer ->H64 references(rad)')]
    edges=[('rgb','dino'),('dino','patch'),('depth','spatial'),('patch','spatial'),('spatial','held'),('held','visual'),
        ('touch','touchhist'),('touchhist','memory'),('touchhist','cond'),('held','memory'),('prop','memory'),
        ('visual','cond'),('prop','cond'),('memory','cond'),('cond','time'),('time','down'),('time','mid'),('time','up'),
        ('down','mid'),('down','up'),('mid','up'),('up','out'),('out','sample')]
    diagram(output,'architecture_dp','G00-B / G10-B: trainable policy71,166,285 /71,242,445; shared frozen DINO22,056,576',nodes,edges)
    nodes=[
        ('base',(0,0),'Current native base command29\nIndependent Linear29->128 token\nPlan cursor=t mod12; original plan intact'),
        ('plan',(1,0),'Remaining absolute H64 plan, radians\nNormalize; Linear29->128\nLinear(relative index/60)->128 +type5'),
        ('motion',(2,0),'Motion177 = normalized q/qd58\n+normalized target29 +tracking29(rad)\n+past issued29/delayed29/ages2/valid1'),
        ('vision',(0,1),'Frozen G00 spatial context384\nLinear384->128 +type3\nNo new DINO or raw image branch'),
        ('force',(1,1),'Per finger20 = world3/local3(N)\n+FK xyz3(m)/rotation9 +age1/valid1\nForce uses signed log1p; Masked zeros6'),
        ('motiontok',(2,1),'Motion token: Linear177->128\nLayerNorm128; SiLU +type2\nSource strictly current/past observed data'),
        ('touch',(1,2),'Recent13ticks x2samples x5 fingers\n130 tokens; Linear20->128/LN/SiLU\n+finger Emb5x128 +type6; padding mask'),
        ('memory',(2,2),'Native current motion177+vision384\n+current tactile2x5x20=200 ->761\nGRU761->128, 1layer; persistent every60Hz'),
        ('query',(0,2),'6 learned output queries[6,128]\n+type0; current token type1\nMemory token type4; type Emb7x128'),
        ('tokens',(0,3),'204 tokens x128\n6query+1base+1motion+1visual+1memory\n+64plan+130touch; key-padding only'),
        ('attn',(1,3),'Encoder layer1: pre-LN128\nQKV Linear128->384;4heads x32\nsoftmax(QK^T/sqrt32)V; proj128 +skip'),
        ('ffn',(2,3),'pre-LN128 ->Linear128->512\nGELU ->Linear512->128 +skip\nAll dropout0; then independent layer2'),
        ('mask',(0,4),'No triangular attention mask\nAll sensor tokens <=current time\nFuture tokens are nominal plan / queries'),
        ('head',(1,4),'Select first6 query outputs\nLayerNorm128 ->Linear128->29\n[6,29] raw unbounded delta, FP32'),
        ('exec',(2,4),'10Hz refresh; consume cursor=t mod6\ng=valid & max finger force norm>0.1N\nwhere(g, clip(base+delta,-1,1), base)')]
    diagram(output,'architecture_transformer','Force / Masked Transformer:828,189 trainable parameters; same architecture, different continuous-force visibility',nodes,
        [('motion','motiontok'),('force','touch'),('motion','memory'),('force','memory'),('vision','memory'),
         ('base','tokens'),('plan','tokens'),('vision','tokens'),('motiontok','tokens'),('query','tokens'),('touch','tokens'),('memory','tokens'),
         ('tokens','attn'),('attn','ffn'),('ffn','head'),('head','exec')])
    for mode in ('training','deployment'):
        teacher=mode=='training'
        nodes=[('obs',(0,0),'Full expert episode observations' if teacher else 'Student native sensor observations'),
            ('history',(1,0),'Past actual expert issued/delayed\nExplicit teacher forcing; never future' if teacher else 'Past actual submitted/delayed mixed\nReal env.step acknowledgment; t-1 only'),
            ('base',(0,1),'Frozen DINO + frozen G00-B\nNative tick recurrence; H64/5Hz'),
            ('res',(1,1),'Independent FP32 Transformer\nCurrent base +plan +motion +touch\nMemory60Hz / residual10Hz'),
            ('label',(2,0),'Future expert commands: labels only\ndelta*=expert_pre_delay-base\nValid contact-only balanced MSE' if teacher else 'No expert, future truth or rewards\nallowed in policy inputs'),
            ('loss',(2,1),'0.5 arm7 MSE +0.5 hand22 MSE\n+1e-4 mean(delta^2); raw supervision\nAdamW; no base gradients' if teacher else 'Native command: where(g,clip(u+d),u)\narm/hand order identical29 dimensions'),
            ('out',(0,2),'96tick TBPTT, carried/detached state\nEpisode boundary only resets memory' if teacher else 'Original delay queue +smoothing\nNative target radians ->120Hz physics'),
            ('loop',(1,2),'No executed student state in cache\nTeacher/student distribution gap remains' if teacher else 'New mixed history influences next nominal\nNo-contact exact current-base bypass\ndoes not isolate full trajectories')]
        diagram(output,'architecture_'+mode,mode.title()+' graph: explicit history and supervision boundaries',nodes,
            [('obs','base'),('history','base'),('obs','res'),('history','res'),('base','res'),('res','loss'),('label','loss'),
             ('loss','out'),('out','loop')]+([] if teacher else [('loop','history')]))
    nodes=[('rgb',(0,0),'Historical B0 RGB-D frozen encoder\nRGB ResNet18 through layer2\nDepth Conv2->32->64->128, GN/SiLU'),
        ('spatial',(0,1),'Fuse Conv256->64 GN8/SiLU\n16 soft keypoint moments x6=96\nGrid Conv64->16 pool3x5=240'),
        ('frame',(0,2),'Linear336->128 LN/SiLU per view\n3views384 +proprio64 +context64\nFrame512 ->trainable GRUCell512->128'),
        ('touch',(1,0),'Finger features27 +Embedding5x8\nLinear35->64/LN/SiLU->32/SiLU\nworld/local force,FK,delta,velocity,events'),
        ('touchmem',(1,1),'Shared GRUCell32->32 per finger120Hz\nConcat5x32 ->Linear160->128/SiLU\nLinear128->64/LN; select every2nd sample'),
        ('intent',(2,0),'Remaining B0 planH12 native commands\nGRUCell29->64 across plan slots\nPersistent GRUCell64->64 every60Hz'),
        ('motion',(2,1),'q/qd58 +past command29 +valid1\n+target29 +tracking29 =146\nPersistent GRUCell146->64 every60Hz'),
        ('head',(1,2),'Concat visual128+touch64+intent64+motion64\nLinear320->256/SiLU->128/SiLU->29\nscaled tanh; contact gate; native clip')]
    diagram(output,'architecture_historical_gru','Historical GRU residual:1,344,597 total /489,125 trainable; different base and targets',nodes,
        [('rgb','spatial'),('spatial','frame'),('touch','touchmem'),('frame','head'),('touchmem','head'),('intent','head'),('motion','head')])


def plots(output):
    report=json.loads((output/'network_audit.json').read_text())
    figure,axes=plt.subplots(1,2,figsize=(11,4))
    modes=('real','zero','reverse_past')
    for label in ('force','masked'):
        model=report['models'][label]
        for axis,part in zip(axes,('arm','hand')):
            axis.plot(modes,[model['intervention_metrics'][mode]['metrics']['contact_combined'][part] for mode in modes],marker='o',label=label)
            axis.set_ylabel(part+' contact composed-command RMSE')
            axis.grid(alpha=.2);axis.legend()
    figure.suptitle('Fixed-weight input intervention:4 selected train +4 validation episodes\nCPU FP32 recalculation; same base/plan/gate/boundary state; native-command units')
    figure.tight_layout(rect=(0,0,1,.88));figure.savefig(output/'force_interventions.png',dpi=150);plt.close(figure)
    for label in ('force','masked'):
        with (output/(label+'_fixed_trajectory.csv')).open() as stream:
            rows=list(csv.DictReader(stream))
        figure,axes=plt.subplots(4,2,figsize=(13,11),sharex=True)
        for axis,joint in zip(axes.flat,(0,3,7,12,16,20,24)):
            for key in ('expert','base','corrected'):
                axis.plot([float(row['time_s']) for row in rows],[float(row[f'{key}_{joint}']) for row in rows],label=key,linewidth=1)
            axis.set_title('canonical joint '+str(joint));axis.set_ylabel('Native command');axis.grid(alpha=.2);axis.legend(fontsize=8)
        axes[-1,-1].plot([float(row['time_s']) for row in rows],[row['gate']=='True' for row in rows])
        axes[-1,-1].set_ylabel('Observed contact gate')
        for axis in axes[-1]:axis.set_xlabel('Expert demonstration time (s)')
        figure.suptitle(label+' epoch5 /3730 updates; CPU FP32 recalculation, teacher-forced train sample\n'+rows[0]['identity']+'; checkpoint '+rows[0]['checkpoint_sha256'][:12])
        figure.tight_layout(rect=(0,0,1,.94));figure.savefig(output/(label+'_fixed_trajectory.png'),dpi=150);plt.close(figure)
    with (output/'dataset_episodes.csv').open() as stream:
        rows=list(csv.DictReader(stream))
    figure,axes=plt.subplots(2,2,figsize=(11,8))
    for split in ('train','validation','test'):
        subset=[row for row in rows if row['split']==split]
        axes[0,0].scatter([float(row['object_x_m']) for row in subset],[float(row['object_y_m']) for row in subset],s=12,label=split,alpha=.6)
        for axis,key in ((axes[0,1],'length'),(axes[1,0],'contact_fraction'),(axes[1,1],'saturation_fraction')):
            axis.hist([float(row[key]) for row in subset],bins=15,density=True,histtype='step',label=split)
            axis.set_xlabel(key);axis.set_ylabel('Density')
    axes[0,0].set_xlabel('Initial local object x (m)');axes[0,0].set_ylabel('Initial local object y (m)')
    for axis in axes.flat:axis.grid(alpha=.2);axis.legend()
    figure.suptitle('All400 successful demonstrations; fixed320/40/40 split\nOne geometry/camera/DR configuration; different native initial states')
    figure.tight_layout(rect=(0,0,1,.92));figure.savefig(output/'dataset_distribution.png',dpi=150);plt.close(figure)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--curve-source',type=Path)
    args=parser.parse_args()
    architecture(args.output)
    plots(args.output)
    if args.curve_source:
        curve_sources(args.curve_source,args.output.parent/'dp_loss_trt_20261031')
