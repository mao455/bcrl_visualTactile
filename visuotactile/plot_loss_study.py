"""CPU-only figures from recorded metrics; no model or trajectory loading."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .training_resources import atomic_json


def plot(root, final=False):
    output = root/'figures'
    output.mkdir(exist_ok=True)
    session = json.loads((root/'session.json').read_text())
    histories = {line:json.loads((root/line/'metrics.json').read_text())
                 for line in session['dp_lines'] if (root/line/'metrics.json').exists()}
    colors = dict(zip(session['dp_lines'],('tab:blue','tab:orange','tab:green','tab:red')))
    metadata = dict(source_root=session['source_root'],lines={},scope='recorded train/validation only; no test-driven selection')
    rows = []
    for line,history in histories.items():
        metadata['lines'][line] = dict(source_checkpoint=history[0]['source_checkpoint'],
                                     last_additional_updates=history[-1]['additional_updates'])
        for record in history:
            for split in ('validation','train_diagnostic'):
                measured = record[split]
                row = dict(line=line,epoch=record['epoch'],additional_updates=record['additional_updates'],
                    split=split,ema=True,episodes=measured['episodes'],source_checkpoint=record['source_checkpoint'],
                    diffusion_loss_A=measured['diffusion_loss']['A'],diffusion_loss_B=measured['diffusion_loss']['B'],
                    command_saturation_fraction=measured['command_saturation_fraction'])
                for space,parts in measured['metrics'].items():
                    for part,scopes in parts.items():
                        for scope,value in scopes.items():
                            row[f'{space}_{part}_{scope}'] = value
                rows.append(row)
    if rows:
        with (output/'dp_metrics.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def finish(figure,name,title,x_scope='additional optimizer updates'):
        figure.suptitle(title+'\nInitialization: original epoch40 EMA; x-axis = '+x_scope,fontsize=10)
        for axis in figure.axes:
            axis.grid(alpha=.2)
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=6)
        figure.tight_layout(rect=(0,0,1,.94))
        figure.savefig(output/name,dpi=150)
        plt.close(figure)

    figure,axes = plt.subplots(1,2,figsize=(11,4))
    for line,history in histories.items():
        objective = line[-1]
        axis = axes[0 if objective=='A' else 1]
        updates = [row['additional_updates'] for row in history]
        axis.plot(updates,[row['validation']['diffusion_loss'][objective] for row in history],label=line+' val40 EMA',color=colors[line])
        axis.plot(updates,[row['train_diagnostic']['diffusion_loss'][objective] for row in history],label=line+' train16 EMA',color=colors[line],linestyle='--')
        trained = [row for row in history if 'train' in row]
        axis.plot([row['additional_updates'] for row in trained],[row['train']['loss'] for row in trained],
                  label=line+' train320 evolving raw',color=colors[line],linestyle=':')
        axis.set_title('Loss '+objective+' (its own definition)')
        axis.set_ylabel('Normalized x0 denoising MSE')
        axis.set_xlabel('Additional updates')
    finish(figure,'dp_losses.png','Train/validation losses; A and B magnitudes are not a performance ranking')

    for space,unit in (('command','normalized native command'),('reference','rad'),('normalized','normalized absolute reference')):
        figure,axes = plt.subplots(2,2,figsize=(11,7))
        for column,part in enumerate(('arm','hand')):
            for row_index,scope in enumerate(('prefix12_rmse','full_rmse')):
                axis = axes[row_index,column]
                for line,history in histories.items():
                    axis.plot([row['additional_updates'] for row in history],
                              [row['validation']['metrics'][space][part][scope] for row in history],label=line,color=colors[line])
                axis.set_title(part+' '+scope)
                axis.set_ylabel('RMSE ('+unit+')')
                axis.set_xlabel('Additional updates')
        finish(figure,'dp_'+space+'_rmse.png','Validation40 EMA; first12 executed prefix versus full H64')

    figure,axes = plt.subplots(2,5,figsize=(16,6))
    for column,part in enumerate(('thumb','index','middle','ring','pinky')):
        for row_index,scope in enumerate(('prefix12_rmse','full_rmse')):
            for line,history in histories.items():
                axes[row_index,column].plot([row['additional_updates'] for row in history],
                    [row['validation']['metrics']['command'][part][scope] for row in history],label=line,color=colors[line])
            axes[row_index,column].set_title(part+' '+scope)
            axes[row_index,column].set_ylabel('Native command RMSE')
    finish(figure,'dp_fingers.png','Validation40 EMA; joint-mean errors within each anatomical finger')

    selected = {line:min(history,key=lambda row:row['validation']['selection_mse']) for line,history in histories.items()}
    figure,axes = plt.subplots(2,2,figsize=(11,7))
    for row_index,space in enumerate(('command','reference')):
        for column,part in enumerate(('arm','hand')):
            for line,record in selected.items():
                axes[row_index,column].plot(np.arange(64)/60,record['validation']['horizon'][space][part],
                    label=line+' +'+str(record['additional_updates']),color=colors[line])
            axes[row_index,column].axvline(11/60,color='grey',linestyle=':')
            axes[row_index,column].set_xlabel('Future offset (s); dotted = last committed position')
            axes[row_index,column].set_ylabel('Native command RMSE' if space=='command' else 'Reference RMSE (rad)')
            axes[row_index,column].set_title(part)
    finish(figure,'dp_horizon.png','Validation-selected EMA; horizon-specific valid counts (not zero-filled tails)',
           x_scope='future offset in seconds')

    figure,axes = plt.subplots(1,3,figsize=(13,4))
    for line,history in histories.items():
        trained = [row for row in history if 'train' in row]
        updates = [row['additional_updates'] for row in trained]
        axes[0].plot(updates,[row['train']['lr'] for row in trained],label=line,color=colors[line])
        axes[1].plot(updates,[row['train']['gradient_norm'] for row in trained],label=line,color=colors[line])
        axes[2].plot([row['additional_updates'] for row in history],
                     [row['validation']['command_saturation_fraction'] for row in history],label=line,color=colors[line])
    for axis,label in zip(axes,('Learning rate (epoch end)','Mean pre-clip gradient norm','Validation40 EMA saturation fraction')):
        axis.set_ylabel(label)
        axis.set_xlabel('Additional updates')
    finish(figure,'dp_optimizer.png','Fresh optimizer and learning-rate schedule; identical paired update budget')

    figure,axes = plt.subplots(1,2,figsize=(11,4))
    for axis,part in zip(axes,('arm','hand')):
        for line,history in histories.items():
            for split,style in (('validation','-'),('train_diagnostic','--')):
                axis.plot([row['additional_updates'] for row in history],
                    [row[split]['metrics']['command'][part]['prefix12_rmse'] for row in history],
                    label=line+' '+('val40' if split=='validation' else 'fixed train16'),linestyle=style,color=colors[line])
        axis.set_title(part+' prefix12')
        axis.set_ylabel('Native command RMSE; EMA')
    finish(figure,'dp_generalization.png','Matched evaluation definition; fixed train16 is not full train320')

    for line,record in selected.items():
        sample = record['validation']['representative']
        expert = np.asarray(sum([block['expert'] for block in sample['blocks']],[]))
        predicted = np.asarray(sum([block['predicted'] for block in sample['blocks']],[]))
        figure,axes = plt.subplots(4,1,figsize=(10,8),sharex=True)
        for axis,joint in zip(axes,(0,3,7,12)):
            axis.plot(np.arange(len(expert))/60,expert[:,joint],label='Expert submitted command')
            axis.plot(np.arange(len(predicted))/60,predicted[:,joint],label=line+' EMA predicted command')
            axis.set_ylabel('joint'+str(joint)+' command')
        axes[-1].set_xlabel('Recorded episode time (s)')
        figure.suptitle(line+' fixed validation episode '+sample['episode']+'\nTeacher-forced future target conversion; prefix12 only',fontsize=9)
        for axis in axes:
            axis.legend(fontsize=7)
            axis.grid(alpha=.2)
        figure.tight_layout(rect=(0,0,1,.94))
        figure.savefig(output/(line+'_validation_trace.png'),dpi=150)
        plt.close(figure)
    figure,axes = plt.subplots(1,3,figsize=(13,4))
    for group in ('G00-new','G10-new'):
        history = json.loads((Path(session['source_root'])/(group+'_full')/'metrics.json').read_text())
        for axis,key in zip(axes,('command_arm_rmse','command_hand_rmse','first12_command_rmse')):
            axis.plot([row['updates'] for row in history],[row['validation'][key] for row in history],label=group)
            axis.set_ylabel(key+' (native command)')
            axis.set_xlabel('Original optimizer updates')
            axis.legend()
            axis.grid(alpha=.2)
    figure.suptitle('Original40 epochs: validation40 EMA only; missing historical training/finger curves are not reconstructed')
    figure.tight_layout(rect=(0,0,1,.92))
    figure.savefig(output/'original40_validation.png',dpi=150)
    plt.close(figure)
    atomic_json(output/'metadata.json',metadata)
    if final:
        atomic_json(root/'plots_complete.json',dict(files=sorted(path.name for path in output.iterdir())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    arguments = parser.parse_args()
    plot(arguments.root,arguments.final)
