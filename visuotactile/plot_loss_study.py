"""CPU-only figures from recorded metrics and bounded traces; no model loading."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from .training_resources import atomic_json


def plot_dp_lift_event(root, output, trials):
    selected = json.loads((root/'selected.json').read_text())['selected']['G10']
    trial = next((row for row in trials if row['line']==selected and row['seed']==981004 and 'trace' in row),None)
    if trial is None:
        return
    directory = root/'dp_closed_loop'
    with np.load(directory/trial['trace']) as trace, np.load(directory/(trial['attempt']+'_initial.npz')) as initial:
        peak = trial['height_peak_step']
        indices = np.arange(max(0,peak-60),min(trial['steps'],peak+61))
        height = trace['object_pose'][:,2]-float(initial['object_initial_z_m'])
        forces = np.linalg.norm(trace['force'],axis=-1).max(axis=1)
        difference = np.diff(trace['command'],axis=0,prepend=trace['command'][:1])
        arm_jump = np.sqrt(np.mean(difference[:,:7]**2,axis=1))
        hand_jump = np.sqrt(np.mean(difference[:,7:]**2,axis=1))
        external_force = np.linalg.norm(trace['applied_force_w_N'],axis=-1).max(axis=(1,2))
        external_torque = np.linalg.norm(trace['applied_torque_w_Nm'],axis=-1).max(axis=(1,2))
        rows = [dict(step=int(step),height_after_step_m=float(height[step]),gate_before_step=bool(trace['gate'][step]),
            arm_command_change_rms=float(arm_jump[step]),hand_command_change_rms=float(hand_jump[step]),
            actual_force_peak_N=float(external_force[step]),actual_torque_peak_Nm=float(external_torque[step]),
            **{f'sensor{finger}_force_before_N':float(forces[step,finger]) for finger in range(5)}) for step in indices]
    with (output/'dp_G10_lift_event.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    figure,axes = plt.subplots(5,1,figsize=(11,10),sharex=True)
    axes[0].plot(indices,height[indices],label=selected+' object height after step')
    axes[0].axhline(.05,color='grey',linestyle=':')
    bodies = json.loads((root/'metadata.json').read_text())['contact_bodies']
    for finger,body in enumerate(bodies):
        axes[1].plot(indices,forces[indices,finger],label=body)
    axes[2].plot(indices,arm_jump[indices],label='arm')
    axes[2].plot(indices,hand_jump[indices],label='hand')
    axes[3].plot(indices,external_force[indices],label='actual applied force')
    axes[4].plot(indices,external_torque[indices],label='actual applied torque')
    axes[3].set_ylim(0,max(1.,float(external_force.max())*1.1))
    axes[4].set_ylim(0,max(1.,float(external_torque.max())*1.1))
    for axis,label in zip(axes,('Height change\n(m)','Finger force\n(N)',
                               'Command change\nRMS','Applied force\n(N)','Applied torque\n(Nm)')):
        axis.set_ylabel(label)
        axis.axvline(peak,color='red',linestyle=':',alpha=.5)
        if trial['first_drop_below_2cm_after_5cm'] is not None:
            axis.axvline(trial['first_drop_below_2cm_after_5cm'],color='black',linestyle='--',alpha=.5)
        axis.grid(alpha=.2)
        axis.legend(fontsize=7)
    axes[-1].set_xlabel('Native control step; red = peak height, black = drop below reset +2cm')
    figure.suptitle(selected+' seed981004; measured associations, no identified drop cause\nForces observed before action; height after action; external wrench sampled within the step',fontsize=10)
    figure.tight_layout(rect=(0,0,1,.94))
    figure.savefig(output/'dp_G10_lift_event.png',dpi=150)
    plt.close(figure)


def plot_dp_closed_loop(root, output):
    directory = root/'dp_closed_loop'
    path = directory/'trials.json'
    if not path.exists():
        return
    trials = json.loads(path.read_text())
    seeds = sorted({row['seed'] for row in trials})
    figure,axes = plt.subplots(len(seeds),3,figsize=(14,3.5*len(seeds)),squeeze=False)
    summaries = []
    for row_index,seed in enumerate(seeds):
        for trial in trials:
            if trial['seed']!=seed or 'trace' not in trial:
                continue
            with np.load(directory/trial['trace']) as trace, np.load(directory/(trial['attempt']+'_initial.npz')) as initial:
                stamps = (trace['time_before']-trace['time_before'][0]).reshape(-1)
                values = (trace['object_pose'][:,2]-float(initial['object_initial_z_m']),
                          trace['finger_distance'],np.cumsum(trace['gate'])/60.)
                for axis,value in zip(axes[row_index],values):
                    axis.plot(stamps,value,label=trial['line'])
            summaries.append({name:trial[name] for name in ('line','seed','steps','success','paired','native_lift',
                'max_subgoals','max_object_height_change_m','longest_height_above_5cm_s','contact_steps',
                'max_contact_streak','minimum_finger_distance_m','actual_force_peak_N','actual_torque_peak_Nm')})
        for axis,label in zip(axes[row_index],('Object height relative reset (m)',
                                             'Mean fingertip to object origin (m)','Cumulative observed contact (s)')):
            axis.set_title(f'Seed {seed}')
            axis.set_ylabel(label)
            axis.set_xlabel('Native simulation time (s)')
            axis.grid(alpha=.2)
            axis.legend(fontsize=7)
    figure.suptitle('Autonomous validation-selected DP checkpoints; native termination\nAggregate contact and distance do not prove object grasp',fontsize=10)
    figure.tight_layout(rect=(0,0,1,.94))
    figure.savefig(output/'dp_closed_loop_progress.png',dpi=150)
    plt.close(figure)
    if summaries:
        with (output/'dp_closed_loop_outcomes.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(summaries[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(summaries)
    plot_dp_lift_event(root,output,trials)


def plot_residual(root, output):
    histories = {variant:json.loads((root/'transformer_training'/variant/'metrics.json').read_text())
                 for variant in ('force','masked') if (root/'transformer_training'/variant/'metrics.json').exists()}
    if not histories:
        return
    colors = {'force':'tab:blue','masked':'tab:orange'}
    rows = []
    for variant,history in histories.items():
        for record in history:
            for split in ('validation','train_diagnostic'):
                metrics = record[split]
                row = dict(variant=variant,epoch=record['epoch'],updates=record['updates'],split=split,ema=False,
                    loss=metrics['loss'],gate_fraction=metrics['gate_fraction'],tactile_valid_fraction=metrics['tactile_valid_fraction'],
                    delta_mean_abs=metrics['delta_mean_abs'],execution_clip_fraction=metrics['execution_clip_fraction'],
                    wrong_direction_saturation_fraction=metrics['wrong_direction_saturation_fraction'],
                    batched_forward_ms_per_anchor=metrics['batched_forward_ms_per_anchor'])
                for space,parts in metrics['metrics'].items():
                    for part,value in parts.items():
                        row[space+'_'+part+'_rmse'] = value
                rows.append(row)
    configs = {variant:json.loads((root/'transformer_training'/variant/'config.json').read_text()) for variant in histories}
    for row in rows:
        row['base_sha256'] = configs[row['variant']]['base_sha256']
    training_rows = [dict(variant=variant,epoch=record['epoch'],optimizer_updates=record['updates'],
        base_sha256=configs[variant]['base_sha256'],split='train320 evolving raw',ema=False,**record['train'])
        for variant,history in histories.items() for record in history if 'train' in record]
    if training_rows:
        with (output/'transformer_training.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(training_rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(training_rows)
    with (output/'transformer_metrics.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    def finish(figure,name,title):
        figure.suptitle(title+'\nG01 frozen validation-selected G00; raw residual weights, native-command units',fontsize=10)
        for axis in figure.axes:
            axis.grid(alpha=.2)
            axis.set_xlabel('Residual optimizer updates')
            if axis.get_legend_handles_labels()[0]:
                axis.legend(fontsize=6)
        figure.tight_layout(rect=(0,0,1,.93))
        figure.savefig(output/name,dpi=150)
        plt.close(figure)
    figure,axes = plt.subplots(1,2,figsize=(11,4))
    for variant,history in histories.items():
        for split,style in (('validation','-'),('train_diagnostic','--')):
            axes[0].plot([row['updates'] for row in history],[row[split]['loss'] for row in history],
                color=colors[variant],linestyle=style,label=variant+' '+split)
        trained = [row for row in history if 'train' in row]
        axes[0].plot([row['updates'] for row in trained],[row['train']['loss'] for row in trained],
            color=colors[variant],linestyle=':',label=variant+' train320 evolving')
        for part,style in (('arm','-'),('hand','--')):
            axes[1].plot([row['updates'] for row in history],
                [row['validation']['metrics']['contact_combined'][part] for row in history],
                color=colors[variant],linestyle=style,label=variant+' '+part)
    axes[0].set_ylabel('Contact balanced raw delta MSE +1e-4 amplitude')
    axes[1].set_ylabel('Val40 contact composed-command RMSE')
    finish(figure,'transformer_loss_and_command.png','Real-force versus masked continuous-force; identical true gate')
    figure,axes = plt.subplots(2,3,figsize=(13,7))
    for axis,part in zip(axes.flat,('arm','thumb','index','middle','ring','pinky')):
        for variant,history in histories.items():
            axis.plot([row['updates'] for row in history],
                [row['validation']['metrics']['contact_residual'][part] for row in history],
                label=variant,color=colors[variant])
        axis.set_title(part)
        axis.set_ylabel('Val40 contact residual RMSE')
    finish(figure,'transformer_fingers.png','Raw residual supervision remains differentiable outside execution clip')
    figure,axes = plt.subplots(2,3,figsize=(13,7))
    for axis,key in zip(axes.flat,('gate_fraction','tactile_valid_fraction','delta_mean_abs','execution_clip_fraction',
                                  'wrong_direction_saturation_fraction','batched_forward_ms_per_anchor')):
        for variant,history in histories.items():
            axis.plot([row['updates'] for row in history],[row['validation'][key] for row in history],
                      label=variant,color=colors[variant])
        axis.set_ylabel(key)
    finish(figure,'transformer_diagnostics.png','Validation40; batch latency is not online latency or a real-time guarantee')
    path = root/'residual_closed_loop/trials.json'
    if path.exists():
        trials = json.loads(path.read_text())
        for seed in sorted({row['seed'] for row in trials}):
            figure,axes = plt.subplots(4,1,figsize=(11,9),sharex=True)
            for trial in trials:
                if trial['seed']!=seed or 'trace' not in trial:
                    continue
                with np.load(path.parent/trial['trace']) as trace:
                    stamps = (trace['time_before']-trace['time_before'][0]).reshape(-1)
                    height = trace['object_pose'][:,2]-trace['object_pose'][0,2]
                    axes[0].plot(stamps,height,label=trial['variant'])
                    axes[1].plot(stamps,np.linalg.norm(trace['force'],axis=-1).max(axis=(1,2)),label=trial['variant'])
                    axes[2].plot(stamps,np.abs(trace['command']-trace['base']).mean(-1)*trace['student'],label=trial['variant'])
                    axes[3].plot(stamps,np.linalg.norm(trace['applied_force_w_N'],axis=-1).max(axis=(1,2)),label=trial['variant'])
                    if trial['mode']!='autonomous' and trial['switch_step'] is not None:
                        for axis in axes:
                            axis.axvline(trial['switch_step']/60.,color='grey',linestyle=':',alpha=.4)
            for axis,label in zip(axes,('Height vs first\nafter-step (m)','Peak finger force (N)',
                                       'Applied correction\nmean absolute command','Applied external\nworld force (N)')):
                axis.set_ylabel(label)
                axis.grid(alpha=.2)
                axis.legend(fontsize=7)
            axes[-1].set_xlabel('Native simulation time (s); dotted = expert handoff if present')
            figure.suptitle(f'Residual diagnostic seed{seed}; aggregate force does not identify a grasp\n'
                           'Physical initial states match; precontact commands differ: no strict causal pairing',fontsize=11)
            figure.tight_layout(rect=(0,0,1,.94))
            figure.savefig(output/f'transformer_closed_loop_s{seed}.png',dpi=150)
            plt.close(figure)


def plot(root, final=False):
    output = root/'figures'
    output.mkdir(exist_ok=True)
    session = json.loads((root/'session.json').read_text())
    histories = {line:json.loads((root/line/'metrics.json').read_text())
                 for line in session['dp_lines'] if (root/line/'metrics.json').exists()}
    colors = dict(zip(session['dp_lines'],('tab:blue','tab:orange','tab:green','tab:red')))
    metadata = dict(source_root=session['source_root'],lines={},scope='recorded train/validation only; no test-driven selection')
    rows = []
    training_rows = []
    for line,history in histories.items():
        metadata['lines'][line] = dict(source_checkpoint=history[0]['source_checkpoint'],
                                     last_additional_updates=history[-1]['additional_updates'])
        for record in history:
            if 'train' in record:
                training_rows.append(dict(line=line,epoch=record['epoch'],additional_updates=record['additional_updates'],
                    source_checkpoint=record['source_checkpoint'],split='train320 evolving raw',**record['train']))
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
            writer = csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
    if training_rows:
        with (output/'dp_training.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(training_rows[0]),lineterminator='\n')
            writer.writeheader()
            writer.writerows(training_rows)

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
        finish(figure,'dp_'+space+'_rmse.png','Validation40 EMA, teacher-forced targets; first12 committed prefix versus full H64')

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
    plot_dp_closed_loop(root,output)
    plot_residual(root,output)
    if final:
        atomic_json(root/'plots_complete.json',dict(files=sorted(path.name for path in output.iterdir())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--final',action='store_true')
    arguments = parser.parse_args()
    plot(arguments.root,arguments.final)
