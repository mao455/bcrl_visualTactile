"""Small audit evidence and exportable curves from actual files, including blocked runs."""
import json
import subprocess
from pathlib import Path
import numpy as np
from .training_resources import atomic_json


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    session=json.loads((root/'session.json').read_text());source=Path(session['source_root'])
    out=root/'report';out.mkdir(exist_ok=True)
    gates={p.stem:json.loads(p.read_text()) for p in (root/'gates').glob('*.json')}
    path=root/'diagnostics/result.json';diagnostics=json.loads(path.read_text()) if path.exists() else dict(trials=[],software_complete=False)
    groups={}
    for r in diagnostics['trials']:
        name='replay' if r['kind']=='replay' else r['cohort']
        g=groups.setdefault(name,dict(trials=0,successes=0,lift=0,subgoal=0,alignment=0,fall=0))
        g['trials']+=1;g['successes']+=bool(r['success']);g['lift']+=bool(r['native_lift'])
        g['subgoal']+=r.get('native_subgoals',r.get('subgoals',0))>0
        g['alignment']+=bool(r.get('native_alignment',False));g['fall']+=bool(r['reasons'].get('fall'))
    resource_path=root/'monitor_exit.json'
    resources=json.loads(resource_path.read_text()) if resource_path.exists() else None
    atomic_json(out/'summary.json',dict(gates=gates,groups=groups,new_test_success_rate=None,
        diagnostic_complete=diagnostics['software_complete'],resources=resources,
        checkpoint_identity=session['checkpoints'],formal_test_scope='No V2 held-out test executed',
        blocked=json.loads((root/'blocked.json').read_text()) if (root/'blocked.json').exists() else None))
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    manifest=json.loads((source/'manifest.json').read_text())['stages']
    for group in ('S-NativeDP','I-OfficialDP','I-HistoryDP'):
        stage=manifest.get(group+'_formal',{});directory=Path(stage.get('output','/nonexistent'))
        train=directory/'train.jsonl';val=directory/'validation.json'
        if train.exists():
            rows=[json.loads(line) for line in train.read_text().splitlines()]
            axes[0].plot([r['updates'] for r in rows],[r['loss'] for r in rows],label=group,alpha=.5,linewidth=.6)
        if val.exists():
            rows=json.loads(val.read_text());axes[1].plot([r['updates'] for r in rows],[r['metrics']['selection_mse'] for r in rows],label=group)
    for ax,title in zip(axes,('Historical native DP training objective','Historical validation sampled command MSE')):
        ax.set_title(title);ax.set_xlabel('Optimizer updates');ax.legend()
    fig.tight_layout();fig.savefig(out/'old_training.png',dpi=170);plt.close(fig)
    if groups:
        fig,ax=plt.subplots(figsize=(11,4));names=list(groups);x=np.arange(len(names));width=.16
        for j,key in enumerate(('successes','lift','subgoal','alignment','fall')):
            ax.bar(x+(j-2)*width,[groups[k][key]/groups[k]['trials'] for k in names],width,label=key)
        ax.set_xticks(x,names,rotation=25,ha='right');ax.set_ylim(0,1.05);ax.legend();ax.set_ylabel('Fraction of actual diagnostic trials')
        fig.tight_layout();fig.savefig(out/'stage_outcomes.png',dpi=170);plt.close(fig)
    traces=list((root/'diagnostics').glob('*/expert_comparison.npz')) if (root/'diagnostics').exists() else []
    if traces:
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for path in traces[:4]:
            with np.load(path) as data:
                for ax,key in zip(axes.flat,('command_arm_rmse','command_hand_rmse','target_arm_rmse','object_position_m')):
                    ax.plot(np.arange(len(data[key]))/60,data[key],label=path.parent.name);ax.set_title(key);ax.set_xlabel('Simulated seconds')
        for ax in axes.flat:ax.legend(fontsize=6)
        fig.tight_layout();fig.savefig(out/'first_four_student_errors.png',dpi=170);plt.close(fig)
    telemetry=root/'resources/resources.jsonl'
    if telemetry.exists():
        rows=[json.loads(line) for line in telemetry.read_text().splitlines()]
        fig,ax=plt.subplots(figsize=(10,4))
        for key in ('gpu_util_percent','rolling_60s_percent'):
            ax.plot([r['elapsed_s']/60 for r in rows],[r.get(key) for r in rows],label=key,linewidth=.7)
        ax.axhline(50,color='red',linestyle='--');ax.set_xlabel('Monitor minutes');ax.set_ylabel('Whole-card GPU %');ax.legend()
        fig.tight_layout();fig.savefig(out/'resources.png',dpi=170);plt.close(fig)
    videos=[r['video'] for r in diagnostics['trials'] if 'video' in r]
    # Existing reference success is an independent sample, not paired with students.
    expert=source/'stages/oracle_instrumented'
    reference_videos=list(expert.glob('attempt_*/*expert*.mp4'))
    videos += [str(p.resolve()) for p in reference_videos]
    media_path=source/'video_tools.json'
    verification=[]
    if media_path.exists():
        media=json.loads(media_path.read_text())
        for video in videos:
            try:
                probe=json.loads(subprocess.check_output([media['ffprobe']['path'],'-v','error','-show_streams','-of','json',video],text=True,timeout=30))
                subprocess.run([media['ffmpeg']['path'],'-v','error','-threads','2','-i',video,'-f','null','-'],check=True,timeout=60,capture_output=True)
                verification.append(dict(path=video,decoded=True,streams=probe['streams']))
            except (OSError,ValueError,subprocess.SubprocessError) as error:
                verification.append(dict(path=video,decoded=False,error=repr(error)))
    atomic_json(out/'video_verification.json',verification)
    (out/'videos.txt').write_text('\n'.join(videos)+'\n')
    return out
