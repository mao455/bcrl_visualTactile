"""Read-only final verification and small sanitized result export for this study."""
import argparse
import json
from pathlib import Path

import numpy as np

from .compact import sha256
from .training_resources import atomic_json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit_traces(root, phase):
    directory = root/(phase+'_closed_loop')
    report = json.loads((directory/'report.json').read_text())
    require(report['complete'],'Closed-loop cohort incomplete')
    verified,videos = [],[]
    initials = {}
    for row in report['trials']:
        if 'trace' not in row:
            verified.append(dict(attempt=row['attempt'],outcome=row['outcome'],trace_available=False))
            continue
        path = directory/row['trace']
        require(sha256(path)==row['trace_sha256'],'Native trace changed')
        with np.load(directory/(row['attempt']+'_initial.npz')) as initial:
            state = dict(initial)
        if row['seed'] not in initials:
            initials[row['seed']] = state
        initial_error = max(float(np.max(np.abs(value.astype(float)-initials[row['seed']][name].astype(float))))
                            for name,value in state.items())
        with np.load(path) as trace:
            count = row['steps']
            require(all(np.isfinite(trace[name]).all() for name in trace.files),'Nonfinite trace value')
            require(np.array_equal(trace['step'],np.arange(count)),'Native steps skipped or duplicated')
            require(np.allclose(trace['time_after']-trace['time_before'],1/60,atol=1e-9,rtol=0),'Native control clock changed')
            require(np.allclose(trace['time_after'][:-1],trace['time_before'][1:],atol=1e-9,rtol=0),'Native time discontinuity')
            require(np.array_equal(trace['nominal_cursor'],np.arange(count)%12),'Nominal cursor mismatch')
            require(np.array_equal(trace['command'],trace['clipped']),'Submitted pre-delay command mismatch')
            require(np.array_equal(trace['command'],trace['submitted']),'Actual native submission differs')
            require(np.all(np.abs(trace['command'])<=1),'Native command outside range')
            reference = trace['plans'][np.arange(count)//12,np.arange(count)%12]
            require(np.array_equal(reference,trace['reference']),'Executed reference was not original nominal plan')
            gate = trace['gate'].astype(bool)
            student = trace['student'].astype(bool)
            inactive = student&~gate
            require(np.array_equal(trace['command'][inactive],trace['base'][inactive]),'No-contact exact bypass violated')
            require(trace['applied_time_s'].shape==(count,2),'Missing120Hz applied-wrench samples')
            require(np.allclose(np.diff(trace['applied_time_s'],axis=1),1/120,atol=1e-9,rtol=0),'Wrench sample rate changed')
            if phase=='residual':
                require(np.array_equal(trace['residual_cursor'],np.arange(count)%6),'Residual cursor mismatch')
                delta = trace['delta_windows'][np.arange(count)//6,np.arange(count)%6]
                require(np.array_equal(delta,trace['delta']),'Stale or overwritten residual window')
                require(np.array_equal(trace['actual_previous'][1:],trace['clipped'][:-1]),'Actual residual feedback misaligned')
                require(np.array_equal(trace['private_previous'][1:],trace['base'][:-1]),'Mixed feedback entered private nominal history')
                expected = trace['base'] if row['variant']=='base' else np.where(gate[:,None],
                    np.clip(trace['base']+trace['delta'],-1,1),trace['base'])
                require(np.array_equal(expected[student],trace['command'][student]),'Residual composition differs')
            video = row.get('video')
            if video:
                import av
                movie = directory/video['file']
                with av.open(str(movie)) as container:
                    frames = list((frame.pts,frame.width,frame.height) for frame in container.decode(video=0))
                require(len(frames)==count==video['frames'],'Video does not fully cover the native rollout')
                require(all(after[0]>before[0] for before,after in zip(frames,frames[1:])),'Video timestamps not increasing')
                videos.append(dict(file=video['file'],sha256=sha256(movie),decoded_frames=len(frames),
                                   width=frames[0][1],height=frames[0][2]))
        verified.append(dict(attempt=row['attempt'],steps=count,trace_available=True,finite=True,
            control_clock_hz=60,physics_trace_hz=120,command_reference_alignment=True,
            initial_match_max_error=initial_error,paired_claim=row['paired'],no_contact_exact=True))
    return dict(traces=verified,videos=videos)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    root = args.root
    require(json.loads((root/'pipeline.json').read_text())['state']=='complete','Owner pipeline is not complete')
    require((root/'monitor_exit.json').exists(),'Independent monitor has not exited')
    session = json.loads((root/'session.json').read_text())
    monitor = json.loads((root/'monitor_exit.json').read_text())
    result = dict(run_id=root.name,session=session,dp={},residual={},resources=monitor,
        interface_audit=json.loads((root/'interface_audit.json').read_text()),
        transformer_interface_audit=json.loads((root/'transformer_interface_audit.json').read_text()),
        residual_admission=json.loads((root/'residual_admission.json').read_text()),
        selection=json.loads((root/'selected.json').read_text()),verification={})
    for line in session['dp_lines']:
        directory = root/line
        report = json.loads((directory/'report.json').read_text())
        require(report['state']=='complete' and report['epochs']==12 and report['updates']==9984,'Unequal DP training budget')
        test = json.loads((directory/'test.json').read_text())
        require(test['checkpoint_sha256']==sha256(directory/'best.pt'),'Selected DP test hash differs')
        history = json.loads((directory/'metrics.json').read_text())
        selected = min(history,key=lambda row:row['validation']['selection_mse'])
        require(test['additional_updates']==selected['additional_updates'],'DP checkpoint selection differs from validation argmin')
        def compact_metrics(row):
            return {key:value for key,value in row.items() if key!='representative'}
        result['dp'][line] = dict(training=report,test=test,
            baseline_validation=compact_metrics(history[0]['validation']),
            selected_validation=compact_metrics(selected['validation']),
            final_validation=compact_metrics(history[-1]['validation']),
            selected_additional_epoch=selected['epoch'],selected_additional_updates=selected['additional_updates'],
            selected_train_diagnostic=compact_metrics(selected['train_diagnostic']),
            source_sha256=history[0]['source_checkpoint'])
    cache_index = json.loads((root/'transformer_cache/index.json').read_text())
    require(len(cache_index['entries'])==400,'Residual cache does not cover fixed400')
    for variant in ('force','masked'):
        directory = root/'transformer_training'/variant
        report = json.loads((directory/'report.json').read_text())
        require(report['complete'] and report['epochs']==5,'Residual training budget incomplete')
        test = json.loads((directory/'test.json').read_text())
        require(test['checkpoint_sha256']==sha256(directory/'best.pt'),'Selected residual test hash differs')
        history = json.loads((directory/'metrics.json').read_text())
        chosen = min(history[1:],key=lambda row:row['validation']['selection_mse'])
        require(chosen['epoch']==test['epoch'],'Residual selection differs from trained validation argmin')
        result['residual'][variant] = dict(training=report,test=test,selected_validation=chosen['validation'],
            initial_validation=history[0]['validation'],selected_train_diagnostic=chosen['train_diagnostic'])
    require(result['residual']['force']['training']['updates']==result['residual']['masked']['training']['updates'],
            'Unequal residual update budgets')
    for phase in ('dp','residual'):
        result[phase+'_closed_loop'] = json.loads((root/(phase+'_closed_loop/trials.json')).read_text())
        result['verification'][phase] = audit_traces(root,phase)
    protected = {}
    original_cache = json.loads((root/'cache_index.json').read_text())
    manifest = Path('outputs/visuotactile/data/close_v4/manifest.json')
    protected['manifest_unchanged'] = sha256(manifest)==original_cache['manifest_sha256']
    protected['original_B0_unchanged'] = sha256(Path('outputs/visuotactile/run/dp_command_20261029/b0/best.pt'))==original_cache['source_split_sha256']
    historical = json.loads(Path('research_docs/evidence/baseline_final_20261030.json').read_text())
    protected['original_B1_unchanged'] = sha256(Path('outputs/visuotactile/run/dp_command_20261029/b1/best.pt'))==historical['variants']['b1']['checkpoint_sha256']
    protected['formal_report_unchanged'] = sha256(Path('outputs/visuotactile/run/dp_command_20261029/evaluation/report.json'))=='a93cba0f89fbc501e6f43bf94e8176edccce26dbf47ab6e72a690112302f9366'
    initial = json.loads((root/'initial_sources.json').read_text())
    for group in ('G00','G10'):
        protected[group+'_epoch40_unchanged'] = sha256(Path(session['source_root'])/(group+'-new_full/best.pt'))==initial[group]
    require(all(protected.values()),'A protected source identity changed')
    result['protected'] = protected
    incidents = root/'resource_incidents.jsonl'
    result['resource_incident_records'] = len(incidents.read_text().splitlines()) if incidents.exists() else 0
    result['figures'] = json.loads((root/'plots_complete.json').read_text())['files']
    result['within_original_deadline'] = monitor['utc_epoch']<=session['deadline']
    result['limitations'] = ['diagnostic sample size3 per DP, no formal0/50 denominator changes',
        'aggregate force lacks contact-partner identity','old GRU uses a different frozen base and is not a matched comparator',
        'DP teacher-forced offline conversion is not autonomous command prediction',
        'private nominal proposal history is a new declared residual protocol',
        'BF16 batch/native recurrence is not bitwise identical; residual cache uses actual native-tick nominal replay']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,result)
    atomic_json(root/'artifact_audit.json',dict(passed=True,evidence_sha256=sha256(args.output),
        protected=protected,verified_trials=sum(len(value['traces']) for value in result['verification'].values()),
        decoded_videos=sum(len(value['videos']) for value in result['verification'].values())))
    print(json.dumps(dict(passed=True,output=str(args.output),bytes=args.output.stat().st_size)))


if __name__=='__main__':
    main()
