"""Ordinary postprocessor waits for the owner; it never launches/extends GPU work.

Only measured tables are appended to the requested results document. Git changes
still require review; this program never commits or pushes automatically.
"""
import argparse
import json
from pathlib import Path
import time

from state_history_report import summarize


def decode_video(path):
    """Use the installed recording codec library without an external executable."""
    import av
    count = 0
    try:
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                frame.to_ndarray(format='rgb24')
                count += 1
        return dict(decoded_frames=count, exit_code=0 if count else 1,
            error='' if count else 'No video frames', decoder='PyAV')
    except Exception as error:
        return dict(decoded_frames=count, exit_code=1, error=repr(error), decoder='PyAV')


def finish(root, output, document):
    session = json.loads((root/'session.json').read_text())
    while not (root/'owner_exit.json').exists():
        if time.time() > session['deadline']+120:
            raise TimeoutError('Owner did not report cleanup by its fixed deadline')
        time.sleep(5)
    summary = summarize(root, output)
    rows = ['\n## 自动收尾实测表（待人工复核与Git提交）\n',
        '此表只读取普通owner结束后的产物，不新增训练或rollout。旧/前置失效样本不混入新分母。\n',
        '|组|cohort|已尝试|合法|完整成功|native lift|至少1子目标|',
        '|---|---|---:|---:|---:|---:|---:|']
    for stage, values in summary['stages'].items():
        for cohort, counts in values.get('cohorts', {}).items():
            rows.append(f"|{stage}|{cohort}|{counts['attempted']}|{counts['legal']}|{counts['successes']}|{counts['lifts']}|{counts['subgoals']}|")
    rows += ['\n|组|updates|train arm/hand RMSE|validation arm/hand RMSE|', '|---|---:|---|---|']
    for kind in sorted(p.name[:-6] for p in root.glob('*_train') if (p/'metrics.json').exists()):
        path = root/(kind+'_train')/'metrics.json'
        if path.exists():
            for row in json.loads(path.read_text()):
                a, b = row['train'], row['validation']
                rows.append(f"|{kind}|{row['update']}|{a['arm_rmse']:.6f}/{a['hand_rmse']:.6f}|{b['arm_rmse']:.6f}/{b['hand_rmse']:.6f}|")
    status = json.loads((root/'status.json').read_text())
    rows.append('\n最终owner状态：`'+json.dumps(status, ensure_ascii=False)+'`。')
    rows.append('\n逐条reset误差、接触前后首差及阶段误差见公开`summary.json/trace_diagnostics`。\n'
        '这些观测不唯一证明网络失败机制；同输入/同命令验收与完整成功分开报告。')
    decode = []
    for video in summary['videos']:
        # Decode only videos whose trial has a completed result, never an
        # interrupted writer's partial output.
        path = Path(video['path'])
        if not (path.parent/'result.json').exists():
            continue
        decode.append(dict(path=str(path), sha256=video['sha256'], **decode_video(path)))
    (output/'video_decode.json').write_text(json.dumps(decode, indent=2)+'\n')
    text = document.read_text().split('\n## 自动收尾实测表')[0]
    document.write_text(text+'\n'+'\n'.join(rows)+'\n')
    (root/'postprocess_complete.json').write_text(json.dumps(dict(utc=time.time(),
        output=str(output), document=str(document), git_published=False), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--document', type=Path, default=Path('research_docs/STATE_TRAINING_RESULTS.md'))
    a = p.parse_args(); finish(a.root, a.output, a.document)
