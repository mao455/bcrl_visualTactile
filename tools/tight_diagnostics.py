"""Finite diagnostics for the existing StateDP. No training/Kit launch path.

Ordinary systemd execution owns its resource monitor and completes its report.
The no-training decision is frozen from existing validation evidence, not test.
"""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback

from tools.runtime import CODE, RUN, read, write, sha, supervised_owner, kill_owned

BEGIN = '<!-- tight-dp3-diagnostics:start -->'
END = '<!-- tight-dp3-diagnostics:end -->'


def table(path, rows):
    if not rows:
        raise ValueError('Empty metric table: '+str(path))
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def source_hashes():
    return {str(p.relative_to(CODE)): sha(p) for folder in ('tools', 'visuotactile')
        for p in sorted((CODE/folder).rglob('*.py'))}


def beat(root, **fields):
    old = read(root/'status.json') if (root/'status.json').exists() else {}
    write(root/'status.json', dict(old, **fields, updated_utc=time.time(), pid=os.getpid()))


def prepare(root, config):
    if root.exists():
        raise FileExistsError('Run already exists; inspect status instead of restarting')
    cfg = read(config)
    if cfg['decision'] != 'diagnostics_then_stop_no_training' or cfg['condition_mode'] != 'existing_robot_geometry_short_windows_v1':
        raise ValueError('Only reviewed existing-equivalent diagnostics are supported')
    parent = (CODE/cfg['source_root']).resolve()
    data = read(parent/'state_data/data.json')
    if [len(data['splits'][s]) for s in ('train', 'validation', 'test')] != [156, 30, 22]:
        raise ValueError('Frozen Core split mismatch')
    for split, ids in cfg['evaluation']['cohorts'].items():
        if not ids or not set(ids) <= set(data['splits'][split]):
            raise ValueError('Diagnostic IDs outside declared Core split')
    if sha(parent/cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('Reference checkpoint changed')
    rows = [json.loads(x) for x in (parent/'training/validation.jsonl').read_text().splitlines()]
    gain = 1-rows[-1]['metrics']['selection_mse']/rows[-4]['metrics']['selection_mse']
    ratio = rows[-1]['metrics']['selection_mse']/rows[-1]['train8']['selection_mse']
    if gain >= .01 or read(parent/'training/sufficiency.json')['still_improving']:
        raise ValueError('Historical no-continuation evidence differs; do not silently train')
    root.mkdir(parents=True)
    write(root/'configuration.json', cfg)
    write(root/'decision.json', dict(training_updates=0, architecture_changed=False,
        reason='Equivalent direct path already exists; existing complete-sampling curve has <1% last-four gain and large train/validation gap',
        last_four_relative_gain=gain, validation_train_mse_ratio=ratio,
        source_validation_sha256=sha(parent/'training/validation.jsonl'), final_test_used=False))
    paths = [parent/p for p in ('manifest.json', 'composition_manifest.json', 'state_data/data.json',
        'state_data/numeric.pt', 'state_data/statistics.pt', 'training/best.pt', 'training/validation.jsonl',
        'training/training.jsonl', 'training/sufficiency.json', 'evaluation_summary.json')]
    write(root/'inputs.json', {str(p): sha(p) for p in paths})
    beat(root, state='PREPARED', stage='acceptance', formal_updates=0, episodes_completed=0,
        episodes_total=sum(map(len, cfg['evaluation']['cohorts'].values())), exit_code=None,
        reason='Awaiting focused acceptance and source freeze', run=str(root))


def load_cpu(root):
    import torch
    import h5py
    from visuotactile.state_history import PersistentState
    from visuotactile.position_control import contract
    from visuotactile.state_image_loss import GeometryObjective
    cfg = read(root/'configuration.json'); parent = CODE/cfg['source_root']
    data = read(parent/'state_data/data.json')
    saved = torch.load(parent/cfg['checkpoint'], map_location='cpu', weights_only=False)
    cache = torch.load(parent/'state_data/numeric.pt', map_location='cpu', weights_only=False)
    if saved['contract'] != contract(24, True) or cache['contract'] != saved['contract']:
        raise ValueError('H24 absolute-target contract mismatch')
    if cache['data_sha256'] != sha(parent/'state_data/data.json'):
        raise ValueError('Numeric data fingerprint mismatch')
    if saved['configuration']['input_builder'] != 'canonical_joint_state_v1':
        raise ValueError('Reference checkpoint input builder mismatch')
    for key, path in [('data_sha256', 'data.json'), ('statistics_sha256', 'statistics.pt')]:
        if saved['configuration'][key] != sha(parent/'state_data'/path):
            raise ValueError('Checkpoint/data normalization identity mismatch')
    model = PersistentState('dp', saved['contract']).eval()
    model.load_state_dict(saved['ema'], strict=True)
    stats = torch.load(parent/'state_data/statistics.pt', map_location='cpu', weights_only=False)
    for key, value in stats.items():
        if not torch.equal(getattr(model, key), value):
            raise ValueError('EMA normalizer mismatch: '+key)
    if saved['updates'] != 8500 or saved['ema_optimization_step'] != 8500:
        raise ValueError('Reference EMA update mismatch')
    ids = [i for values in cfg['evaluation']['cohorts'].values() for i in values]
    records = {i: cache['records'][i] for i in ids}
    entry = data['entries'][ids[0]]
    with h5py.File(entry['path']) as f:
        meta = json.loads(f.attrs['metadata_json'])
    objective = GeometryObjective(meta); objective.weights = saved['configuration']['loss_weights']
    return cfg, data, records, model, objective, saved


def acceptance(root):
    """One CPU optimizer step is disposable; never modifies a reference model."""
    import copy
    import torch
    from visuotactile.state_history import HistoryBank, short_windows
    from visuotactile.anchor_sampling import anchor_batch
    from visuotactile.models.vendor.ema_model import EMAModel
    from visuotactile.checkpointing import training_state, restore_training, save
    cfg, data, records, model, objective, old = load_cpu(root)
    import hashlib
    for name in ('state_history.py', 'models/native_dp.py', 'native_data.py', 'narrow_data.py', 'position_control.py'):
        path = 'visuotactile/'+name
        original = subprocess.check_output(['git', 'show', cfg['reference_commit']+':'+path], cwd=CODE)
        if hashlib.sha256(original).hexdigest() != sha(CODE/path):
            raise ValueError('Existing-equivalent policy/data code changed: '+path)
    del old
    rec = records[cfg['evaluation']['cohorts']['train'][0]]
    with torch.no_grad():
        full, anchors, hidden = model.encode(rec)
        bank = HistoryBank(); online = []
        for tick in range(len(rec['time'])):
            c = bank.observe(0, 1, tick, model, rec['actor'][tick], rec['robot'][tick], rec['geometry'][tick], rec['time'][tick])
            if c is not None: online.append(c)
        parity = float((full-torch.cat(online)).abs().max())
        if parity >= 2e-5: raise ValueError('Full-prefix train/deploy condition mismatch')
    # Hold Actor and recurrent output constant: only the direct current-state
    # input can produce this derivative/intervention response.
    robots, geos = short_windows(rec, torch.tensor([12]), model)
    robots = robots.detach().requires_grad_(); geos = geos.detach().requires_grad_()
    fixed = torch.zeros(1, 384)
    def condition(r, g):
        return model.condition_projection(torch.cat((model.robot_encoder(r), model.state_encoder(g), fixed), -1))
    c = condition(robots, geos)
    gr, gg = torch.autograd.grad(c.square().mean(), (robots, geos))
    r2, g2 = robots.detach().clone(), geos.detach().clone()
    r2[0, -180] += .1; g2[0, -87] += .1
    delta_r = float((condition(r2, geos)-c).abs().max().detach())
    delta_g = float((condition(robots, g2)-c).abs().max().detach())
    current_grad = [float(gr[0, -180:-2].abs().sum()), float(gg[0, -87:-2].abs().sum())]
    if min(delta_r, delta_g, *current_grad) <= 0: raise ValueError('Direct current state path is inactive')
    model.train(); model.zero_grad(set_to_none=True)
    averaged = copy.deepcopy(model).eval().requires_grad_(False); ema = EMAModel(averaged, power=.75)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-6, betas=(.95, .999), weight_decay=1e-6)
    generator = torch.Generator().manual_seed(101); noise = torch.Generator().manual_seed(102)
    before = next(model.condition_projection.parameters()).detach().clone()
    identity = cfg['evaluation']['cohorts']['train'][0]
    cond, y, mask = anchor_batch(model, records, [(identity, 12)])
    loss, _, _ = objective.forward(model, cond, y, mask, noise)
    loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    if not torch.isfinite(loss) or not torch.isfinite(norm): raise FloatingPointError('Acceptance nonfinite loss/gradient')
    opt.step(); ema.step(model)
    change = float((next(model.condition_projection.parameters()).detach()-before).abs().max())
    if change <= 0: raise ValueError('Disposable update did not change parameters')
    temporary = root/'disposable_acceptance.pt'
    save(temporary, training_state(model, averaged, opt, ema, generator, noise, updates=1))
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    restore_training(saved, model, averaged, opt, ema, generator, noise)
    if ema.optimization_step != 1 or not all(int(v['step']) == 1 for v in opt.state.values()):
        raise ValueError('Disposable optimizer/EMA restore mismatch')
    with torch.no_grad():
        c, _, _ = averaged.encode(rec, stop=13)
        predicted = averaged.predict(c[-1:], torch.Generator().manual_seed(7091))
    if predicted.shape != (1, 24, 29) or not torch.isfinite(predicted).all():
        raise ValueError('Complete prediction after checkpoint restore failed')
    report = dict(passed=True, source_sha256=source_hashes(), full_prefix_ticks=len(rec['time']),
        condition_shape=list(full.shape), condition_projection_input=896, condition_projection_output=512,
        robot_window_shape=list(robots.shape), geometry_window_shape=list(geos.shape),
        native_history_hidden_shape=list(hidden.shape), train_deployment_max_abs=parity,
        current_robot_condition_delta=delta_r, current_geometry_condition_delta=delta_g,
        direct_current_input_gradient_l1=current_grad, recurrent_and_actor_held_constant=True,
        disposable_updates=1, formal_updates=0, disposable_loss=float(loss), gradient_norm=float(norm),
        parameter_change=change, save_load_optimizer_ema=True, normalization_matches=True,
        complete_prediction_shape=list(predicted.shape), original_model_source_unchanged=True)
    temporary.unlink()  # Only this explicitly disposable smoke checkpoint.
    write(root/'acceptance.json', report)
    beat(root, state='CHECKED', stage='ready_to_freeze', reason='CPU acceptance passed')


def historical_curves(root, parent):
    rows = []
    for line in (parent/'training/validation.jsonl').read_text().splitlines():
        r = json.loads(line)
        for name, key, loss_key in [('train8', 'train8', 'train8_loss'), ('validation30', 'metrics', 'validation_loss')]:
            rows.append(dict(update=r['update'], split=name, loss=r.get(loss_key, {}).get('loss'),
                arm_rmse_rad=r[key]['arm_rmse'], hand_rmse_rad=r[key]['hand_rmse'],
                selection_mse_rad2=r[key]['selection_mse'], protocol='historical_episode_batch_seed7091'))
    table(root/'historical_curves.csv', rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # All plots read the emitted raw CSV, including historical curves.
    with (root/'historical_curves.csv').open() as f: plotted = list(csv.DictReader(f))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for split in ('train8', 'validation30'):
        selected = [r for r in plotted if r['split'] == split]
        for ax, key in zip(axes, ('loss', 'arm_rmse_rad', 'hand_rmse_rad')):
            ax.plot([int(r['update']) for r in selected], [float(r[key]) if r[key] else float('nan') for r in selected], label=split)
            ax.set(xlabel='Optimizer updates', ylabel=key); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(root/'historical_curves.png', dpi=140); plt.close(fig)


def target_plot(root, identity, sample, phases, contact, lift):
    import matplotlib.pyplot as plt
    rows = []
    for tick in range(len(phases)):
        row = dict(tick=tick, seconds=tick/60, phase=str(phases[tick]), contact=bool(contact[tick]), lift=bool(lift[tick]), replan=tick%12 == 0)
        for kind in ('truth', 'prediction', 'hold'):
            # Increments within a planned chunk start from the real anchor target.
            before = sample['hold'][tick] if tick%12 == 0 else sample[kind][tick-1]
            for j in range(29):
                row[f'{kind}_{j}_rad'] = float(sample[kind][tick, j])
                row[f'{kind}_delta_{j}_rad'] = float(sample[kind][tick, j]-before[j])
        rows.append(row)
    path = root/(identity+'_targets.csv'); table(path, rows)
    with path.open() as f: rows = list(csv.DictReader(f))
    # Fixed representative joints, chosen before viewing predictions.
    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    for col, joint in enumerate((3, 7)):
        for row_index in range(2):
            ax = axes[row_index, col]
            for kind in ('truth', 'prediction', 'hold'):
                key = f'{kind}_{joint}_rad' if row_index == 0 else f'{kind}_delta_{joint}_rad'
                ax.plot([float(r['seconds']) for r in rows], [float(r[key]) for r in rows], label=kind, linewidth=.8)
            for r in rows:
                if r['replan'] == 'True': ax.axvline(float(r['seconds']), color='gray', alpha=.15, linewidth=.5)
            for field, color in (('contact', 'green'), ('lift', 'purple')):
                first = next((r for r in rows if r[field] == 'True'), None)
                if first: ax.axvline(float(first['seconds']), color=color, label='first '+field)
            ax.set(xlabel='Expert-history time (s)', ylabel=('Target' if row_index == 0 else 'Within-plan change')+' (rad)', title=f'Canonical joint {joint}')
            ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/(identity+'_targets.png'), dpi=140); plt.close(fig)


def reuse_closed_loop(root, parent):
    import numpy as np
    from visuotactile.narrow_decision import failure_stage
    rows = []; selected = []
    for p in sorted((parent/'evaluation').glob('*/result.json')):
        r = read(p)
        rows.append(dict(identity=r['identity'], split=r['split'], legal=r['legal'], success=r['success'],
            approach=r.get('approach_proxy'), object_contact=r.get('contact', {}).get('object_contact'),
            stable_grasp=r.get('contact', {}).get('stable_grasp_diagnostic'), lift=r.get('native_lift'),
            alignment=r.get('native_alignment'), subgoals=r.get('native_subgoals'),
            actuator_error_rad=r.get('actuator_target_max_error'), submitted_target_error_rad=r.get('submitted_target_max_error'),
            tracking_rmse_rad=r.get('tracking_rmse_rad'), failure_stage=failure_stage(r),
            source=str(p), source_sha256=sha(p)))
        if r['split'] in ('train', 'validation') and r['split'] not in [x[0] for x in selected]: selected.append((r['split'], p.parent))
    table(root/'closed_loop_reused.csv', rows)
    cohorts = {}
    for split in sorted({r['split'] for r in rows}):
        group = [r for r in rows if r['split'] == split]
        cohorts[split] = dict(attempts=len(group), valid=sum(r['legal'] for r in group),
            rejected=sum(not r['legal'] for r in group), successes=sum(r['success'] for r in group),
            reached={k: sum(bool(r[k]) for r in group) for k in ('approach', 'object_contact', 'stable_grasp', 'lift', 'alignment', 'subgoals')})
    events = []
    for split, directory in selected:
        with np.load(directory/'trace.npz') as z:
            initial_z = read(directory/'initial_observable.json')['object_pose_w'][0][2]
            physical = []
            for t in range(len(z['joint'])):
                err = z['joint'][t]-z['target'][t]
                physical.append(dict(tick=t, seconds=(t+1)/60, object_rise_m=float(z['object'][t, 2]-initial_z),
                    fingertip_min_distance_m=float(z['fingertip_distance'][t]),
                    contact_force_max_N=float(np.linalg.norm(z['object_contact'][t], axis=-1).max()),
                    arm_tracking_rmse_rad=float(np.sqrt(np.mean(err[:7]**2))),
                    hand_tracking_rmse_rad=float(np.sqrt(np.mean(err[7:]**2))),
                    native_lift=bool(np.any(z['lifted'][t])), native_alignment=bool(np.any(z['alignment'][t])),
                    subgoals=int(np.max(z['subgoals'][t]))))
            events.append(dict(identity=directory.name, split=split,
                first_contact_tick=next((r['tick'] for r in physical if r['contact_force_max_N']>.1), None),
                first_native_lift_tick=next((r['tick'] for r in physical if r['native_lift']), None),
                first_native_alignment_tick=next((r['tick'] for r in physical if r['native_alignment']), None),
                maximum_object_rise_m=max(r['object_rise_m'] for r in physical),
                earliest_expert_physical_divergence='not measured: autonomous/expert histories are not same-tick pairs'))
        path = root/(directory.name+'_physical.csv'); table(path, physical)
        with path.open() as f: plotted = list(csv.DictReader(f))
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        for ax, keys, unit in [(axes[0], ['object_rise_m', 'fingertip_min_distance_m'], 'm'),
            (axes[1], ['contact_force_max_N'], 'N'), (axes[2], ['arm_tracking_rmse_rad', 'hand_tracking_rmse_rad'], 'rad')]:
            for key in keys: ax.plot([float(r['seconds']) for r in plotted], [float(r[key]) for r in plotted], label=key)
            ax.set(xlabel='Student time (s)', ylabel=unit); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout(); fig.savefig(root/(directory.name+'_physical.png'), dpi=140); plt.close(fig)
    write(root/'closed_loop_summary.json', dict(cohorts=cohorts, representative_events=events,
        new_starts=0, scope='Reused old autonomous trials; no teacher target error across different-progress trajectories'))


def report(root):
    s = read(root/'status.json'); decision = read(root/'decision.json')
    content = f'''{BEGIN}
## 2026-11-08 DP3风格StateDP条件与诊断

状态 **{s['state']}**；阶段 `{s['stage']}`；已完成离线轨迹 {s.get('episodes_completed', 0)}/{s.get('episodes_total', 38)}。
原因：{s.get('reason', '')}。正式新增更新0；新Kit/闭环启动0。
等价当前robot/geometry直达condition已存在，保留原模型/GRU/U-Net与旧checkpoint。
旧最近四次验证改善 {decision['last_four_relative_gain']:.4%}，验证/train8采样MSE比 {decision['validation_train_mse_ratio']:.2f}；依据train/validation冻结为诊断后停止，不自动续训。

本轮root：`{root}`；状态`status.json`，日志`service.log`，验收`acceptance.json`，
冻结身份`session.json`/`inputs.json`/`source_freeze.json`。
图均由本run CSV生成：`historical_curves.csv/png`为原17点完整DDIM曲线（旧episode-batch噪声）；
`sampled_ticks.csv`/`stage_metrics.json`为当前同部署batch1顺序、固定seed7091、真实完整专家前缀、前12步目标误差；
`*_targets.csv/png`含29关节目标/变化量原始数值及固定joint3/7示例、接触/抬升/重规划标记。
`closed_loop_reused.csv`/`closed_loop_summary.json`与`*_physical.csv/png`复用旧27次自主轨迹。
未生成文件代表该阶段未完成，不代表零误差或零成功。离线目标FK误差、命令接口误差和物理tracking各自独立；
自主与专家不同进度下第一次物理偏离未测，不按同tick作动作误差。阶段事件只作诊断，且含transition接触/对齐信息，绝不输入网络。
资源上限：整卡批次/完整60s均值50%，45%准入、35%目标、6GiB显存/14GiB RAM/60GiB磁盘储备；最多1h含等待。
实测资源见`resources/resources.json`与`resource_summary.json`，启动不等于完成或全程资源合规。
{END}'''
    if (root/'stage_metrics.json').exists():
        metrics = read(root/'stage_metrics.json')
        lines = ['\n|专家历史离线集/阶段|Arm RMSE / hold (rad)|Hand RMSE / hold (rad)|有效tick|', '|---|---:|---:|---:|']
        for arm in (r for r in metrics if r['group'] == 'arm' and r['motion'] == 'all'):
            hand = next(r for r in metrics if r['split'] == arm['split'] and r['phase'] == arm['phase'] and r['group'] == 'hand' and r['motion'] == 'all')
            fmt = lambda r: f"{r['rmse_rad']:.6f} / {r['hold_rmse_rad']:.6f}" if r['ticks'] else '未测'
            lines.append(f"|{arm['split']} / {arm['phase']}|{fmt(arm)}|{fmt(hand)}|{arm['ticks']}|")
        content = content.replace(END, '\n'.join(lines)+'\n\n'+END)
    if (root/'closed_loop_summary.json').exists():
        cohorts = read(root/'closed_loop_summary.json')['cohorts']
        text = '\n旧闭环复用：'+ '；'.join(f"{k} {v['successes']}/{v['valid']}成功，拒绝{v['rejected']}" for k,v in cohorts.items())+'。不加入新实验分母。\n\n'
        content = content.replace(END, text+END)
    cfg = read(root/'configuration.json'); path = CODE/cfg['output_report']; old = path.read_text()
    if BEGIN in old:
        start, end = old.index(BEGIN), old.index(END)+len(END)
        updated = old[:start]+content+old[end:]
    else:
        first, rest = old.split('\n', 1); updated = first+'\n\n'+content+'\n\n'+rest
    temporary = path.with_suffix('.md.tmp'); temporary.write_text(updated); temporary.replace(path)


@supervised_owner
def run(root):
    import torch
    import h5py
    import numpy as np
    from visuotactile.state_resources import Lease
    from visuotactile.state_image_offline import deployment_sample, diagnostic_rows, summarize_diagnostics, no_kit
    from tools.narrow_repair_audit import stages
    lock = (root/'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'session.json').exists(): raise RuntimeError('No automatic rerun/resume of completed or failed diagnostics')
    monitor = None; code = 1
    try:
        cfg = read(root/'configuration.json'); check = read(root/'acceptance.json')
        # Acceptance covers the unmodified policy/data implementation and the
        # real checkpoint. Final CLI/plot tests cover the orchestration separately.
        current_sources = source_hashes()
        if not check['passed'] or any(digest != current_sources.get(path) for path, digest in check['source_sha256'].items()
                if path.startswith('visuotactile/')):
            raise ValueError('Acceptance model/evaluator source changed')
        final_checks = read(root/'runtime_checks.json')
        if not final_checks['passed'] or final_checks['source_sha256'] != current_sources:
            raise ValueError('Final runtime checks do not cover current sources')
        if final_checks['configuration_sha256'] != sha(root/'configuration.json') or sha(root/'configuration.json') != sha(CODE/'configs/tight_dp3_diagnostics.json'):
            raise ValueError('Reviewed run configuration differs from committed configuration')
        for path, digest in read(root/'inputs.json').items():
            if sha(path) != digest: raise ValueError('Frozen input changed: '+path)
        head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=CODE, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=CODE, text=True).strip()
        if dirty: raise ValueError('Commit the reviewed runtime before launch: '+dirty)
        now = time.time(); deadline = now+3600*cfg['evaluation']['maximum_hours']
        write(root/'session.json', dict(started_utc=now, training_deadline=deadline, deadline=deadline,
            source_head=head, cwd=str(CODE), python=str(Path('/proc/self/exe').resolve()),
            configuration_sha256=sha(root/'configuration.json'),
            resource_target_percent=35, formal_updates=0, seed=7091))
        write(root/'source_freeze.json', source_hashes())
        write(root/'gpu_schedule.json', cfg['resources'])
        beat(root, state='RUNNING', stage='historical_curves', reason='Frozen existing-model diagnostic queue', exit_code=None)
        parent = CODE/cfg['source_root']; historical_curves(root, parent); reuse_closed_loop(root, parent)
        log = (root/'monitor.log').open('a')
        monitor = subprocess.Popen(RUN+['-m', 'visuotactile.state_resources', '--root', str(root)],
            cwd=CODE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        class BoundedLease(Lease):
            def check(self, deadline=None):
                super().check(deadline)
                if shutil.disk_usage(root).free < cfg['resources']['min_disk_gib']*1024**3:
                    raise RuntimeError('60GiB disk reserve reached')
                if monitor.poll() is not None: raise RuntimeError('Resource monitor exited')
        lease = BoundedLease(root, deadline=deadline)
        beat(root, stage='resource_admission', monitor_pid=monitor.pid,
            reason='Require a complete measured 60-second window before reference load and CUDA')
        with lease.block(cuda=False):
            cfg, data, records, model, objective, saved = load_cpu(root); del saved
        with lease.block(): model.to(cfg['evaluation']['device'])
        rows = []; completed = 0; anchors_total = 0
        def progress(**kw):
            beat(root, **kw)
            if not (root/'first_progress.json').exists():
                write(root/'first_progress.json', read(root/'status.json'))
        for split, ids in cfg['evaluation']['cohorts'].items():
            for index, identity in enumerate(ids):
                beat(root, stage='offline_sampling', identity=identity, split=split, native_ticks=0,
                    sampled_anchors=0, reason='Deployment-order DDIM on a complete expert prefix')
                with lease.block(): rec = {k:v.to(cfg['evaluation']['device']) for k,v in records[identity].items()}
                sample = deployment_sample(model, rec, objective, lease, seed=cfg['evaluation']['seed'],
                    progress=progress)
                e = data['entries'][identity]
                with h5py.File(e['path']) as f:
                    g = f['episodes'][e['episode']]; phases = stages(g)
                    contact = np.linalg.norm(g['transition/object_contact_force_w'][:], axis=-1).reshape(len(phases), -1).max(-1)>.1
                    lift = g['transition/diagnostics_before/lifted_object'][:].reshape(len(phases), -1).any(-1)
                episode_rows = diagnostic_rows(sample, phases, identity=identity, split=split)
                path = root/'sampled_ticks.csv'
                with path.open('a', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(episode_rows[0]))
                    if completed == 0: writer.writeheader()
                    writer.writerows(episode_rows)
                rows.extend(episode_rows)
                if index == 0: target_plot(root, identity, sample, phases, contact, lift)
                completed += 1; anchors_total += (len(phases)+11)//12
                write(root/'stage_metrics.json', summarize_diagnostics(rows))
                beat(root, episodes_completed=completed, anchors_total=anchors_total)
                del rec, sample
        no_kit()
        if source_hashes() != read(root/'source_freeze.json'): raise ValueError('Runtime source drift')
        code = 0
        beat(root, state='COMPLETE', stage='diagnostics_complete_no_training', exit_code=0,
            reason='Finite diagnosis complete; equivalent path and existing plateau do not justify retraining')
    except BaseException as exc:
        code = 2 if isinstance(exc, (TimeoutError, InterruptedError)) else 1
        write(root/'error.json', dict(type=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc()))
        beat(root, state='BUDGET_STOP' if code == 2 else 'FAILED', exit_code=code, reason=str(exc))
    finally:
        if monitor is not None:
            (root/'STOP_MONITOR').touch()
            try: monitor.wait(timeout=15)
            except subprocess.TimeoutExpired: kill_owned(monitor)
            log.close()
            path = root/'resources/resources.json'
            if path.exists():
                summary = read(path)
                samples = [json.loads(line) for line in (root/'resources/resources.jsonl').read_text().splitlines()]
                summary['configured_limits'] = cfg['resources']
                summary['below_vram_reserve_samples'] = sum(r['gpu_free_mib'] is not None and r['gpu_free_mib']<6144 for r in samples)
                summary['complete_60s_above_cap_samples'] = sum(r['rolling_60s_percent'] is not None and r['rolling_60s_percent']>50 for r in samples)
                summary['actual_min_gpu_free_mib'] = min((r['gpu_free_mib'] for r in samples if r['gpu_free_mib'] is not None), default=None)
                summary['hard_gpu_quota_guaranteed'] = False
                write(root/'resource_summary.json', summary)
        beat(root, owned_monitor_exited=monitor is None or monitor.poll() is not None)
        report(root)
        lock.close()
    return code


def trained_diagnostics(root):
    """Final train8/validation30 only; no test selection or physical rollouts."""
    import torch
    import h5py
    import numpy as np
    from visuotactile.state_history import PersistentState
    from visuotactile.state_image_loss import GeometryObjective, PALM_LOSS_ID
    from visuotactile.state_image_offline import deployment_sample,diagnostic_rows,summarize_diagnostics,sampled_metrics,no_kit
    from visuotactile.state_resources import lease_for
    from tools.narrow_repair_audit import stages
    cfg=read(root/'configuration.json');s=read(root/'session.json');data=read(root/'state_data/data.json')
    plan=cfg['palm_training'];out=root/'offline';out.mkdir(exist_ok=False)
    lease=lease_for(Path(s.get('resource_root',root)),deadline=s['offline_deadline']);cohorts=plan['diagnostic_cohorts']
    saved=torch.load(root/'training/best.pt',map_location='cpu',weights_only=False)
    if saved['configuration']['loss_id']!=PALM_LOSS_ID or saved['configuration']['data_sha256']!=sha(root/'state_data/data.json'):
        raise ValueError('Selected model/data identity mismatch')
    cache=torch.load(root/'state_data/numeric.pt',map_location='cpu',weights_only=False)
    if cache['data_sha256']!=sha(root/'state_data/data.json'):raise ValueError('Numeric data identity changed')
    with h5py.File(data['entries'][cohorts['train'][0]]['path']) as f:metadata=json.loads(f.attrs['metadata_json'])
    objective=GeometryObjective(metadata,mode=PALM_LOSS_ID);objective.weights=cfg['loss_weights']
    if saved['contract'].get('modality'):
        from visuotactile.state_history import PersistentImage
        from visuotactile.native_data import visual_records
        model=PersistentImage('I-HistoryDP',saved['contract'])
        cache['records']=visual_records(root/'state_data',sum(cohorts.values(),[]),saved['contract']['modality'])
    else:model=PersistentState('dp',saved['contract'])
    model.load_state_dict(saved['ema'],strict=True);model.eval()
    stats=torch.load(root/'state_data/statistics.pt',map_location='cpu',weights_only=False)
    if not all(torch.equal(getattr(model,k),v) for k,v in stats.items()):raise ValueError('Checkpoint normalization mismatch')
    with lease.block():model.cuda()
    rows=[];full_rows=[];summaries=[];completed=0;plans=0;ticks=0
    for split,ids in cohorts.items():
        if not set(ids)<=set(data['splits'][split]):raise ValueError('Fixed diagnostic cohort changed')
        for index,identity in enumerate(ids):
            with lease.block():rec={k:v.cuda() if torch.is_tensor(v) else v for k,v in cache['records'][identity].items()}
            def progress(**values):write(out/'progress.json',dict(offline_episodes=completed,offline_total=38,identity=identity,utc=time.time(),**values))
            sample=deployment_sample(model,rec,objective,lease,full_horizon=True,progress=progress)
            entry=data['entries'][identity]
            with h5py.File(entry['path']) as f:
                g=f['episodes'][entry['episode']];phases=stages(g)
                contact=np.linalg.norm(g['transition/object_contact_force_w'][:],axis=-1).reshape(len(phases),-1).max(-1)>.1
                lift=g['transition/diagnostics_before/lifted_object'][:].reshape(len(phases),-1).any(-1)
            rows.extend(diagnostic_rows(sample,phases,identity=identity,split=split))
            full=sample['full_horizon']
            full_rows.extend(diagnostic_rows(full,np.asarray(phases)[full['target_ticks']],identity=identity,split=split))
            summaries.append(dict(identity=identity,split=split,executed12=sampled_metrics(sample),all_valid24=sampled_metrics(full)))
            if index==0:target_plot(out,identity,sample,phases,contact,lift)
            completed+=1;ticks+=len(phases);plans+=math.ceil(len(phases)/12)
            write(out/'progress.json',dict(offline_episodes=completed,offline_total=38,sampled_anchors=plans,native_ticks=ticks,utc=time.time()))
            table(out/'sampled_ticks.csv',rows);table(out/'full_horizon_ticks.csv',full_rows)
            write(out/'stage_metrics.json',summarize_diagnostics(rows));write(out/'full_horizon_stage_metrics.json',summarize_diagnostics(full_rows))
            write(out/'episode_metrics.json',summaries)
    old=read(CODE/plan['previous_diagnostics']/'stage_metrics.json');new=read(out/'stage_metrics.json')
    comparisons=[]
    for current in new:
        if current['motion']!='all':continue
        reference=next(r for r in old if all(r[k]==current[k] for k in ('split','phase','group','motion')))
        comparisons.append(dict(split=current['split'],phase=current['phase'],group=current['group'],ticks=current['ticks'],
            old_rmse_rad=reference['rmse_rad'],new_rmse_rad=current['rmse_rad'],hold_rmse_rad=current['hold_rmse_rad'],
            old_palm_rmse_mm=reference['ee_rmse_mm'],new_palm_rmse_mm=current['ee_rmse_mm'],new_palm_angle_mean_deg=current['palm_angle_mean_deg']))
    table(out/'stage_comparison.csv',comparisons)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for ax,group in zip(axes[:2],('arm','hand')):
        rr=[r for r in comparisons if r['split']=='validation' and r['group']==group and r['ticks']]
        reference_label='StateDP Palm19865' if saved['contract'].get('modality') else 'old Core8500'
        x=np.arange(len(rr));ax.bar(x-.2,[r['old_rmse_rad'] for r in rr],.4,label=reference_label)
        ax.bar(x+.2,[r['new_rmse_rad'] for r in rr],.4,label='new selected');ax.set_ylabel(group+' RMSE (rad)')
        ax.set_xticks(x,[r['phase'] for r in rr],rotation=70);ax.legend()
    rr=[r for r in comparisons if r['split']=='validation' and r['group']=='arm' and r['ticks']]
    axes[2].plot([r['phase'] for r in rr],[r['old_palm_rmse_mm'] for r in rr],label=reference_label)
    axes[2].plot([r['phase'] for r in rr],[r['new_palm_rmse_mm'] for r in rr],label='new selected')
    axes[2].set_ylabel('Command FK palm RMSE (mm)');axes[2].tick_params(axis='x',rotation=70);axes[2].legend()
    fig.tight_layout();fig.savefig(out/'stage_comparison.png',dpi=140);plt.close(fig)
    write(out/'result.json',dict(complete=True,episodes=completed,ticks=ticks,plans=plans,
        selected_update=saved['updates'],checkpoint_sha256=sha(root/'training/best.pt'),selection='fixed validation4 before final38',
        new_closed_loop=0,no_new_task_success_evidence=True,**no_kit()))


def publish_metrics(root, evaluation, output, *, pair_root=None):
    """CPU-only publication from completed receipts; never load a policy or Kit.

    Raw runs remain immutable. The exported CSV/JSON is sufficient to redraw the
    figures; large checkpoints, images and simulator trajectories stay local.
    """
    import gzip
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if pair_root is None and (read(root/'status.json')['state'] != 'COMPLETE' or read(evaluation/'status.json')['state'] != 'COMPLETE'):
        raise ValueError('Publication requires completed training and evaluation')
    if pair_root is not None:
        finished=read(root/'training/finished.json');offline=read(root/'offline/result.json')
        summary=read(evaluation/'evaluation_summary.json')
        if not offline['complete'] or summary['attempts']!=10 or len(summary['results'])!=10:
            raise ValueError('Visual publication requires complete per-group evidence')
        if not all(r['software_complete'] for r in summary['results']):
            raise ValueError('Incomplete visual trial; do not publish as completed')
    reference_label='StateDP Palm19865' if pair_root else 'old Core8500'
    selected_label=root.name+' selected' if pair_root else 'new selected19865'
    output.mkdir(parents=True, exist_ok=True)

    def public(value):
        if isinstance(value, dict):
            return {public(k): public(v) for k, v in value.items()}
        if isinstance(value, list):
            return [public(v) for v in value]
        if isinstance(value, str):
            # Shared outputs can resolve into a different frozen worktree.
            for prefix in ('/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup/',
                           '/home/CNS2025331827/docs/bc+rl_RSS_palm_eval/',
                           '/home/CNS2025331827/docs/bc+rl_RSS/'):
                value = value.replace(prefix, '')
            value=value.replace('/home/CNS2025331827/', '~/')
        return value

    def csv_rows(name):
        with (output/name).open(newline='') as stream:
            return list(csv.DictReader(stream))

    def compressed_copy(source, name):
        with source.open('rb') as src, (output/name).open('wb') as dst:
            with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as archive:
                shutil.copyfileobj(src, archive)

    validation = [json.loads(line) for line in (root/'training/validation.jsonl').read_text().splitlines()]
    training = [json.loads(line) for line in (root/'training/training.jsonl').read_text().splitlines()]
    curves, losses = [], []
    previous = 0
    for row in validation:
        for key, cohort in (('train8', 'fixed_train2'), ('metrics', 'fixed_validation4')):
            for horizon, values in (('commit12', row[key]), ('valid24', row[key]['full_horizon'])):
                curves.append(dict(update=row['update'], anchor_epochs=row['anchor_equivalent_epochs'],
                    cohort=cohort, horizon=horizon, **{k: values[k] for k in (
                        'arm_rmse', 'hand_rmse', 'ee_distance_rmse_mm', 'ee_rotation_mean_deg',
                        'selection_score', 'ticks', 'hold_arm_rmse', 'hold_hand_rmse')}))
        part = [r for r in training if previous < r['update'] <= row['update']]
        previous = row['update']
        losses.append(dict(update=row['update'], anchor_epochs=row['anchor_equivalent_epochs'], updates_in_interval=len(part),
            interval_base=sum(r['components']['base'] for r in part)/len(part),
            interval_weighted_position=sum(r['components']['position']*r['loss_weights']['position'] for r in part)/len(part),
            interval_weighted_rotation=sum(r['components']['rotation']*r['loss_weights']['rotation'] for r in part)/len(part),
            fixed_train2_loss=row['train8_loss']['loss'], fixed_validation4_loss=row['validation_loss']['loss']))
    table(output/'sampling_curves.csv', curves)
    table(output/'loss_curves.csv', losses)
    compressed_copy(root/'training/training_curves.csv', 'every_update_loss.csv.gz')
    for name in ('sampled_ticks.csv', 'full_horizon_ticks.csv'):
        compressed_copy(root/'offline'/name, name+'.gz')
    for p in sorted((root/'offline').glob('*_targets.*')):
        if p.suffix in ('.csv', '.png'):
            if pair_root and p.suffix=='.csv':compressed_copy(p,p.name+'.gz')
            else:shutil.copyfile(p, output/p.name)
    for name in ('stage_comparison.csv', 'stage_metrics.json', 'full_horizon_stage_metrics.json', 'episode_metrics.json'):
        shutil.copyfile(root/'offline'/name, output/name)

    # All plots below read the published CSV/JSON, not model outputs in memory.
    curves = csv_rows('sampling_curves.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout='constrained')
    fields = ('arm_rmse', 'hand_rmse', 'ee_distance_rmse_mm', 'ee_rotation_mean_deg')
    titles = ('Arm RMSE (rad)', 'Hand RMSE (rad)', 'Command FK palm RMSE (mm)', 'Command FK palm mean angle (deg)')
    for ax, field, title in zip(axes.flat, fields, titles):
        for cohort, color in (('fixed_train2', 'tab:blue'), ('fixed_validation4', 'tab:orange')):
            for horizon, style in (('commit12', '-'), ('valid24', '--')):
                rr = [r for r in curves if r['cohort'] == cohort and r['horizon'] == horizon]
                ax.plot([float(r['anchor_epochs']) for r in rr], [float(r[field]) for r in rr],
                        style, color=color, label=cohort+' / '+horizon)
        ax.set(title=title, xlabel='Anchor-equivalent epochs'); ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Expert-history deployment sampling: DDIM16, seed7091; full real prefixes')
    fig.savefig(output/'sampling_curves.png', dpi=150); plt.close(fig)

    losses = csv_rows('loss_curves.csv')
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout='constrained')
    for field in ('interval_base', 'interval_weighted_position', 'interval_weighted_rotation'):
        axes[0].plot([float(r['anchor_epochs']) for r in losses], [float(r[field]) for r in losses], label=field)
    for field in ('fixed_train2_loss', 'fixed_validation4_loss'):
        axes[1].plot([float(r['anchor_epochs']) for r in losses], [float(r[field]) for r in losses], label=field)
    for ax, title in zip(axes, ('Every-update loss, mean within validation interval', 'Fixed-cohort deterministic denoising loss')):
        ax.set(title=title, xlabel='Anchor-equivalent epochs', ylabel='Weighted training objective', yscale='log')
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output/'loss_curves.png', dpi=150); plt.close(fig)

    comparisons = csv_rows('stage_comparison.csv')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for row_axes, split in zip(axes, ('train', 'validation')):
        for ax, group in zip(row_axes[:2], ('arm', 'hand')):
            rr = [r for r in comparisons if r['split'] == split and r['group'] == group and int(r['ticks'])]
            for key, label in (('old_rmse_rad', reference_label), ('new_rmse_rad', selected_label), ('hold_rmse_rad', 'hold current actuator target')):
                ax.plot(range(len(rr)), [float(r[key]) for r in rr], 'o-', label=label)
            ax.set(ylabel=group+' RMSE (rad)', title=split+' / commit12')
            ax.set_xticks(range(len(rr)), [r['phase'].replace('_', ' ')+'\nn='+r['ticks'] for r in rr], rotation=55, ha='right', fontsize=7)
            ax.grid(alpha=.25)
        rr = [r for r in comparisons if r['split'] == split and r['group'] == 'arm' and int(r['ticks'])]
        for key, label in (('old_palm_rmse_mm', reference_label), ('new_palm_rmse_mm', selected_label)):
            row_axes[2].plot(range(len(rr)), [float(r[key]) for r in rr], 'o-', label=label)
        row_axes[2].set(ylabel='Command FK palm RMSE (mm)', title=split+' / commit12')
        row_axes[2].set_xticks(range(len(rr)), [r['phase'].replace('_', ' ')+'\nn='+r['ticks'] for r in rr], rotation=55, ha='right', fontsize=7)
        row_axes[2].grid(alpha=.25)
    axes[0, 0].legend(fontsize=7); axes[0, 2].legend(fontsize=7)
    fig.savefig(output/'stage_errors.png', dpi=150); plt.close(fig)

    stages = read(output/'stage_metrics.json')
    motion = [r for r in stages if r['phase'] == 'all' and r['motion'] != 'all']
    table(output/'motion_errors.csv', motion)
    motion = csv_rows('motion_errors.csv')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for ax, group in zip(axes, ('arm', 'hand')):
        rr = [r for r in motion if r['group'] == group and int(r['ticks'])]
        for key, label in (('rmse_rad', selected_label), ('hold_rmse_rad', 'hold current actuator target')):
            ax.plot(range(len(rr)), [float(r[key]) for r in rr], 'o-', label=label)
        ax.set_xticks(range(len(rr)), [r['split']+' / '+r['motion']+'\nn='+r['ticks'] for r in rr], rotation=25, ha='right')
        ax.set_ylabel(group+' RMSE (rad)'); ax.grid(alpha=.25); ax.legend(fontsize=8)
    fig.savefig(output/'motion_errors.png', dpi=150); plt.close(fig)

    trials = []
    trial_root=evaluation/'evaluation' if pair_root else evaluation
    for p in sorted(trial_root.glob('palm_fresh_*/result.json')):
        r = read(p)
        trials.append({k: r[k] for k in ('identity', 'seed', 'software_complete', 'legal', 'success', 'steps',
            'native_lift', 'native_subgoals', 'native_alignment', 'actuator_target_max_error',
            'submitted_target_max_error', 'tracking_rmse_rad')} | {'termination': ','.join(k for k, v in r['reasons'].items() if v)})
    table(output/'closed_loop.csv', trials)
    if pair_root is not None:
        write(output/'summary.json',public(dict(training=finished,progress=read(root/'training/progress.json'),
            offline=offline,closed_loop=summary,training_source=read(pair_root/'session.json')['source_head'],
            evaluation_source=read(evaluation.parent/'session.json')['source_head'],
            normalizer_sha256=sha(pair_root/'state_data/statistics.pt'))))
        write(output/'frozen_configuration.json',public(read(root/'training/configuration.json')))
        for p in output.glob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
        return
    manifest = read(root/'manifest.json')
    table(output/'dataset_index.csv', [public({k: r[k] for k in ('id', 'family', 'split', 'steps', 'size', 'sha256', 'path', 'reference_path', 'reference_sha256')}) for r in manifest['episodes']])
    write(output/'summary.json', public(dict(
        training=read(root/'status.json'), progress=read(root/'training/progress.json'),
        training_source=read(root/'session.json')['source_head'],
        selected_checkpoint=read(evaluation/'selected_checkpoint.json'),
        offline=read(root/'offline/result.json'), closed_loop=read(evaluation/'evaluation_summary.json'),
        training_resources=read(root/'resource_summary.json'), evaluation_resources=read(evaluation/'resource_summary.json'),
        normalizer_sha256=sha(root/'state_data/statistics.pt'),
        notes=['Offline errors use expert histories, not autonomous same-tick labels.',
               'Command interface error and physical tracking error are distinct.',
               'Dataset expansion, fresh initialization, loss and exposure changed together; no causal ablation.',
               'Train2/validation4 select weights; final train8/validation30 diagnose; no test selection.'])))
    write(output/'frozen_configuration.json', public(read(root/'configuration.json')))
    write(output/'source_hashes.json', public(read(root/'execution_source_freeze.json')))
    cleanup = root/'cleanup_previous_core.json'
    if cleanup.exists():
        ledger = read(cleanup)
        if ledger['state'] != 'COMPLETE':
            raise ValueError('Do not publish incomplete cleanup as complete')
        write(output/'cleanup.json', public(ledger))
        fields = ('path', 'action', 'bytes', 'sha256', 'archive', 'archive_bytes',
                  'archive_sha256', 'roundtrip_sha256_verified', 'reclaimed_allocated_bytes', 'removed')
        table(output/'cleanup_files.csv', [{k: r.get(k, '') for k in fields} for r in ledger['entries']])
    # Keep public text diffs clean without changing any numeric CSV values.
    # Compressed raw metrics and the original run files retain their bytes.
    for p in output.glob('*.csv'):
        p.write_bytes(p.read_bytes().replace(b'\r\n', b'\n'))
    write(output/'artifacts.json', {p.name: dict(bytes=p.stat().st_size, sha256=sha(p))
          for p in sorted(output.iterdir()) if p.is_file() and p.name != 'artifacts.json'})


def publish_visual_metrics(root, evaluation, output):
    """Publish completed measurements while preserving the failed owner status."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    completion=read(evaluation/'evaluation_complete.json')
    if not completion['complete'] or completion['groups']!=['G00','G10']:
        raise ValueError('Both visual evaluation cohorts must be complete')
    if not read(evaluation/'status.json')['owned_workers_exited']:
        raise ValueError('Do not export a live experiment')
    expected=[r['identity'] for r in read(evaluation/'evaluation_plan.json')]
    if len(expected)!=10 or len(set(expected))!=10:
        raise ValueError('Expected original ten development identities')
    groups={};inputs={};videos=[]

    def public(value):
        if isinstance(value,dict):return {public(k):public(v) for k,v in value.items()}
        if isinstance(value,list):return [public(v) for v in value]
        if isinstance(value,str):
            for prefix in ('/home/CNS2025331827/docs/bc+rl_RSS_palm_eval/',
                           '/home/CNS2025331827/docs/bc+rl_RSS_narrow_followup/',
                           '/home/CNS2025331827/docs/bc+rl_RSS/'):
                value=value.replace(prefix,'')
            return value.replace('/home/CNS2025331827/','~/')
        return value

    def evidence(path):
        inputs[public(str(path))]=dict(bytes=path.stat().st_size,sha256=sha(path))

    for group in ('G00','G10'):
        g=root/group;e=evaluation/group
        offline=read(g/'offline/result.json');summary=read(e/'evaluation_summary.json')
        episodes=read(g/'offline/episode_metrics.json')
        if not offline['complete'] or offline['episodes']!=38:
            raise ValueError('Incomplete final train8/validation30 diagnosis')
        if {split:sum(r['split']==split for r in episodes) for split in ('train','validation')}!={'train':8,'validation':30}:
            raise ValueError('Final diagnostic cohort mismatch')
        if [r['identity'] for r in summary['results']]!=expected:
            raise ValueError('Development queue differs from registered identities')
        for r in summary['results']:
            path=e/'evaluation'/r['identity']/'result.json'
            if read(path)!=r or not r['software_complete']:
                raise ValueError('Trial receipt missing, changed or incomplete')
            if r['legal'] and r['checkpoint_sha256']!=offline['checkpoint_sha256']:
                raise ValueError('Offline and deployed checkpoint differ')
            evidence(path)
        valid=[r for r in summary['results'] if r['legal'] and r['software_complete']]
        if (summary['attempts'],summary['valid'],summary['rejected'],summary['successes'])!=(
                10,len(valid),10-len(valid),sum(bool(r['success']) for r in valid)):
            raise ValueError('Closed-loop counts do not match individual receipts')
        if sha(g/'training/best.pt')!=offline['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        for name in ('training/best.pt','training/finished.json','training/progress.json','training/configuration.json',
                     'training/training.jsonl','training/validation.jsonl','offline/result.json','offline/stage_metrics.json'):
            evidence(g/name)
        for p in sorted((e/'evaluation').glob('*/video_verified.json')):
            receipt=read(p);trial=read(p.parent/'result.json');movie=p.parent/'rollout.mp4'
            if receipt['decoded_frames']!=trial['steps']:
                raise ValueError('Video frame count differs from completed trial')
            evidence(movie);videos.append(dict(group=group,identity=trial['identity'],path=public(str(movie)),
                sha256=sha(movie),**receipt,uploaded=False))
        publish_metrics(g,e,output/group,pair_root=root)
        groups[group]=read(output/group/'summary.json')

    if [r['identity'] for r in read(root/'G00/offline/episode_metrics.json')]!=[
            r['identity'] for r in read(root/'G10/offline/episode_metrics.json')]:
        raise ValueError('Final offline cohorts differ between modalities')
    stage_rows=[];sampling=[]
    for group in groups:
        stage_rows.extend(dict(model=group,**r) for r in read(output/group/'stage_metrics.json'))
        with (output/group/'sampling_curves.csv').open() as f:
            sampling.extend(dict(model=group,**r) for r in csv.DictReader(f))
    table(output/'stage_metrics.csv',stage_rows);table(output/'sampling_curves.csv',sampling)
    fields=('arm_rmse','hand_rmse','ee_distance_rmse_mm','ee_rotation_mean_deg')
    titles=('Arm RMSE (rad)','Hand RMSE (rad)','Command FK palm RMSE (mm)','Command FK palm mean angle (deg)')
    with (output/'sampling_curves.csv').open() as f:sampling=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for ax,field,title in zip(axes.flat,fields,titles):
        for group,color in (('G00','tab:blue'),('G10','tab:orange')):
            for cohort,style in (('fixed_train2','--'),('fixed_validation4','-')):
                rows=[r for r in sampling if r['model']==group and r['cohort']==cohort and r['horizon']=='commit12']
                ax.plot([float(r['anchor_epochs']) for r in rows],[float(r[field]) for r in rows],style,color=color,label=group+' '+cohort)
        ax.set(title=title,xlabel='Anchor-equivalent epochs');ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Expert-prefix DDIM16 sampling; unequal training exposure, no causal tactile claim')
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    manifest=read(root/'manifest.json')
    table(output/'dataset_index.csv',[public({k:r[k] for k in ('id','family','split','steps','size','sha256','path')}) for r in manifest['episodes']])
    receipt_paths={'original_status':root/'status.json','original_error':root/'vision_evaluate_error.json',
        'evaluation_status':evaluation/'status.json','evaluation_error':evaluation/'vision_evaluate_error.json',
        'evaluation_complete':evaluation/'evaluation_complete.json','training_launch':root/'launch.json',
        'evaluation_launch':evaluation/'launch.json','training_resources':root/'monitor_exit.json',
        'evaluation_resources':evaluation/'monitor_exit.json','throughput_comparison':root/'throughput_comparison.json'}
    receipts={k:public(read(p)) for k,p in receipt_paths.items()}
    for p in receipt_paths.values():evidence(p)
    for p in (root/'state_data/data.json',root/'state_data/statistics.pt',root/'configuration.json',evaluation/'evaluation_plan.json'):
        evidence(p)
    write(output/'summary.json',dict(groups=groups,receipts=receipts,videos=videos,
        classification='Measurements complete; evaluation owner state='+receipts['evaluation_status']['state'],
        notes=['Original statuses and failure receipts are preserved, not relabeled COMPLETE.',
            'Use recorded anchor exposure for comparisons; these final models are not an equal-update tactile ablation.',
            'Offline errors use full expert prefixes; autonomous traces have no same-tick expert-label claim.',
            'Final train8/validation30 is diagnostic; test22 unused; no new training or simulation.',
            'Native complete-success counts are reported per cohort; lift/subgoals are separate.',
            'Training resource telemetry has gaps; monitor RSS is not total worker memory.']))
    write(output/'frozen_configuration.json',public(read(root/'configuration.json')))
    write(output/'evaluation_plan.json',read(evaluation/'evaluation_plan.json'))
    write(output/'source_hashes.json',dict(training=read(root/'source_freeze.json'),evaluation=read(evaluation/'source_freeze.json')))
    write(output/'source_artifacts.json',inputs)
    for p in output.rglob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    write(output/'artifacts.json',{str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifacts.json'})


def publish_residual_metrics(root, evaluation, output):
    """Publish completed residual evidence on CPU; no training or simulation."""
    import gzip
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from visuotactile.video import verify_video
    status=read(root/'status.json');completion=read(evaluation/'evaluation_complete.json')
    if root!=evaluation or status['state']!='COMPLETE' or not status['owned_workers_exited'] or not completion['complete']:
        raise ValueError('Residual publication requires its completed, exited queue')
    if completion['groups']!=['G01','G11']:raise ValueError('Unexpected residual cohort')
    data=read(root/'state_data/data.json');plan=read(root/'evaluation_plan.json');expected=[r['identity'] for r in plan]
    if len(expected)!=10 or len(set(expected))!=10:raise ValueError('Development identities differ')
    output.mkdir(parents=True,exist_ok=True);sources={};groups={};videos=[];closed=[];stages=[];curves=[];force=[]
    def public(value):
        if isinstance(value,dict):return {public(k):public(v) for k,v in value.items()}
        if isinstance(value,list):return [public(v) for v in value]
        if isinstance(value,str):
            for prefix in ('/home/CNS2025331827/docs/bc+rl_RSS_palm_eval/','/home/CNS2025331827/docs/bc+rl_RSS/'):
                value=value.replace(prefix,'')
            return value.replace('/home/CNS2025331827/','~/')
        return value
    def record(path):sources[public(str(path))]=dict(bytes=path.stat().st_size,sha256=sha(path))
    def copy(source,dest):
        record(source)
        if dest.suffix=='.gz':
            with source.open('rb') as src,dest.open('wb') as dst:
                with gzip.GzipFile(filename='',fileobj=dst,mode='wb',mtime=0) as archive:shutil.copyfileobj(src,archive)
        elif source.suffix=='.json':write(dest,public(read(source)))
        else:shutil.copyfile(source,dest)
    summary=read(root/'result.json')
    for group in completion['groups']:
        g=root/group;out=output/group;out.mkdir(exist_ok=True)
        offline=read(g/'offline/result.json');evaluation_rows=read(g/'evaluation_summary.json')['results']
        finished=read(g/'training/finished.json');history=read(g/'training/validation.json')
        if not offline['complete'] or sha(g/'training/best.pt')!=offline['checkpoint_sha256']:
            raise ValueError('Selected residual identity differs')
        if [r['identity'] for r in evaluation_rows]!=expected:raise ValueError('Evaluation plan mismatch')
        for split,number in (('train',8),('validation',30)):
            with (g/'offline/sampled_ticks.csv').open() as f:
                identities={r['identity'] for r in csv.DictReader(f) if r['split']==split}
            if identities!=set(data['splits'][split][:number]):raise ValueError('Offline cohort mismatch')
        for i,r in enumerate(evaluation_rows):
            path=g/'evaluation'/r['identity']/'result.json'
            if read(path)!=r or not r['software_complete'] or not r['legal']:
                raise ValueError('Publication requires all ten completed legal trial receipts')
            if r['checkpoint_sha256']!=offline['checkpoint_sha256'] or r['seed']!=plan[i]['spec']['seed']:
                raise ValueError('Deployed model/seed differs')
            if bool(r['success'])!=bool(r['reasons']['max_successes']):raise ValueError('Native success mismatch')
            if r['post_success_actions']!=0 or r['residual_ticks']!=r['steps']:raise ValueError('Tick/termination mismatch')
            record(path)
            closed.append(dict(group=group,identity=r['identity'],seed=r['seed'],success=r['success'],steps=r['steps'],
                native_lift=r['native_lift'],native_subgoals=r['native_subgoals'],alignment=r['native_alignment'],
                stable_grasp=r['contact']['stable_grasp_diagnostic'],**r['reasons'],
                first_success_tick=r['first_success_tick'],post_success_actions=r['post_success_actions'],
                submitted_target_max_error_rad=r['submitted_target_max_error'],tracking_rmse_rad=r['tracking_rmse_rad'],
                contact_gate_ticks=r['contact_gate_ticks'],checkpoint_sha256=r['checkpoint_sha256']))
        for p in sorted((g/'evaluation').glob('*/video_verified.json')):
            movie=p.parent/'rollout.mp4';trial=read(p.parent/'result.json');receipt=verify_video(movie,expected_frames=trial['steps'])
            if receipt!=read(p):raise ValueError('Video decode differs from stored receipt')
            record(movie);videos.append(dict(group=group,identity=trial['identity'],success=trial['success'],
                path=public(str(movie)),sha256=sha(movie),uploaded=False,**receipt))
        for name in ('training/best.pt','training/last.pt'):record(g/name)
        for name in ('training/finished.json','training/configuration.json','training/validation.json','offline/result.json',
                     'offline/stage_errors.csv','offline/base_stage_errors.csv','offline/force_sensitivity.json','nominal/scales.json'):
            copy(g/name,out/Path(name).name)
        for name in ('sampled_ticks.csv','base_sampled_ticks.csv'):
            copy(g/'offline'/name,out/(name+'.gz'))
        copy(g/'training/loss.jsonl',out/'every_update_loss.jsonl.gz')
        copy(g/'training/curves.png',out/'training_curves.png')
        for p in sorted((g/'offline').glob('*')):
            if p.suffix=='.png' or p.name.endswith('_targets.csv'):
                copy(p,out/(p.name+('.gz' if p.suffix=='.csv' else '')))
        for row in history:
            for split in ('train','validation'):
                for scope,metrics in row[split].items():
                    curves.append(dict(group=group,epoch=row['epoch'],update=row['update'],cohort='fixed_'+split+('2' if split=='train' else '4'),scope=scope,**metrics))
        for kind,key in (('residual','metrics'),('frozen_base','base_metrics')):
            stages.extend(dict(model=group,kind=kind,**r) for r in offline[key])
        force.extend(dict(group=group,**r) for r in offline['force_sensitivity'])
        selected=min(history,key=lambda r:r['validation']['contact']['normalized_equal_mse'])
        groups[group]=dict(training=public(finished),selection_record=selected,checkpoint_sha256=offline['checkpoint_sha256'],
            closed_loop=summary['groups'][group]['closed_loop'],frozen_base_closed_loop=summary['groups'][group]['frozen_base_closed_loop'])
    if len(videos)!=4 or len(closed)!=20:raise ValueError('Expected four complete videos and twenty trials')
    table(output/'closed_loop.csv',closed);table(output/'sampling_curves.csv',curves)
    table(output/'stage_metrics.csv',stages);table(output/'force_sensitivity.csv',force)
    # One compact comparison regenerated from the exported raw CSV only.
    with (output/'stage_metrics.csv').open() as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(11,3.6),layout='constrained')
    for ax,part,key,title in zip(axes,('arm','hand','arm'),('rmse_rad','rmse_rad','ee_rmse_mm'),('Arm RMSE (rad)','Hand RMSE (rad)','Target palm FK RMSE (mm)')):
        for offset,kind in ((-.18,'frozen_base'),(.18,'residual')):
            values=[next(float(r[key]) for r in rows if r['model']==g and r['kind']==kind and r['split']=='validation'
                and r['phase']=='all' and r['motion']=='all' and r['group']==part) for g in completion['groups']]
            ax.bar([offset,1+offset],values,width=.36,label=kind)
        ax.set_xticks([0,1],['G01 / G00','G11 / G10']);ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('Fixed expert prefixes, validation30; not autonomous trajectory errors')
    fig.savefig(output/'comparison.png',dpi=150);plt.close(fig)
    launch=read(root/'launch.json');finish=status['utc']
    telemetry=[json.loads(line) for line in (root/'resources/shared.jsonl').read_text().splitlines()]
    telemetry=[r for r in telemetry if launch['started']<=r['utc_epoch']<=finish]
    intervals=[(b['utc_epoch']-a['utc_epoch'],a['gpu_util_percent']) for a,b in zip(telemetry,telemetry[1:])]
    valid=[(dt,value) for dt,value in intervals if dt<=8 and value is not None]
    resources=dict(scope='formal launch through exit; gate monitor restarts reported separately',samples=len(telemetry),
        gpu_time_weighted_percent=sum(dt*v for dt,v in valid)/sum(dt for dt,_ in valid),
        max_gap_s=max(dt for dt,_ in intervals),gap_seconds_above8=sum(dt for dt,_ in intervals if dt>8),
        leading_uncovered_s=telemetry[0]['utc_epoch']-launch['started'],trailing_uncovered_s=finish-telemetry[-1]['utc_epoch'],
        min_gpu_free_mib=min(r['gpu_free_mib'] for r in telemetry),min_ram_available_bytes=min(r['ram_available_bytes'] for r in telemetry),
        full_authorization=True,inherited_monitor_summary=read(root/'monitor_exit.json'))
    errors=[line for line in (root/'evaluation.log').read_text(errors='replace').splitlines() if 'PhysX error:' in line]
    for name in ('status.json','launch.json','evaluation_complete.json','evaluation_plan.json','configuration.json','source_freeze.json',
                 'base_identities.json','throughput_comparison.json','monitor_exit.json','process_exit_requested.json'):
        copy(root/name,output/name)
    record(root/'evaluation.log');record(root/'resources/shared.jsonl');record(root/'state_data/data.json')
    write(output/'summary.json',dict(groups=groups,videos=videos,resources=resources,
        formal_elapsed_s=finish-launch['started'],source_commit=launch['source_head'],status=status,
        physx_log_errors=errors,physx_error_impact='Not specifically audited; retain native receipts and original log, do not infer no impact from exit0',
        data=dict(core=218,train=166,validation=30,test=22,data_sha256=sha(root/'state_data/data.json'),test_used=False),
        scope='Existing development measurements only; no new model inference, training, Kit or cleanup; same seeds, hidden-state bitwise pairing unproven'))
    write(output/'source_artifacts.json',sources)
    for p in output.rglob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    write(output/'artifacts.json',{str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.rglob('*')) if p.is_file() and p.name!='artifacts.json'})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'check', 'run', 'status', 'final', 'publish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CODE/'configs/tight_dp3_diagnostics.json')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode == 'prepare': prepare(root, args.config)
    elif args.mode == 'status': print(json.dumps(read(root/'status.json'), indent=2))
    elif args.mode == 'publish':
        if args.evaluation is None or args.output is None:
            parser.error('publish requires --evaluation and --output')
        if read(root/'configuration.json').get('residual_pair') and (root/'base_identities.json').exists():
            publish_residual_metrics(root,args.evaluation.resolve(),args.output.resolve())
        elif read(root/'configuration.json').get('vision_pair'):
            publish_visual_metrics(root, args.evaluation.resolve(), args.output.resolve())
        else:publish_metrics(root, args.evaluation.resolve(), args.output.resolve())
    else:
        import torch
        from visuotactile.precision import state_fp32
        torch.set_num_threads(2); state_fp32()
        if args.mode == 'final':
            trained_diagnostics(root)
        elif args.mode == 'check': acceptance(root)
        else: raise SystemExit(run(root))


if __name__ == '__main__': main()
