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


def target_plot(root, identity, sample, phases, contact, lift, *, target_semantics='u_expert'):
    import matplotlib.pyplot as plt
    rows = []
    for tick in range(len(phases)):
        row = dict(tick=tick, seconds=tick/60, phase=str(phases[tick]), contact=bool(contact[tick]), lift=bool(lift[tick]), replan=tick%12 == 0, target_semantics=target_semantics)
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
            ax.set(xlabel='Expert-history time (s)', ylabel=(target_semantics+' target' if row_index == 0 else 'Within-plan target change')+' (rad)', title=f'Canonical joint {joint}')
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


def configuration_geometry_rows(sample, context, geometry, *, identity, split,
                                penetration_tolerance_m, contact_tolerance_m):
    """Measure saved predictions at every12 ticks in the expert's after-state scene.

    This is a surface-consistency diagnostic, not autonomous physics or an
    action-to-force model. No projection, policy call, or optimizer is used.
    """
    import torch
    indices=torch.arange(0,len(sample['prediction']),12)
    q=sample['prediction'][indices].detach().double().cpu()
    truth=sample['truth'][indices].detach().double().cpu()
    pose=context['object_pose_base'][indices].double().cpu()
    contacts=context['contact_mask'][indices].bool().cpu()
    distal=[[i for i in group if geometry.links[i].endswith(('_DP','_elastomer'))]
            for group in geometry.finger_indices]
    if any(not group for group in distal):raise ValueError('Surface diagnostic lacks a distal collider')
    with torch.no_grad():
        predicted=geometry.distances(q,pose).cpu()
        reference=geometry.distances(truth,pose).cpu()
    rows=[]
    for row,tick in enumerate(indices.tolist()):
        item=dict(identity=identity,split=split,tick=tick,phase=int(context['stage'][tick]),
                  required_contacts=int(contacts[row].sum()))
        for label,distances in (('prediction',predicted),('reference_reached',reference)):
            d=distances[row]
            finger=torch.stack([d[group].min() for group in distal])
            kept=(finger>=-penetration_tolerance_m)&(finger<=contact_tolerance_m)&contacts[row]
            item.update({label+'_min_surface_distance_m':float(d.min()),
                label+'_maximum_penetration_m':float((-d.min()).clamp_min(0)),
                label+'_within_penetration_tolerance':bool(d.min()>=-penetration_tolerance_m),
                label+'_kept_required_contacts':int(kept.sum())})
        rows.append(item)
    return rows



def accepted_matrix_physics_gate(session):
    """Resolve the matrix owner's selected attempt, including repaired gates."""
    owner = Path(session['matrix_owner'])
    owner_session = read(owner/'session.json') if (owner/'session.json').is_file() else {}
    return owner/owner_session.get('accepted_physics_gate', session.get('accepted_physics_gate', 'physics_gate'))


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
    target_semantics = saved['contract'].get('supervision', 'u_expert')
    surface = None; surface_rows = []; surface_options = None
    surface_report = dict(state='unmeasured',reason='No admitted cooked collision geometry for this configuration target')
    if target_semantics == 'q_reached' and s.get('matrix_owner'):
        live_path = accepted_matrix_physics_gate(s)/'geometry_live.json'
        if live_path.exists():
            live = read(live_path)
            if live.get('live_collision_parity') and live.get('inventory',{}).get('cooked_hulls'):
                from visuotactile.kinematics import HandCollisionGeometry
                surface = HandCollisionGeometry(metadata, live['inventory'])
                matrix_options=read(Path(s['matrix_owner'])/'configuration.json')['method_matrix']
                surface_options=matrix_options.get('geometry_soft_options',matrix_options.get('geometry_projection',
                    dict(penetration_tolerance_m=.001,contact_tolerance_m=.002)))
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
            if surface is not None:
                from visuotactile.state_image_loss import geometry_loss_context
                surface_rows.extend(configuration_geometry_rows(sample, geometry_loss_context(entry), surface,
                    identity=identity,split=split,penetration_tolerance_m=surface_options['penetration_tolerance_m'],
                    contact_tolerance_m=surface_options['contact_tolerance_m']))
            rows.extend(diagnostic_rows(sample,phases,identity=identity,split=split))
            full=sample['full_horizon']
            full_rows.extend(diagnostic_rows(full,np.asarray(phases)[full['target_ticks']],identity=identity,split=split))
            summaries.append(dict(identity=identity,split=split,executed12=sampled_metrics(sample),all_valid24=sampled_metrics(full)))
            if index==0:target_plot(out,identity,sample,phases,contact,lift,target_semantics=target_semantics)
            completed+=1;ticks+=len(phases);plans+=math.ceil(len(phases)/12)
            write(out/'progress.json',dict(offline_episodes=completed,offline_total=38,sampled_anchors=plans,native_ticks=ticks,utc=time.time()))
            table(out/'sampled_ticks.csv',rows);table(out/'full_horizon_ticks.csv',full_rows)
            write(out/'stage_metrics.json',summarize_diagnostics(rows));write(out/'full_horizon_stage_metrics.json',summarize_diagnostics(full_rows))
            write(out/'episode_metrics.json',summaries)
    if surface_rows:
        table(out/'geometry_rows.csv',surface_rows)
        surface_report=dict(state='complete',geometry_identity=surface.identity,geometry_assets=surface.assets,
            sampled_ticks=len(surface_rows),sampling='Every12 recorded control ticks, fixed before evaluation',
            scope='Predicted configuration in recorded expert after-state object scene; not autonomous physical penetration or internal force',
            surface='PhysX cooked hull surface; contact/rest offsets excluded',splits={})
        for split in cohorts:
            selected=[r for r in surface_rows if r['split']==split];required=sum(r['required_contacts'] for r in selected)
            surface_report['splits'][split]=dict(sampled_ticks=len(selected),required_contacts=required)
            for label in ('prediction','reference_reached'):
                kept=sum(r[label+'_kept_required_contacts'] for r in selected)
                surface_report['splits'][split][label]=dict(
                    maximum_penetration_m=max(r[label+'_maximum_penetration_m'] for r in selected),
                    mean_maximum_penetration_m=sum(r[label+'_maximum_penetration_m'] for r in selected)/len(selected),
                    ticks_exceeding_penetration_tolerance=sum(not r[label+'_within_penetration_tolerance'] for r in selected),
                    kept_required_contacts=kept,contact_retention=kept/required if required else None)
    if target_semantics == 'q_reached': write(out/'geometry_metrics.json',surface_report)
    if target_semantics == 'q_reached':
        write(out/'stage_comparison_omitted.json', dict(reason='Previous reference fits u_expert; q_reached target errors are not comparable',
            current_target='q_reached', previous_target='u_expert', current_metrics='stage_metrics.json'))
    else:
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
        target_semantics=target_semantics, old_command_reference_comparable=target_semantics=='u_expert',
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


def dagger_publish(root, output=None):
    """Publish retained DAgger receipts on CPU, including failed/partial rounds.

    No checkpoint is deserialized and no model, simulator, decoder, collection,
    training, selection, cleanup or git operation is invoked here.
    """
    import gzip
    root=Path(root).resolve()
    output=Path(output).resolve() if output is not None else CODE/'research_docs/evidence'/root.name
    if output==root or root in output.parents:raise ValueError('Publication must not overwrite run evidence')
    output.mkdir(parents=True,exist_ok=True)
    issues=[];sources={};artifacts={};hashes={}
    def public(value):
        if isinstance(value,dict):return {str(public(k)):public(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [public(v) for v in value]
        if isinstance(value,float) and not math.isfinite(value):return None
        if isinstance(value,str):
            for prefix in (str(CODE)+'/',str(root.parent.parent.parent.parent)+'/',
                '/home/CNS2025331827/docs/bc+rl_RSS_palm_eval/','/home/CNS2025331827/docs/bc+rl_RSS/'):
                value=value.replace(prefix,'')
            return value.replace(str(Path.home())+'/','~/')
        return value
    def evidence(path,kind='metric',declared=None):
        path=Path(path);key=public(str(path));row=dict(path=key,kind=kind,exists=path.is_file(),uploaded=False)
        if row['exists']:
            stat=path.stat();token=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
            if kind=='trajectory':digest=declared;scope='Collector manifest SHA; raw trajectory not opened by publisher'
            else:
                if token not in hashes:hashes[token]=sha(path)
                digest=hashes[token];scope='Read and SHA256 verified during CPU publication'
            row.update(bytes=stat.st_size,sha256=digest,hash_scope=scope)
            if declared and digest!=declared:issues.append(dict(problem='declared_SHA_mismatch',path=key,declared=declared,actual=digest))
        else:row.update(sha256=declared,hash_scope='Declared only; missing file')
        (artifacts if kind in ('checkpoint','trajectory','video') else sources)[key]=row
        return row
    def optional(path,default=None):
        if not path.is_file():return default
        evidence(path)
        try:return read(path)
        except (ValueError,OSError) as error:
            issues.append(dict(problem='unreadable_json_receipt',path=str(path),error=str(error)));return default
    def lines(path):
        if not path.is_file():return []
        evidence(path);rows=[]
        with path.open() as stream:
            for number,line in enumerate(stream,1):
                if not line.strip():continue
                try:
                    value=json.loads(line)
                    if not isinstance(value,dict):raise ValueError('Expected metric object')
                    rows.append(value)
                except ValueError:issues.append(dict(problem='incomplete_or_invalid_jsonl_line',path=str(path),line=number))
        return rows
    def number(value):
        try:result=float(value)
        except (TypeError,ValueError):return None
        return result if math.isfinite(result) else None
    def numeric(value,prefix=''):
        result={}
        for key,item in value.items():
            if isinstance(item,dict):result.update(numeric(item,prefix+key+'.'))
            elif isinstance(item,(int,float)):result[prefix+key]=number(item)
        return result
    def emit(name,rows):
        path=output/name
        if not rows:path.unlink(missing_ok=True);return
        fields=list(dict.fromkeys(k for row in rows for k in row))
        clean=[{k:public(row.get(k)) for k in fields} for row in rows]
        if path.suffix=='.gz':
            import io
            with path.open('wb') as raw,gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped:
                with io.TextIOWrapper(zipped,encoding='utf-8',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(clean)
        else:table(path,clean)
    def csv_read(path):
        if not path.is_file():return []
        evidence(path)
        with path.open(newline='') as stream:return list(csv.DictReader(stream))
    cfg=optional(root/'configuration.json',{});status=optional(root/'status.json',{})
    pool_mode=cfg.get('version')=='successful_takeover_pool_v1'
    if not pool_mode and status.get('state')=='RUNNING' and not status.get('owned_workers_exited'):
        raise ValueError('Publish after the DAgger owner stops its workers')
    session=optional(root/'session.json',{});launch=optional(root/'launch.json',{})
    registration=optional(root/'registration.json',{})
    plans=optional(root/'collection_plan.json',[]);eval_plan=optional(root/'evaluation_plan.json',[])
    groups=list(cfg.get('groups',{})) or sorted(p.name for p in (root/'round_01').glob('*') if p.is_dir())
    round_names=[f'round_{i:02d}' for i in range(1,int(cfg.get('rounds',1))+1)] if pool_mode else ['round_01']
    if pool_mode:round_names=sorted(set(round_names)|{p.name for p in root.glob('round_[0-9][0-9]') if p.is_dir()})
    round_reports={name:dict(status=optional(root/name/'round_status.json'),
        finished=optional(root/name/'round_finished.json'),decision=optional(root/name/'decision.json'),
        snapshot=optional(root/name/'snapshot.json'),groups={}) for name in round_names} if pool_mode else {}
    group_reports={};collection_rows=[];closed_rows=[];stage_rows=[];curves=[];losses=[];errors=[];videos=[];all_plans={};sampling_rows=[]
    def video(node,group,cohort,identity,result):
        for movie in sorted(node.glob('*.mp4')):
            receipt=optional(node/'video_verified.json',{})
            videos.append(dict(round=round_name,group=group,cohort=cohort,identity=identity,success=result.get('success'),
                **evidence(movie,'video'),decoded_frames=receipt.get('decoded_frames'),frame_rate=receipt.get('frame_rate'),
                decode_scope='Existing retained CPU decode receipt; publisher did not decode' if receipt else 'No retained complete decode receipt'))
    def evaluated(node,group,cohort,selected):
        aggregate=optional(node/'evaluation_summary.json',{});rows=[]
        indexed={r.get('identity'):r for r in aggregate.get('results',[]) if isinstance(r,dict)}
        known={p.name for p in (node/'evaluation').glob('*') if p.is_dir()}|set(indexed)
        registered={case['identity']:case for case in eval_plan if isinstance(case,dict) and 'identity' in case}
        if len(registered)!=len(eval_plan):issues.append(dict(group=group,problem='evaluation_plan_missing_or_duplicate_identity'))
        for identity in list(registered)+sorted(known-set(registered)):
            trial=node/'evaluation'/identity;r=optional(trial/'result.json')
            started=optional(trial/'started.json')
            attempted=r is not None or started is not None or identity in indexed
            if r is None and identity in indexed:issues.append(dict(group=group,cohort=cohort,identity=identity,problem='trial_receipt_missing_summary_preserved'))
            if r is not None and identity in indexed and r!=indexed[identity]:issues.append(dict(group=group,identity=identity,problem='trial_summary_mismatch'))
            r=r or {};valid=r.get('legal') is True and r.get('software_complete') is True and isinstance(r.get('success'),bool)
            if valid and r.get('reasons',{}).get('max_successes')!=r['success']:
                valid=False;issues.append(dict(group=group,identity=identity,problem='native_success_field_mismatch'))
            expected_seed=registered.get(identity,{}).get('spec',{}).get('seed')
            matched=(expected_seed is None or r.get('seed')==expected_seed) and bool(selected.get('exists')) and r.get('checkpoint_sha256')==selected.get('sha256')
            classification=('unmeasured' if not attempted else 'rejected' if r.get('legal') is False else
                'valid' if valid and matched else 'unverified_model_or_seed' if valid else 'software_incomplete')
            if valid and not matched:issues.append(dict(group=group,cohort=cohort,identity=identity,problem='evaluation_model_or_seed_unverified'))
            row=dict(round=round_name,group=group,cohort=cohort,identity=identity,registered=identity in registered,attempted=attempted,
                classification=classification,legal=r.get('legal'),software_complete=r.get('software_complete'),
                native_success=r.get('success'),success=r.get('success') if classification=='valid' else None,
                seed=r.get('seed'),steps=r.get('steps'),native_lift=r.get('native_lift'),native_subgoals=r.get('native_subgoals'),
                native_alignment=r.get('native_alignment'),stable_grasp=r.get('contact',{}).get('stable_grasp_diagnostic'),
                termination=','.join(k for k,v in r.get('reasons',{}).items() if v),post_success_actions=r.get('post_success_actions'),
                checkpoint_sha256=r.get('checkpoint_sha256'))
            rows.append(row);closed_rows.append(row);video(trial,group,cohort,identity,r)
        counted=[r for r in rows if r['registered']];valid=sum(r['classification']=='valid' for r in counted)
        attempted=sum(r['attempted'] for r in counted);successes=sum(r['success'] is True for r in counted) if valid else None
        complete=len(registered)==10 and len(counted)==10 and all(r['classification'] in ('valid','rejected') for r in counted)
        allocation=len(registered);completion=None
        if pool_mode and cohort=='autonomous_development_after_dagger':
            completion=optional(node.parent/'evaluation_complete.json',{})
            group_completion=completion.get('group_summary',{}).get(group,{})
            expansion=group_completion.get('expansion',{})
            allocation=len(registered) if expansion.get('expanded') is True else expansion.get('initial_trials')
            allowed_initial=cfg.get('evaluation',{}).get('attempts_per_group',4)
            complete=(completion.get('complete') is True and group_completion.get('complete') is True
                and aggregate.get('complete') is True and aggregate.get('expansion')==expansion
                and allocation in (allowed_initial,len(registered)) and allocation in (4,10)
                and len(registered)==10 and group_completion.get('planned')==len(registered)
                and group_completion.get('completed')==allocation and group_completion.get('unrun')==len(registered)-allocation
                and group_completion.get('software_failures')==0 and attempted==allocation
                and all(r['classification'] in ('valid','rejected') and r['software_complete'] is True for r in counted[:allocation])
                and all(not r['attempted'] for r in counted[allocation:]))
        return dict(planned=len(registered),attempted=attempted,valid=valid,successes=successes,
            allocated=allocation,allocation_completion=completion,
            success_rate=successes/valid if valid else None,unstarted=sum(not r['attempted'] for r in counted),
            rejected=sum(r['classification']=='rejected' for r in counted),
            software_incomplete=sum(r['classification']=='software_incomplete' for r in counted),
            unverified_model_or_seed=sum(r['classification']=='unverified_model_or_seed' for r in counted),
            unregistered_attempts=sum(r['attempted'] and not r['registered'] for r in rows),
            state='complete' if complete else 'partial' if attempted else 'unmeasured',declared_summary=aggregate,
            denominator='Registered ten-case development cohort; actual per-trial receipts, legal/software complete, native success and model/seed identity')
    for round_name,group in ((r,g) for r in round_names for g in groups):
        job=root/round_name/group;pair=optional(job/'initial_pair.json',{})
        report_key=round_name+'/'+group if pool_mode else group
        export_key=round_name+'_'+group if pool_mode else group
        local_plan=optional(job/'collection_plan.json',[] if pool_mode else plans)
        all_plans[report_key]=local_plan
        initial={key:evidence(pair[key],'checkpoint',pair.get(key+'_sha256')) for key in ('frontend','backend') if pair.get(key)}
        rows=optional(job/'collection_results.json',[]);attempts=lines(job/'attempts.jsonl')
        by_id={r['identity']:r for r in rows if isinstance(r,dict) and 'identity' in r}
        started={r['identity']:r for r in attempts if 'identity' in r}
        registered={p['identity']:p for p in local_plan if 'identity' in p}
        if len(registered)!=len(local_plan):issues.append(dict(group=group,round=round_name,problem='collection_plan_missing_or_duplicate_identity'))
        local=[]
        for identity in list(registered)+sorted((set(by_id)|set(started))-set(registered)):
            row=by_id.get(identity,{});result=row.get('result',{});dagger=result.get('dagger',{});exposure=row.get('exposure',{})
            attempted=identity in by_id or identity in started
            valid=result.get('legal') is True and result.get('software_complete') is True and isinstance(result.get('success'),bool)
            accepted=row.get('accepted') is True and valid and result.get('success') is True and dagger.get('expert_takeover_success') is True and row.get('label_contract_checked') is True
            if row.get('accepted') and not accepted:issues.append(dict(group=group,identity=identity,problem='accepted_correction_receipt_inconsistent'))
            record=dict(round=round_name,group=group,cohort='training_takeover_collection',identity=identity,registered=identity in registered,
                pool_partition=registered.get(identity,{}).get('pool_partition','train'),
                attempted=attempted,legal=result.get('legal'),software_complete=result.get('software_complete'),
                classification='unmeasured' if not attempted else 'rejected' if result.get('legal') is False else 'valid' if valid else 'software_incomplete',
                mixed_success=result.get('success') if valid else None,accepted=accepted,
                student_autonomous_success=dagger.get('student_autonomous_success'),takeover_triggered=dagger.get('takeover_triggered'),
                expert_takeover_success=dagger.get('expert_takeover_success'),student_ticks=dagger.get('student_ticks'),
                expert_ticks=dagger.get('expert_ticks'),expert_valid_ticks=dagger.get('expert_valid_ticks'),
                supervised_ticks=exposure.get('supervised_ticks'),supervised_anchors=exposure.get('supervised_anchors'),
                seed=result.get('seed',started.get(identity,registered.get(identity,{})).get('seed')),
                exclusion_reason=row.get('exclusion_reason'),error=row.get('error'),
                termination=','.join(k for k,v in result.get('reasons',{}).items() if v))
            local.append(record);collection_rows.append(record)
            video(job/'episodes'/identity,group,'training_takeover_collection',identity,result)
        admitted=[r for r in local if r['registered'] and r['accepted']]
        entries=optional(job/'correction_entries.json',{})
        for entry in entries.values():
            if entry.get('path'):evidence(entry['path'],'trajectory',entry.get('source_sha256'))
        collection=dict(planned=len(registered),attempted=sum(r['attempted'] for r in local if r['registered']),
            unstarted=sum(not r['attempted'] for r in local if r['registered']),
            accepted_episodes=len(admitted),effective_expert_ticks=sum(r['supervised_ticks'] or 0 for r in admitted),
            effective_dp_anchors=sum(r['supervised_anchors'] or 0 for r in admitted),
            mixed_successes=sum(r['mixed_success'] is True for r in local if r['registered']),
            expert_takeover_successes=sum(r['expert_takeover_success'] is True for r in local if r['registered']),
            student_autonomous_successes=sum(r['student_autonomous_success'] is True for r in local if r['registered']),
            software_incomplete=sum(r['classification']=='software_incomplete' for r in local if r['registered']),
            unregistered_attempts=sum(r['attempted'] and not r['registered'] for r in local),
            finished=optional(job/'collection_finished.json'),gate=optional(job/'collection_gate.json'),
            selection='Only successful expert-executed corrections admitted; student prefix retained with zero supervision; not autonomous evaluation')
        stages={}
        for stage,node,target,unit in (('frontend',job/'dp/training',pair.get('supervision','unknown'),'supervised_planning_anchor_equivalent_epochs'),
                ('backend',job/'backend'/group/'training','u_expert','supervised_control_tick_equivalent_epochs')):
            done=optional(node/'finished.json',{});progress=optional(node/'progress.json',{})
            configuration=optional(node/'configuration.json',{});selected=optional(node/'selected.json',{})
            budget=optional(node/'budget.json',optional(job/'dp/budget.json') if stage=='frontend' else None)
            checkpoint=evidence(node/'best.pt','checkpoint');evidence(node/'last.pt','checkpoint')
            updates=lines(node/('training.jsonl' if stage=='frontend' else 'loss.jsonl'))
            validation=lines(node/'validation.jsonl') if stage=='frontend' else optional(node/'validation.json',[])
            emit(export_key+'_'+stage+'_every_update.csv.gz',[numeric(r) for r in updates])
            emit(export_key+'_'+stage+'_validation_raw.csv.gz',[numeric(r) for r in validation])
            sampling=optional(node/'sampling_exposure.json',done.get('coverage',progress.get('coverage')))
            if sampling is None and validation:sampling=validation[-1].get('coverage')
            if pool_mode and validation:
                with (output/(export_key+'_'+stage+'_validation.json.gz')).open('wb') as raw:
                    with gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as zipped:
                        zipped.write(json.dumps(public(validation),sort_keys=True,allow_nan=False).encode())
            if sampling is not None:
                write(output/(export_key+'_'+stage+'_sampling_exposure.json'),public(sampling))
                sampling_rows.append(dict(round=round_name,group=group,stage=stage,**numeric(sampling)))
            if stage=='frontend':
                emit(export_key+'_frontend_geometry.csv.gz',[numeric(r) for r in lines(node/'geometry_projection.jsonl')])
                emit(export_key+'_frontend_probe.csv.gz',[numeric(r) for r in lines(job/'dp/probe/training.jsonl')])
            cumulative=0
            for row in updates:
                cumulative+=row.get('supervised_ticks',0)
                population=done.get('supervised_tick_population',done.get('control_tick_population'))
                exposure=cumulative/population if stage=='backend' and population else row.get('anchor_equivalent_epochs')
                losses.append(dict(round=round_name,stage=stage,model=report_key+' '+stage,target=target,exposure_unit=unit,
                    first_update=row.get('update'),update=row.get('update'),updates_in_interval=1,exposure=exposure,mean_loss=number(row.get('loss'))))
            for row in validation:
                if pool_mode:
                    cohort_keys=(('train8','fixed_train_diagnostic'),('metrics','old_validation'),('correction_dev','correction_dev')) if stage=='frontend' else (
                        ('train','fixed_train_diagnostic'),('validation','old_validation'),('correction_dev','correction_dev'))
                else:
                    cohort_keys=(('train8','fixed_train2'),('metrics','fixed_validation4')) if stage=='frontend' else (
                        ('train','fixed_train2'),('validation','fixed_validation4'))
                for key,cohort in cohort_keys:
                    metric=row.get(key,{})
                    if stage=='backend':metric=metric.get('all',{})
                    curves.append(dict(round=round_name,stage=stage,model=report_key+' '+stage,target=target,exposure_unit=unit,cohort=cohort,
                        update=row.get('update'),exposure=row.get('anchor_equivalent_epochs',row.get('coverage',{}).get('anchor_equivalent_epochs')) if stage=='frontend' else row.get('control_tick_epochs'),
                        arm_rmse_rad=number(metric.get('arm_rmse' if stage=='frontend' else 'arm_rmse_rad')),
                        hand_rmse_rad=number(metric.get('hand_rmse' if stage=='frontend' else 'hand_rmse_rad')),
                        palm_rmse_mm=number(metric.get('ee_distance_rmse_mm')),
                        selection_metric=number(metric.get('selection_score',metric.get('selection_mse')) if stage=='frontend' else metric.get('normalized_equal_mse'))))
                if pool_mode:
                    for cohort,value in row.get('selection_components',{}).items():
                        curves.append(dict(round=round_name,stage=stage,model=report_key+' '+stage,target=target,exposure_unit=unit,
                            cohort='selection_component_'+cohort,update=row.get('update'),
                            exposure=row.get('anchor_equivalent_epochs',row.get('coverage',{}).get('anchor_equivalent_epochs')) if stage=='frontend' else row.get('control_tick_epochs'),
                            selection_metric=number(value)))
            exposure=done.get('coverage',{}).get('anchor_equivalent_epochs',progress.get('anchor_equivalent_epochs')) if stage=='frontend' else done.get('control_tick_epochs',progress.get('control_tick_epochs'))
            train_state=('disposable' if done.get('disposable') or configuration.get('disposable') else
                'budget_reached' if done.get('target_reached') is True else 'stopped_before_ceiling' if done else 'partial' if progress or updates else 'unmeasured')
            report=dict(state=train_state,updates=done.get('updates',progress.get('update')),exposure=exposure,exposure_unit=unit,
                target=target,finished=done,progress=progress,configuration=configuration,budget=budget,selection=selected,
                selected_checkpoint=checkpoint,restore=optional(node/'restore.json'),
                continuation_audit=configuration.get('continuation_audit'),sampling_exposure=sampling,
                probe=optional(job/('dp/throughput_probe.json' if stage=='frontend' else 'backend/throughput_probe.json')))
            stages[stage]=report
            stage_rows.append(dict(round=round_name,group=group,stage=stage,target=target,state=train_state,updates=report['updates'],
                new_optimizer_updates=done.get('new_optimizer_updates'),exposure=exposure,exposure_unit=unit,
                supervised_target_ticks=done.get('supervised_target_ticks'),supervised_ticks=done.get('supervised_ticks'),
                stop_reason=done.get('stop_reason'),selected_update=selected.get('update'),checkpoint_sha256=checkpoint.get('sha256')))
        backend=job/'backend'/group
        offline=optional(backend/'offline/result.json',{})
        for name in ('stage_errors.csv','base_stage_errors.csv','force_sensitivity.json','geometry_metrics.json'):
            path=backend/'offline'/name
            if path.suffix=='.csv':
                metric_rows=csv_read(path);emit(export_key+'_'+name,metric_rows)
                for row in metric_rows:
                    errors.append(dict(round=round_name,stage='backend',model=report_key,target='u_expert',prediction_kind='final_command' if name=='stage_errors.csv' else 'nominal_against_command',
                        **{key:row.get(key) for key in ('split','phase','group','motion')},
                        **{key:number(row.get(key)) for key in ('ticks','rmse_rad','ee_rmse_mm')}))
            else:
                receipt=optional(path)
                if receipt is not None:write(output/(export_key+'_'+name),public(receipt))
                else:(output/(export_key+'_'+name)).unlink(missing_ok=True)
        for name in ('sampled_ticks.csv','base_sampled_ticks.csv'):
            emit(export_key+'_'+name+'.gz',csv_read(backend/'offline'/name))
        if pool_mode:
            emit(export_key+'_contact_phase_metrics.csv',csv_read(backend/'offline/contact_phase_metrics.csv'))
        evaluated_new=evaluated(backend,group,'autonomous_development_after_dagger',stages['backend']['selected_checkpoint'])
        previous=evaluated(Path(pair['backend']).parent.parent,group,'historical_autonomous_development',initial['backend']) if pair.get('backend') else None
        round_finished=round_reports.get(round_name,{}).get('finished') or {}
        round_group=round_finished.get('groups',{}).get(group,{})
        owner_state=round_group.get('state','UNMEASURED') if pool_mode else status.get('groups',{}).get(group,'UNMEASURED')
        group_reports[report_key]=dict(round=round_name,group=group,owner_state=owner_state,initial_pair=pair,
            initialization_artifacts=initial,collection=collection,stages=stages,offline=offline,
            autonomous_development=evaluated_new,historical_autonomous_development=previous,
            evaluation_completion=optional(job/'backend/evaluation_complete.json'),failure=optional(job/'failure.json'),
            input_acceptance=optional(job/'dp/vision_acceptance.json'),selection=optional(job/'selection.json'),
            pool_ingestion=optional(job/'ingestion.json'),data=optional(job/'dp/state_data/data.json') if pool_mode else None,
            preparation={name:optional(job/name) for name in ('dp/prepared.json','backend/prepared.json',
                'dp/preparation_recovery.json','backend_preparation_recovery.json')} if pool_mode else None)
        if pool_mode:round_reports[round_name]['groups'][group]=group_reports[report_key]
    telemetry=lines(root/'resources/shared.jsonl')
    if not telemetry:telemetry=lines(root/'shared_resources.jsonl')
    start=number(launch.get('started',session.get('started')));end=number(status.get('utc'))
    valid_samples=[r for r in telemetry if number(r.get('utc_epoch')) is not None and (start is None or r['utc_epoch']>=start) and (end is None or r['utc_epoch']<=end)]
    intervals=[(b['utc_epoch']-a['utc_epoch'],number(a.get('gpu_util_percent'))) for a,b in zip(valid_samples,valid_samples[1:])]
    usable=[(dt,value) for dt,value in intervals if 0<dt<=8 and value is not None]
    covered=sum(dt for dt,_ in usable)
    def minimum(key):
        values=[number(r.get(key)) for r in valid_samples];values=[v for v in values if v is not None]
        return min(values) if values else None
    resources=dict(samples=len(valid_samples),time_weighted_whole_card_percent=sum(dt*v for dt,v in usable)/covered if covered else None,
        measured_interval_s=covered,max_sample_gap_s=max((dt for dt,_ in intervals),default=None),
        gap_seconds_above8=sum(dt for dt,_ in intervals if dt>8),
        leading_uncovered_s=max(0,valid_samples[0]['utc_epoch']-start) if valid_samples and start is not None else None,
        trailing_uncovered_s=max(0,end-valid_samples[-1]['utc_epoch']) if valid_samples and end is not None else None,
        min_gpu_free_mib=minimum('gpu_free_mib'),min_ram_available_bytes=minimum('ram_available_bytes'),
        monitor_exit=optional(root/'monitor_exit.json'),
        scope='Retained launch-to-status telemetry, positive intervals <=8s; whole card, not per-process utilization; missing coverage retained')
    emit('resource_samples.csv.gz',[numeric(r) for r in valid_samples])
    log_rows=[];physx=[]
    for path in sorted(root.rglob('*.log')):
        evidence(path,'log');counts=dict(path=str(path),error_lines=0,physx_error_lines=0)
        with path.open(errors='replace') as stream:
            for line_number,line in enumerate(stream,1):
                if '[Error]' in line or 'Traceback (most recent call last)' in line:counts['error_lines']+=1
                if 'PhysX error:' in line:
                    counts['physx_error_lines']+=1;physx.append(dict(path=str(path),line=line_number,message=line.rstrip()))
        log_rows.append(counts)
    receipts={name:optional(root/(name+'.json')) for name in ('source_freeze','integration','owned_cleanup','owner_error','active_stage','worker_processes','publication_error','cleanup','preparation_receipt','time_estimate','successor_service','rounds','pool_import','gpu_gate','timing','registration')}
    if pool_mode:
        preparation={}
        for name in ('controller_acceptance','takeover_timing','owner_data_acceptance','cleanup_candidates'):
            receipt=optional(root/'preparation'/(name+'.json'));preparation[name]=receipt
            if receipt is not None:write(output/(name+'.json'),public(receipt))
        receipts['preparation']=preparation
        if receipts['cleanup'] is not None:write(output/'cleanup.json',public(receipts['cleanup']))
    verified_complete=status.get('state')=='COMPLETE' and bool(groups) and not issues and status.get('owned_workers_exited') is True and (receipts.get('owned_cleanup') or {}).get('verified') is True and all(
        r['owner_state']=='COMPLETE' and r['autonomous_development']['state']=='complete' and r['evaluation_completion'] is not None and r['evaluation_completion'].get('complete') is True
        and (r['collection']['finished'] or {}).get('complete') is True and (r['collection']['finished'] or {}).get('sufficient') is True
        and (r['collection']['gate'] or {}).get('passed') is True
        and all(s['finished'] and not s['finished'].get('disposable') and s['selected_checkpoint'].get('exists') for s in r['stages'].values()) for r in group_reports.values())
    if pool_mode:
        verified_complete=verified_complete and all((r.get('finished') or {}).get('state')=='RESOLVED'
            and r.get('snapshot') and r.get('decision') is not None for r in round_reports.values())
    pool_path=Path(cfg.get('pool_root',CODE/'outputs/visuotactile/data/tight_dagger_pool')) if pool_mode else None
    if pool_path is not None and not pool_path.is_absolute():pool_path=CODE/pool_path
    pool_receipts={name:optional(pool_path/name) for name in ('manifest.json','acceptance/readback.json','acceptance/imported_legacy_snapshot.json')} if pool_mode else None
    summary=dict(schema='dagger_pool_cpu_publication_v1' if pool_mode else 'dagger_cpu_publication_v1',status=status,verified_round_complete=verified_complete,configuration=cfg,
        registration=registration,session=session,source_run=str(root),source_commit=launch.get('source_head',session.get('source_head')),
        groups=group_reports,rounds=round_reports if pool_mode else None,pool=pool_receipts,
        resources=resources,root_receipts=receipts,publication_issues=issues,
        partial_snapshot=pool_mode and not status.get('owned_workers_exited',False),
        inference_or_training_performed=False,kit_loaded=False,selection_performed_by_exporter=False,
        test22_used=status.get('test22_used'),no_next_round=status.get('no_next_round'),
        log_scope='Retained log line occurrences; duplicate sinks are not distinct simulator events',physx_error_lines=len(physx),
        limitations=['Collection successes include expert takeover and are not autonomous policy success',
            'Only successful corrections are aggregated; failed/illegal/interrupted attempts remain in collection denominator',
            'u_expert and q_reached frontend supervision retain separate target semantics; backend target remains u_expert',
            'Offline errors use recorded actual prefixes, not aligned autonomous trajectories',
            'Ten-case development evaluation is not formal test22 or OOD evidence',
            'Same reset specifications and seeds do not establish bitwise hidden-state pairing',
            'Null/unmeasured is missing measurement, never a zero success rate; exit0 does not establish harmless log errors'])
    write(output/'summary.json',public(summary));write(output/'plans.json',public(dict(collection=all_plans if pool_mode else plans,autonomous_development=eval_plan)))
    emit('collection.csv',collection_rows);emit('closed_loop.csv',closed_rows);emit('training_stages.csv',stage_rows)
    emit('validation_curves.csv',curves);emit('training_loss.csv.gz',losses);emit('offline_metrics.csv',errors);emit('sampling_exposure.csv',sampling_rows)
    if pool_mode:
        write(output/'round_decisions.json',public({key:dict(decision=value['decision'],finished=value['finished'],
            snapshot_id=(value.get('snapshot') or {}).get('snapshot_id'),
            groups={g:r.get('selection') for g,r in value['groups'].items()}) for key,value in round_reports.items()}))
        write(output/'pool_identity.json',public(pool_receipts))
    emit('video_index.csv',videos);emit('log_counts.csv',log_rows);emit('physx_errors.csv',physx)
    write(output/'source_artifacts.json',public(sources));write(output/'local_artifacts.json',public(artifacts))
    matrix_plots(output,curves,losses,errors)
    for path in output.glob('*.csv'):path.write_bytes(path.read_bytes().replace(b'\r\n',b'\n'))
    write(output/'artifacts.json',{str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.iterdir()) if p.is_file() and p.name!='artifacts.json'})
    document=CODE/'research_docs/STATE_DP_RESULTS.md'
    marker='tight-dagger-pool-results' if pool_mode else 'tight-dagger-results'
    start_marker='<!-- '+marker+':start -->';end_marker='<!-- '+marker+':end -->'
    text=document.read_text() if document.exists() else ''
    if text.count(start_marker)!=text.count(end_marker) or text.count(start_marker)>1:
        raise ValueError('Malformed DAgger managed block; preserve existing document')
    relative=os.path.relpath(output,document.parent)
    lines_out=[start_marker,'## 2026-11-11 DAgger：共享持久纠偏池与多轮增量续训' if pool_mode else '## 2026-11-11 DAgger：一轮前端与后端增量续训','',
        f"本轮 owner 状态为 **{status.get('state','UNMEASURED')}**；产物核验{'完整' if verified_complete else '尚不完整'}。以下按已有凭据整理，缺失项不记为零成功。",'',
        '|组|采集已启动 / 机会|接管成功 / 已启动|准入纠偏 episode / 有效专家 tick / DP anchor|新 DP 更新 / anchor 遍|新后端更新 / tick 遍|自主开发成功 / 有效 / 计划|',
        '|---|---:|---:|---:|---:|---:|---:|']
    def show(value):return '未测' if value is None else f'{value:.3f}' if isinstance(value,float) else str(value)
    for group,row in group_reports.items():
        c=row['collection'];f=row['stages']['frontend'];b=row['stages']['backend'];e=row['autonomous_development']
        lines_out.append(f"|{group}|{c['attempted']} / {c['planned']}|{c['expert_takeover_successes']} / {c['attempted']}|{c['accepted_episodes']} / {c['effective_expert_ticks']} / {c['effective_dp_anchors']}|{show(f['updates'])} / {show(f['exposure'])}|{show(b['updates'])} / {show(b['exposure'])}|{show(e['successes'])} / {e['valid']} / {e['planned']}|")
    if pool_mode:
        legacy=(pool_receipts or {}).get('acceptance/readback.json') or {}
        lines_out+=['',f"旧池 CPU 读回准入 {show(legacy.get('pool_entries'))} 条成功纠偏；每轮同一不可变快照供两组使用，预登记 correction-dev 按完整 episode 与训练隔离。实际抽样曝光、update0 候选、轮次继续/停止决定见机器凭据；尚未发生的轮次均标作未测。",
            f'[池与快照身份]({relative}/pool_identity.json) · [各轮选择与决定]({relative}/round_decisions.json)。']
    lines_out+=['','采集成功包含专家接管，不等于学生自主成功；只接纳成功专家纠偏会产生选择偏差。学生前缀保留作真实历史，不监督；前端沿各组既有监督语义续训，后端监督仍为 `u_expert`。',
        '自主列为预登记十次开发机会；非法、软件中断、模型身份未核验和未启动均单列保留，不替换样本，也不作为正式 test22 成功率。',
        f"整卡时间加权利用率：{show(resources['time_weighted_whole_card_percent'])}%；资源样本 {resources['samples']} 条，>8s 内部缺口 {show(resources['gap_seconds_above8'])}s。这不是每进程利用率；头尾未覆盖时间及原 monitor 汇总见机器结果。",'',
        f'[原始机器结果]({relative}/summary.json) · [采集逐次记录]({relative}/collection.csv) · [自主逐次记录]({relative}/closed_loop.csv) · [两阶段曝光]({relative}/training_stages.csv) · [权重/视频本地索引]({relative}/local_artifacts.json)。']
    if (output/'training_curves.png').is_file():lines_out+=['',f'![DAgger 训练与固定验证曲线]({relative}/training_curves.png)']
    lines_out+=['','仅 CPU 导出已有数值、日志与 SHA；未新增训练、推理、Kit、评估、清理或模型选择。初始化身份、失败原因、视频解码凭据和全部缺失项见公开结果。',end_marker]
    block='\n'.join(lines_out)
    if start_marker in text:
        start=text.index(start_marker);end=text.index(end_marker,start)+len(end_marker);text=text[:start]+block+text[end:]
    else:text=text.rstrip()+'\n\n'+block+'\n'
    document.parent.mkdir(parents=True,exist_ok=True)
    temporary=document.with_suffix('.dagger.tmp');temporary.write_text(text);temporary.replace(document)
    return public(summary)


def matrix_publish(root, output):
    """CPU-only export of a finite method matrix, including incomplete stages.

    Read measured receipts only. No model deserialization, inference, simulator,
    seed selection, or best-checkpoint selection occurs in this publication path.
    """
    root, output = Path(root).resolve(), Path(output).resolve()
    cfg, status = read(root/'configuration.json'), read(root/'status.json')
    if not cfg.get('method_matrix'):
        raise ValueError('Expected an explicitly registered method matrix')
    if status.get('owned_workers_exited') is False or status.get('state') == 'RUNNING' and not status.get('owned_workers_exited'):
        raise ValueError('Publish after the owner has stopped its workers')
    if root == output or root in output.parents:
        raise ValueError('Publication must not overwrite run evidence')
    output.mkdir(parents=True, exist_ok=True)
    sources, artifacts, issues = {}, {}, []
    digest_cache = {}
    def public(value):
        if isinstance(value, dict): return {public(k):public(v) for k,v in value.items()}
        if isinstance(value, list): return [public(v) for v in value]
        if isinstance(value, str):
            for prefix in (str(CODE)+'/', str((CODE/'outputs').resolve().parent)+'/'):
                value = value.replace(prefix, '')
            return value.replace(str(Path.home())+'/', '~/')
        return value
    def evidence(path, kind='metric', declared_sha=None):
        path = Path(path); key = public(str(path))
        row = dict(path=key, kind=kind, exists=path.is_file(), uploaded=False)
        if row['exists']:
            st = path.stat(); row['bytes'] = st.st_size
            if declared_sha or kind=='source_trajectory':
                row.update(sha256=declared_sha, hash_scope='retained source manifest; not rehashed during publication')
            else:
                identity = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)
                if identity not in digest_cache: digest_cache[identity] = sha(path)
                row.update(sha256=digest_cache[identity], hash_scope='read and SHA256 verified during CPU publication')
        else:
            row.update(sha256=declared_sha, hash_scope='declared only; missing file')
        (artifacts if kind in ('checkpoint','source_trajectory','rollout','video') else sources)[key] = row
        return row
    def optional(path, fallback=None):
        if not path.is_file(): return fallback
        evidence(path); return read(path)
    def lines(path):
        if not path.is_file(): return []
        evidence(path)
        rows=[]
        with path.open() as stream:
            for index,line in enumerate(stream,1):
                if not line.strip(): continue
                try: rows.append(json.loads(line))
                except json.JSONDecodeError:
                    issues.append(dict(problem='incomplete_or_malformed_metric_line',path=str(path),line=index))
        return rows
    def csv_rows(path):
        if not path.is_file(): return []
        evidence(path)
        with path.open(newline='') as stream: return list(csv.DictReader(stream))
    def emit(name, rows):
        if not rows:
            (output/name).unlink(missing_ok=True)
            return
        fields = list(dict.fromkeys(key for row in rows for key in row))
        table(output/name, [{key:public(row.get(key)) for key in fields} for row in rows])
    def number(value):
        if value in ('', None): return None
        value = float(value)
        return value if math.isfinite(value) else None
    def metric_rows(rows, stage, group, target, kind):
        numeric = ('ticks','episodes','rmse_rad','hold_rmse_rad','ee_rmse_mm','fingertip_rmse_mm','palm_angle_mean_deg')
        return [dict(stage=stage, model=group, target=target, prediction_kind=kind,
            **{k:r.get(k) for k in ('split','phase','group','motion')},
            **{k:number(r.get(k)) for k in numeric}) for r in rows]
    def numeric_fields(value, prefix=''):
        result={}
        for key,item in value.items():
            key=prefix+key
            if isinstance(item,dict):result.update(numeric_fields(item,key+'.'))
            elif isinstance(item,(int,float)):result[key]=int(item) if isinstance(item,bool) else number(item)
        return result

    master = optional(root/'result.json', {})
    registration = optional(root/'preregistration.json', {})
    session = optional(root/'session.json', {})
    budget = optional(root/'budget.json')
    group_rows, closed_rows, error_rows, curve_rows, loss_rows, resource_rows = [], [], [], [], [], []
    trace_diagnostics, videos, evaluation_plans = [], [], {}
    exported_stages = {}
    main_groups = cfg['method_matrix'].get('groups', registration.get('groups', []))
    reserved = set(cfg.get('vision_pair',{}).get('groups',[])+cfg.get('residual_pair',{}).get('groups',[]))
    default_groups = dict(command=[g for g in main_groups if g in cfg.get('vision_pair',{}).get('groups',[])],
        command_residual=[g for g in main_groups if g in cfg.get('residual_pair',{}).get('groups',[])],
        configuration=[g for g in main_groups if g not in reserved and g.endswith('00')],
        configuration_residual=[g for g in main_groups if g not in reserved and g.endswith('01')])
    for stage in ('command','configuration','command_residual','configuration_residual','formal_test'):
        node = root/stage
        recorded = optional(node/'stage_result.json', master.get('stages',{}).get(stage,{}))
        stage_status = optional(node/'status.json', recorded.get('status',dict(state='UNMEASURED')))
        sc = optional(node/'configuration.json', {})
        ss = optional(node/'session.json', {})
        reuse = optional(node/'reuse_manifest.json', {}) if stage=='command' else {}
        residual_stage = stage.endswith('_residual') or recorded.get('residual',False)
        v = ss.get('residual_pair' if residual_stage else 'vision_pair', sc.get('residual_pair' if residual_stage else 'vision_pair',{}))
        frozen = optional(node/'model_freeze.json', {}) if stage == 'formal_test' else {}
        groups = list(dict.fromkeys(v.get('groups',[])+list(recorded.get('groups',{}))+list(frozen.get('models',{}))))
        if not groups: groups = default_groups.get(stage, [])
        plan = optional(node/'evaluation_plan.json', [])
        evaluation_plans[stage]=plan
        ids = [case['identity'] for case in plan]
        if len(ids) != len(set(ids)): raise ValueError('Duplicate registered evaluation identities in '+stage)
        evaluate = v.get('evaluate',True)
        stage_out = dict(status=stage_status, groups={}, declared_groups=groups,
            acceptance={name:optional(node/(name+'.json')) for name in ('vision_acceptance','residual_acceptance')},
            evaluation_plan_sha256=sha(node/'evaluation_plan.json') if plan else None,
            cohort='historical_development_reused' if reuse else 'formal_test' if stage == 'formal_test' else 'development',
            reuse_manifest=reuse or None,throughput_probe=optional(node/'throughput_probe.json'))
        for group in groups:
            d = node/group; recorded_group = recorded.get('groups',{}).get(group,{})
            finished = optional(d/'training/finished.json', recorded_group.get('training',{}))
            historical_training = finished if reuse else None
            if reuse:finished=dict(updates=0,stop_reason='existing_selected_EMA_reused',new_optimizer_updates=0)
            progress = optional(d/'training/progress.json', {})
            training_cfg = optional(d/'training/configuration.json', {})
            selected_receipt = optional(d/'training/selected.json', {})
            training_budget = optional(d/'training/budget.json', {})
            adaptation_scope = optional(d/'training/adaptation_scope.json')
            offline = optional(d/'offline/result.json', recorded_group.get('offline',{}))
            evaluation_root=(Path(reuse.get('evaluation_root',str(Path(reuse['source_root'])/'evaluation_resume_01')))/group) if reuse else d
            evaluation = optional(evaluation_root/'evaluation_summary.json', recorded_group.get('evaluation',{}))
            failure = optional(d/'failure.json')
            auxiliary={name:optional(d/'offline'/(name+'.json')) for name in ('geometry_metrics','force_sensitivity')}
            execution={name:optional(root/'queue'/group/(name+'.json')) for name in
                ('status','node','throughput_probe','residual_acceptance','evaluation_process_reaped')}
            geometry_rows=csv_rows(d/'offline/geometry_rows.csv')
            if geometry_rows and stage != 'formal_test':
                emit(stage+'_'+group+'_geometry.csv',geometry_rows)
            residual = bool(training_cfg.get('residual_version')) or residual_stage or bool(training_cfg.get('architecture'))
            target = 'u_expert' if residual else training_cfg.get('supervision',v.get('supervision',recorded.get('supervision')))
            if target not in ('u_expert','q_reached'):
                target = 'u_expert' if stage == 'command' else 'unknown'
            unit = 'control_tick_equivalent_epochs' if residual else 'planning_anchor_equivalent_epochs'
            selected = evidence(d/'training/best.pt', 'checkpoint')
            for p in sorted((d/'training').glob('*.pt')): evidence(p,'checkpoint')
            if selected.get('sha256') and offline.get('checkpoint_sha256') and selected['sha256'] != offline['checkpoint_sha256']:
                issues.append(dict(stage=stage,group=group,problem='offline_selected_checkpoint_SHA_mismatch'))
            # Read the source manifest only; large original HDF5 files are never opened.
            manifest = optional(node/'manifest.json')
            if manifest:
                for entry in manifest.get('episodes',[]):
                    if 'path' in entry:
                        evidence(Path(entry['path']), 'source_trajectory', entry.get('sha256',entry.get('source_sha256')))
            stage_metrics = optional(d/'offline/stage_metrics.json', offline.get('metrics',[]))
            if not stage_metrics: stage_metrics = csv_rows(d/'offline/stage_errors.csv')
            if stage != 'formal_test':
                error_rows += metric_rows(stage_metrics,stage,group,target,'final_command' if residual else 'nominal')
                if residual:
                    base_metrics = offline.get('base_metrics') or csv_rows(d/'offline/base_stage_errors.csv')
                    # Historical/current residual offline metrics explicitly compare base to u_expert.
                    error_rows += metric_rows(base_metrics,stage,group,'u_expert','nominal_against_command_not_configuration_fit')
                validation = [] if reuse else optional(d/'training/validation.json') if residual else lines(d/'training/validation.jsonl')
                for r in validation or []:
                    for source,cohort in (('train','fixed_train2'),('validation','fixed_validation4')) if residual else (('train8','fixed_train2'),('metrics','fixed_validation4')):
                        metrics = r.get(source,{})
                        if residual: metrics = metrics.get('all',{})
                        curve_rows.append(dict(stage=stage,model=group,target=target,exposure_unit=unit,cohort=cohort,
                            update=r.get('update'),exposure=r.get('control_tick_epochs' if residual else 'anchor_equivalent_epochs'),
                            arm_rmse_rad=number(metrics.get('arm_rmse_rad' if residual else 'arm_rmse')),
                            hand_rmse_rad=number(metrics.get('hand_rmse_rad' if residual else 'hand_rmse')),
                            palm_rmse_mm=number(metrics.get('ee_distance_rmse_mm')),
                            selection_metric=number(metrics.get('normalized_equal_mse' if residual else 'selection_score'))))
                updates = [] if reuse else lines(d/'training'/('loss.jsonl' if residual else 'training.jsonl'))
                numeric_name=stage+'_'+group+'_updates.csv.gz'
                if updates:
                    import gzip
                    numeric_updates=[numeric_fields(row) for row in updates]
                    fields=list(dict.fromkeys(key for row in numeric_updates for key in row))
                    with gzip.open(output/numeric_name,'wt',encoding='utf-8',newline='') as stream:
                        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(numeric_updates)
                else:(output/numeric_name).unlink(missing_ok=True)
                width = max(1, math.ceil(len(updates)/1000)); cumulative = 0
                for offset in range(0,len(updates),width):
                    part=updates[offset:offset+width]
                    cumulative += sum(r.get('supervised_ticks',0) for r in part)
                    exposure = cumulative/finished['control_tick_population'] if residual and finished.get('control_tick_population') else part[-1].get('anchor_equivalent_epochs')
                    loss_rows.append(dict(stage=stage,model=group,target=target,exposure_unit=unit,
                        first_update=part[0].get('update'),update=part[-1].get('update'),updates_in_interval=len(part),
                        exposure=exposure,mean_loss=sum(r['loss'] for r in part)/len(part)))
                for p in (d/'offline').glob('*.csv'): evidence(p)
            # Every registered case remains visible, even if it never started.
            summary_results = {r['identity']:r for r in evaluation.get('results',[])}
            actual_ids = {p.name for p in (evaluation_root/'evaluation').glob('*') if p.is_dir()}
            if set(summary_results)-set(ids) or actual_ids-set(ids):
                if plan: issues.append(dict(stage=stage,group=group,problem='unregistered_trial_ids',identities=sorted((set(summary_results)|actual_ids)-set(ids))))
            cases = ids+sorted((set(summary_results)|actual_ids)-set(ids)) if evaluate else []
            measured = []
            for identity in cases:
                trial = evaluation_root/'evaluation'/identity
                r = optional(trial/'result.json')
                started = (trial/'started.json').is_file()
                if started: evidence(trial/'started.json')
                if r is None and identity in summary_results:
                    issues.append(dict(stage=stage,group=group,identity=identity,problem='trial_result_missing_summary_preserved'))
                if r is not None and identity in summary_results and r != summary_results[identity]:
                    issues.append(dict(stage=stage,group=group,identity=identity,problem='trial_summary_receipt_mismatch'))
                attempted = r is not None or started or identity in summary_results
                r = r or {}
                valid = r.get('legal') is True and r.get('software_complete') is True and isinstance(r.get('success'),bool)
                if valid and r.get('success') is not None and r.get('reasons',{}).get('max_successes') != r['success']:
                    issues.append(dict(stage=stage,group=group,identity=identity,problem='native_success_field_mismatch'));valid=False
                classification = 'unmeasured' if not attempted else 'rejected' if r.get('legal') is False else 'valid' if valid else 'software_incomplete'
                row=dict(stage=stage,model=group,cohort=stage_out['cohort'],identity=identity,attempted=attempted,
                    registered=identity in ids if plan else None,
                    classification=classification,seed=r.get('seed'),legal=r.get('legal'),software_complete=r.get('software_complete'),
                    success=r.get('success') if valid else None,steps=r.get('steps'),native_lift=r.get('native_lift'),
                    native_subgoals=r.get('native_subgoals'),native_alignment=r.get('native_alignment'),
                    termination=','.join(k for k,value in r.get('reasons',{}).items() if value),
                    checkpoint_sha256=r.get('checkpoint_sha256'),post_success_actions=r.get('post_success_actions'))
                closed_rows.append(row);measured.append(row)
                trace_diagnostics.append(dict(stage=stage,model=group,identity=identity,
                    **{k:r.get(k) for k in ('native_inference','residual_diagnostics','contact','planning_latency_s','native_history_latency_s','plan_consistency')}))
                for name in ('initial_collision.json','initial_observable.json','native_inference_timing.json'):
                    if (trial/name).is_file():evidence(trial/name)
                for p in sorted(trial.glob('*')):
                    if p.suffix in ('.h5','.npz','.mp4'): evidence(p,'video' if p.suffix=='.mp4' else 'rollout')
                if (trial/'rollout.mp4').is_file():
                    verified=optional(trial/'video_verified.json',{})
                    videos.append(dict(stage=stage,model=group,identity=identity,success=r.get('success'),
                        path=str(trial/'rollout.mp4'),sha256=evidence(trial/'rollout.mp4','video')['sha256'],
                        decoded_frames=verified.get('decoded_frames'),frame_rate=verified.get('frame_rate'),
                        decode_scope='Existing complete CPU decode receipt; not newly decoded by publisher' if verified else 'unmeasured'))
            counted=[r for r in measured if r['registered']] if plan else measured
            attempted=sum(r['attempted'] for r in counted);valid=sum(r['classification']=='valid' for r in counted)
            rejected=sum(r['classification']=='rejected' for r in counted)
            incomplete=sum(r['classification']=='software_incomplete' for r in counted)
            successes=sum(r['success'] is True for r in counted) if attempted else None
            counts=dict(planned_attempts=len(ids) if evaluate and plan else None,attempts=attempted,valid=valid,
                rejected=rejected,software_incomplete=incomplete,unstarted=sum(not r['attempted'] for r in counted),
                unregistered_attempts=sum(r['attempted'] and r['registered'] is False for r in measured),
                new_attempts=0 if reuse else attempted,historical_attempts=attempted if reuse else 0,
                successes=successes,success_rate=successes/valid if valid else None,
                declared_counts={k:evaluation.get(k) for k in ('attempts','valid','rejected','successes')},
                evaluation_state='unmeasured' if not attempted else 'complete' if plan and attempted==len(ids) and incomplete==0 else 'partial',
                evaluation_state_scope='Registered-case receipt coverage only; owner status is separate',
                owner_state=stage_status.get('state'),
                denominator_rule='Registered cases only when a plan exists; valid=legal and software_complete native receipts; rejected/software-incomplete/unregistered remain visible; no replacement')
            for key in ('attempts','valid','successes'):
                if evaluation.get(key) is not None and evaluation[key]!=counts[key]:
                    issues.append(dict(stage=stage,group=group,problem='declared_count_mismatch',field=key,declared=evaluation[key],verified=counts[key]))
            row=dict(stage=stage,model=group,target=target,training_role='reused_selected_model' if reuse else 'selected_model_reference' if stage=='formal_test' else 'trained_here',
                training_state=('stopped_before_target' if finished.get('target_reached') is False else 'complete') if finished else 'partial' if progress else 'unmeasured',
                updates=finished.get('updates',progress.get('update')),exposure_unit=unit,
                exposure=None if reuse else finished.get('control_tick_epochs') if residual else progress.get('anchor_equivalent_epochs'),
                stop_reason=finished.get('stop_reason'),selection=finished.get('selection',training_cfg.get('selection',offline.get('selection'))),
                selected_update=selected_receipt.get('update',offline.get('selected_update')),
                selected_epoch=selected_receipt.get('epoch'),selection_receipt=selected_receipt,selected_checkpoint=selected,
                training=finished,historical_training=historical_training,training_progress={} if reuse else progress,
                historical_progress=progress if reuse else None,training_configuration=training_cfg,training_budget=training_budget,
                warm_start=training_cfg.get('warm_start'),adaptation_scope=adaptation_scope,independent_execution=execution,
                offline_state='complete' if offline.get('complete') else 'partial' if stage_metrics else 'unmeasured',
                offline_target=target,closed_loop=counts,failure=failure,auxiliary_measurements=auxiliary)
            stage_out['groups'][group]=row
            group_rows.append(dict(stage=stage,model=group,target=target,training_state=row['training_state'],
                training_role=row['training_role'],updates=row['updates'],exposure=row['exposure'],exposure_unit=unit,
                stop_reason=row['stop_reason'],selection=row['selection'],offline_state=row['offline_state'],
                cohort=stage_out['cohort'],**{k:v for k,v in counts.items() if k not in ('declared_counts','denominator_rule','evaluation_state_scope')}))
        for name in ('monitor_exit.json','resources/resources.json','throughput_comparison.json','throughput_probe.json'):
            receipt=optional(node/name)
            if receipt is not None: resource_rows.append(dict(stage=stage,path=str(node/name),receipt=receipt))
        exported_stages[stage]=stage_out
    log_rows, physx = [], []
    for path in sorted(root.rglob('*.log')):
        if not path.is_file(): continue
        evidence(path,'log'); count=0; errors=0
        with path.open(errors='replace') as stream:
            for line_number,line in enumerate(stream,1):
                if '[Error]' in line or 'Traceback (most recent call last)' in line: errors+=1
                if 'PhysX error:' in line:
                    count+=1;physx.append(dict(path=str(path),line=line_number,message=line.rstrip()))
        log_rows.append(dict(path=str(path),physx_error_lines=count,error_or_traceback_lines=errors))
    root_receipts={name:optional(root/(name+'.json')) for name in ('physics_acceptance','geometry_acceptance','geometry_blocked','force_calibration_blocked','monitor_exit','owner_error','launch','execution_plan','concurrency_comparison','owned_cleanup','development_kit_owner')}
    gate_attempts={}
    for gate in sorted(root.glob('physics_gate*')):
        if not gate.is_dir(): continue
        receipts={}
        for p in sorted(gate.glob('*.json')):
            receipt=optional(p)
            # Keep large geometry/process inventories local, while retaining every
            # attempt receipt's exact identity and all acceptance/failure fields.
            if p.name in ('geometry_live.json','geometry_acceptance.json','inotify.json'):
                receipt={k:v for k,v in receipt.items() if k not in ('inventory','before','after')}
                receipt.update(omitted_fields='Inventory/process listings stay in the indexed local receipt',artifact=evidence(p))
            root_receipts[gate.name+'/'+p.stem]=receipt
            receipts[p.stem]=receipt
        result=receipts.get('result',{});acceptance=receipts.get('acceptance',{});exit_receipt=receipts.get('exit',{})
        passed=acceptance.get('passed');code=exit_receipt.get('returncode',acceptance.get('worker_returncode'))
        inconsistent=passed is True and (code != 0 or result.get('complete') is not True or result.get('software_complete') is False)
        if inconsistent: issues.append(dict(problem='physics_gate_acceptance_receipts_inconsistent',attempt=gate.name))
        state=('inconsistent' if inconsistent else 'accepted' if passed is True else 'failed' if passed is False or code not in (None,0)
            else 'functional_complete_without_accepted_exit' if result.get('complete') else 'unmeasured')
        geometry=receipts.get('geometry_acceptance',{})
        gate_attempts[gate.name]=dict(state=state,selected_by_session=gate.name==session.get('accepted_physics_gate','physics_gate'),
            functional_complete=result.get('complete'),software_complete=result.get('software_complete'),acceptance_passed=passed,
            worker_returncode=code,geometry_passed=geometry.get('passed'),live_collision_parity=geometry.get('live_collision_parity'),
            receipts=list(receipts))
    summary=dict(schema='method_matrix_cpu_publication_v1',status=status,source_run=str(root),session=session,
        configuration=cfg,preregistration=registration,actual_budget=budget,stages=exported_stages,root_receipts=root_receipts,
        physics_gate_attempts=gate_attempts,
        resource_receipts=resource_rows,publication_issues=issues,log_summary=log_rows,physx_error_lines=len(physx),
        trace_diagnostics=trace_diagnostics,
        physx_count_scope='All lines in every retained run log; duplicated log sinks are line occurrences, not proven distinct simulator events',
        inference_or_training_performed=False,kit_loaded=False,selection_performed_by_exporter=False,
        missing_value_semantics='null/unmeasured means unmeasured; never a 0 percent task success rate',
        limitations=['q_reached nominal configuration errors and u_expert command errors have separate targets and plots',
            'Residual frozen-base errors against u_expert are command-gap diagnostics, not actual-configuration fitting error',
            'Expert-history FK/surface errors are not autonomous physical deviation',
            'Formal test is reported separately and is never consulted to choose a model',
            'Native resets pair source specifications and event seeds, not hidden simulator state',
            'A normal exit never establishes harmless PhysX errors'])
    write(output/'summary.json',public(summary))
    emit('groups.csv',group_rows);emit('closed_loop.csv',closed_rows);emit('offline_metrics.csv',error_rows)
    emit('validation_curves.csv',curve_rows);emit('training_loss_intervals.csv',loss_rows)
    emit('log_counts.csv',log_rows);emit('physx_errors.csv',physx);emit('video_index.csv',videos)
    write(output/'evaluation_plans.json',public(evaluation_plans))
    write(output/'source_artifacts.json',public(sources));write(output/'local_artifacts.json',public(artifacts))
    matrix_plots(output,curve_rows,loss_rows,error_rows)
    for p in output.glob('*.csv'):p.write_bytes(p.read_bytes().replace(b'\r\n',b'\n'))
    write(output/'artifacts.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.iterdir()) if p.is_file() and p.name!='artifacts.json'})
    return public(summary)


def matrix_commit_publication(root, output):
    """Publish only verified CPU report files after the finite owner has exited.

    The caller must first run matrix_publish. Failures never alter experiment
    status or rewind commits; a failed push retains its local report commit.
    """
    import gzip
    import io
    import re
    root, output = Path(root).resolve(), Path(output).absolute()
    repository = CODE.resolve(); target = 'refs/heads/feature/tight-state-foundation'
    allowed_remotes = {'https://github.com/mao455/bcrl_visualTactile.git',
        'git@github.com:mao455/bcrl_visualTactile.git', 'ssh://git@github.com/mao455/bcrl_visualTactile.git'}
    receipt = dict(complete=False, target=target, remote='origin', source_head=None,
        report_commit=None, pushed=False, files=[], started=time.time())
    staged_by_us = False
    def git(*args):
        return subprocess.run(['git', *args], cwd=repository, check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=45).stdout
    def require(condition, reason):
        if not condition: raise ValueError(reason)
    def names(*args):
        return set(filter(None, git(*args, '-z').split('\0')))
    try:
        status, session = read(root/'status.json'), read(root/'session.json')
        source = session.get('source_head'); receipt['source_head'] = source
        require(status.get('state') in ('COMPLETE','FAILED','BUDGET_STOP') and status.get('owned_workers_exited') is True,
            'Report publication requires terminal owner status and exited owned workers')
        require(isinstance(source,str) and re.fullmatch(r'[0-9a-f]{40}',source), 'Missing frozen source commit')
        require(git('rev-parse','HEAD').strip()==source, 'HEAD differs from frozen source; preserve user commits')
        for mode in ((),('--push',)):
            require(git('remote','get-url',*mode,'--all','origin').splitlines() in [[url] for url in allowed_remotes],
                'Unexpected origin repository or multiple remote URLs')
        remote = git('ls-remote','--exit-code','origin',target).splitlines()
        require(remote==[source+'\t'+target], 'Remote target differs from frozen source; no automatic reconciliation')
        require(not names('diff','--cached','--name-only'), 'Pre-existing staged files; preserve user index')
        frozen = read(root/'source_freeze.json')
        require(bool(frozen), 'Missing frozen runtime hash manifest')
        def verify_source():
            require(git('rev-parse','HEAD').strip()==source, 'Source HEAD changed before report commit')
            for name, digest in frozen.items():
                path = repository/name
                require(path.resolve().is_relative_to(repository) and not path.is_symlink() and sha(path)==digest,
                    'Frozen runtime changed: '+name)
        verify_source()
        relative = output.relative_to(repository)
        require(len(relative.parts)>=3 and relative.parts[:2]==('research_docs','evidence'), 'Output must be a report folder under research_docs/evidence')
        for path in (output,*output.parents):
            if path==repository: break
            require(not path.is_symlink(), 'Publication path contains a symlink: '+str(path))
        state_doc = repository/'research_docs/STATE_DP_RESULTS.md'
        require(state_doc.is_file() and not state_doc.is_symlink(), 'State document missing or linked')
        original_doc=git('show',source+':research_docs/STATE_DP_RESULTS.md')
        def unmanaged_doc(value):
            return re.sub(r'<!-- METHOD_MATRIX_START -->.*?<!-- METHOD_MATRIX_END -->\n?', '', value, flags=re.S)
        require(unmanaged_doc(state_doc.read_text())==unmanaged_doc(original_doc),
            'State document changed outside the owner-managed result block; preserve user edits')
        manifest = read(output/'artifacts.json')
        files = sorted(output.iterdir())
        require({p.name for p in files}==set(manifest)|{'artifacts.json'}, 'Output differs from publisher artifact manifest')
        summary = read(output/'summary.json')
        require(summary.get('schema')=='method_matrix_cpu_publication_v1' and summary.get('status')==status
            and summary.get('session',{}).get('source_head')==source, 'Publication is stale or belongs to another source')
        source_run = Path(summary['source_run'])
        require((repository/source_run).resolve()==root, 'Publication belongs to another run')
        generated = {'summary.json','groups.csv','closed_loop.csv','offline_metrics.csv','validation_curves.csv',
            'training_loss_intervals.csv','log_counts.csv','physx_errors.csv','video_index.csv','evaluation_plans.json',
            'source_artifacts.json','local_artifacts.json','artifacts.json','training_curves.png','offline_errors.png'}
        generated.update(stage+'_'+group+'_geometry.csv' for stage,row in summary['stages'].items() for group in row['groups'])
        generated.update(stage+'_'+group+'_updates.csv' for stage,row in summary['stages'].items() for group in row['groups'])
        require(all(p.name.removesuffix('.gz') in generated for p in files), 'Unknown files are not generated matrix reports')
        total=0
        for path in files+[state_doc]:
            require(path.is_file() and not path.is_symlink(), 'Only regular report files may be published')
            size=path.stat().st_size;total+=size
            require(size<=20*1024**2, 'Report file exceeds 20 MiB: '+path.name)
            if path==state_doc: continue
            require(path.suffix in ('.json','.csv','.png','.gz'), 'Raw/model/unknown file type: '+path.name)
            if path.name!='artifacts.json':
                require(manifest[path.name]==dict(bytes=size,sha256=sha(path)), 'Generated artifact changed: '+path.name)
            if path.suffix=='.json': read(path)
            elif path.suffix=='.csv':
                with path.open(newline='',encoding='utf-8') as stream:
                    for row in csv.reader(stream): require(all('\0' not in cell for cell in row), 'Binary CSV rejected')
            elif path.suffix=='.png':
                from PIL import Image
                with Image.open(path) as picture:
                    require(picture.format=='PNG', 'Non-PNG binary rejected');picture.verify()
            else:
                require(path.name.endswith('.csv.gz'), 'Only compressed numeric CSV is allowed')
                with gzip.open(path,'rb') as stream: payload=stream.read(100*1024**2+1)
                require(len(payload)<=100*1024**2, 'Compressed numeric report exceeds decoded limit')
                rows=csv.reader(io.StringIO(payload.decode('utf-8')));header=next(rows,None)
                require(bool(header), 'Empty compressed numeric table')
                for row in rows:
                    require(len(row)==len(header), 'Invalid compressed numeric row')
                    for cell in row:
                        if cell: require(math.isfinite(float(cell)), 'Non-numeric compressed payload')
        require(total<=100*1024**2, 'Total report exceeds 100 MiB')
        explicit = {str(p.relative_to(repository)) for p in files+[state_doc]}
        changed = names('diff','--name-only')
        require(changed<=explicit, 'Other tracked changes exist: '+', '.join(sorted(changed-explicit)))
        snapshots = {str(p.relative_to(repository)):sha(p) for p in files+[state_doc]}
        receipt.update(files=sorted(explicit),bytes=total,report_sha256=snapshots)
        verify_source()
        require(not names('diff','--cached','--name-only'), 'Index changed before report staging')
        require(all(sha(repository/p)==digest for p,digest in snapshots.items()), 'Report changed before staging')
        git('add','--',*receipt['files']);staged_by_us=True
        require(names('diff','--cached','--name-only')<=explicit, 'Unrelated files entered the index')
        require(names('diff','--name-only')<=explicit, 'Unrelated tracked changes appeared before commit')
        verify_source()
        require(all(sha(repository/p)==digest for p,digest in snapshots.items()), 'Report changed before commit')
        require(bool(names('diff','--cached','--name-only')), 'No generated report changes to commit')
        git('commit','--only','-m','Publish measured method matrix results: '+root.name,'--',*receipt['files'])
        receipt['report_commit']=git('rev-parse','HEAD').strip();staged_by_us=False
        require(git('rev-parse','HEAD^').strip()==source, 'Report commit does not directly follow frozen source')
        require(names('diff-tree','--no-commit-id','--name-only','-r','HEAD')<=explicit, 'Report commit contains unrelated paths')
        require(all(sha(repository/name)==digest for name,digest in frozen.items()), 'Frozen runtime changed during commit')
        require(not names('diff','--cached','--name-only') and not names('diff','--name-only'), 'Working tree or index changed during commit')
        require(git('rev-parse','HEAD').strip()==receipt['report_commit'], 'HEAD changed before report push')
        git('push','origin','HEAD:'+target)
        receipt.update(complete=True,pushed=True,ended=time.time())
        write(root/'public_git.json',receipt)
    except Exception as error:
        receipt.update(error=repr(error),ended=time.time())
        if isinstance(error,subprocess.CalledProcessError): receipt['git_stderr']=error.stderr
        if staged_by_us and not receipt['report_commit']:
            try: git('restore','--staged','--',*receipt['files'])
            except Exception as cleanup_error: receipt['index_cleanup_error']=repr(cleanup_error)
        write(root/'public_git_error.json',receipt)
    return receipt


def matrix_plots(output, curves, losses, errors):
    """One plotter for exported scalar rows; each target/exposure has its own axes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels=list(dict.fromkeys((r['target'],r['exposure_unit']) for r in curves+losses))
    if not panels: (output/'training_curves.png').unlink(missing_ok=True)
    if panels:
        fig,axes=plt.subplots(len(panels),2,figsize=(12,3.5*len(panels)),squeeze=False,layout='constrained')
        for axis,(target,unit) in zip(axes,panels):
            for model in dict.fromkeys(r['model'] for r in losses if (r['target'],r['exposure_unit'])==(target,unit)):
                rows=[r for r in losses if r['model']==model and (r['target'],r['exposure_unit'])==(target,unit) and r['exposure'] is not None]
                axis[0].plot([r['exposure'] for r in rows],[r['mean_loss'] for r in rows],label=model)
            for model,cohort in dict.fromkeys((r['model'],r['cohort']) for r in curves if (r['target'],r['exposure_unit'])==(target,unit)):
                rows=[r for r in curves if r['model']==model and r['cohort']==cohort and (r['target'],r['exposure_unit'])==(target,unit)
                      and r['exposure'] is not None and r['selection_metric'] is not None]
                axis[1].plot([r['exposure'] for r in rows],[r['selection_metric'] for r in rows],label=model+' '+cohort)
            for ax,title in zip(axis,('Training objective interval mean','Fixed-cohort validation metric')):
                ax.set(title=target+' / '+title,xlabel=unit);ax.grid(alpha=.25)
                if ax.lines:ax.legend(fontsize=7)
                else:ax.text(.5,.5,'Unmeasured',ha='center',transform=ax.transAxes)
        fig.savefig(output/'training_curves.png',dpi=140);plt.close(fig)
    targets=list(dict.fromkeys(r['target'] for r in errors))
    if not targets: (output/'offline_errors.png').unlink(missing_ok=True)
    if targets:
        fig,axes=plt.subplots(len(targets),3,figsize=(13,3.6*len(targets)),squeeze=False,layout='constrained')
        for axis,target in zip(axes,targets):
            for ax,part,key,title in zip(axis,('arm','hand','arm'),('rmse_rad','rmse_rad','ee_rmse_mm'),
                    ('Arm RMSE (rad)','Hand RMSE (rad)','Target-configuration palm FK RMSE (mm)')):
                rows=[r for r in errors if r['target']==target and r['split']=='validation' and r['phase']=='all'
                    and r['motion']=='all' and r['group']==part and r[key] is not None and r['ticks']]
                labels=[r['model']+(' nominal vs command' if r['prediction_kind'].startswith('nominal_against') else '') for r in rows]
                ax.bar(range(len(rows)),[r[key] for r in rows]);ax.set_xticks(range(len(rows)),labels,rotation=35,ha='right',fontsize=7)
                ax.set(title=target+' / '+title);ax.grid(axis='y',alpha=.25)
                if not rows:ax.text(.5,.5,'Unmeasured',ha='center',transform=ax.transAxes)
        fig.suptitle('Fixed expert histories; separate target semantics; no autonomous deviation claim')
        fig.savefig(output/'offline_errors.png',dpi=140);plt.close(fig)

def qg_publish(root, output=None):
    """Read completed QG evidence on CPU; no policy, torch or simulator imports."""
    import gzip
    import io
    import sys
    import h5py
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root).resolve()
    terminal=read(root/'status.json')
    if terminal.get('state')!='COMPLETE' or terminal.get('owned_workers_exited') is not True:
        raise ValueError('QG detailed publication requires the completed owner and exited owned workers')
    output=Path(output).resolve() if output else CODE/'research_docs/evidence'/root.name/'qg_details'
    if output==root or root in output.parents:
        raise ValueError('Reports must not modify the completed experiment root')
    output.mkdir(parents=True,exist_ok=True)
    sources={};generated={};events=[];videos=[];groups={}

    def require(ok, message):
        if not ok:raise ValueError(message)

    def public(value):
        if isinstance(value,dict):return {k:public(v) for k,v in value.items()}
        if isinstance(value,list):return [public(v) for v in value]
        if isinstance(value,str) and value.startswith('/'):
            marker='/outputs/'
            if marker in value:return 'outputs/'+value.split(marker,1)[1]
            if value.startswith(str(CODE)+'/'):return value[len(str(CODE))+1:]
        return value

    def provenance(path):
        path=Path(path)
        key=str(path)
        if key not in sources:sources[key]=dict(path=public(key),sha256=sha(path),bytes=path.stat().st_size)
        return sources[key]

    def flat(value, prefix=''):
        result={}
        if isinstance(value,dict):
            for key,item in value.items():result.update(flat(item,(prefix+'.' if prefix else '')+str(key)))
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):result.update(flat(item,prefix+'.'+str(index)))
        else:
            if isinstance(value,(float,np.floating)):require(math.isfinite(value),'Nonfinite source: '+prefix)
            result[prefix]=value.item() if isinstance(value,np.generic) else value
        return result

    def emit(path, rows):
        # A factory permits two bounded passes over the large microbatch log.
        factory=rows if callable(rows) else lambda:iter(rows)
        fields={};count=0
        for row in factory():
            for key,value in row.items():
                if isinstance(value,(float,np.floating)):require(math.isfinite(value),'Nonfinite export: '+key)
                fields.setdefault(key,None)
            count+=1
        require(count>0,'Empty export: '+str(path))
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('wb') as target:
            compressed=gzip.GzipFile(filename='',mode='wb',fileobj=target,mtime=0) if path.suffix=='.gz' else target
            with io.TextIOWrapper(compressed,encoding='utf-8',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(fields),lineterminator='\n');writer.writeheader();writer.writerows(factory())
        generated[str(path.relative_to(output))]=dict(rows=count,columns=len(fields))
        return count

    def lines(path):
        with Path(path).open() as stream:
            for line in stream:
                if line.strip():yield json.loads(line)

    def copy_numeric(path,destination):
        provenance(path);destination.parent.mkdir(parents=True,exist_ok=True)
        if path.suffix=='.csv':
            with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
            for row in rows:
                for value in row.values():
                    try:number=float(value)
                    except (TypeError,ValueError):continue
                    require(math.isfinite(number),'Nonfinite copied CSV: '+str(path))
            emit(destination.with_suffix('.csv.gz'),rows)
        else:
            if path.suffix=='.json':
                value=read(path);flat(value);write(destination,public(value))
            else:shutil.copyfile(path,destination)

    def first(mask):
        indices=np.flatnonzero(mask)
        return int(indices[0]+1) if len(indices) else None

    def stable_grasp_numpy(a,palm,initial):
        # Same rule as native_evaluation.stable_grasp, implemented without torch.
        contact=np.linalg.norm(a['object_contact'],axis=-1)>.1
        two=contact.sum(-1)>=2;pose=a['object'];inverse=palm[:,3:].copy();inverse[:,1:]*=-1
        qv=inverse[:,1:];v=pose[:,:3]-palm[:,:3]
        t=2*np.cross(qv,v);relative=v+inverse[:,:1]*t+np.cross(qv,t)
        q=pose[:,3:];relative_q=np.concatenate((inverse[:,:1]*q[:,:1]-(qv*q[:,1:]).sum(-1,keepdims=True),
            inverse[:,:1]*q[:,1:]+q[:,:1]*qv+np.cross(qv,q[:,1:])),axis=-1)
        stable=np.zeros(len(pose),dtype=bool);longest=current=0
        for active in two.reshape(-1):
            current=current+1 if active else 0;longest=max(longest,current)
        for stop in range(12,len(pose)+1):
            start=stop-12
            if not two[start:stop].all() or not (pose[start:stop,2]>=initial+.02).all():continue
            drift=np.linalg.norm(relative[start:stop]-relative[start],axis=-1).max()
            cosine=np.clip(np.abs((relative_q[start:stop]*relative_q[start]).sum(-1)),0,1)
            stable[stop-1]=drift<=.01 and (2*np.arccos(cosine)).max()<=.2
        return stable,dict(object_contact=bool(contact.any()),max_fingers=int(contact.sum(-1).max()),
            two_finger_continuous_s=longest/120,stable_grasp_diagnostic=bool(stable.any()),
            object_contact_threshold_N=.1,stable_window_s=.2,scope='filtered normal forces; not native success')

    def train_export(directory,dest,group):
        training=directory/'training';out=dest/'training';out.mkdir(parents=True,exist_ok=True)
        configuration=read(training/'configuration.json');provenance(training/'configuration.json')
        finished=read(training/'finished.json');provenance(training/'finished.json')
        loss_path=training/('training.jsonl' if group=='QG00' else 'loss.jsonl');provenance(loss_path)
        emit(out/'loss.csv.gz',lambda:(flat(r) for r in lines(loss_path)))
        val_path=training/('validation.jsonl' if group=='QG00' else 'validation.json');provenance(val_path)
        validation=list(lines(val_path)) if group=='QG00' else read(val_path)
        emit(out/'validation.csv',[flat(r) for r in validation])
        copy_numeric(training/'curves.png',out/'curves.png')
        if group=='QG00':
            fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
            for ax,key,title in zip(axes.flat,('arm_rmse','hand_rmse','ee_distance_rmse_mm','ee_rotation_mean_deg'),
                    ('Arm RMSE (rad)','Hand RMSE (rad)','Reached-configuration palm FK RMSE (mm)','Palm FK mean rotation error (deg)')):
                for cohort,label in [('train8','fixed train2'),('metrics','fixed validation4')]:
                    for horizon,style in [('commit12','-'),('valid24','--')]:
                        ax.plot([r['update'] for r in validation],
                            [(r[cohort] if horizon=='commit12' else r[cohort]['full_horizon'])[key] for r in validation],
                            style,label=label+' '+horizon)
                ax.set(xlabel='Optimizer update',ylabel=title);ax.grid(alpha=.25)
            axes[0,0].legend(fontsize=7);fig.suptitle('QG00: expert-history DDIM16 predictions; q_reached labels')
            fig.savefig(out/'validation_pose_curves.png',dpi=140);plt.close(fig)
            geo_path=training/'geometry_projection.jsonl';provenance(geo_path)
            emit(out/'geometry_projection.csv.gz',lambda:(flat(dict(log_index=i,**r)) for i,r in enumerate(lines(geo_path))))
            aggregate={}
            for log in lines(geo_path):
                update=log['update'];r=aggregate.setdefault(update,dict(update=update,logged_microbatches=0,
                    sampled_rows=0,required_contact_rows=0,nonzero_local_gradient_rows=0,penetrating_rows=0,
                    penetration_loss=0.,contact_loss=0.,geometry_loss=0.,contact_local_gradient_norm_sum=0.,
                    penetration_local_gradient_norm_sum=0.,maximum_absolute_penetration_m=0.,
                    maximum_penetration_violation_m=0.,maximum_required_contact_gap_violation_m=0.,
                    minimum_signed_surface_distance_m=None,required_fingertip_distance_sum_m=0.,required_fingertip_count=0))
                r['logged_microbatches']+=1
                for sample in log.get('rows',[]):
                    r['sampled_rows']+=1;r['required_contact_rows']+=int(sample['required_contacts']>0)
                    c=sample['contact_hand_gradient_norm_per_rad'];p=sample['penetration_hand_gradient_norm_per_rad']
                    r['nonzero_local_gradient_rows']+=int(c>0 or p>0)
                    r['contact_local_gradient_norm_sum']+=c;r['penetration_local_gradient_norm_sum']+=p
                    for key in ('penetration_loss','contact_loss','geometry_loss'):r[key]+=sample[key]
                    distance=sample['minimum_distance_m'];r['penetrating_rows']+=int(distance<0)
                    r['minimum_signed_surface_distance_m']=distance if r['minimum_signed_surface_distance_m'] is None else min(distance,r['minimum_signed_surface_distance_m'])
                    r['maximum_absolute_penetration_m']=max(r['maximum_absolute_penetration_m'],max(0.,-distance))
                    for key in ('maximum_penetration_violation_m','maximum_required_contact_gap_violation_m'):r[key]=max(r[key],sample[key])
                    for distance,required in zip(sample['fingertip_distance_m'],sample['required_contact_mask']):
                        if required:r['required_fingertip_distance_sum_m']+=distance;r['required_fingertip_count']+=1
            training_rows={r['update']:r for r in lines(loss_path)}
            for r in aggregate.values():
                count=r['sampled_rows'];raw=training_rows[r['update']]
                for key in ('penetration_loss','contact_loss','geometry_loss'):r[key]=r[key]/count if count else None
                r['mean_required_fingertip_distance_m']=r['required_fingertip_distance_sum_m']/r['required_fingertip_count'] if r['required_fingertip_count'] else None
                r['hand_geometry_gain']=raw['hand_geometry_gain'];r['hand_geometry_weight']=raw['loss_weights']['hand_geometry']
                r['weighted_geometry_loss']=raw['components']['hand_geometry']*r['hand_geometry_weight']
                r['nonzero_local_gradient_fraction']=r['nonzero_local_gradient_rows']/count if count else None
                if count:
                    require(count==raw['geometry_sampling']['selected_count'],'Geometry full-update sample count mismatch')
                    require(math.isclose(r['geometry_loss'],raw['components']['hand_geometry'],rel_tol=2e-5,abs_tol=1e-7),'Geometry reduction differs from trainer')
            geometry=list(aggregate.values());emit(out/'geometry_by_update.csv.gz',geometry)
            fig,axes=plt.subplots(2,2,figsize=(12,7),layout='constrained')
            panels=[(('penetration_loss','contact_loss'), 'Two sampled rows: mean hinge losses'),
                (('contact_local_gradient_norm_sum','penetration_local_gradient_norm_sum'),'Separate local gradient norm sums / rad'),
                (('maximum_absolute_penetration_m','maximum_penetration_violation_m'),'Sampled maximum depth / tolerance violation (m)'),
                (('mean_required_fingertip_distance_m','maximum_required_contact_gap_violation_m'),'Required fingertip distance / gap violation (m)')]
            for ax,(keys,title) in zip(axes.flat,panels):
                for key in keys:
                    rows=[r for r in geometry if r[key] is not None and r['sampled_rows']]
                    ax.plot([r['update'] for r in rows],[r[key] for r in rows],label=key,alpha=.7,linewidth=.65)
                ax.set(xlabel='Optimizer update',title=title);ax.legend(fontsize=6);ax.grid(alpha=.25)
            active=[r for r in geometry if r['sampled_rows']]
            fraction=axes[0,1].twinx()
            fraction.plot([r['update'] for r in active],[r['nonzero_local_gradient_fraction'] for r in active],
                color='gray',linewidth=.4,alpha=.3)
            fraction.set(ylabel='Rows with nonzero local gradient / selected rows',ylim=(0,1))
            fig.suptitle('QG00 sampled x0 geometry in expert context; local gradients are not parameter-gradient norms')
            fig.savefig(out/'geometry_curves.png',dpi=140);plt.close(fig)
            geometry_summary=dict(log_rows=generated[str((out/'geometry_projection.csv.gz').relative_to(output))]['rows'],
                updates=len(geometry),active_updates=sum(bool(r['sampled_rows']) for r in geometry),
                sampled_configurations=sum(r['sampled_rows'] for r in geometry),
                nonzero_local_gradient_configurations=sum(r['nonzero_local_gradient_rows'] for r in geometry),
                normalization='Per-update selected-row mean; not mean of microbatch means or population-unbiased',
                scope='Denoised training x0 configurations in expert object/contact context; not autonomous physical trajectories',
                penetration='Absolute depth=max(0,-minimum_distance_m); violation excludes 1mm tolerance',
                contact='Required-finger gap violation excludes 2mm tolerance; separate contact/penetration local gradients')
        else:geometry_summary=None
        for p in sorted((directory/'offline').iterdir()):
            if p.suffix in ('.csv','.json','.png') and p.name!='progress.json':copy_numeric(p,dest/'offline'/p.name)
        selected=training/('validation.json' if group=='QG00' else 'selected.json');provenance(selected)
        weights={}
        for name in ('best.pt','last.pt'):
            p=training/name
            if p.exists():weights[name]=provenance(p)
        config={k:v for k,v in configuration.items() if k not in ('source_sha256','warm_start')}
        warm=configuration.get('warm_start')
        if warm:config['warm_start']={k:v for k,v in warm.items() if k!='learned_parameter_names'}
        return dict(finished=finished,selected=read(selected),configuration=config,weights=weights,geometry=geometry_summary,
            validation_scope='Fixed train2/validation4; selection before development evaluation; no test22',
            target='q_reached' if group=='QG00' else 'u_expert combined actuator command; base q_reached prediction is a different target',
            offline_scope='Expert-history predictions; QG01 base_* files compare nominal q output against u_expert, not configuration fitting')

    provenance(root/'status.json')
    for stage,group in [('configuration','QG00'),('configuration_residual','QG01')]:
        directory=root/stage/group;dest=output/group;dest.mkdir(exist_ok=True)
        evaluation=read(directory/'evaluation_summary.json');provenance(directory/'evaluation_summary.json')
        complete=root/'queue'/group/'evaluation_complete.json'
        if not complete.exists():complete=root/stage/'evaluation_complete.json'
        require(read(complete)['complete'],'Evaluation incomplete');provenance(complete)
        require(evaluation['attempts']==10 and evaluation['valid']==10,'Expected completed ten-case QG cohort')
        reported={r['identity']:r for r in evaluation['results']};group_events=[]
        fig,axes=plt.subplots(10,2,figsize=(12,24),layout='constrained')
        for index,path in enumerate(sorted((directory/'evaluation').glob('*/result.json'))):
            trial=path.parent;r=read(path);provenance(path);require(r==reported[trial.name],'Trial/summary mismatch')
            for name in ('trace.npz','numeric.h5','initial_observable.json'):provenance(trial/name)
            initial=read(trial/'initial_observable.json')['object_pose_w'][0][2]
            with np.load(trial/'trace.npz',allow_pickle=False) as z,h5py.File(trial/'numeric.h5','r') as h:
                a={k:z[k] for k in z.files};tr=h['episode/transition'];n=len(a['joint'])
                require(n==r['steps'],'Trace/result tick count mismatch')
                for value in a.values():require(np.isfinite(value).all(),'Nonfinite trace')
                for key,field in [('joint','joint_position_rad'),('object','object_pose_w'),('lifted','lifted_object'),('subgoals','successes'),('retract','retract_succeeded')]:
                    require(np.array_equal(a[key],tr['diagnostics_after/'+field][:]),'Trace/HDF mismatch: '+key)
                for key,field in [('target','target_after_rad'),('object_contact','object_contact_force_w')]:
                    require(np.array_equal(a[key],tr[field][:]),'Trace/HDF mismatch: '+key)
                require(np.array_equal(a['alignment'],tr['native_alignment_condition'][:]),'Alignment trace mismatch')
                clock=tr['time_after_s'][:];before=tr['time_before_s'][:];palm=tr['diagnostics_after/palm_body_pose_w'][:]
                require(np.isfinite(clock).all() and np.all(np.diff(clock)>0),'Invalid recorded time')
                reasons={k:v[:] for k,v in tr['termination_reasons'].items()}
                for key,value in reasons.items():require(bool(value.any())==r['reasons'][key],'Termination mismatch: '+key)
            stable,contact=stable_grasp_numpy(a,palm,initial)
            if r.get('contact'):require(contact==r['contact'],'Recomputed grasp diagnostic differs')
            require(bool(a['lifted'].any())==r['native_lift'] and int(a['subgoals'].max())==r['native_subgoals'],'Native stage mismatch')
            require(bool(a['alignment'].any())==r['native_alignment'],'Native alignment/result mismatch')
            require(bool(reasons['max_successes'].any())==r['success'],'Native success/result mismatch')
            forces=np.linalg.norm(a['object_contact'],axis=-1);fingers=(forces>.1).sum(-1)
            tracking=a['joint']-a['target'];rows=[]
            for tick in range(n):
                row=dict(tick=tick+1,time_before_s=float(before[tick]),time_after_s=float(clock[tick]),
                    elapsed_after_s=float(clock[tick]-before[0]),object_z_m=float(a['object'][tick,2]),
                    object_rise_from_initial_m=float(a['object'][tick,2]-initial),
                    contact_fingers_max_substep=int(fingers[tick].max()),two_finger_both_substeps=bool((fingers[tick]>=2).all()),
                    arm_tracking_rmse_rad=float(np.sqrt(np.mean(tracking[tick,:7]**2))),
                    hand_tracking_rmse_rad=float(np.sqrt(np.mean(tracking[tick,7:]**2))),
                    native_lift=bool(a['lifted'][tick]),native_subgoals=int(a['subgoals'][tick]),native_alignment=bool(a['alignment'][tick]),
                    native_retract=bool(a['retract'][tick]),computed_stable_grasp_window_end=bool(stable[tick]))
                row.update({f'finger_{finger}_object_force_max_N':float(forces[tick,:,finger].max()) for finger in range(5)})
                row.update({'termination_'+key:bool(value[tick]) for key,value in reasons.items()})
                if group=='QG01':
                    for part,selection in [('arm',slice(0,7)),('hand',slice(7,29))]:
                        for name in ('residual_applied_rad','residual_proposed_rad'):
                            row[name+'_'+part+'_rms']=float(np.sqrt(np.mean(a[name][tick,selection]**2)))
                    row.update(residual_clip_components=int(a['residual_clip_mask'][tick].sum()),
                        residual_tanh_saturation_components=int((np.abs(a['residual_tanh_fraction'][tick])>=.95).sum()),
                        residual_contact_observed=bool(a['residual_contact_gate'][tick]),residual_contact_valid=bool(a['residual_contact_valid'][tick]))
                rows.append(row)
            emit(dest/'rollouts'/(trial.name+'.csv.gz'),rows)
            event=dict(group=group,identity=trial.name,seed=r['seed'],valid=r['legal'] and r['software_complete'],success=r['success'],
                ticks=n,first_contact_tick=first((fingers>0).any(-1)),first_two_finger_tick=first((fingers>=2).any(-1)),
                first_native_lift_tick=first(a['lifted']),first_subgoal_tick=first(a['subgoals']>=1),second_subgoal_tick=first(a['subgoals']>=2),
                first_alignment_tick=first(a['alignment']),first_retract_tick=first(a['retract']),first_computed_stable_tick=first(stable),
                max_contact_fingers=contact['max_fingers'],two_finger_continuous_s=contact['two_finger_continuous_s'],
                computed_stable_grasp=contact['stable_grasp_diagnostic'],stable_source='CPU recomputation of original rule from recorded object/palm/forces',
                recorded_contact_summary_present='contact' in r,termination_tick=n,termination=';'.join(k for k,v in r['reasons'].items() if v),
                maximum_rise_from_initial_m=float((a['object'][:,2]-initial).max()),
                observed_stage='object_contact_without_native_lift' if contact['object_contact'] and not r['native_lift'] else 'see_native_events',
                interpretation='Observed stage only; no causal failure attribution',source_result_sha256=sha(path))
            event.update({'first_termination_'+key+'_tick':first(value) for key,value in reasons.items()})
            events.append(event);group_events.append(event)
            x=[row['elapsed_after_s'] for row in rows]
            axes[index,0].plot(x,[row['object_rise_from_initial_m']*1000 for row in rows],linewidth=1)
            axes[index,0].axhline(20,color='gray',linestyle=':',label='stable diagnostic height prerequisite')
            axes[index,0].set(title=trial.name+' / '+event['termination'],ylabel='Height from reset (mm)')
            for finger in range(5):axes[index,1].plot(x,forces[:,:,finger].max(-1),label='finger '+str(finger),linewidth=.65)
            axes[index,1].axhline(.1,color='gray',linestyle=':');axes[index,1].set(ylabel='Filtered object force (N)')
            for ax in axes[index]:
                for key,color in [('first_contact_tick','tab:green'),('first_native_lift_tick','tab:orange'),('first_subgoal_tick','tab:purple')]:
                    if event[key] is not None:ax.axvline(x[event[key]-1],color=color,linestyle='--',alpha=.7)
                ax.set(xlabel='Recorded elapsed time (s)');ax.grid(alpha=.2)
            for video in sorted(trial.glob('*.mp4')):
                receipt=trial/'video_verified.json'
                videos.append(dict(group=group,identity=trial.name,**provenance(video),
                    verification=json.dumps(public(read(receipt)),sort_keys=True) if receipt.exists() else None))
                if receipt.exists():provenance(receipt)
        require(len(group_events)==10,'Wrong number of completed results')
        axes[0,1].legend(fontsize=7);fig.suptitle(group+' autonomous development: green=first contact; native stages absent when unreached')
        fig.savefig(dest/'rollouts.png',dpi=130);plt.close(fig)
        groups[group]=dict(attempts=evaluation['attempts'],valid=evaluation['valid'],rejected=evaluation['rejected'],
            successes=evaluation['successes'],native_lifts=evaluation['native_lifts'],first_subgoal_reached=evaluation['first_subgoal_reached'],
            second_subgoal_reached=sum(e['second_subgoal_tick'] is not None for e in group_events),alignments=evaluation['alignments'],
            stable_grasp_computed=sum(e['computed_stable_grasp'] for e in group_events),
            terminations={key:sum(e['first_termination_'+key+'_tick'] is not None for e in group_events) for key in reasons},
            ticks=sum(e['ticks'] for e in group_events),training=train_export(directory,dest,group))
    emit(output/'events.csv',events);emit(output/'video_index.csv',videos)
    write(output/'sources.json',public(list(sources.values())))
    require(not any(name=='torch' or name.split('.')[0] in ('omni','carb','isaacsim','isaaclab') for name in sys.modules),
        'Publisher unexpectedly imported a policy/GPU/simulator dependency')
    files=[]
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name!='summary.json':
            files.append(dict(path=str(path.relative_to(output)),bytes=path.stat().st_size,sha256=sha(path),**generated.get(str(path.relative_to(output)),{})))
    summary=dict(version='qg_cpu_detail_v1',source_root=str(root),groups=groups,files=files,
        tick_convention='1-based completed 60Hz control transitions; first contact is the containing control tick of either 120Hz substep',
        contact_force='Per-finger maximum vector norm across the two recorded physics substeps; filtered object force, threshold >0.1N',
        stable_grasp_rule='12 complete ticks: both substeps >=2 fingers, object >=reset height+0.02m, palm-relative drift <=0.01m and <=0.2rad',
        tracking_error='Post-integration actual joint minus issued actuator target; not autonomous/expert same-tick imitation error',
        qualification='All results are development evidence. Contact/stage/tolerance diagnostics do not establish a causal failure mechanism.',
        test22_used=False,checkpoint_deserialized=False,torch_or_kit_imported=False,raw_trajectories_or_weights_copied=False,
        validation=dict(result_summary_trace_hdf_agreement=True,
            checked_trace_hdf_fields=['joint','object','target','object_contact','lifted','subgoals','retract','alignment'],
            checked_result_fields=['steps','native_lift','native_subgoals','native_alignment','success','reasons','contact_when_recorded'],
            finite_exported_numbers=True,events=len(events),ticks=sum(e['ticks'] for e in events)))
    summary=public(summary);write(output/'summary.json',summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'check', 'run', 'status', 'final', 'publish', 'matrix_publish', 'dagger_publish', 'qg_publish'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=CODE/'configs/tight_dp3_diagnostics.json')
    parser.add_argument('--evaluation', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode == 'prepare': prepare(root, args.config)
    elif args.mode == 'status': print(json.dumps(read(root/'status.json'), indent=2))
    elif args.mode == 'matrix_publish':
        if args.output is None: parser.error('matrix_publish requires --output')
        matrix_publish(root, args.output.resolve())
    elif args.mode == 'dagger_publish':dagger_publish(root,args.output)
    elif args.mode == 'qg_publish':qg_publish(root,args.output)
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
