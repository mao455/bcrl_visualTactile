"""CPU-only control, dataset, RNG provenance and repository inventory audit."""
import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import torch

from .action_reference import command_to_reference, reference_to_command
from .long_data import MANIFEST, shift_actions
from .training_resources import atomic_json


UPSTREAM = 'ff2fc62873f6c6cd93164123c8358e9ce2d65c9b'


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sanitized(value):
    if isinstance(value,dict):
        return {str(sanitized(key)):sanitized(item) for key,item in value.items()}
    if isinstance(value,list):
        return [sanitized(item) for item in value]
    if isinstance(value,str):
        return value.replace(str(Path.cwd())+'/', '').replace('/home/CNS2025331827/', '<local-home>/')
    return value


def source_audit():
    paths = subprocess.check_output(['git', 'ls-files', 'isaacsimenvs'], text=True).splitlines()
    rows = []
    for name in paths:
        original = subprocess.run(['git', 'show', f'{UPSTREAM}:{name}'], capture_output=True)
        current = Path(name).read_bytes()
        rows.append(dict(path=name,sha256=digest(current),
            upstream_sha256=digest(original.stdout) if original.returncode == 0 else None,
            identical=original.returncode == 0 and original.stdout == current))
    return dict(upstream=UPSTREAM,head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        files=rows,changed=[row['path'] for row in rows if not row['identical']])


def inventory(output):
    tracked = subprocess.check_output(['git','ls-files'],text=True).splitlines()
    roots = ('visuotactile/', 'tests/', 'isaacsimenvs/tasks/play/', 'isaacsimenvs/tasks/precise_assembly/',
             'isaacsimenvs/cfg/', 'evaluation/eval_isaacsim.py', 'evaluation/problems/__init__.py',
             'evaluation/problems/peg/', 'evaluation/objects.py', 'isaacsimenvs/utils/rlgames_utils.py',
             'rl_games/rl_games/torch_runner.py','rl_games/rl_games/algos_torch/players.py',
             'rl_games/rl_games/algos_torch/network_builder.py','rl_games/rl_games/algos_torch/models.py')
    paths = sorted({name for name in tracked if name.startswith(roots) and Path(name).suffix in ('.py','.yaml','.yml','.json','.sh')}
                   | {'visuotactile/audit_system.py'})
    modules = {name[:-3].replace('/','.'):name for name in paths if name.endswith('.py')}
    incoming = defaultdict(set)
    rows = []
    documents = {str(path):path.read_text() for path in Path('research_docs').glob('*.md')}
    tests = {str(path):path.read_text() for path in Path('tests').glob('test_*.py')}
    for name in paths:
        source = Path(name).read_text()
        definitions, dependencies, calls = [], set(), set()
        purpose = ''
        if name.endswith('.py'):
            tree = ast.parse(source)
            purpose = (ast.get_docstring(tree) or '').split('\n')[0]
            package = name[:-3].replace('/','.').split('.')[:-1]
            for node in ast.walk(tree):
                if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    definitions.append(f'{node.name}:{node.lineno}')
                if isinstance(node,ast.Import):
                    dependencies.update(item.name for item in node.names)
                elif isinstance(node,ast.ImportFrom):
                    base = '.'.join(package[:len(package)-node.level+1]) if node.level else ''
                    dependencies.add('.'.join(part for part in (base,node.module) if part))
                elif isinstance(node,ast.Call):
                    called = ast.unparse(node.func)
                    if any(token in called.lower() for token in ('save','write','dump','step','subprocess','mkdir','unlink','remove','popen')):
                        calls.add(called)
            for dependency in dependencies:
                if dependency in modules:
                    incoming[modules[dependency]].add(name)
        elif name.endswith(('.yaml','.yml')):
            purpose = '原生任务/训练配置；由 Hydra / _load_env_cfg / _configure_agent 解析'
        else:
            purpose = '运行入口或显式配置；具体 shell / JSON 字段见源文件'
        original = subprocess.run(['git','show',f'{UPSTREAM}:{name}'],capture_output=True)
        status = 'upstream identical' if original.returncode == 0 and original.stdout == source.encode() else 'local addition' if original.returncode else 'modified upstream'
        rows.append(dict(path=name,purpose=purpose or '包入口/测试或辅助定义',definitions=definitions,
            imports=sorted(dependencies),io_calls=sorted(calls),upstream=status,
            documents=[path for path,text in documents.items() if name in text or Path(name).name in text],
            tests=[path for path,text in tests.items() if name[:-3].replace('/','.') in text],sha256=digest(source.encode())))
    for row in rows:
        row['imported_by'] = sorted(incoming[row['path']])
    atomic_json(output/'file_inventory.json',rows)
    sections = ['# 研究项目逐文件地图', '',
        '历史审计基线为 a07d0aa；文档首版6022330后按本轮最终工作树刷新，包含Loss B及诊断记录修复。此表覆盖全部 visuotactile Python/shell、研究测试、实际原生任务/配置及专家加载入口。',
        '每项列出路径、源码定义行、静态模块依赖/调用者、潜在 I/O 调用及关联文档/测试。依赖是 AST import 图，不冒充完整动态调用图；CLI、注册表和 shell 子进程调用另见下方主链。',
        '所有源文件 SHA256、完整定义和依赖在 [file_inventory.json](evidence/system_audit_20261031/file_inventory.json)。第三方完整源码不逐文件复制。', '',
        '## 从入口追踪系统', '',
        '| 环节 | 入口 → 被调用模块 | 输入 → 输出/副作用 |',
        '|---|---|---|',
        '| 原专家采集 | collect_success_batch → collect → evaluation.eval_isaacsim / rl_games → sim.environment → recording → compact/storage_audit → manifest | 原生观察→专家命令→真实 env.step→成功审核→本地 HDF5/manifest；可启动 Kit/审核进程 |',
        '| 旧命令DP | run_baselines → train_dp → dataset/history/models.nominal或tactile_dp；evaluate_baselines → rollout | 固定320/40/40→训练checkpoint；GPU策略/CPU物理→原生轨迹 |',
        '| H64 DP | run_long → long_data/train_long/validate_long；evaluate_long → long_controller | 冻结DINO缓存→完整episode训练→绝对reference→共享action_reference→native command |',
        '| DP损失 | train_long / models.long_horizon / study_metrics → dp_objectives | canonical29预测/目标/valid→arm7与hand22组内MSE各50%；正式默认B，A仅显式历史复现；不改推理参数 |',
        '| A/B历史对照 | run_loss_study → train_loss_study/study_metrics → evaluate_study | 相同epoch40 EMA→四条受控续训；只作为显式历史复现 |',
        '| GRU残差 | residual_session → residual_data/train_residual/evaluate_residual → residual_controller | 冻结B0因果计划→专家teacher forcing→contact-gated delta；不回填B0 |',
        '| Transformer | transformer_data/train_transformer_residual → transformer_controller → long_controller | 专家过去实际动作/DP计划→独立残差；部署ack回填实际混合命令 |',
        '| T0/T1 | evaluate_training_initials → restore_observable_state → native env.step；audit_training_initials | 近似初态+原命令或DP→删失/timeout独立报告；不是精确重放 |',
        '| 未来诊断记录v2 | evaluate_training_initials → diagnostic_recording | 每tick传感/记忆/采样与原生RNG、queue、target→本地HDF5；RGB-D只存12tick刷新帧；只读快照，CPU测试通过，未新增Kit验收 |',
        '| 资源 | collection_resources / training_resources / long_resources / study_resources | 只管理自有进程、锁、墙钟等待；不改模拟dt或他人任务 |', '',
        '## 状态和边界', '',
        '`train.py`/早期 probes 是保留的历史接口；当前正式候选从 `train_long.py` 进入。`models/correction.py` 的 full-candidate shadow 与 `models/residual.py` 的已训练 additive delta 不可混称。',
        '`private_history.py` 保存隔离历史的历史实现/检查；当前 Transformer 使用已授权 actual mixed feedback。禁止把旧版本权重切换新输入语义。',
        '`models/vendor` 是固定 MIT DP源码；DINO来自固定torch hub revision。原生任务只多 action_utils 的只读trace hook；环境wrapper是本地扩展，world_hold是显式兼容模式。', '',
        '## 逐文件索引', '']
    for row in rows:
        name=row['path']
        definitions=', '.join(row['definitions']) or '无函数定义/配置值'
        related=', '.join(Path(path).name for path in row['documents']) or '本系统审计/文件地图'
        sections.extend([f'### `{name}`', '', row['purpose'], '',
            f'- 定义（函数/类:行）：{definitions}。',
            f'- 来源：{row["upstream"]}；依赖：'+(', '.join(row['imports']) or '无静态导入')+'。',
            '- 静态调用者：'+(', '.join(row['imported_by']) or 'CLI/动态调用或无静态调用；需结合主链')+'。',
            '- I/O或状态操作候选：'+(', '.join(row['io_calls']) or '未检出写入/step命名调用；以源码为准')+'。',
            f'- 文档：{related}；测试：'+(', '.join(row['tests']) or '未检出直接import测试；不表示已验证')+'。', ''])
    Path('research_docs/PROJECT_FILE_MAP.md').write_text('\n'.join(sections))
    return rows


def dataset_audit(output):
    manifest = json.loads(MANIFEST.read_text())
    source = Path('outputs/visuotactile/run/dp_loss_trt_20261031_01')
    splits = json.loads((source/'splits.json').read_text())
    flattened = sum(splits.values(),[])
    assert len(flattened) == len(set(flattened)) == 400
    assert set(flattened) == {row['id'] for row in manifest['episodes']}
    configurations, identities, rows = {}, {}, []
    maxima = Counter()
    delay_counts = Counter()
    errors = []
    for entry in manifest['episodes']:
        path = MANIFEST.parent/entry['directory']/'data.h5'
        with h5py.File(path,'r') as stream:
            meta = json.loads(stream.attrs['metadata_json'])
            group = stream['episodes'][entry['episode']]
            length = entry['length']
            def read(name):
                return torch.from_numpy(np.asarray(group[name][:length]))
            command = read('transition/command_clipped_pre_delay')
            submitted = read('transition/submitted_action')
            delayed = read('transition/control_trace/delayed_command')
            before = read('transition/target_before_rad')
            after = read('transition/target_after_rad')
            limits = torch.tensor(meta['joint_limits_rad'][0])
            scale,dt = meta['config']['action']['dof_speed_scale'],meta['control_dt']
            reference = command_to_reference(command,before,limits,speed_scale=scale,control_dt=dt)
            recovered = reference_to_command(reference,before,limits,speed_scale=scale,control_dt=dt)
            raw = command_to_reference(delayed,before,limits,speed_scale=scale,control_dt=dt)
            raw[:,:7] = raw[:,:7].clamp(limits[:7,0],limits[:7,1])
            mix = torch.tensor([meta['config']['action']['arm_moving_average']]*7+[meta['config']['action']['hand_moving_average']]*22)
            expected = (mix*raw+(1-mix)*before).clamp(limits[:,0],limits[:,1])
            source_steps = read('transition/control_trace/selected_command_step').long()
            queue_length = read('transition/control_trace/queue_length')
            delay = read('transition/control_trace/delay_index')
            delay_counts.update(delay.tolist())
            actions = shift_actions(command,delayed,read('transition/time_before_s'),
                read('transition/control_trace/selected_command_time_s'),read('obs/time_s'))
            values = dict(clipped=float((command-submitted.clamp(-1,1)).abs().max()),
                target_input=float((before-read('obs/actuator_target_rad')).abs().max()),
                roundtrip=float((recovered-command).abs().max()),native_formula=float((expected-after).abs().max()),
                selected_command=float((delayed-command[source_steps]).abs().max()),
                obs_action_time=float((read('obs/time_s')-read('transition/time_before_s')).abs().max()),
                previous_action=float((actions[1:,:29]-command[:-1]).abs().max()),
                target_continuity=float((before[1:]-after[:-1]).abs().max()),
                q_continuity=float((read('transition/diagnostics_before/joint_position_rad')[1:]-read('transition/diagnostics_after/joint_position_rad')[:-1]).abs().max()))
            for key,value in values.items():
                maxima[key] = max(maxima[key],value)
            succeeded = bool(read('transition/diagnostics_after/retract_succeeded')[-1])
            completed_goals = int(read('transition/diagnostics_after/successes')[-1])
            maximum_goals = int(read('transition/diagnostics_after/max_goals')[-1])
            valid = succeeded and completed_goals >= maximum_goals and bool(read('transition/control_trace/valid').all())
            valid &= bool((source_steps>=0).all() and (source_steps<=torch.arange(length)).all())
            valid &= all(value <= (1e-5 if key=='roundtrip' else 1e-6) for key,value in values.items())
            if not valid:
                errors.append(entry['id'])
            configuration = {key:meta['config'][key] for key in ('action','assets','domain_randomization','reset','termination','precise_assembly','sim','scene')}
            configuration.update(physics_dt=meta['physics_dt'],control_dt=dt,wrench_frame_mode=meta['wrench_frame_mode'],
                joint_order=meta['joint_order'],joint_order_lab=meta['joint_order_lab'],joint_limits_rad=meta['joint_limits_rad'][0],
                cameras=meta['cameras'],camera_rig=meta['camera_rig'],contact_bodies=meta['contact_bodies'])
            configuration = json.loads(json.dumps(configuration))
            configuration['sim'].pop('device',None)
            configuration['sim'].pop('render',None)
            configuration['camera_rig'].get('wrist',{}).pop('parent_prim',None)
            for camera in configuration['cameras']:
                camera.pop('pos_env',None)
                camera.pop('quat_ros_wxyz',None)
            configuration = sanitized(configuration)
            key = digest(json.dumps(configuration,sort_keys=True).encode())
            configurations.setdefault(key,dict(count=0,value=configuration))['count'] += 1
            code = meta.get('code_identity',{})
            code = dict(commit=code.get('commit'),branch=code.get('branch'),
                working_tree_dirty=bool(code.get('working_tree')),
                source_sha256={name:value for name,value in code.get('source_sha256',{}).items()
                    if name.startswith(('visuotactile/','isaacsimenvs/','evaluation/eval_isaacsim','rl_games/'))})
            code_key = digest(json.dumps(code,sort_keys=True).encode())
            identities.setdefault(code_key,dict(count=0,value=code))['count'] += 1
            rows.append(dict(identity=entry['id'],length=length,success=valid,queue_lengths=sorted(queue_length.unique().tolist()),
                source_env_index=meta.get('source_env_index',0),num_envs=meta['config']['scene']['num_envs'],
                config_identity=key,code_identity=code_key,**values))
        if len(rows)%100 == 0:
            print(f'Audited {len(rows)}/400 records',flush=True)
    report = dict(manifest_sha256=digest(MANIFEST.read_bytes()),split_sha256=digest((source/'splits.json').read_bytes()),
        counts={key:len(value) for key,value in splits.items()},episodes=len(rows),steps=sum(row['length'] for row in rows),
        errors=errors,maximum_errors=dict(maxima),delay_index_counts=dict(delay_counts),
        configurations=configurations,code_identities=identities,rows=rows,
        scope='All400 numeric control fields re-read; full70GB HDF5 hashes reused from protected_artifacts/previous verification, not rehashed here')
    atomic_json(output/'control_data_audit.json',report)
    return report


def rng_audit(output):
    source = Path('outputs/visuotactile/run/network_audit_20261031_01')
    session = json.loads((source/'session.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    entries = {row['id']:row for row in manifest['episodes']}
    rows = []
    for identity in session['selected_train']:
        entry = entries[identity]
        with h5py.File(MANIFEST.parent/entry['directory']/'data.h5','r') as stream:
            meta=json.loads(stream.attrs['metadata_json'])
            group=stream['episodes'][entry['episode']]
            lane=meta['source_env_index']
            count=meta['config']['scene']['num_envs']
            rng=group['transition/torch_rng_before_step'][:]
            delay=group['transition/control_trace/delay_index'][:]
            replayed=[]
            singleton=[]
            for state in rng:
                generator=torch.Generator().set_state(torch.from_numpy(state))
                replayed.append(int(torch.randint(0,3,(count,),generator=generator)[lane]))
                generator.set_state(torch.from_numpy(state))
                singleton.append(int(torch.randint(0,3,(1,),generator=generator)[0]))
            rows.append(dict(identity=identity,lane=lane,num_envs=count,steps=len(delay),
                recorded_rng_native_draw_matches=int((np.asarray(replayed)==delay).sum()),
                singleton_same_rng_matches=int((np.asarray(singleton)==delay).sum()),
                first12_source_delay=delay[:12].tolist(),first12_native_reconstructed=replayed[:12],
                first12_singleton_same_rng=singleton[:12],
                scope='Independent per-step CPU RNG reconstruction, not physical replay or replacement delay'))
    atomic_json(output/'delay_rng_audit.json',rows)
    return rows


def historical_sources(output):
    base=Path('outputs/visuotactile/run')
    selections = ['dp_command_20261029/evaluation/plan.json','dp_command_20261029/b0/report.json',
        'dp_command_20261029/b1/report.json','long_horizon_20261030_v2/G00-new_full/config.json',
        'long_horizon_20261030_v2/G10-new_full/config.json','long_horizon_20261030_v2/closed_loop/plan.json',
        'residual_20261030_01/training_full/config.json','residual_20261030_01/closed_loop/environment.json',
        'residual_20261030_01/closed_loop_remaining_01/environment.json',
        'dp_loss_trt_20261031_01/G00-A/config.json','dp_loss_trt_20261031_01/G00-B/config.json',
        'dp_loss_trt_20261031_01/G10-A/config.json','dp_loss_trt_20261031_01/G10-B/config.json',
        'dp_loss_trt_20261031_01/residual_closed_loop/plan.json',
        'network_audit_20261031_01/session.json']
    selections.extend(str(path.relative_to(base)) for path in base.glob('dp_loss_trt_20261031_01/*/environment.json'))
    selections.extend(str(path.relative_to(base)) for path in base.glob('dp_loss_trt_20261031_01/transformer_training/*/config.json') if 'pilot' not in str(path))
    selections.extend(str(path.relative_to(base)) for path in base.glob('dp_loss_trt_20261031_01/dp_closed_loop/plan.json'))
    selections.extend(str(path.relative_to(base)) for path in base.glob('dp_command_20261029/evaluation/environment.json'))
    rows=[]
    for name in sorted(set(selections)):
        path=base/name
        if not path.exists():
            rows.append(dict(path=str(path),missing=True))
            continue
        data=json.loads(path.read_text())
        keys=('source_commit','source_sha256','source_hashes','source','source_epoch','source_updates','checkpoints',
              'checkpoint_hashes','checkpoint_sha256','config','model','version','epoch','best_epoch','manifest_sha256',
              'loss','physics_dt','control_dt','num_envs','env_spacing_m','native_stage_timeout_steps')
        item={key:data[key] for key in keys if key in data}
        if 'config' in item:
            item['config']={key:value for key,value in item['config'].items()
                if key in ('action','assets','domain_randomization','reset','termination','precise_assembly','scene','decimation')}
        rows.append(dict(path=str(path),file_sha256=digest(path.read_bytes()),evidence=sanitized(item)))
    atomic_json(output/'historical_sources.json',rows)


def evaluation_audit(output):
    base=Path('outputs/visuotactile/run')
    roots=['dp_command_20261029/evaluation','long_horizon_20261030_v2/closed_loop',
           'residual_20261030_01/closed_loop','residual_20261030_01/closed_loop_remaining_01',
           'dp_loss_trt_20261031_01/dp_closed_loop','dp_loss_trt_20261031_01/residual_closed_loop',
           'network_audit_20261031_01']
    report=[]
    for name in roots:
        root=base/name
        metadata_path=root/'environment.json'
        meta=json.loads(metadata_path.read_text()) if metadata_path.exists() else None
        traces=[]
        for path in sorted(root.glob('*.npz')):
            if any(token in path.stem for token in ('initial','switch')):
                continue
            with np.load(path) as record:
                command_key=next((key for key in ('clipped','command_clipped_pre_delay','command') if key in record),None)
                delayed_key=next((key for key in ('delayed','delayed_command') if key in record),None)
                if command_key is None or delayed_key is None:
                    continue
                command,delayed=record[command_key],record[delayed_key]
                indices=np.arange(len(command))[:,None]-np.arange(3)[None]
                candidates=command[indices.clip(0)]
                error=np.max(abs(candidates-delayed[:,None]),axis=-1).min(-1)
                traces.append(dict(path=path.name,steps=len(command),queue_candidate_max_error=float(error.max()),
                    selected_delay_values=np.unique(record['selected_delay']).tolist() if 'selected_delay' in record else None))
        report.append(dict(root=str(root),environment_recorded=meta is not None,
            action=meta['config']['action'] if meta else None,
            delay={key:meta['config']['domain_randomization'][key] for key in ('use_action_delay','action_delay_max')} if meta else None,
            num_envs=meta['config']['scene']['num_envs'] if meta else None,
            wrench_frame_mode=meta['wrench_frame_mode'] if meta else None,traces=traces,
            scope='Delayed command must equal a queue candidate; equal/saturated commands do not identify the sampled index'))
    atomic_json(output/'evaluation_control_audit.json',report)


def parameter_contracts(output):
    saved=torch.load('pretrained_assembly/tight_insertion/model.pth',map_location='cpu',weights_only=False)[0]
    with (output/'expert_tensors.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['name','shape','elements','scope'])
        for name,value in saved['model'].items():
            writer.writerow([name,str(list(value.shape)),value.numel(),
                'network' if name.startswith('a2c_network.') else 'normalization buffer'])
    data=json.loads((output/'control_data_audit.json').read_text())
    configuration=next(iter(data['configurations'].values()))['value']
    constants={}
    for node in ast.parse(Path('isaacsimenvs/tasks/play/utils/scene_utils.py').read_text()).body:
        if isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try:
                constants[node.target.id]=ast.literal_eval(node.value)
            except (ValueError,TypeError):
                continue
    with (output/'joint_contract.csv').open('w',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(['canonical_index','joint','lab_index','lower_rad','upper_rad','stiffness',
                         'damping','armature','friction','reset_center_rad'])
        for index,name in enumerate(configuration['joint_order']):
            prefix='ARM' if index<7 else 'HAND'
            writer.writerow([index,name,configuration['joint_order_lab'].index(name),
                *configuration['joint_limits_rad'][index],constants[prefix+'_JOINT_STIFFNESS'][name],
                constants[prefix+'_JOINT_DAMPING'][name],constants.get(prefix+'_JOINT_ARMATURE',{}).get(name,'URDF/default'),
                constants.get(prefix+'_JOINT_FRICTION',{}).get(name,'URDF/default'),constants['ARM_DEFAULT_JOINT_POS'].get(name,0)])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    atomic_json(args.output/'native_source_audit.json',source_audit())
    inventory(args.output)
    dataset_audit(args.output)
    rng_audit(args.output)
    historical_sources(args.output)
    evaluation_audit(args.output)
    parameter_contracts(args.output)


if __name__=='__main__':
    main()
