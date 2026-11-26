# Final15 补充绑定计划22：保留root monitor中断事实

PLAN22逐项核验并原样继承PLAN21的106项SHA／字节数，新增PLAN21对、root monitor源文件和中断复核对，共111项。六个冻结final15工具保持。只写本计划和新的中断复核材料；未执行actual builder/root review/export，未读取main64结局或模型缓存，未发信号或修改任何运行源码。

具体修正：`ROOT_MONITOR15_EXIT.json`不存在，历史root monitor当前精确调用census为空，但没有自然退出证明。最后指标为2026-11-24 16:56:38(+08)、原wave52/stage35，自己的PID/birth没有被源码捕获；退出时刻、原因、状态均未知。新的`ROOT_MONITOR15_INTERRUPTION_REVIEW21.json/.md`如实绑定源文件、完整本地metrics SHA、identity SHA、当前0匹配/0错误census及这些限制，不生成任何exit文件。

冻结关闭工具的精确`own_command/own_census`只要求当前没有root monitor和其他指定自有actor；没有读取或要求`ROOT_MONITOR15_EXIT.json`。因此PLAN22从**生效的**`post_closure_supplements`删除该不存在路径，保留其原pending只作历史，并新增中断复核对和源码。不要按PLAN21旧叙述要求补造第六份关闭回执。其余五个补充支撑仍为实际activation、root历史identity、watcher stop intent、watcher实际certificate和native monitor exit，须在真实闭合后由root审核；它们不能把历史root monitor中断改称自然关闭。

以下命令替代PLAN21构建片段，仅在主队列和tail helpers真正结束、相关自有actor退出、watcher stop/certify实际完成且root审阅后运行。它从PLAN22的111固定项与五个实际关闭支撑取具体路径，不要求缺失root-monitor exit。保留builder原项，不修改冻结门。若另有cleanup20实际运行，root先审核最终源/测试/登记/全部实际回执并加入`extra`，再第一次写binding；本计划没有批准或启动该helper。

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python - <<'PY'
import hashlib, json, sys
from pathlib import Path
project = Path('/home/CNS2025331827/docs/bc+rl_RSS')
audit = project/'outputs/diagnostics/tight_paired_dagger_20261122_01'
root = project/'outputs/visuotactile/run/tight_paired_dagger_20261122_01'
sys.path.insert(0, str(audit))
import final_recovery15_common as common
output = audit/common.BINDING_NAME
assert not output.exists() and not output.is_symlink()
assert not (audit/common.REVIEW_NAME).exists()
plan = json.loads((audit/'FINAL15_SUPPLEMENT_PLAN22.json').read_bytes())
binding = common.build_binding(root, audit)
assert binding.get('version') == 'paired_final_recovery15_bindings_v1', binding
for filename, identity in plan['fixed_existing'].items():
    path = Path(filename)
    assert not path.is_symlink() and path.is_file()
    data = path.read_bytes()
    assert len(data) == identity['bytes']
    assert hashlib.sha256(data).hexdigest() == identity['sha256']
    previous = binding['small_file_sha256'].setdefault(filename, identity['sha256'])
    assert previous == identity['sha256']
extra = [Path(row['path']) for row in plan['post_closure_supplements']]
extra += [Path(name) for name in plan['self_plan_binding']['paths']]
# Add only root-reviewed FINALIZED cleanup20 material here if it subsequently exists/runs.
for path in extra:
    assert path.resolve().is_relative_to(project) and not path.is_symlink()
    assert path.is_file() and path.suffix in ('.json', '.py', '.md')
    data = path.read_bytes()
    assert len(data) <= 16*1024*1024
    digest = hashlib.sha256(data).hexdigest()
    previous = binding['small_file_sha256'].setdefault(str(path.resolve()), digest)
    assert previous == digest
with output.open('x') as stream:
    json.dump(binding, stream, indent=2)
    stream.write('\n')
print(json.dumps({'path': str(output), 'sha256': common.sha(output),
                  'small_file_count': len(binding['small_file_sha256'])}, indent=2))
PY
```

root实际检查首次写出的binding与六工具SHA后，再排他创建`ROOT_EVIDENCE_TOOLS_REVIEW15.json`，精确字段保持：`status='cpu_review_passed'`、`binding_sha256=common.sha(audit/common.BINDING_NAME)`、`source_sha256={name:common.sha(audit/name) for name in common.FINAL_TOOLS}`；六工具键是basename。中断复核不是root最终binding批准，也不替代稍后的真实全actor census。

后续8载荷extractor→compute→复用8表assembler→root科学审阅→dry inventory→apply仍用`FINAL15_EXECUTION_COMMANDS17.md`；没有重跑学生机会、改变训练年龄或成功判据。cleanup58仍明确为异常后自有进程SIGTERM/SIGKILL清理、软件未完成和机会已消耗，不能与缺失root-monitor退出记录混写。

`.jsonl`原metrics留本机，由中断审阅JSON固定其SHA，不塞入small-file映射。small-file仅原允许后缀，曲线CSV/PNG走原allowlist；local-retained logs集合不扩，裸STOP只由JSON证书绑定。16MiB单文件/256MiB总量、新文件不覆盖及原截止保持。PLAN22自己两份文件完成后可从外部登记SHA，不嵌入循环自SHA。
