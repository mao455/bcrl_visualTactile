# Final15 补充绑定计划21

本计划逐项核验并原样继承 PLAN19 的91项小文件 SHA256／字节数，新增15项，共106项；六个 frozen final15 工具 SHA保持，未修改。新增为 PLAN19 两份、aid19最终关闭审阅对、实际全局exit与wave48 intent/exit、wave47只读诊断对、wave58只读诊断对，以及cleanup58登记、实际执行和独立审阅对。JSON记录106个具体规范路径，`missing_expected=[]`。未打开main64结局、模型或缓存载荷，未执行builder、root review、export或任何进程操作。

aid19仅有wave46/47/48三个持久intent，均有exit；全局退出码1及错误保留。46和47分别扩展/恢复134/134、133/133次，约0.44秒，48零写。精确helper/worker已不存在、unit cgroup已消失，不存在待恢复的活跃扩展证据。wave47历史逐TID恢复来自冻结协议和持久摘要，没有独立同期逐TID快照；不能称零争用、证明提速或隐藏物理状态等价。独立关闭结论只涉及aid19，不代替主15队列与watcher关闭。

wave58在已经完整发出`RAM reserve`异常后，原native cleanup长期未退出。root预登记精确自有进程组身份后发送SIGTERM，15秒后再次核验再SIGKILL，原owner实际reap返回-9并推进原wave59。独立审阅13项通过。必须写作“异常后的自有进程清理、软件未完成、机会已消耗”，不得写成自然退出、startup timeout、正常完成或重新执行。信号事实依据root实际回执，独立review没有追踪系统调用。原截止、295科学源码、原后续5波plan/session和总64×4分母保持。

`final_error_cleanup20.py`在本计划快照时仍是未完成CPU fixtures/自审的inactive源码，因此**未绑定**。若其后定稿或启动，root应另审最终源码、测试、登记及全部实际intent/signal/exit，再在第一次写actual binding前补充固定SHA；不要把本计划的空SHA解释为已经通过或已经运行。

原PLAN19 pending清单作为历史原样保留；当前明确pending为主15 activation、root monitor identity/exit、watcher stop intent/certificate及native monitor exit六条具体路径。这些存在与否均不等同于真实闭合批准，root须等待真实队列和两个tail helper结束、回收所有自有相关actor并完成watcher stop/certify后，再审核实际材料。PLAN21自己两份文件可在字节固定后从外部加入binding，不嵌入自身SHA。

冻结common没有构建binding的CLI，只有`build_binding(root,audit)`返回字典。下面是**供root在真实闭合与实际审阅后使用、此处未执行**的最小构建命令；它保留builder全部条目，验证PLAN21的固定106项，并要求六个实际关闭支撑文件全部存在。若cleanup20后来定稿/运行，先把root已审实际文件加入`extra`再运行；不要创建第二份覆盖binding来补漏。

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
plan = json.loads((audit/'FINAL15_SUPPLEMENT_PLAN21.json').read_bytes())
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

root随后须实际检查刚写binding及六工具SHA，再独立用新文件创建`ROOT_EVIDENCE_TOOLS_REVIEW15.json`。冻结门要求如下精确字段；`source_sha256`键是六个**basename**，不是绝对路径或某个嵌套工具对象。不要先生成通过审阅再回头假装已经看过binding。

```python
review = {
    'status': 'cpu_review_passed',
    'binding_sha256': common.sha(audit/common.BINDING_NAME),
    'source_sha256': {name: common.sha(audit/name) for name in common.FINAL_TOOLS},
}
# Only after root actually inspected the concrete binding and exact sources:
with (audit/common.REVIEW_NAME).open('x') as stream:
    json.dump(review, stream, indent=2)
    stream.write('\n')
```

后续精确CLI沿用`FINAL15_EXECUTION_COMMANDS17.md`：一次详细8载荷extractor→compute→复用8表的assembler→root科学审阅→dry inventory→apply。退出码不足以证明export成功，分别要求`dry_inventory_ready`、`export_complete`。不得改冻结6工具、36继承依赖、机会分母或成功判据。

`small_file_sha256`允许`.json/.py/.md/.txt`，但冻结exporter不导出`.txt`；本计划未新增txt。曲线CSV/PNG走原allowlist，不加入small-file映射。`local_retained_log_sha256`严格保持三个ROOT边界`.log`并集加mandatory failed13 helper；aid19/cleanup20普通日志继续留本机，由JSON记录SHA，不额外扩集合。裸STOP只由已绑定intent/certificate记身份与SHA。单文件16MiB、总256MiB、规范普通路径和新输出不覆盖规则保持。
