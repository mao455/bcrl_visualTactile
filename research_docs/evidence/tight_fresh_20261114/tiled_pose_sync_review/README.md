# GPU Tiled 固定相机同步候选：位姿修复，三视角仍未通过

只读复核 session06 `gpu_tiled_pose_sync_probe01`，冻结源码 `82d505241d2910b96c9b7f562dfe41bd2e111dad`，并与 session05 `scaling_gpu_tiled_n8_raw_diagnostic_03` 比较。新增的实例级 Fabric→USD 同步修复了两个固定相机的实际 CameraParams 位姿；第二视角黑图与腕部异常内容仍然存在。**Tiled 继续不采用；本次没有完整学生动作、训练数据或自主成功样本。**

候选为 GPU 物理、8 lanes、视觉微批量 M8、Kit 16/物理 2 线程、Tiled 三视角，descriptor sets/reserved descriptors 为 360000/900000。物理 120Hz、控制 60Hz 未变。owner 21.0098 秒后退出 1，启动完成耗时 13.3094 秒。8 个 numeric HDF5 的 `episode` 组全部为空；lane6 初态被拒绝，其余 7 lane 也均在首个控制动作前因三视角不完整有效而停止。这里不能计算自主任务成功率或有效控制吞吐。

`review.json` 保存来源 SHA256、逐 lane 小型数值统计与原始失败/渲染诊断的路径；没有图像、轨迹或权重。CPU 审计复核了 206 个冻结源码文件、3 个实际安装源码文件、原失败 packet/report 的哈希以及 4 次 render 前后的已暴露状态指纹。旧实验文件保持原样。

| 指标 | 旧 GPU Tiled raw03 | 本次同步候选 |
|---|---:|---:|
| 固定视角0：位置误差 | 约 0.807055 m | float32 配置位置逐值一致 |
| 固定视角1：位置误差 | 约 1.105667 m | float32 配置位置逐值一致 |
| 固定视角0：最大朝向误差 | 2.603539 rad | 2.276e-7 rad |
| 固定视角1：最大朝向误差 | 2.426649 rad | 9.691e-8 rad |
| 腕部：相对当前 parent 的最大位置误差 | 4.932e-7 m | 4.932e-7 m |
| 腕部：相对当前 parent 的最大朝向误差 | 1.529e-7 rad | 1.529e-7 rad |

两个实验保存的全部初始 observable、腕部 parent 位姿逐值相同，不能据此声称完整 PhysX/渲染器内部状态相同。固定相机目标位置按已保存的环境网格原点、冻结 `wrist_bracket_close` rig 及实际 float32 `eye + origin` 构造；朝向按 Z-up look-at 计算。网格 XY 从桌面中心读取，并与机器人底座偏移核对，Z=0。此次两个固定相机实例 `_sync_usd_on_fabric_write` 从 false 变 true，腕部保持 false；共享安装没有修改。旧的约米级错位已经消失，且首包与随后 4 次 CameraParams 读回都一致。源码身份还包含其他诊断代码变化，这不是仅一个变量的性能比较。

首包原始图像输出的结论如下，每个 lane 为 448×252 像素：

- 视角0：8 lane 均有空间变化的 RGB，metric depth 全部有限且有效；首包最大 RGB 空间标准差约 0.268–0.270。
- 视角1：8 lane 的 RGB 都严格为 0，alpha=255；深度全部有限且有效。失败是 RGB 内容缺失，不能由深度有效推断视角可用。
- 腕部视角：lane3/4 分别只有 342/118 个灰色像素，其余 6 lane 的 RGB 全黑；两个非黑 lane 通过原有内容阈值，不等于腕视角已正确成像。原生深度仅对应 342/118 个像素非有限，其余有效。

实际为 3 个独立 3×3 atlas，各 1344×756、8 个有效 tile 加 1 个空槽；另有 24 个 1×1 CameraParams 产品。全部 lane/view 的相机 prim、image RP 与标定 RP 绑定正确，产品总数 27。请求启动参数 `/rtx/viewTile/limit=8`，现场实际读回为 **1000000**；因此不能把请求值 8 当作实际 tile 上限。renderer descriptor 两项读回均匹配。CameraParams 来自正确绑定的独立 1×1 产品，尚不能证明图像 atlas 使用了完全同一渲染曝光或相机矩阵。

首包的 24 张 raw RGBA 与 cached RGBA、24 张 raw depth 与 cached depth，CPU 逐 tile 比较均严格相等。黑图在 Lab 缓存拆片之前的 `LdrColorSDbuffPtr` 输出已经出现，本次没有拆片错误证据。

随后 **4 次只调用 `sim.render()`** 的有界诊断没有 step、reset、controller、camera.update 或替换研究 packet。每次暴露的物体/关节/时间/目标/延迟队列/原 packet/相机计数及已记录 RNG 指纹均不变；原 packet/report 文件哈希也不变。它检查不了未暴露的 PhysX、渲染器或 RNG 内部状态。render 序号不是新的 60Hz 控制帧，追加控制帧和训练样本均为 0。

这 4 次读取中，视角0 raw RGB 相对旧 cache 都出现变化，第一次每 lane 有约 13.4–21.9 万个 RGBA 元素不同；这是额外渲染后未调用 camera.update 所预期的旧缓存差异，不是拆片失效。视角1 raw RGB 始终全黑且与原 cache 相等，腕部仍保留同样的 342/118 灰像素模式且逐值等于原 cache。三个视角的 depth 一直与原 cache 相等。故这次有限追加渲染没有恢复完整三视角，但不能外推任意等待或不同渲染配置的结果。

下一项若继续研究 Tiled，应具体区分多个 atlas 的原生颜色渲染和 annotator 输出问题，并预先限定诊断预算；当前证据不足以指定唯一根因或直接批准单 atlas 重构。正式采集仍使用已经通过的 standard 路径，不放宽相机有效性阈值。

重现本报告只需要 CPU，不导入 Kit 或 torch，不读取 NPZ 图像数组（哈希按文件流式计算）。输出文件必须不存在，避免覆盖原证据：

```bash
/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python \
  research_docs/evidence/tight_fresh_20261114/tiled_pose_sync_review/review.py \
  --candidate /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_06/gpu_tiled_pose_sync_probe01 \
  --prior /home/CNS2025331827/docs/bc+rl_RSS/outputs/visuotactile/run/tight_fresh_20261114_05/scaling_gpu_tiled_n8_raw_diagnostic_03 \
  --output /tmp/tiled_pose_sync_review_reproduced.json
```
