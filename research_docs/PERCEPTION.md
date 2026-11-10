# 当前视觉、触觉与执行历史

G11 前端使用三路 RGB-D、robot178 和直接触觉，QG01 的 QG00 前端使用三路 RGB-D 与 robot178；两组 Transformer 后端均读取观测触觉。前端保留现有 U-Net，DINOv2-S/14 backbone 冻结，既有视觉投影、深度编码、持续历史和策略参数参与本轮前端微调。旧 StateDP 的 Actor141/geometry85 不进入这两组学生输入。

相机顺序固定为 `fixed_high/fixed_side/wrist_rgbd`，每路448×252。冻结 DINO 特征和确定性深度网格按12控制tick采样；共享缓存须核验源轨迹、backbone、内容 SHA、相机顺序、分辨率、stride 和无随机增强。模型更新后不能复用旧可训练投影、hidden、condition 或名义计划。

robot178 包含当前实测 q/qd、动作前执行器目标及误差、严格过去 issued/applied 和年龄/有效位。接管前后都保留真实执行历史，不用新 DP 的重预测覆盖。不同的 `previous_nominal` 表示上一控制 tick 的名义输出，不代表过去已经执行的命令。

触觉每个60Hz观察有两个120Hz子步，来源为已完成的上一 transition；首 tick 无效，禁止未来传感样本。world/local 力、编码器 FK、有效位和时间都保留。接管成功后的未来 q、物体姿态、接触阶段只可作标签或独立 loss context，不能进入学生编码器。

采集准入检查真实像素/depth/有效位、frame counter 和时钟；`camera_valid=true` 不能替代内容检查。每条 episode 保存完整历史和监督 mask，不能复制帧、补假图或借 reset 帧补 terminal 图。原400和仍用 Core 的源数据保持原样；旧相机失败数据不因新 loader 而变为合格数据。

CPU 已验证数据路由、mask、H24 边界与 DINO 复用；实际新接管的图像/触觉内容和物理接口仍须由新阶段真实采集验收。参见 [合同](CONTRACTS.md) 和 [实际结果](STATE_DP_RESULTS.md)。
