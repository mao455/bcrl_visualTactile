# 当前模态与历史边界

StateDP保持Actor141/robot178/geometry85，不读取RGB-D或触觉，不把仿真几何冒称视觉估计。三路RGB-D与五指触觉用于逐条数据准入；ImageDP/触觉模型本轮未训练。

当前显式配置fixed_high、fixed_side、wrist_rgbd三路448×252 RGB-D。验收真实逐帧像素、深度/有效位、frame counter与时钟；camera_valid全True不足以放行。触觉为120Hz每控制tick两子步，首tick无效，其后观测等于上一transition末完成的两子步。

原400源数据只读。旧窄域200条两固定RGB为空白，完整三模态有效数0/200，不补假图；后来75条三视角有效，但都不能仅凭模态有效进入当前1.5–3倍慢速集。本轮21条开发capture的State、三路RGB-D与触觉验收通过；慢速/任务准入是独立门控，仍未过关。

实际规则见[合同](CONTRACTS.md)，逐条结果见[当前结果](STATE_DP_RESULTS.md)。历史图像模型和相机布局文档留在Git642f5a2。
