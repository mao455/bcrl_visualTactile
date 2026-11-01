# Native-command 三组控制合同 v1

新身份为 S-NativeDP、I-OfficialDP、I-HistoryDP；旧 G00/G10 绝对参考权重不兼容。
`visuotactile/native_data.py:CONTRACT` 是机器可读合同。当前验收尚未结束，不声称已训练或成功。

监督直接取 `transition/command_clipped_pre_delay[t:t+64]`，尾部补零并 mask。
第一个输出就是 t 的命令，不额外应用官方 observation-horizon 索引偏移。
预测经 train320 limits-affine 逆变换，执行时裁剪到 [-1,1]，顺序提交前12步。
原生 env.step 继续随机延迟、arm7增量积分、hand22绝对映射、平滑及限位。
不调用 reference_to_command，不以 delayed 命令作标签，不添加第二层延迟。

120Hz物理、60Hz控制保持原值。策略在 tick 0/12/24 消费新图像，相机采集配置仍为 update_period=0。
规划周期200ms是模拟时间合同；真实推理超时必须单列，不能改dt隐藏计算耗时。

共同机器人行178维：q29、qd29、动作前真实target29、target−q29、
t−1 issued29、t−1 delayed29、issued年龄1、selected-command年龄1、历史有效位1、episode相对时间1。
13行窗口另加行年龄和行有效位，形状13×180；t0过去命令和年龄为0且无效。
训练历史明确是专家已执行命令的teacher forcing；部署来自学生真实执行ack。
没有当前/未来命令、未来delay index、未来target、奖励或成功阶段输入。

S组另行允许当前及过去两次5Hz的85维模拟几何，统一robot-base坐标。
这是有特权的诊断基线，不是可直接部署的视觉模型。I组不接受这些字段。
旧组的信息权限不因新组扩大而被追溯修改。

checkpoint保存完整合同、模型组、训练统计、划分、数据/源码hash、EMA及恢复状态。
加载器逐项比较合同；不能通过切换配置把旧reference权重解释为native-command模型。

## 2026-11-01 DINO缓存数值修复

冻结DINO的历史缓存实际使用FP32 transformer、允许TF32的卷积、FP16 patch存储，
不是完整严格FP32前端。原缓存两次观测合为6张独立图像，奇数末帧则只有3张；
本机两种batch shape选择不同卷积计算，旧缓存与新单帧严格FP32路径的最大差异达0.4375。

Native专用`NativeDinoSpatial`固定6张图像：仅重复当前三视角，无未来图像、无跨图注意力，
仍用原revision/weights和18×32 patch。合法uint8/255相机输入先无损恢复像素字节。
精度开关限定在冻结DINO调用内并恢复；机器人编码、GRU、condition及U-Net仍为FP32。
原完整双帧缓存复用；奇数尾帧从原图重算为单独overlay，原400数据、旧缓存和旧reference标签不变。
新增`native_perception` Gate检查400份旧cache hash和首帧复现；overlay与索引均进入artifact hash。
History checkpoint记录该索引hash，部署须一致；完整episode Gate使用原图重算线上patch，
与训练缓存比较memory/condition≤1e−5、动作≤1e−4，缓存复现原0.005阈值保持不变。
