# 持物后段继续采集：独立时段02

第十一轮已完整结束并审查：A持物2/7→5/7，首次一次进入原生预插入目标15mm范围，但仅累计7/10控制步，仍0原生对准/子目标/插入；B持物5/8→4/8。选A11/B09作为继续产生真实偏差的完整学生，不宣称整体晋升；共用215条训练池，保留A05/B06等已有回退。

新root为tight_fresh_20261115_02，独立6小时从followup实际创建session起算，不改01原14:58:38.966截止。单轮12，32个尚未采集的训练family（29000032–29000063）、A/B共64机会；held0=24、held_root60=24、fixed80=16。held_root60使用已验证入口的0.06m根位置接近条件，仍要求真实持续抬起及双指接触；不是放宽任务成功或让验证初态进入采集。

沿用正式GPU16/M8标准相机、有界异步写入/2验证worker、完整每60Hz三视角及历史；前端20遍B2与随机合法起点、A半数几何、最新数据repeat2、顺序A/B；刷新各自新前端后端输入，15遍后端并行，CPU8同8开发初态/角色完整自主。单轮后停在实际结果边界，下一方法由结果决定。六小时不是必须耗满；保留110GiB启动盘余量和原60GiB运行余量。

从本包已提交、干净的新冻结树运行：

```bash
.venv_isaacsim/bin/python scripts/tight_fresh_20261115_02/launch.py \
  --worker-tree /absolute/frozen/tree --expected-commit FULL_40_CHARACTER_SHA --launch
```

默认不带--launch只读配置。入口复用已验证followup/session、research_loop和collection→cycle owner，单次新来源绑定及统一配置检查；不增加仿真测试或重训原四组。旧worker和monitor已退出才启动，新owner和monitor均Restart=no。源码/命令/阶段/失败/退出在root同级_launch与root/research_loop；不要重复调用已有身份。

实际已用冻结提交1d4028c启动：2026-11-15 12:24:06.380建立session，固定18:24:06.380截止（Asia/Shanghai）；12:27读取到wave0/control tick100。A11/B09实际来源与64机会计划一致。见research_docs/evidence/tight_fresh_20261114/held_followup02/startup.json；当前只确认实际仿真已推进，采集、训练与自主结果仍待结束，后续以_launch/status、research_loop和各阶段回执为准。
