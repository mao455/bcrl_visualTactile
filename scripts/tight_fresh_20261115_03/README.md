# 第十二轮结论与预取接续

第十二轮64机会、38接管、36成功纠偏（其中持物相关21条/4286合法专家步）已完整训练，池251条/127训练family。16开发机会完成：A0/7、B0/8插入，原生lift1/7和3/8；持续持物相对实际来源A11/B09为5→4、5→3，均无原生near/对准/子目标。新增标签参加训练不等于自主改善，12不采用为下一采集来源。

新独立session03沿用A11/B09完整来源及251数据。单轮13、32个新训练family（29000064–29000095）、64机会：held0=32、held_root60=24、fixed80=8。保持根距离60mm的真实持物入口与原任务判据。前端新增成功后缀latest_repeat从2改4；仍全部专家合法后缀，学生前缀只作历史，不能解释为只加权接手最初12步。20遍B2的真实更新量由aggregate后实际人口登记，增加曝光而不静默减少Adam/EMA更新；后端15遍、同8个开发初态/角色。旧6验证family与test22隔离。

训练主要瓶颈实测为冻结特征读取：12的A/B分别占更新耗时52.4%/61.1%，各读约2.94TB。新frozen_visual_eager_prefetch_v2只提前读取下一B2的原CPU存储，最多3GiB、保留12GiB可用RAM，不改dtype/模型输入/随机抽样/参数更新次数，不缓存可学习历史。最大两个真实train文件、两批合4visit的统一CPU检查2.568秒，tensor字节/layout与原读取完全相同，torch RNG不变、低内存回退及提前退出通过；0GPU/0参数更新。尚无正式整周期加速结论。

旧派生文件已按精确名单退休：550个旧项目冻结视觉文件及20个已结束非best last，释放38.23GiB选定磁盘块；保留原始数据、best、05回退和最新12全部模型。删除名单与日志留在session02/completed_review，retire.py只接受固定SHA；不要再次执行。

启动方式与02一致，从本包已提交的干净冻结树运行：

```bash
.venv_isaacsim/bin/python scripts/tight_fresh_20261115_03/launch.py \
  --worker-tree /absolute/frozen/tree --expected-commit FULL_COMMIT_SHA --launch
```

实际已用冻结提交19736be启动：2026-11-15 16:23:23.594建立session，固定22:23:23.594截止，旧02的18:24:06.380截止及字节保持。16:24:53已观察wave0/control tick25；当前只是实际采集推进，新成功数与预取完整训练/自主结果待结束。单轮完成即停，错误/阶段结果持久保存，Restart=no；后续读取session03_launch/status、research_loop、new_family_13。

report_completed12.py汇总已保存的completed_review与原公开来源报告；report_cycle11.py/report_frontend11.py/report_collection11.py新增--round 12，复用现有算法。只读已完成结果，不重采、重训或改变旧指标。
