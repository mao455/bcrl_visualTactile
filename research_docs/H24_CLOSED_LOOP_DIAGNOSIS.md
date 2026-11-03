# H24 闭环失败诊断与视频

2026-11-04新任务；实现准备完成，尚未以此文字宣称录像已录制。
本轮无训练、权重/模型/控制/物理修改、超参数搜索或test40选择。
使用feature/native-dp-training-v2，原Sample8000/10000 SHA分别为
00783c7db01b8b269746718a3322238beb9906a88eac1fa8e9adc58bfc5b4399和
9a7cc8c6123563f93b77256104e61e6b357bb38303ede4231fe19fddeb0e17ca。

先录六段独立Kit视频：train r931005_1793209376_e02_p000059/982403的8000、10000、Expert；
validation固定r931005_1793209376_e00_p000055/984101的同三组。
validation按历史两模型均接触但无lift的共同ID排序首个选取，为描述性诊断，不作总体成功率。
历史342tick成功没有原始录像；8000最多两次本轮重现，结果全部保留。
无法重现时只提供明确标注的历史数值可视化，不冒充原始成功录像或物理重放。

每个run使用独立Kit、既有120/60Hz、H24/commit12、EMA、DDIM16/eta0/clipFalse/seed7091。
既有3路448×252图像CPU拼成1280×720/30fps，原控制/相机采样不变。
先视频，再有限Expert prefix。学生自己的GRU每60Hz沿实际专家前缀更新，
严格过去的position历史取真实postprocessed expert target；切换不重置GRU/采样器，不瞬移状态。
先两个train precontact探针；通过前缀一致性且有阶段改善或24tick目标误差下降至少20%，
才扩展到train lift和一个validation初态的precontact/lift；最多8次takeover、总15次Kit启动。
连续两次同类pre-scene FileThread故障停止仿真批次；一个新4小时窗口包含资源等待。

首次显著分歧预设：末端5mm、Arm RMS.01rad、Hand RMS.05rad、物体5mm/5deg，持续3ticks。
固定规则下，已有train8000/10000的724个导出初始字段及环境RNG相同；
原始目标/实际提交输入tick0即不同，实际末端tick16、Arm19、Hand37、物体位置51/姿态52出现持续分歧。
PhysX solver/contact/broadphase隐藏状态未被序列化，不能称完全相同模拟器状态。
固定validation的原专家历史离线目标FK8000/10000为13.184/12.228mm；
自主目标对同一原专家target为41.937/42.009mm，实际对原专家actual为44.173/43.084mm。
这不是全val40接触前17.33mm与共同7例实际约46mm的同口径相减；跟踪、采样/历史、时间与数据子集均分列。

新增四项CPU诊断测试通过：阈值持续性、真实expert target反馈、前缀不符拒绝、记录hook/RNG中性。
复用了旧39项数学和batch审计，没有重跑。
运行目录为`outputs/visuotactile/native_dp_v2/h24_diagnosis_20261104_01`；
持久owner完成后自动生成分析、图表、clean/annotated MP4、ffprobe/全帧解码结果和README/manifest。
现场进度以新root/status.json、results.json及实际文件为准；不能把已发启动命令当作完成。
