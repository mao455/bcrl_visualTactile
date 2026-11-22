"""Read-only forensic figures and a self-contained evidence report."""
from pathlib import Path
import json,hashlib,datetime,subprocess
import numpy as np,h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CAN=Path('/home/CNS2025331827/docs/bc+rl_RSS');SRC=CAN/'outputs/experiments/tight_adaptive_dagger_20261119/source/bc+rl_RSS';RUN=CAN/'outputs/visuotactile/run/tight_adaptive_dagger_20261119_01';OUT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def link(rel,line=None):return f'[{rel}'+(f':{line}' if line else '')+f']({SRC/rel}'+(f':{line}' if line else '')+')'
def local(name):return f'[{name}]({OUT/name})'
def main():
 training=read(OUT/'training_verification.json');data=read(OUT/'data_chain.json');sources=read(OUT/'source_and_launch.json');final=read(OUT/'final_verification.json');criteria=read(OUT/'native_criteria.json')
 fig,axes=plt.subplots(2,4,figsize=(17,7),constrained_layout=True)
 for item,ax in zip(sorted(training,key=lambda x:(x['round'],x['component']!='DP',x['role'])),axes.flat):
  r,role,component=item['round'],item['role'],item['component'];root=RUN/f'round{r:02d}/learning'/role
  if component=='DP':
   hist=[json.loads(l) for l in (root/'training/validation.jsonl').read_text().splitlines()];values=[x['validation']['selection'] for x in hist]
  else:
   hist=read(root/'backend'/role/'training/validation.json');values=[x['selection_score'] for x in hist]
  epochs=[x['epoch'] for x in hist];ax.plot(epochs,values,lw=1.5);ax.axvspan(50,60,alpha=.1,color='green');ax.scatter([item['epoch']],[item['recorded_validation']],c='red',s=22,zorder=3);ax.set_title(f'R{r} {role} {component}: selected {item["epoch"]}');ax.set_xlabel('Completed epoch');ax.set_ylabel('Fixed validation selection loss');ax.grid(alpha=.2)
  if component=='DP':
   twin=ax.twinx();twin.step(epochs,[x['learning_rates'][0] for x in hist],where='mid',c='orange',alpha=.7);twin.set_ylim(0,1.8e-5);twin.ticklabel_format(axis='y',style='sci',scilimits=(-5,-5));twin.set_ylabel('LR used in epoch',color='orange');ax.axvline(48.5,c='orange',ls=':',alpha=.6)
 fig.savefig(OUT/'validation_and_lr.png',dpi=160);plt.close(fig)
 priority=[('A',28839004),('B',28839005),('B',28839007),('B',28839002)];details=[]
 fig,axes=plt.subplots(4,2,figsize=(14,12),constrained_layout=True)
 for row,(role,seed) in enumerate(priority):
  records=sorted([x for x in final['records'] if x['seed']==seed and x['role'].startswith(role)],key=lambda x:x['role'].endswith('_new'));series={};detail=dict(role=role,seed=seed,models=[])
  for rec in records:
   p=Path(rec['path']);t=np.load(p/'trace.npz')
   with h5py.File(p/'numeric.h5') as f:
    tr=f['episode/transition'];key=tr['native_keypoints_max_dist'][:]*1000;contact=(np.linalg.norm(tr['object_contact_force_w'][:],axis=-1).max(1)>.1).sum(-1);q=tr['diagnostics_after/joint_position_rad'][:];target=tr['target_after_rad'][:]
    z=tr['diagnostics_after/object_pose_w'][:,2];tracking=np.abs(q-target);steps=np.arange(len(key));name=rec['role'];series[name]=(t,key,contact)
    axes[row,0].plot(steps,key,label=name);axes[row,1].plot(steps,contact,label=name)
    x=dict(role=name,frames=len(key),events=rec['events'],command_tick0=t['command'][0].tolist(),nominal_tick0=t['nominal'][0].tolist(),residual_tick0=t['residual_normalized'][0].tolist(),tracking_arm_rms_rad=float(np.sqrt((q[:,:7]-target[:,:7])**2).mean()) if False else float(np.sqrt(((q[:,:7]-target[:,:7])**2).mean())),tracking_hand_rms_rad=float(np.sqrt(((q[:,7:]-target[:,7:])**2).mean())),z_first_below_0_5_m=int(np.flatnonzero(z<.5)[0]) if (z<.5).any() else None)
    detail['models'].append(x)
  a,b=series[role+'_parent'][0],series[role+'_new'][0];n=min(len(a['command']),len(b['command']));detail['output_differences']=dict(command_tick0_max_rad=float(np.abs(a['command'][0]-b['command'][0]).max()),nominal_tick0_max_rad=float(np.abs(a['nominal'][0]-b['nominal'][0]).max()),normalized_residual_tick0_max=float(np.abs(a['residual_normalized'][0]-b['residual_normalized'][0]).max()),command_max_common_prefix_rad=float(np.abs(a['command'][:n]-b['command'][:n]).max()))
  details.append(detail);axes[row,0].set_yscale('log');axes[row,0].axhline(15,c='gray',ls=':');axes[row,0].axhline(7.5,c='gray',ls='--');axes[row,0].set_title(f'{role} {seed}: active-goal keypoint distance');axes[row,0].set_ylabel('mm (goal changes cause jumps)');axes[row,1].set_title(f'{role} {seed}: fingers with object contact >0.1 N');axes[row,1].set_ylim(-.2,5.2)
  for ax in axes[row]:ax.set_xlabel('0-based control transition');ax.legend(fontsize=8);ax.grid(alpha=.2)
 fig.savefig(OUT/'priority_pair_timelines.png',dpi=150);plt.close(fig);(OUT/'priority_case_details.json').write_text(json.dumps(details,indent=2)+'\n')
 selected_table='\n'.join(f'| {x["round"]} | {x["role"]} {x["component"]} | {x["epoch"]}/{x["updates"]} | {x["exposure"]["minimum_visits"]}/{x["exposure"]["minimum"]} | {x["recomputed_validation"]:.12g} | {x["validation_abs_error"]:.3g} |' for x in training)
 pair_table='\n'.join(f'| {x["seed"]} {x["roles"][0][0]} | {x["differences"]["command_first_gt1e_6_rad"]} | {x["differences"]["joint_pre_first_gt1e_4_rad"]} | {x["differences"]["object_pre_first_gt1mm"]} | {x["differences"]["finger_contact_count_first_difference"]} |' for x in final['priority_pairs'])
 hash_table='\n'.join(f'| {Path(p).relative_to(CAN)} | `{x["actual_sha256"]}` |' for p,x in sources['model_identities'].items())
 replay=[]
 for filename in ['online_self_replay_exact_vision.json','online_self_replay_A_exact_vision.json']:
  if (OUT/filename).exists():replay+=read(OUT/filename)
 replay_table='\n'.join(f'| R{x["round"]} {x["identity"]} | {x["frames"]} | {x["maximum_command_error_rad"]:.9g} | {x["maximum_nominal_error_rad"]:.9g} | {x["maximum_robot_error"]:.9g} |' for x in replay)
 gradient=[dict(x['gradient'],role=x['role']) for x in training if 'gradient' in x]+read(OUT/'extra_gradients.json')
 gradient_table='\n'.join(f'| {x["role"]} | {x["identity"]} | {x["loss"]:.9g} | {x["gradient_norm_before_clip"]:.7g} | {x["clip_multiplier"]:.4g} | {x["invalid_gradient_max"]:.4g} | {min(x["per_joint_output_gradient_rms"]):.4g} |' for x in gradient)
 actual_batches=[]
 for p in sorted((RUN/'final_test').glob('paired2_wave*/plan.json')):
  d=read(p);actual_batches.append(dict(path=str(p),lanes=d['batch']['lanes'],vision_microbatch=d['batch']['vision_microbatch']))
 (OUT/'actual_final_batches.json').write_text(json.dumps(actual_batches,indent=2)+'\n')
 text=f'''# Tight-Insertion 两轮 adaptive DAgger 端到端取证审计

审计对象：实验提交 `248aec3`，父 130-50 提交 `9417461`；实际根 `{RUN}`。本次只读既有产物、执行 CPU 重算及 GPU 固定权重推理/反传，**0 optimizer/EMA 更新、0 新物理步、0 新采集、0 选模**。生产源码、原始数据、模型与缓存没有修改；本目录是独立诊断产物。审计判断以每个实际阶段的源文件 SHA、owner 启动命令、plan 与恢复回执为准。

## 结论与优先级

确认两项数值/训练接口偏差，另确认一项报告事实错误；未发现能单独证明三个旧成功丢失的代码错误。

1. **DP 接管入口的部署相位覆盖缺口。** 新 50 条中 48 条首合法 tick 不在 12-tick 规划格点。target-window 支持其前一个原生 anchor 的带 mask 监督，但实际随机采样和覆盖补齐只允许 anchor 本身有专家标签，所以这些入口 anchor 从未训练。绝对 tick ≥50 曝光仍成立；缺失的是“同一标签在实际入口历史/计划槽位下”的覆盖。优先补入口 anchor 与相位覆盖验收，不把随机 anchor 增广整体判为错误。
2. **采集与缓存 DINO 算术路径不一致。** 两轮采集实际 microbatch=1（3 图 kernel），缓存默认 microbatch=2（尾 lane 复制、6 图 kernel）。固定 raw RGB 后能复现其差异及约 1e-4 rad 级在线命令差异；采用实际采集路径后四例完整自重放命令均 <1e-6 rad。终测实际全部 microbatch=2，与缓存一致，故本项不是已经证实的终测 train/eval 路径错配。
3. **报告把在线噪声错误写成7093。** 实际 checkpoint 合同 `sampler_seed=7091` 覆盖默认回退7093。新纠偏拟合数字来自已记录在线输出，因此这个硬编码说明错误不自动使数字失效，但其“不同 seed”解释不可靠；队列复用 generator 的位置与 DINO kernel 才是应记录的实际区别。LR 的正确事实是四次 DP 都在 epoch48 结束后减半，49–60 使用7.5e-6。

最值得先修：①入口相位覆盖；②统一并登记视觉算术路径、自重放设门；③修正自动报告并补齐可精确重放的输入/RNG记录。修复后若要证明恢复旧能力，应另行登记匹配开发对照；本次没有修改或执行这些修复。

## 1. 实际执行身份与入口

{link('tools/tight_project.py',17)} 将 frontend → `tools.tight_fresh_train train`，backend_prepare → `tools.tight_fresh_backend`，backend_cache → `tools.tight_finish residual_cache`，backend_train → `visuotactile.anchor_continue residual-train`，dagger → `tools.tight_fresh_batch_run run`。入口只是调度，实际参数完整保存在 {local('source_and_launch.json')} 的 `owners[].record.commands/job`，不根据当前 HEAD 猜测执行路径。

关键 owner 源提交：R1训练 `e7502d78f4`，R2训练 `a076a8274c`，R2磁盘恢复 `fed146f224`，终测 `0e914563c2`。R1首个真实采集 plan 记录 `d364c0c`，实际 owner 为 `0a48e13`；后续有准入/闭合恢复提交，不能把所有物理阶段写成248aec3。

核对31份采集/终测 plan、全部已加载模型身份及4份 DP配置中的源码 SHA：数值模型、数据、控制路径无差异。早期3份R1 plan 共6个 SHA差异仅涉及 `tools/tight_adaptive_dagger.py` 和 `tools/tight_fresh_report.py` 的 CPU准入、已完成机会重新登记与恢复路径；没有物理/标签实现变化。原始失败 plan 与回执均保留。

R1采集加载原130-50完整A/B；R2采集加载R1各自选中完整pair；终测加载原130-50与R2完整pair。每个DP与后端实文件重新SHA并对齐plan/result，未发现混角色或混轮次。

| 实际模型路径（相对当前项目） | SHA256 |
|---|---|
{hash_table}

## 2. 采集、触发、控制与首合法标签

实查73个完整采集机会的触发；50个准入纠偏的每个 transition 都追到两训练 loader、后端 target 与执行器反馈。{local('trigger_verification.json')} 独立用保存的已完成学生历史重算原触发函数，73/73触发回执一致。R1 48机会/31接管/29准入/4855合法tick，R2 25/22/21/3500。接管后专家成功不计学生自主成功。

队列不是物理清空：shadow专家队列在学生前缀持续前进，长度3；切换tick及下一tick是过渡，mask=0；`first_legal=switch+2`。过渡已有专家控制但存在旧排队动作来源，不能把control_source=expert直接当合法标签。{local('control_verification.json')} 按实际 action配置独立复算原专家流水线：raw clip → delay源索引 → arm增量/hand绝对映射 → EMA → arm progress0.5/hand1 → hand2rad/s限速。50条所有数值最大误差0；学生prefix提议=实际命令，专家后缀映射=实际命令；共享可达投影没有再改这些已处理标签；position executor没有第二次delay。

{local('data_chain.json')} 对全部50条核对：A `clean_sample=q_reached[t]=diagnostics_after.q[t]=q[t+1]`；B clean_sample与两个后端 target 均为该transition最终实际专家绝对命令。合法mask排除prefix/过渡，所有帧的真实prefix保留在history。两物理子步actuator target、requested/applied/target与trace执行值相等；相邻q_after=q_before、target_after=next target_before。

逐值样本例：R1 `r10_B_layout10_train_28817004_contact`，switch108，109仍过渡，110首合法；JSON保留106/107/108/109/110、下一原生规划边界120与末帧的29维q_before/q_after/u_exec/q_label/u_label、mask/source/queue/source_step与相机时钟。其首合法tick110的A标签等于同次q_after，B标签等于同次u_exec，误差均0。其余49条按同样规则自动选取switch±2、相邻12边界及terminal并保存实值；子目标切换另见终测与post_subgoal梯度样本。

预期与实际：合法标签时序、裁剪/限速/投影、执行器落地值均通过。这里没有发现标签差一帧、队列过渡泄漏或重复执行器滤波。

## 3. 入库、family、三路视觉与触觉

共同池130→159→180，新29+21=50，合法4855+3500=8355。A/B的identity/split、原始 resolved(path,episode)去重一致，原130全部保留，validation固定6条。30个训练family与6个验证family隔离，没有把同family纠偏当独立family。R1全训练32363合法tick、R2 35863。

两个loader逐帧time、robot q/current actuator target/previous requested、supervision_mask/control_segment和后端target全部与raw一致（数值最大差0）。学生前缀作为真实历史，loss mask独立；后端32tick分块不会因prefix无标签而清历史。原生子目标变化不等于episode id变化，在线历史重置只在新episode tick0执行；终测全链q/target/contact连续，未发现子目标切换意外清历史。

{local('vision_verification.json')} 对R1/R2各A/B来源4条纠偏，在0、1、11、12、switch−1、switch、first_legal与末帧，从raw RGB/depth重新生成缓存：同帧DINO缓存最大差0、depth_grid最大差0；与相邻帧特征RMS距离约0.45–0.64。可排除这些样本“索引一致但内容错一帧”。50条所有相机frame严格递增、camera_time=obs_time，无重复帧号。相机顺序固定 `fixed_high,fixed_side,wrist_rgbd`，原始depth是米，采样中值约0.93–1.05m，没有毫米/米混用。

腕相机使用每帧实际外参，需减lane base世界偏移再比较。相对手掌rig全帧最大平移约0.127mm、旋转约0.000976rad偏离字面配置；这是可测小量，符合USD姿态精度可能性，但本次没有证明它是标定bug或失败原因。触觉每个观察对应前一个transition的两个物理子步，间隔1/120s；world/local force和finger joint chain逐值一致。首帧初始化与后续有效样本分别处理，没有把当前动作之后的未来触觉灌入当前观察。

跨lane核查覆盖上述不同lane来源原始内容及四例从实际并发capture到独立controller的完整自重放；未发现串lane。未穷举所有lane排列与所有历史时刻，因此不宣称形式证明。

## 4. DP训练、入口相位与梯度

实际H24，线上每12个60Hz控制tick规划一次并commit前12个；训练允许随机anchor增广，不要求所有训练anchor都在原生grid。连续完整前缀的训练B2 encoder与逐时encode_dense在实查0/1/11/12/switch/firstlegal的condition完全一致。

### 确认问题P1：恢复入口anchor被实际训练采样排除

位置：{link('visuotactile/anchor_sampling.py',69)}、{link('visuotactile/anchor_sampling.py',102)}；覆盖补齐 {link('visuotactile/effective_exposure.py',39)} 也只从合法tick选anchor。相比之下 {link('visuotactile/target_windows.py',26)} 明确支持首合法tick之前的原生anchor并只监督其合法后缀。

最小真实例：`r10_B_layout10_train_28817004_contact` switch108，first_legal110。线上anchor108的slot2对应110（slot0/1无合法专家标签）；`labels(record,[108],24)` 给出slot2起的合法后缀，但 `validate_training_anchors(record,[108])` 抛出 `Selected training starts must have true action supervision`。选中R1 A/B的52/50次访问中，最早训练anchor是110，108从未出现。50条中48条有此情形，角色合计96/100条审计行，详见 {local('onset_phase_gap.json')}。

这是一项接口/分布覆盖缺口：同一个绝对合法tick即使≥50次进入loss，也没有在实际在线入口历史、当前commit槽位下曝光。较晚anchor已经读到后续专家执行反馈，会使恢复入口条件更容易。现有数据可以训练这个带mask窗口，不需要伪造prefix专家标签。影响两种DP的新增50纠偏；父模型不含这些新数据。旧池若有同类非grid交接，同一设计也可能影响父模型，未将新50结论外推为父模型所有历史的审计。

修复建议：保留原随机增广，额外纳入首合法tick前的实际原生anchor，用现有window_mask严格排除prefix，另记`(anchor,slot,absolute tick)`的部署相位覆盖，避免只用absolute tick年龄替代该验收。需重新登记采样与预算后验证，不在本次实施。它可能解释“纠偏入口没有学成”，尚不足以证明终测从tick0开始的旧成功丢失。

### 选模、曝光、LR与梯度独立复算

四次DP逐提交更新日志重建真实anchor→24窗口mask→每optimizer step唯一绝对tick，checkpoint计数差0；四次后端先重建已提交step贡献日志，再在本次审计重新从raw HDF5合法mask、实际seeded deck、4轨迹group、32tick chunk与128合法tick窗口独立生成每步参与集合：逐步与原日志及checkpoint计数完全一致。该CPU复算未初始化CUDA，四份证据为backend_raw_round01_A/B.json、backend_raw_round02_A/B.json。未计祖先、重叠slot、padding或未提交partialepoch。

八个选中checkpoint均重新跑完整固定validation6；DP最大数值差4.6566e-10，后端差0，选中epoch均为曝光门达标集合中的最低验证损失。

| 轮 | 组件 | 选中epoch/update | 最低访问/逐tick提交曝光 | 独立重算验证loss | 与原日志差 |
|---|---|---|---|---|---|
{selected_table}

四次DP实际初始LR均1.5e-5；epoch48验证后 `Plateau.observe` 设置next LR=7.5e-6；epoch49至60的更新使用半LR。不存在“首轮全程半LR”或磁盘恢复导致提前减半。实现见 {link('visuotactile/bounded_extension.py',95)} 与 {link('tools/tight_fresh_train.py',692)}。曲线的橙色轴是该epoch实际使用LR，竖线在48/49之间；绿色区表示门槛可达阶段，红点是选中模型。

![验证与实际LR]({OUT/'validation_and_lr.png'})

本次5份真实loader→完整history→生产objective→backward诊断覆盖4条独立新纠偏（普通不稳/弱支撑、A子目标后、B子目标后）。使用选中checkpoint中的raw训练state，固定诊断噪声8091，无optimizer/EMA step；不伪称还原原始瞬时训练梯度。

| 角色 | 真实纠偏 | loss | clip前参数梯度norm | clip系数 | 非法slot梯度max | 29维中最小输出梯度RMS |
|---|---|---|---|---|---|---|
{gradient_table}

所有29输出维有非零梯度、prefix/padding无梯度，未发现关键维被global clip压死。R2 A弱支撑样本中，hand_geometry加权输出梯度L2=1.1577e-6，base hand=1.9131e-4，约0.6%；其权重确实小，但这是已登记目标权重，不能据单例宣布致因bug。原冻结归一化与scale逐维值、各几何分量及gradient均保存在 {local('training_verification.json')} 和 {local('extra_gradients.json')}。后端实际是linear reachable_nominal_centered + ideal_box_span_fy_roundoff loss，没有把旧tanh/hardclip实现误当本轮路径。实际Transformer dropout=0（models/native_dp.py:533），使用LayerNorm而非运行均值BatchNorm；推理显式eval，DINO冻结eval。没有观察到train/eval模式导致的额外状态差异，DINO kernel/precision差异单列P2。

## 5. R2 B-DP磁盘中断与恢复

{link('tools/tight_fresh_train.py',438)} 调用 {link('visuotactile/checkpointing.py',47)}。保留断点 `round02/disk_recovery04/B_interrupted_last.pt` 的SHA `e9c34db9091b840cbda02fa353b1053ae9ceebe87c85accb86ea980d0d617e6c`，epoch1/update120，已提交visits240。实际恢复不是取partialepoch临时文件。

本次独立调用恢复函数后，raw/EMA参数、Adam的188项状态及step120、EMA optimization_step、schedule/noise/torch/CUDA RNG全部逐tensor相等，LR1.5e-5，Plateau状态保持。保存schedule RNG重算next deck与实际 `deck_002.json` 全序相同；首个durable回执为epoch2/update240/visits480/EMA240。证据 {local('restore_verification.json')}。

限制：epoch2当时具体数值checkpoint后来被原子替换，现只剩其durable回执/hash；无法独立比较原第一条恢复后梯度或更新tensor。保存断点+恢复函数+下一deck+连续日志支持正确恢复，不能冒称有已经不存在的逐更新tensor证据。没有发现optimizer/EMA/LR/RNG/采样位置重置。

## 6. DP→nominal→Transformer→线上接口

{local('nominal_provenance.json')} 逐一加载R1 A/B各165个（159train+6val）、R2各186个（180+6）nominal文件的identity，所有base SHA指向本轮本角色选中DP；A supervision=q_reached，B=u_expert，backend target仍都是专家最终绝对命令。未发现旧DP缓存或模型配错。

四例（每轮各A/B）用实际选中DP、固定seed7091，从保存的真实完整历史重建nominal：robot/time/visual/24plan/current nominal/previous nominal/target最大差0，mask/plan_valid完全一致；训练B2与serial condition最大差0。证据 {local('nominal_round01.json')}、{local('nominal_round02.json')}、{local('nominal_A_round01.json')}、{local('nominal_A_round02.json')}。

R2 B真实完整历史280tick，在同模型/同相位/同噪声下比较cache与online：robot0；视觉latent≤1.82e-7，condition≤1.08e-6，nominal与previous nominal≤7.92e-7；离线32tick chunk后端与逐tick线上最终命令≤7.16e-7rad。history fullprefix与chunk detach的边界没有造成状态错位，证据 {local('cache_online_parity.json')}。该文件中父模型对“候选缓存”的大误差只是跨模型差异，不能读成父模型自重放错误。

先自重放再跨模型：下表从raw RGB、metricdepth与真实本体/触觉输入重建实际采集时完整controller输出，反馈使用真实已执行专家命令的ack，而不是拿student shadow预测冒充反馈。四例分别来自并发采集中的A/B首个lane任务，避免把后续queue的持续RNG误当初始seed。

| 实际采集样本 | 完整tick数 | command最大误差rad | nominal最大误差rad | robot178误差 |
|---|---|---|---|---|
{replay_table}

### 确认问题P2：DINO采集与缓存kernel未统一

{link('tools/tight_adaptive_dagger.py',149)} 固定采集microbatch1；缓存 {link('visuotactile/native_data.py',541)} 调用dino(rgb)，经 {link('visuotactile/models/native_vision.py',78)} 默认microbatch2，最后lane复制成6图。该类的注释本来要求改变kernel形状经过真实command parity gate。

最小复现：对同一raw RGB tick调用 `dino.forward_lanes(rgb[None],vision_microbatch=1)[0]` 与 `dino(rgb)`，再将两组patch送入固定controller、同相位/7091噪声/真实ack。B的R1/R2完整自重放中，用缓存patch对比原在线：command最大误差分别0.00016635656/0.00023174286rad，nominal0.00020480156/0.00011301041rad；改回原始RGB的实际microbatch1后，command均7.1526e-7rad。patch自身最大差1.5234375/1.25是原始特征单位，不是关节角。

影响：共享代码可以影响父与候选被采集/缓存观察到的输入。实际终测16个两lane波**全部microbatch2**，与缓存同形状，见 {local('actual_final_batches.json')}；不能把采集端发现误写成终测microbatch1。它能解释既有自重放约1e-4rad偏差，但没有物理反事实证明能解释三个旧成功丢失。

修复建议：明确统一camera preprocessing/kernel shape与precision配置，将其纳入缓存identity与deployment合同；先设完整真实history命令自重放门，再考虑更改microbatch。不能只核对缓存文件SHA和索引。

### 同输入父子比较与噪声报告错误

{local('same_history_original130_A.json')} 与 {local('same_history_original130_B.json')} 使用现有真实开发/训练family采集历史，原130-50与R2固定完整pair、同7091噪声/相位/输入/ack比较；详细command/nominal差异在各自contrasts。它们是teacher-forced历史的输出比较，没有仿真反事实，不等于某个闭环成功会恢复。

实际模型合同指定7091，{link('visuotactile/native_controller.py',76)} 的7093只是 `.get` 回退。{link('research_docs/evidence/tight_adaptive_dagger_20261119_01/quantitative_report.py',110)} 与该文件155行、{link('research_docs/evidence/tight_adaptive_dagger_20261119_01/publish_results.py',79)} 把“online7093”写死。这是确认报告bug（P3），应从checkpoint与每lane generator state/episode位置生成说明。队列内新episode历史会重置，但generator持续推进；所以也不能把所有后续episode简单说成“与离线7091同噪声”。这项解释错误不改变最终32个成功判定。

## 7. final_test完整32条、16配对与四个重点结果

{local('final_verification.json')} 重新从真实worker plan/result、numeric HDF5、trace/plans与模型SHA核对全部32条；16份实际初态配对回执通过。相邻lane世界偏移先扣base再比较，未要求隐藏PhysX/RNG精确重放。重新统计A原1/8→新0/8，B原2/8→新1/8。没有将资源中断/准备失败算入32条完整结果。

逐tick链：24plan按tick//12与tick%12取current nominal并先作joint bounds；nominal+normalized residual/scale→jointclip→reachable projection→command→requested→actual actuator两子步→applied ack。全32记录中nominal/cursor、jointclip、request、actuator、ack、q/target/contact chain差0；最后投影重算最大1.1921e-7rad。trace的`execution_preprojection_target_rad`实际上是jointclip之后、reachable projection之前，不能与未经jointclip的raw proposal直接比较。

{local('native_criteria.json')} 从raw数值独立重算near_goal、retract_condition与fullcomplete，32条三种判据mismatch均0。原生成功不是仅接近孔口：每个子目标near阈值随原生goal变化，要求累计10个达标step（`force_consecutive=False`，不是10连续步）；最终retract需要keypoint≤7.5mm且mean finger-object distance>100mm，并维持原生retract状态。首子目标会重置原生600tick时钟，不能把>601帧误当延长截止。此时controller历史不重置。

最早差异是“可定位输出分叉”，不是已经证明的首次有害bug：

| 配对 | command>1e-6rad首tick | q_before>1e-4rad首tick | 物体位置>1mm首tick | 指尖接触数不同首tick |
|---|---|---|---|---|
{pair_table}

- **A 28839004丢失：** 原策略首子目标203，第二子目标247，完整成功253（254帧）；候选首子目标213，未进入第二子目标，随后失去支撑，420 transition后物体z=0.09875m并终止。原最小keypoint1.280mm；候选在第一子目标阶段可到8.932mm，但之后goal已变，不能把跨goal全局最小值误当最终孔底精度。最早输出分叉0、物体分叉62，失败不只发生在晚期。
- **B 28839005丢失：** 原首子目标236、完整成功285；候选216曾near一次，累计达标仅6步未到10，首子目标都未完成；601帧超时。候选末keypoint51.870mm、z0.66601m，**不是跌落**。接触数最后≥2在277，282后该>0.1N指尖接触诊断为0；物体仍留在基座高度，不能将接触消失直接称为跌落。原最小1.217mm，候选当前goal最小10.189mm。最早输出0、物体68。
- **B 28839007丢失：** 原首子目标234、第二282、完整287；候选始终无near/子目标，最小keypoint19.275mm，末94.988mm/z0.65007m，601帧超时。两指以上接触最后445；对准误差在接触完全消失前已不达门槛。最早输出0、物体66。
- **B 28839002新增：** 原无子目标，最小18.777mm，601帧超时；候选首子目标224，368时最终goal keypoint2.277mm，373累计第二子目标并完成retract（指尖平均距离113.726mm），374帧成功。最早输出0、物体65。说明参数改变可造成双向能力变化，不能把所有新输出差异一概解释成有害。

{local('priority_case_details.json')} 保存四对t0 nominal/residual/command的29维实值、跟踪误差和关键事件；{local('native_criteria.json')} 保存goal切换/retract实值，{local('final_verification.json')} 保存物体/接触/关节快照。

![四个重点配对的几何与接触]({OUT/'priority_pair_timelines.png'})

## 8. 尚缺字段、最小补录与因果边界

最终自主numeric记录由 {link('visuotactile/queued_rollout.py',341)} 明确剔除image/depth；终测无原始逐帧RGB-D，也没有完整DINO patch/visual latent或每次diffusion noise/RNG state。MP4为有损图像，不能支持精确full-history网络自重放。因此本次精确自重放使用保存完整rawframes的既有采集轨迹；终测能追到输出/执行/物理结果，不能把终测t0差异进一步唯一归到视觉encoder、DP参数或后端参数。

最小补录建议（未执行）：固定原130-50/R2 pair，在**已有开发初态**做约96–144tick短诊断；记录每lane/episode/tick、三相机frame/time/实际外参、DINO patch fp16及metricdepth_grid（或可无损恢复raw RGB-D）、robot178/两substep触觉时间与真实ack、规划condition、每次noise或generator state、24plan、backend状态/nominal/previous nominal/raw residual、投影区间和执行命令。先同一模型自重放≤1e-6，再固定同输入与噪声做DP/后端逐级替换。短诊断只定位分叉，不计入成功率，也不重新使用终测挑模型。

已证明：三个旧成功的参数/完整输出在tick0就变、反馈放大后几何与成功判据发生偏离。未证明：P1、P2、某一损失权重或某个组件足以因果解释这些丢失。没有证据支持只加epoch、只加相同纠偏、或把encoder列为唯一瓶颈。P3报告事实应立即修正，但不会恢复控制能力。

## 9. 复现与范围

所有脚本只引用SRC而不是较旧CAN源码。环境Python `/home/CNS2025331827/miniconda3/envs/isaaclab/bin/python`，`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2`。生产训练/仿真入口未重启。可按下列诊断脚本分别复算；完整输入路径、样本、预期/实际值、hash与失败限制在对应JSON中。

| 环节 | 本目录复现脚本 | 主要证据 |
|---|---|---|
| raw→两loader→后端 | audit_data.py | data_chain.json |
| 因果触发、真实机会 | trigger_verify.py | trigger_verification.json |
| 专家/学生实际控制 | control_verify.py | control_verification.json |
| 三路raw视觉与触觉 | vision_verify.py | vision_verification.json |
| 八组件验证/曝光/LR | training_verify.py | training_verification.json |
| 后端raw mask/deck独立曝光 | rebuild_backend_raw.py | backend_raw_round01_A/B.json / backend_raw_round02_A/B.json |
| 相位与RNG恢复 | phase_restore_sources.py | onset_phase_gap.json / restore_verification.json |
| B完整自重放、nominal | replay_interfaces.py / replay_exact_vision.py | online_self_replay*.json / nominal_round*.json |
| A重放/缓存+额外梯度 | extra_checks.py | online_self_replay_A_exact_vision.json / nominal_A_round*.json / extra_gradients.json |
| 同输入原130-50/R2 | same_history_original130_A.py / same_history_original130_B.py | same_history_original130_*.json |
| 终测输出/物理/统计 | final_verify.py / native_criteria.py | final_verification.json / native_criteria.json |
| 本报告与图 | render_audit.py | REPORT.md / 两张PNG |

例：`OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /home/CNS2025331827/miniconda3/envs/isaaclab/bin/python {OUT/'phase_restore_sources.py'}`。某些重算会占用GPU做固定权重推理/反传，均不更新参数。原始训练梯度与已覆盖的恢复后checkpoint不能重新创造；对这些限制不作更强结论。
'''
 (OUT/'REPORT.md').write_text(text)
 print('report bytes',len(text.encode()),'figures created',flush=True)
if __name__=='__main__':main()
