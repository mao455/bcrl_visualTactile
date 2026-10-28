# 三视角验证与400条采集协议（2026-10-28）

本报告基于本机实际源码、渲染和完整轨迹。起点 `f39c968`，分支 `feature/triple-camera-validation`。只做观测扩展、原专家采集与零更新检查；不训练、不换手、不改原资产/控制/物理/判据。证据根目录为 `outputs/visuotactile/camera_v3_20261028/`（不公开上传）。当前400批次状态以 SESSION_HANDOFF 和实际 run_status 为准，不能把本文的验收样本计入400。

## 研究判断与选择

推荐保留两个固定外部相机，加一台腕部附近支架承载的斜置相机。已验证的是**仿真观测方案**：相机随现有刚体固定变换运动，没有新增质量、碰撞、关节、驱动器，没有逐帧读取物体/孔座来look_at。尚未验证实体相机型号、支架强度、KUKA左手安装孔、线缆或整个关节范围的运动包络。固定视角仍不可少；腕部会被手背、手指和插入件遮挡，不能保证每个阶段都看清孔口。

### 来源查证（2026-10-28实取网页/仓库索引）

- [SharpaDex](https://sharpadex.sharpa.com.cn/)描述双腕相机与头部相机，页面给出480×480/30fps。其指尖视触觉相机不是本项目腕部RGB相机。
- [Stanford Cloak安装指南](https://tml.stanford.edu/cloak/setup/sharpa.html)：Franka、右SHARPA、ZED Mini和专用加长/倾斜支架，说明腕部外置支架可行，**不证明当前KUKA左手兼容**。
- [SHARPA CAD](https://github.com/sharpa-robotics/sharpa-opencad-library)：读取README和递归文件索引，有左/右Wave STEP、Wrist/Flange/Internal接口STEP/PDF，默认ISO-9409-1-50-4-M6接口。没有下载/改造STEP或复制支架；API未列license，不能据此赋予再分发许可。
- [DROID](https://droid-dataset.github.io/)明确两台外部ZED2＋腕部ZED Mini。借鉴视角分工，未复制其机器人、数据或模型。

网页快照保存在ignored references目录。本轮没有vendored第三方代码，原MIT和既有Diffusion Policy版权文件保留。

## 实际机器人和挂载

实际URDF为 `assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf`。原场景转换/烘焙USD，机器人基座在环境坐标(0,.8,0)，方向单位四元数。`camera_diagnostics.json`保存运行时USD树。

URDF固定链 `iiwa14_link_7 → iiwa14_link_ee → sharpa_mount → left_hand_C_MC` 在USD中仍可见嵌套Xform，但只有 `/World/envs/env_0/Robot/iiwa14_link_7` 是该段刚体，后面固定链接已合并。新相机实际挂在该刚体下的 `ResearchWristCamera`。原 `setup_student_camera` 的world-only异常只是原软件未接线，不能解释为SHARPA硬件不支持腕部相机。

### 当前冻结的采集配置

坐标均为米。A/B的eye和target相对environment origin；C相对**刚体link7坐标系**，不是掌心/COM。target只用于构造一次固定朝向。

| ID | 光心eye | 朝向target | 光学参数 |
| --- | --- | --- | --- |
| fixed_high | (.32,-.45,1.55) | (0,0,.68) | focal40 / aperture33.1973787 |
| fixed_side | (-.58,-.06,1.05) | (0,0,.66) | focal40 / aperture33.1973787 |
| wrist_rgbd | (.12,.16,.07)，link7局部 | (0,-.14,.24)，同局部 | focal24 / aperture33.1973787 |

A俯角约57.6°，B约33.8°；水平面绕工作区方位角最小差约119.5°，不是光轴三维夹角。固定相机移近操作区并收窄视场，不做图像后裁剪。三路统一**448×252、RGB＋米制轴向深度＋bool有效掩码、60Hz采样**；先比较224×126后才提高分辨率。固定近/远裁剪(.1,5)m，腕部(.015,3)m；仿真最小深度不是某款实体RGB-D相机已验证的最小测距。

渲染采用当前Lab的quality preset、AA Off；没有更改原灯光、材质、任务DR。固定K的fx=fy约539.802，腕部约323.881，cx224/cy126。每条metadata存实际内参，不只保留请求参数。`wrist_bracket`为方案名；camera layout3 / perception2 / memory2 / trace2 / compact storage2彼此独立。

### 候选与实际可见性

先跑了side/opposite各168步，掌/指遮挡且孔口不足；这两条还记录了错误的静态腕部位姿，只保留为诊断，不能当正式数据。lower候选拍到物体和指尖，但光心可低到桌面附近，否决。overhead能看孔座却被手背覆盖较多；最终bracket在上方并向侧面偏置，能看到更多手指旁的物体轮廓。无需换手；未继续实跑第三固定备选（接口存在，尚未验收），不声称腕部在所有任务上优于第三固定相机。

实际检查三组完整轨迹：seed920302/168步、17/220步、920101/214步。原生物体位置/姿态和孔座位置不同，均通过初始box重叠检查。查看同步原图、定期全程分镜与完整解码视频；每12步导出RGB、深度预览、bool mask及float32原深度。路径：各目录 `review/synchronized.mp4`、`review/all_stages.png`、`phases.png`。末步之后自动reset图不补入上一episode；原系统没有单独终止曝光。

- 接近/抓取：A看工作台、孔座及手物相对布局；B看指尖高度和夹持侧面；腕部看手指旁L形物体，初始孔口可能出框。
- 抬起/搬运：A较稳定；B可能被机械臂遮挡孔座；腕部相机逐渐看到放大的孔口，补充侧面盲区。
- 对准/插入：A常被手和机械臂遮挡，B能看竖直插入件；腕部可见部分孔口与物体尖端，但手指/物体也会遮住孔口，最终不能保证三路都看到空孔。
- 撤手：只有终止前真实曝光；若任务结束与阶段切换发生同tick，没有额外撤手/终止图，不从新episode补齐。

以seed17为例，hole URDF开口21×31mm。448图像上A的内部投影约15.5×14.8px、B约20.8×7.9px；在step88搬运时腕部约56.2×64.7px。对投影内部去重像素做深度检测，A有123个像素看到开口后方，B为0且160像素被前景挡住，腕部有1054个开口后方像素。step187插入时分别0/37/57，step218腕部203个内部投影像素全被前景遮挡。**投影面积不等于可见性**；深度阈值3/5mm、边缘混合和插入件遮挡也有限制，须结合RGB解释。脚本 `camera_review aperture` 仅作特权诊断，绝不输入策略。红色物体像素计数也只针对本资产，不是通用语义分割。

三组执行轨迹的光心到机器人visual/collision mesh局部AABB最小距离为.1281/.1301/.1270m，到object/table/hole盒子的最小距离分别约.1954/.1456/.0811m（每组取三者最小）。保守AABB正距离说明光心在包围体外，**不能证明实体相机外壳/支架/全指运动包络安全**。此处只读取URDF/FK与已采轨迹，没有改变碰撞。

## 动态标定修复与时间

本机真正导入的Lab路径为 `docs/IsaacLab_DexAssemble-sharpa/source/isaaclab`。其Camera配置虽有update_latest_camera_pose=True，但XformPrimView在CPU禁用Fabric查询并回退USD；因此腕部RGB在动，camera.data.pos_w却停在初始USD姿态。最初审计正确报错；没有用这些旧值继续采集。

修复只在研究wrapper内：为每个相机、每个环境的原render product附加Replicator CameraParams，读取**同一渲染的**cameraViewTransform，按row-vector/OpenGL矩阵转成世界ROS optical（+Z前、+X右、+Y下）pos+wxyz。不增加物理步、不设置每帧look_at、不改共享Lab。保存(N,V,7) camera_pose_w及腕部(N,7) camera_parent_pose_w，后者是诊断真值。逐帧验证与link7当前姿态×固定局部外参的位置/旋转误差<2e−5，固定A/B不动，时间戳与动作前时间一致，post/pre calibration精确衔接。外参字段没有加入Nominal allowlist。

物理仍120Hz、原生动作/专家/相机60Hz，触觉每步两个120Hz子样本。增加渲染使墙钟变慢，不更改仿真dt或动作增量。在线采集为单环境；模型两环境独立状态测试通过，但没有声称三相机多环境实机渲染已验收。

## 原专家回归

released checkpoint SHA256 `7e61015dad660430ab07bab4f715c90c6f76b84cc358da6be81319a6aee09990`；teacher policy原140D，加原SAPG身份给player；研究图像不进入专家。使用原deterministic group0。配对保持同seed、同合法初态、同world_hold（此前已验证的可选世界施力兼容wrapper），未对原RL做训练。

- seed920302旧双相机repeat A/B：168步，128组非图像数组逐位一致。
- 同旧双相机→三路224 quality及三路448 quality：同168步、128组数组逐位一致。
- seed17新三路448→上一批原双路world_hold：220步、128组数组一致，包含非零外力。

比较包括专家输入/动作、q/dq、物体/孔座/桌/目标、target、接触、delay、RNG before inference/step/preprocess、模拟时钟、success/termination。物理配置各section和checkpoint hash也相同。此次重复基线差为0，配对亦为0，未人为设置“宽容差掩盖差异”；不外推GPU物理、其他seed或整机确定性。无渲染CPU路径仍禁止。

## 颗粒诊断

在物理状态固定时连续24次render，无env.step、无新观测/RNG调用，检查关节/object状态和torch RNG不变。保存原始uint8 RGB，再单独编码/解码MP4。颗粒在原图已经存在；原reader只做RGB/255，无额外RGB随机噪声。depth augmentation/pose randomization/delay仍显式拒绝。

旧performance、AA Off下静态RGB时间标准差约13.62/20.82（0..255），MP4约13.26/20.25，MP4相对原图MAE约3.92/5.19，压缩不是主要来源。同A/B固定几何的quality224约.75/.64，quality448三路约.67/.55/.45。原粒状区域随render改变，而物理/RNG不变，证据指向渲染采样/重建路径；不能解释成材质本身随机或物理噪声。

额外受控测试尝试关闭sampledLighting，但读取设置发现被渲染器重新置True，故该项**不是有效关闭对照**；仅把samplesPerPixel改16在performance下仍约11/14，不能宣称加采样已解决。quality同时改变denoiser、cache、GI/reflection等多项，明显改善颗粒，但本轮未把根因唯一归到某一开关，也未证明没有时间重建伪影。没有对图像做模糊或减弱任务DR。静态诊断在自然终止后做，物理状态可能已reset，明确不当作旧episode终止图。

## 资源、数据和网络检查

| 同seed920302完整168步 | 正式HDF5 | 派生MP4总计 | 采集循环墙钟 |
| --- | ---: | ---: | ---: |
| 旧双224 performance |25.96MB|2.15MB|14.42–14.75s|
| 三224 quality |37.66MB|.34MB|21.72s|
| 双448 quality |67.25MB|.44MB|22.03s（无并行CPU验证时）|
| 三448 quality |118.25MB|.76MB|32.85–35.43s|

同448/quality的第三视角使此样本文件总量+75.8%（不能只按1.5倍逻辑像素估算），循环+49–61%；RGB内容、深度可压缩性和逐文件开销不同。单次GPU进程快照双路2932MiB、三路3001MiB，+69MiB，不是峰值；GPU总量含其他任务，不能直接全归本项目。静态render计时受暖机/其他CPU负载影响，有约28/44ms的样本，不能当精确吞吐基准。一次双路46s同时运行CPU模型回放，已排除该次作为相机成本对照。

三个最终高分辨率验收episode主文件约119/151/150MB。400成功仅成功文件约44–56GiB；不假设成功率。若失败占用/次数过多，95GiB批次上限可能先到，届时明确停止而不假称完成400。剩余磁盘持续检查，保留40GiB底线，每次预留12GiB给2400步原记录＋转换暂存。串行一个环境、nice19、两CPU线程/核22,23、每次休息5秒；不操作后台程序。不承诺渲染零竞争。

正式四主文件=data.h5＋camera_0/1/2.mp4；H5保存无损RGB uint8、float32 depth和mask，MP4只观看。每条143字段往返shape/dtype/value精确；末图缺失保持mask。168步三视角完整Dataset顺序回放chunk7/step1最大误差4.92e−7，重复reset精确、参数/buffer hash不变。16步真实三视角因果回放生成实际未训练Nominal计划供Shadow使用，三路图像Jacobian非零，RGB/Depth/tactile/motion/intent梯度有限非零，保存恢复精确；0参数更新、不执行网络候选。旧双视角checkpoint显式拒绝，不重新解释权重。纯专家数据仍缺已训练Nominal自身计划，Correction训练目标未定。

## 新400协议

`collect_success_batch.py`只在camera_gate各项验收后启动新的根目录。固定上述rig/resolution/render/原专家，seed序列930000起，每次+1，原生reset姿态/位置/孔座/DR，零额外XY；非法box初始化动作前拒绝且记录，不修改高度或重新采样同seed。自然成功且全字段/转换/动态标定/物理协议审计通过才计数；任务失败、删失仍保留并进入尝试分母；软件/审计错误停止，未知损坏文件不删。最多1600个seed尝试、95GiB、每条2400步/900s；不能以尝试数替代400成功。源代码hash冻结，改变源码后停止下一次启动。未来划分按seed/初态组，不按帧。

已授权400不再受上一批60尝试/10GiB/不扩量限制；上一批仍原样单列。新批次不混入候选相机、旧失败或验收轨迹。具体PID、首条episode及中途/最终状态写在SESSION_HANDOFF，来源是实际文件与进程。
