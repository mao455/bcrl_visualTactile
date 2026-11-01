# 公开方法源码对照：Native-command三组

这是固定revision源码/配置对照，不是作者checkpoint复现或论文性能复现。
完整下载身份、文件hash及来源位于本机 `outputs/visuotactile/native_dp_3way/source_audit_20261101/sources.json`。
所有引用限下列revision；默认配置不等同论文所有任务的最终设置。没有下载作者模型或移植其大型训练框架。

| 方法/仓库 | 固定revision | 实际入口/配置 |
| --- | --- | --- |
| [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) | 5ba07ac6661db573af695b419a7947ecb704690f | diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py；diffusion_unet_lowdim_policy.py；config/train_diffusion_unet_hybrid_workspace.yaml |
| [robomimic](https://github.com/ARISE-Initiative/robomimic) | d309eaecc18acf4152a830a895a6984b8ac71b05 | robomimic/algo/diffusion_policy.py、bc.py；models/obs_core.py；config/diffusion_policy_config.py |
| [DP3 / SimpleDP3](https://github.com/YanjieZe/3D-Diffusion-Policy) | 47385d9d6f5bde3f2ebdf2400ecb8261cc9e6b97 | 3D-Diffusion-Policy/diffusion_policy_3d/policy/{dp3,simple_dp3}.py；config/{dp3,simple_dp3}.yaml |
| [Octo](https://github.com/octo-models/octo) | 241fb3514b7c40957a86d869fecb7c7fc353f540 | octo/model/octo_module.py；components/{tokenizers,action_heads}.py；scripts/configs/octo_pretrain_config.py |
| [RDT-1B](https://github.com/thu-ml/RoboticsDiffusionTransformer) | cd79363a1387e8f81c7724d070ef7e45fd23150f | models/rdt_runner.py；models/rdt/model.py；configs/base.yaml |
| [Reactive DP](https://github.com/xiaoxiaoxh/reactive_diffusion_policy) | 824c5e8de1fd1811106907a04b5f0186e0138c0b | reactive_diffusion_policy/policy/latent_diffusion_unet_image_policy.py；config/train_latent_diffusion_unet_real_image_workspace.yaml |
| [ImplicitRDP](https://github.com/Chen-Wendi/ImplicitRDP) | 112d6857d914c5dee4c4c172790088adc6bdc3a5 | ImplicitRDP/policy/diffusion_transformer_image_policy.py；config/train_reactive_diffusion_transformer_real_image_workspace.yaml |
| [DexUMI](https://github.com/real-stanford/DexUMI) | acddb8f8a89a8f0186868bbec44306eb7808114a | dexumi/diffusion_policy/diffusion_policy.py；config/diffusion_policy/train_diffusion_policy.yaml |
| [Consistency Policy](https://github.com/Aaditya-Prasad/Consistency-Policy) | eed0c465d3c1aa6dcf22f10e784df07dec06beac | consistency_policy/student/ctm_policy.py；configs/ctmp_square.yaml |
| [VLA-Touch](https://github.com/clear-nus/vla-touch) | c9f970366387fc23f1d64a48a7827d3263410d89 | VLA/models/rdt_runner.py；VLA/residual_controller/bridge_controller.py、bridge_train.py；README |
| [UMI](https://github.com/real-stanford/universal_manipulation_interface) | d095ba9590df789df5189eea5ee7e431689038a6 | diffusion_policy/policy/diffusion_unet_timm_policy.py；config/train_diffusion_unet_timm_umi_workspace.yaml |

## 输入、历史与输出

下表的O/P/A为观察帧/预测horizon/提交chunk。动作含义由任务dataset/env定义；同为29维或H64不代表控制合同相同。
“未见独立target字段”仅指所读配置，没有证明整仓库不能加入此字段。

| 方法 | 视觉与本体/控制器输入 | 历史机制与更新 | 扩散与输出/时域 |
| --- | --- | --- | --- |
| DP | hybrid复用robomimic图像编码＋lowdim，GroupNorm/SpatialSoftmax；lowdim版支持纯状态；未规定本项目controller target | 固定O帧条件；没有本项目跨episode起点的持续GRU | Conditional U-Net1D；所读配置O2/P16/A8，epsilon默认且支持sample；动作取O−1开始 |
| robomimic DP | ObservationGroupEncoder按配置选择RGB/lowdim，BN转GN；控制目标不是框架通用必填项 | 固定stack2，推理action_queue顺序弹出；BC另有RNN/Transformer分支，不能混称DP默认结构 | 默认U-Net [256,512,1024]、time256、O2/P16/A8；100步DDPM epsilon，DDIM可选 |
| DP3/SimpleDP3 | point_cloud，默认可裁去RGB只保留XYZ；机器人状态由shape_meta配置；未声明原生target语义 | 固定短窗经DP3Encoder后作为global condition；不是持续记忆 | 所读SimpleDP3 O2/P16/A8、sample；任务动作由shape_meta决定，不是本项目pre-delay |
| Octo | 示例primary/wrist image tokenizer＋language；所读pretrain配置未添加独立proprio/target tokenizer | window2、时间/模态token、Transformer因果注意力及padding mask；不是GRU隐藏态 | DiffusionActionHead以readout条件生成未来4步；实际执行chunk由部署wrapper决定 |
| RDT-1B | 三相机×2帧、语言、统一128维state/action及mask，ctrl_freqs显式输入；state不等于controller target | 图像历史token与当前state-action token送Transformer，无本项目60Hz GRU | base config P64，hidden2048/depth28/heads32；1000训练噪声、5推理步、sample；执行chunk看部署入口 |
| RDP | ResNet18多视角＋任务lowdim，所选任务含GelSight embedding；扩展触觉/力历史走快速分支 | 慢视觉预测latent action；predict_from_latent_action允许更新extended_obs，再解码动作 | latent U-Net＋独立训练的action autoencoder/temporal decoder；horizon继承at配置，不把主YAML插值当数字 |
| ImplicitRDP | ResNet34 spatial tokens＋任务lowdim；extended_obs由RNN编码后进入diffusion时间条件 | 慢视觉token与快反馈RNN；predict_from_obs_tokens_and_noisy_trajectory处理缓存视觉/噪声轨迹及扩展反馈，缓存时域须与runner一起解释 | Transformer 7层/4heads/768；所读配置O2/P16、A=16−2+1=15，DDIM5；RNN1层512 |
| DexUMI | DINO ViT-small patch8；forward可选proprio与FSR；所读配置skip_proprioception=true、enable_fsr=false | obs特征展平固定窗，无持续GRU | Conditional U-Net；示例action_dim18、O1/P16/A8、50训练步/16DDIM epsilon；relative_hand_action=false |
| Consistency Policy | square示例两路84×84 RGB、eef pos/quat、gripper qpos；无独立controller target | O2固定视觉/低维条件；teacher到student一致性蒸馏 | CTMPPUnetHybridImagePolicy，P16/A8；CTM+DSM，教师路径必须提供，不等于直接训练普通DP |
| VLA-Touch | base RDT语言/图像/state/frequency；README Figure1另有DINO视觉＋marker触觉与base chunk的interpolant controller | 高层触觉语言反馈＋低层快速触觉修正；base与controller独立，不能仅看RDT runner断言全部历史 | base diffusion Transformer＋单独bridge/residual训练入口；整体推理整合README仍标待发布，不能声称完整复现 |
| UMI | 选定Timm CLIP ViT-B/16，attention_pool_2d，RGB及任务lowdim；共享/非共享相机配置明确 | 固定观察窗；图像/机器人时间对齐由dataset/real runner承担，无持续GRU | U-Net [256,512,1024] time128；所读workspace A8、16DDIM、50噪声epsilon；pose/夹爪动作不是本项目关节命令 |

## 触觉、chunk反馈与本项目取舍

DP、robomimic DP、DP3、所读Octo/RDT/Consistency/UMI默认配置没有独立的在线触觉残差执行器；
固定chunk内部是否提前重规划取决于runner，不能从policy预测函数推断硬件闭环频率。
DexUMI代码支持FSR但该示例关闭它，不能把“支持触觉”写成这份配置已用触觉训练。
RDP独立action autoencoder和快反馈解码可参考时域分工；ImplicitRDP参考因果扩展反馈与视觉token，
两者都不能直接等价于本项目加性关节残差。补读RDP `config/at/at_peel.yaml` 后可解析：
原动作P32、O2、dataset observation downsample2，dataset_obs_steps4，提交A29；
autoencoder启用一层32维RNN decoder，不能把latent的horizon当原生动作长度。
UMI `config/task/umi.yaml` 指定图像/lowdim观察各2、动作horizon16、relative pose表示；workspace提交8。
VLA-Touch `bridge_controller.encode_observation`拼接state、两路图像embedding及force，
`predict`另收VLA action chunk；dataset分别读取forces/displacements，独立force decoder存在辅助监督。
源码接口并未将它表达为本项目29维native-command残差。VLA-Touch base不微调的说法来自官方README，
其缺失的完整集成推理不能用本项目实现补齐后称作者复现。

本项目直接复用已审计DP ConditionalUnet1D与官方EMA，噪声/采样用现有diffusers。
normalizer采用官方limits-affine常数通道语义；目标future-only切片从0开始，显式区别O−1偏移。
Official组参考完整ResNet18/GN和SpatialSoftmax，增加独立depth/valid与关联深度，
共享相机权重、448×252全视野、O3/H64/commit12、sample、Loss B均是本项目适配。
History组参考空间token和双频反馈设计，但DINO6×8 token与双GRU是本项目结构，未声称来自某方法原配置。

本轮不复制大型Octo/RDT框架或训练十亿参数网络；不引入点云采集/估计、触觉蒸馏或一致性教师训练。
这些变更需要新的数据/预算/验收，会破坏当前共同控制与预算对照。借鉴动机是工程选择，效果仍待真实实验。
作者论文中的最佳设置、各任务真实执行频率/控制目标、论文成功率均未在本机复现，不能由源码默认值外推。
