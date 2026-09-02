# SLat-aware SS 蒸馏设计

## 目标与边界

该改造让冻结的 SLat Flow 教师为 SS Flow 提供下游任务感知监督，同时保留原 SS flow-matching MSE。实现不对 `decoder(z_s) > 0` 与 `argwhere` 宣称真实梯度；SLat 候选拓扑固定为配对 GT SLat 坐标，SS decoder 的连续 occupancy 只作为这些坐标上的 feature gate。

这是一种可控的代理目标：它能学习“遗漏哪些 GT 体素最会改变 SLat 教师行为”，但当前版本不能用 SLat loss 直接发现候选集合之外的假阳性体素。假阳性由全体素 BCE/Dice 辅助损失约束。

## 文献依据

- VQ-VAE（NeurIPS 2017）使用 straight-through estimator 将重建梯度传过离散表示：<https://proceedings.neurips.cc/paper/2017/hash/7a98af17e63a0ac09ce2e96d03992fbc-Abstract.html>
- Gumbel-Softmax（ICLR 2017）用可退火的连续分布近似 categorical selection：<https://openreview.net/pdf?id=rkE3y85ee>
- DreamFusion（ICLR 2023）展示了冻结生成模型作为先验、用蒸馏梯度优化另一种表示：<https://openreview.net/pdf?id=FjNys5c7VyY>
- DMTet（NeurIPS 2021）说明下游表面目标可以经专门的可微表示监督三维几何与拓扑：<https://proceedings.neurips.cc/paper/2021/hash/30a237d18c50f563cba4531f1db44acf-Abstract.html>
- ReinMax（NeurIPS 2023）分析 straight-through 为离散变量梯度的一阶近似：<https://proceedings.neurips.cc/paper_files/paper/2023/hash/28b5dfc51e5ae12d84fb7c6172a00df4-Abstract-Conference.html>

本实现是上述原则在 TRELLIS SS/SLat 接口上的工程组合，并不是原 TRELLIS 论文已有的训练方法。

## 单步训练路径

1. 按原 SS flow matching 采样 `t_ss` 与噪声，计算 `pred_v_ss` 和基础 MSE。
2. 用 flow 参数化直接恢复单步 `pred_x0_ss`，避免除以 `1-t`：

   ```text
   pred_x0 = (1 - sigma_min) * x_t - sigma_t * pred_v
   ```

3. 冻结 SS decoder 输出 `64^3` occupancy logits。decoder 参数不求梯度，但保留对 `pred_x0_ss` 的输入梯度。
4. 用配对 SLat 坐标构造全体素 occupancy target，计算 BCE 与 soft Dice。
5. 在同一批 SLat latent 上采样一个 flow timestep，冻结教师分别处理：

   - 完整 noisy SLat 输入；
   - 被 SS occupancy gate 扰动的 noisy SLat 输入。

6. 完整分支使用 `torch.no_grad()` 作为稳定目标；扰动分支保持输入梯度，最小化两者 flow prediction 的一致性误差。
7. 总损失为：

   ```text
   L = L_ss_flow
       + lambda_bce * L_occ_bce
       + lambda_dice * L_occ_dice
       + schedule * lambda_slat * L_slat_consistency
   ```

该路径只在蒸馏活动步做两个 SLat 单步前向，不展开 SS/SLat 的完整 25 步采样。默认 `slat_distill_preserve_average=true`，因此每隔 `n` 步计算一次时，活动步权重乘以 `n`，避免仅因降低计算频率就把长期平均蒸馏强度缩小为 `1/n`。如果关闭该选项，`slat_distill_weight` 表示活动步权重，长期平均值会相应降低。

这里假设 feature projection 保存的 SLat `coords` 与 SS 的 64³ target voxelization 使用同一规范化空间和体素约定。首次数据检查必须验证这一点；如果两套坐标来自不同归一化或表面采样规则，不能直接把 SLat coords 当作 SS occupancy target。

## 新增入口

- 数据集：`ImageConditionedFaceScanSLatAwareSparseStructureLatent_ControlNet`
- Trainer：`ImageConditionedSLatAwareSSFlowMatchingCFGTrainer_ControlNet`
- 配置：`configs/generation/ss_flow_finetune_FaceScan_ControlNet_slat_distill.json`
- 静态单元测试：`fine_tuning/tests/test_slat_aware_ss_distillation.py`

冻结的 SS decoder 和 SLat teacher 会在 Trainer 构造时从 `frozen_model_ckpts` 加载，然后从 `BasicTrainer.models` 中移除。因此它们不会进入 DDP、优化器、EMA 或训练 checkpoint；只有 SS/ControlNet denoiser 被训练和保存。

## 数据前置条件

每个训练样本必须同时具备：

- `ss_latents/<ss_model>/<instance>.npz`；
- `latents/<slat_model>/<instance>.npz`，内含 `coords` 和 `feats`；
- metadata 中对应的 `ss_latent_*` 与 `latent_*` 完成标志；
- FaceScan normal condition；
- FaceScan control occupancy。

数据加载时会按实际 SLat 文件检查：坐标为整数且位于 `[0, 63]`、坐标不重复、site 数不超过 `max_num_voxels`、feature 通道数与 teacher 输入一致，以及 normalization 的 mean/std 长度和数值合法。若 metadata 提供 `num_voxels_<slat_model>`，会在构造数据集时提前过滤；否则以加载文件时的强校验为准，不能再把 mesh 的 `num_voxels` 当成 SLat site 数。

当前 FaceScan 数据目录已有一个 16 视角 smoke-test 样本的配对 SLat latent；正式训练前仍需用生产视角数批量完成 feature projection，再运行 `dataset_toolkits/encode_latent.py` 并合并 completion metadata。数据集会在缺少 `latent_<slat_model>` 列时立即报出带处理指引的错误，避免静默训练空数据集。

## 首次有卡验证顺序

RTX 5090 上应显式使用 `SPCONV_ALGO=native`。当前环境的 spconv 2.3.6 在默认 `auto` benchmark 路径中触发过 native `SIGFPE`，切换到项目 README 推荐的 `native` 后，真实 SLat teacher forward/backward 正常：

SLat teacher 引入了比普通 SS 训练更深的 FP16 反向路径。实卡测试发现历史默认的 `inflat_all` 初始 log-scale 20 会在首个蒸馏步溢出，因此该配置显式使用 `fp16_initial_log_scale=12`；其他配置仍保持默认值 20，断点续训时则恢复 checkpoint 保存的动态 scale。

```bash
SPCONV_ALGO=native \
python train.py \
  --config configs/generation/ss_flow_finetune_FaceScan_ControlNet_slat_distill.json \
  --output_dir <output_dir> \
  --data_dir datasets/FaceScan_ControlNet/train \
  --num_gpus 1
```

1. 先执行仅初始化的数据/模型契约检查，确认 paired SLat 数量和坐标范围。
2. 用 batch size 1、`slat_distill_every_n_steps=4` 做单步显存 smoke test；首次应把 step 设置到蒸馏活动步。
3. 检查冻结模型参数梯度始终为空，SS/ControlNet 参数梯度非零。
4. 分别关闭 `slat_distill_weight` 和 occupancy 辅助项做消融。
5. 同时报告 SS IoU/Dice 与最终 mesh 指标；若只提高 SS 指标而最终结果不变，应停止扩大实验。

## 已知风险

- 固定 GT topology 是训练代理，与推理时动态预测 topology 存在差距。
- SLat 一致性只对 GT 候选上的漏检敏感，不能独立处理候选外假阳性。
- `straight_through` 模式梯度有偏；配置默认使用更稳定的 `soft` gate。
- 冻结 SLat 仍需为扰动分支保存输入梯度激活，显存会显著高于普通 SS 微调。
- 训练前期 SS 预测不稳定，因此配置从更新前计数 `step == 500` 开始蒸馏并线性 warmup 1000 步；该次更新完成后的日志 step 为 501。
- 专用 trainer 默认跳过 dataset snapshot，避免启动时把大量 SLat 搬到 GPU，并避免额外加载第二份 SS decoder。需要可视化时可显式设置 `dataset_snapshot_num_samples` 为一个很小的正数；实现会在上 GPU 前移除 `slat_x_0` 并复用冻结 decoder。
- feature gate 是“体素缺失”的连续代理：坐标仍存在于 sparse topology 中，不能把它等同于推理时真实删除 sparse site。
- clean/gated 一致性在所有 gate 为 1 时存在平凡最优点，因此必须与 BCE/Dice-only、简单 positive weighting 做消融，不能只凭一致性 loss 下降判断下游收益。
