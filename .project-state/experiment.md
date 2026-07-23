# Experiments

## EXP-20260721-001 - FaceScape SS flow 5000-step fine-tune

Description:
- 使用 KL=1e-4、step2000 SS encoder latent 对官方 normal SS flow 权重进行 5000 steps FaceScape 单卡微调。

Experiment:
- 输出目录：`outputs/train/ss_flow_finetune`；启动参数显式使用 `--ckpt none --auto_retry 0 --num_gpus 1`。
- 初始化权重：`microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.pt`。
- latent：`ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8_step0002000`；decoder：`outputs/train/ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8/ckpts/decoder_step0002000.pt`。
- 训练配置：`max_steps=5000`、`batch_size_per_gpu=16`、`batch_split=8`、AdamW `lr=1e-5`、`ema_rate=0.999`、`i_log=100`、`i_sample=20000`、`i_save=500`。
Results:
- 日志完整覆盖 step 1-5000，loss 与 grad 均未发现 NaN/Inf；step500 至 step5000 每 500 steps 的 raw denoiser、EMA denoiser 和 misc checkpoint 均齐全。
- 训练耗时 13242.46 秒（约 3.68 小时），除首步外中位 step 时间 2.636 秒，单卡吞吐约 6.07 samples/s。
- 前 100 steps 平均 loss 为 0.280922，后 100 steps 为 0.236871，下降 15.68%；前/后 500 steps 平均 loss 为 0.264629/0.238593，下降 9.84%；最终 step5000 loss 为 0.223324。
- 最佳 100-step 滚动均值为 0.236379（结束于 step4954）；最后 1000 steps 的线性趋势仍为下降，估算每 1000 steps 下降约 0.00315。
- 5 个 step 的 grad_norm 日志字段缺失（641、1012、2494、2551、3782），但对应 loss 有效且训练连续完成；其余记录未见数值发散。
- init/final 条件样本均已输出。final 样本呈现合理的粗粒度头颈/胸肩占用轮廓，但与配对 GT 仍存在明显局部形状差异；仅一组 final 样本不足以判断泛化质量。
- 输出目录占用约 84 GB；记录时数据盘仅剩约 15 GB。
Analysis:
- 训练过程稳定且截至 step5000 仍有缓慢下降趋势，没有从训练 loss 观察到明显发散或过拟合拐点。
- 训练 loss 不能单独确定最佳生成 checkpoint；应在固定 FaceScape test 子集上比较 step1000-5000 的 raw/EMA 权重，并以 SS occupancy IoU/Dice、Chamfer/表面距离及条件一致性为主进行选择。
- `i_sample=20000` 导致训练中没有周期采样，只有初始化和结束钩子的样本，因此当前视觉证据有限；继续训练前还需处理 checkpoint 占用，避免磁盘耗尽。
Related records:
- EXP-20260720-001

## EXP-20260720-001 - SS flow kl1e-4 step1000 sampling gate

Description:
- 评估 `kl=1e-4` SS latent 分布在 SS flow step1000 采样阶段是否具备基础可用性。

Experiment:
- 使用 `outputs/train/ss_flow_finetune_kl1e-4_step1000/ckpts/denoiser_step0001000.pt` 和 `outputs/train/ss_enc_dec_fine_tune_kl1e-4/ckpts/decoder_step0001000.pt`，在 `datasets/Facescape/test` 上固定 seed `20260720` 抽取 16 个样本进行 condition image -> SS flow -> SS decoder -> voxel 采样评估，并导出 `pred.ply`、`gt.ply`、`cond.png`。
Results:
- `num_samples=16`，`empty_pred_count=0`，`overfull_pred_count=0`。
- `occupancy_ratio` mean `1.027656`，median `1.005559`，min `0.880648`，max `1.196455`。
- predicted occupied voxels mean `11754.9375`，GT occupied voxels mean `11478.0`。
- IoU mean `0.218463`，median `0.229206`，min `0.052059`，max `0.383279`；Dice/F1 mean `0.348896`，median `0.372890`。
- 输出目录包含 `per_sample_metrics.csv`、`summary.csv`、`summary.json`、`eval_config.json`，以及 16 组 `pred.ply`/`gt.ply`/`cond.png`；已校验 `pred.ply` vertex count 与 `predicted_occupied_voxels` 一致。
Analysis:
- 观察：采样结果没有出现空体或满体，预测 occupancy 数量级与 GT 非常接近，说明 `kl=1e-4` latent 分布对 SS flow step1000 的基础采样可用性通过。
- 解释：该结果支持继续把 `kl=1e-4` 用于后续 flow 链路；当前问题不再是 KL 分布导致 flow 崩坏。
- 不确定性：IoU/Dice 较低，说明 1000-step flow 的条件对齐或逐体素形状匹配还不强；该实验不能证明最终生成质量已经足够，也不能替代更多样本或下游 SLat/mesh 评估。
Related records:
- RUN-20260720-002
- CFG-20260718-004
- EXE-20260720-001
- AST-20260718-022
- AST-20260720-001
