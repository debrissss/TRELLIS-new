# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持当前 FaceScape SLat encoder + GS decoder fine-tune 的成本/速度对比。

## Current Working Thread
用户正在评估更贵 GPU 的速度收益是否能覆盖成本。当前已提交并推送项目状态，同时准备了一个约 50GB 的 FaceScape SLat GS 训练子集，供低配置机器测试训练吞吐。

## Relevant State
- EXE-20260717-105
- EXE-20260718-001
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
- ART-20260718-001
- ART-20260718-002
- ART-20260718-003
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- EVT-20260718-120400-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/track-untracked-state`。
- 2026-07-18 已提交并推送 commit `837e3f9 Add SLat GS fine-tune config and logs`。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 当前为 batch16 对照配置：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`。
- 用户报告 batch16 在当前 DataLoader 设置下约为 1700 steps/h。
- `outputs/slat_enc_dec_gs_fine_tune_v2` 是已完成的 batch8/lr1e-5 1000-step 对照；最后 100 step 平均 loss 为 0.0208222。
- `outputs/slat_enc_dec_gs_fine_tune_v3` 记录了 batch16 早期因 DataLoader shared memory bus error 失败的输出。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`。
- 该子集的 `train/metadata.csv` 为 1178 个样本加表头，包含 1178 个 `renders/<sha>/` 目录和 1178 个 `features/dinov2_vitl14_reg/<sha>.npz` 文件。
- 一致性检查确认子集 metadata 中每个样本都有 feature 文件和 `renders/<sha>/transforms.json`。
- 该子集不包含 `voxels/`、`renders_cond/` 或预训练 `.pt` checkpoint。

## Interpretations
- SLat encoder + Gaussian decoder 训练数据路径需要 metadata、render 图像/相机 transforms、DINOv2 patch token feature；当前子集覆盖这些必要输入。
- 当前低配测速的关键指标应同时看 `steps/h` 和 `samples/h`：batch16 的 1700 steps/h 约等于 27200 samples/h。
- 若低配机器跑 batch8 或 batch16，需要按有效 batch 统一换算样本吞吐，否则只比较 GPU 利用率或 steps/h 容易误判成本收益。

## Active Hypotheses
- H1: batch16 的吞吐优势主要来自每 step 样本数更大，但样本吞吐与 batch8 可能接近。
  Evidence: 用户报告 batch16 约 1700 steps/h；先前 batch8 约可换算到相近 samples/h 量级。
  Uncertainty: 低配机器上的 CPU/I/O、显存和 `/dev/shm` 瓶颈可能改变这个关系。
- H2: 低配机器若复用 batch16 配置，可能先受显存或 DataLoader 共享内存限制。
  Evidence: 当前机器 batch16 曾触发 DataLoader shm bus error；`batch_split` 不降低 DataLoader 完整 batch 压力。
  Uncertainty: 另一台机器的 `/dev/shm`、CPU 核数、磁盘速度和 PyTorch worker 行为未知。

## Current Decision State
- Accepted: 为低配机器准备约 50GB 子集，而不是搬运完整约 441GB FaceScape 数据集。
- Accepted: 子集只复制当前 SLat GS 训练读取的数据，不复制 `voxels/` 和 `renders_cond/`。
- Accepted: 避免在当前机器实际启动训练或压力检查，以免再次触发内存/共享内存问题影响其它任务。
- Pending: 低配机器应使用 batch8 还是 batch16 做第一轮速度测试，取决于其显存和 `/dev/shm`。

## Next Actions
1. 将 `datasets/Facescape_slat_gs_50gb` 同步到低配置机器。
2. 同步 TRELLIS 代码、`configs/vae/slat_enc_dec_gs_fine_tune.json`、以及 SLat encoder/GS decoder `.pt` 预训练权重。
3. 在低配机器先用 `--auto_retry 0` 跑短程测试，记录 steps/h、samples/h、GPU 利用率、显存峰值和是否出现 DataLoader bus error。
4. 用统一的 samples/h 与单位小时成本比较当前昂贵 GPU 和低配机器的实际性价比。

## Constraints
- 不启动训练或重型数据检查。
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 迁移子集时需要保留 `train/metadata.csv`、`train/renders/` 和 `train/features/dinov2_vitl14_reg/` 的相对路径结构。

## Open Questions
- 低配置机器的 GPU 显存、CPU 核数、磁盘类型和 `/dev/shm` 大小是多少？
- 低配机器上是否已经有 `microsoft/TRELLIS-image-large/ckpts/*.pt` 微调初始化权重？
