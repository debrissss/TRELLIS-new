# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

## Relevant State
- EXE-20260717-105
- EXE-20260717-130
- EXE-20260717-131
- EXE-20260717-133
- EXE-20260717-141
- EXE-20260717-142
- EXE-20260717-143
- EXE-20260717-144
- CFG-20260717-103
- CFG-20260717-106
- CFG-20260717-108
- CFG-20260717-111
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06
- EVT-20260718-000000-07
- EVT-20260718-000000-08
- EVT-20260718-000000-09
- EVT-20260718-000000-10
- EVT-20260718-000000-11

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- 用户报告 batch16 配置在 `dataloader_num_workers=2` 后速度比 workers=0 快一倍多；workers=4 与 workers=2 差别不大，prefetch=true 后 GPU 低占用频率没有明显减少，workers=8/persistent=false 也没有好转。当前已设置 `dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`。
- 用户报告 workers=8、persistent=true、prefetch=true 后 GPU 低占用次数约减少 30%，但仍然频繁。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- batch16 的 DataLoader 调参已有改善但未根治 GPU starvation，继续加 worker 的边际收益可能有限；需要比较总体 wall time 和验证质量，而不只看瞬时 GPU 利用率。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置的 GPU starvation 不是单靠 worker/persistent/prefetch 能完全解决。
  Evidence: workers=8、persistent=true、prefetch=true 只让低占用次数减少约 30%，仍频繁出现。
  Uncertainty: batch16 的验证质量是否足以抵消数据管线成本尚未确定。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 记录 batch16 当前配置的每 100 step wall time、最后 100 step loss 和 final/fixed validation sample，再与 RUN-20260718-001 的 batch8/lr1e-5 比较。
2. 若 batch16 没有明显质量优势，回退 RUN-20260718-001 的 batch8/lr1e-5 配置；若质量明显更好，再考虑数据预处理/缓存方案而不是继续盲目加 worker。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？
