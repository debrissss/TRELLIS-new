# Current State

## Active Goal
构建可靠的固定样本评估工具，用于比较 FaceScape SS encoder + decoder 微调结果，并决定下一轮 KL 权重和 flow 微调方向。

## Current Working Thread
用户已完成 `lambda_kl=0.001`、`5e-4`、`1e-4` 三个 1000-step SS enc/dec fine-tune 运行；当前已新增 `eval/` 工具，用 mini metadata dataset 复用 `SparseStructure`，并输出固定样本 IoU、Dice/F1、occupancy ratio、voxel count 和 trainer-style soft Dice loss。

## Relevant State
- CFG-20260718-001
- CFG-20260718-002
- EXE-20260718-002
- EXE-20260718-003
- RUN-20260718-005
- RUN-20260718-006
- RUN-20260718-007
- RUN-20260718-008
- RUN-20260718-009
- ART-20260718-004
- ART-20260718-005
- ART-20260718-006
- ART-20260717-001

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-ss-enc-dec`。
- 新增 `eval/prepare_ss_eval_dataset.py`：从源 dataset root 的 `metadata.csv` 和 `voxels/` 生成固定样本 mini dataset root。
- 新增 `eval/evaluate_ss_enc_dec_reconstruction.py`：在固定 mini dataset 上评估 SS encoder/decoder checkpoint。
- 新增 `eval/ss_eval_checkpoints.json`：列出 official、`kl1e-3_step1000`、`kl5e-4_step1000`、`kl1e-4_step1000` 四组 encoder/decoder checkpoint。
- 新增 `eval/README.md`：记录固定评估集准备、posterior mean 评估和 sample posterior 评估命令。
- 评估脚本直接复用 `trellis.datasets.SparseStructure`，不重复实现 PLY 到 voxel tensor 的转换。
- 评估脚本默认使用 posterior mean，即 `encoder(ss.float(), sample_posterior=False)`；可通过 `--sample_posterior --seed <seed>` 使用 stochastic posterior。
- 每个样本输出 `iou`、`dice_f1`、`occupancy_ratio`、`gt_occupied_voxels`、`predicted_occupied_voxels`、`intersection_voxels`、`union_voxels`、`soft_dice_loss`。
- `soft_dice_loss` 使用 trainer 中同口径的 sigmoid logits Dice loss 公式，带 `+1` 平滑项。
- 单元测试覆盖 mini dataset 生成、样本不足报错、hard 指标公式、空 GT 边界、summary 忽略 NaN、posterior sampling 开关传递。
- 4 样本 smoke test 验证了 mini dataset 生成、四组 checkpoint deterministic 评估和 official sample posterior 评估路径。
- deterministic 4 样本 smoke 中四组 checkpoint 指标均饱和，hard IoU/Dice 为 `1.0`、`soft_dice_loss=0.0`。
- official sample posterior 4 样本 smoke 中 `iou_mean=0.999956`、`dice_f1_mean=0.999978`、`soft_dice_loss_mean=2.2113e-05`。

## Interpretations
- mini metadata dataset 方案能最大化复用 TRELLIS 现有 `SparseStructure` 数据路径，减少评估代码和训练数据读取逻辑不一致的风险。
- posterior mean 评估可能在 SS VAE 上指标饱和；sample posterior 模式更接近训练 loss 口径，也更容易暴露 latent 分布放松后的随机重建稳定性差异。
- 当前正式模型选择不能只靠 4 样本 smoke；需要生成 64 或更大固定 test mini dataset 后，分别运行 posterior mean 和 sample posterior 两种评估。
- 如果 hard IoU/Dice 在正式集上仍饱和，应优先比较 `soft_dice_loss` 和 sample posterior 模式下的 occupancy/IoU 稳定性。

## Active Hypotheses
- H1: `lambda_kl=1e-4` 已接近当前 SS VAE fine-tune 的合理低 KL 区间。
  Evidence: 训练日志 Dice 均值已从 `2.2405e-05` 降到 `2.4171e-06`，且 436/1000 个 step Dice 小于 `1e-6`；KL 均值升到 `0.449517`。
  Uncertainty: 尚未完成固定 test mini dataset 上的正式 posterior mean 和 sample posterior 评估。
- H2: posterior mean 的 hard voxel 指标可能对当前 SS VAE checkpoint 区分度不足。
  Evidence: 4 样本 smoke 中四组 checkpoint 的 hard IoU/Dice 均为 `1.0`。
  Uncertainty: 4 样本太小，正式 64/256 样本可能仍能暴露差异。
- H3: sample posterior 评估可能更适合判断低 KL 是否破坏 latent 稳定性。
  Evidence: official 4 样本 sample posterior smoke 产生非零 `soft_dice_loss_mean=2.2113e-05`，而 posterior mean 为 `0.0`。
  Uncertainty: 需要四组 checkpoint 在同一固定正式样本集上对比。

## Current Decision State
- Accepted: 固定样本评估使用 mini dataset root，而不是纯 sha list。
- Accepted: 评估代码保留用户指定 hard 指标，并额外输出 trainer-style `soft_dice_loss`。
- Accepted: `lambda_kl=1e-4` 是当前训练日志层面的最佳候选。
- Pending: 正式固定样本评估是否确认 `lambda_kl=1e-4` 优于 `5e-4`。
- Pending: 后续 SS flow 在 `lambda_kl=1e-4` latent 上是否稳定收敛。

## Next Actions
1. 生成正式固定评估集：`datasets/Facescape_ss_eval_test_64`。
2. 运行 posterior mean 评估输出到 `outputs/ss_enc_dec_eval`。
3. 运行 sample posterior 评估输出到 `outputs/ss_enc_dec_eval_sample_posterior`。
4. 根据正式 summary 判断是否保留 `lambda_kl=1e-4` 或继续试 `5e-5`。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。

## Open Questions
- 正式 64 样本 posterior mean hard 指标是否仍全部饱和？
- sample posterior 模式下 `lambda_kl=1e-4` 是否比 `5e-4` 更稳？
