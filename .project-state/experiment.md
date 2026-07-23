# Experiments

## EXP-20260720-001 - SLat enc/dec lambda_kl=1e-8 训练结果

Description:
- FaceScape SLat encoder + Gaussian decoder 使用 `lambda_kl=1e-8` 完成 1000-step 微调后的训练日志分析。

Experiment:
- 使用 `configs/vae/slat_enc_dec_gs_fine_tune.json` 将 `lambda_kl` 从 `1e-7` 降到 `1e-8`，其余主要训练设置保持为 1000 steps、batch size per GPU 16、batch split 8、AdamW lr=1e-5。
- 训练输出目录为 `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-8`，命令记录为 `python train.py --config configs/vae/slat_enc_dec_gs_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/train/slat_enc_dec_gs_fine_tune_kl1e-8 --num_gpus 1 --auto_retry 0`。
- 本次训练保存了 step500/step1000 的 encoder、decoder、EMA 和 misc checkpoint，并生成 init/final reconstruction sample。

Results:
- 训练完整跑到 step 1000，`loss` 与 `log` 均有 1000 条记录，`failed` 或中断迹象未在输出目录中发现。
- final step: loss 0.0208826，rec 0.0208122，L1 0.00403639，SSIM loss 0.0424668，LPIPS 0.0414125，KL 9.93760，grad_norm 0.0425923。
- last100 mean: loss 0.0204808，rec 0.0204094，L1 0.00417656，SSIM loss 0.0426094，LPIPS 0.0385549，KL 9.91137，KL contribution 9.91e-8，grad_norm 0.0384816。
- first100 到 last100: loss 下降 8.54%，rec 下降 8.56%，LPIPS 下降 13.80%，grad_norm 下降 38.63%。
- 总 elapsed 约 2095.85 秒，last100 平均 step time 约 2.0105 秒；final reconstruction sample 非空，能看到完整头颈结构。
- 与 `lambda_kl=1e-7` 训练日志同口径比较：last100 mean loss 低约 2.99e-6，last100 mean rec 低约 2.29e-6，last100 mean LPIPS 低约 1.13e-5，last100 mean KL 低约 0.0153；这些差异都非常小。

Analysis:
- 观察：`lambda_kl=1e-8` 没有在 1000-step 训练日志中表现出崩溃、KL 暴涨或重建明显异常，训练趋势与 `lambda_kl=1e-7` 基本重合。
- 解释：由于 KL 加权贡献已经从 `1e-7` 时约 9.93e-7 降到约 9.91e-8，而总 loss 约 2.05e-2，本轮训练实际上几乎完全由重建项和 regularization 主导；日志层面的微小优势不足以证明 `1e-8` 比 `1e-7` 更好。
- 不确定性：当前结论只基于训练日志和训练快照，不能判断泛化质量；必须用固定 eval50/view0 对 step1000 non-EMA 与 EMA checkpoint 做重建评估，再与 `kl1e-7` 的固定评估结果比较。
- 建议：优先评估 non-EMA step1000；若 fixed eval 与 `kl1e-7` 持平或更好，再考虑用它编码一小批 latent 并测试后续 SLat flow，否则继续沿用 `kl1e-7` 作为更保守选择。

Related records:
- CFG-20260717-116
- EXE-20260717-105
- AST-20260718-004
- RUN-20260719-002

## EXP-20260720-002 - SLat enc/dec lambda_kl=1e-6 batch8 与 batch16 训练日志对比

Description:
- FaceScape SLat encoder + Gaussian decoder 在 `lambda_kl=1e-6` 下对比有效 batch16 与有效 batch8 的 1000-step 训练日志。

Experiment:
- batch16 对照使用 `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6`，配置为 `batch_size_per_gpu=16`、`batch_split=8`、AdamW lr=1e-5、1000 steps、`lambda_kl=1e-6`。
- batch8 试验使用 `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8`，配置为 `batch_size_per_gpu=8`、`batch_split=4`、AdamW lr=1e-5、1000 steps、`lambda_kl=1e-6`。
- 两组训练都使用 `datasets/Facescape/train`，并保存 step500/step1000 的 encoder、decoder、EMA 和 misc checkpoint。

Results:
- batch16 final step: loss 0.0208858，rec 0.0208061，L1 0.00403758，SSIM loss 0.0424877，LPIPS 0.0413547，KL 9.65429。
- batch8 final step: loss 0.0248300，rec 0.0247483，L1 0.00493863，SSIM loss 0.0528900，LPIPS 0.0461586，KL 9.70999。
- batch16 last100 mean: loss 0.0204891，rec 0.0204084，L1 0.00417652，SSIM loss 0.0426043，LPIPS 0.0385552，KL 9.66271，weighted KL 9.66e-6。
- batch8 last100 mean: loss 0.0206530，rec 0.0205708，L1 0.00416704，SSIM loss 0.0427627，LPIPS 0.0392563，KL 9.73487，weighted KL 9.73e-6。
- batch16 last100 step time 均值约 2.0001 秒，batch8 last100 step time 均值约 1.0061 秒；batch8 单 step 时间约为 batch16 的 50.3%。
- batch16 last100 loss std 0.002411，rec std 0.002410；batch8 last100 loss std 0.002956，rec std 0.002954，batch8 波动更大。
- batch16 last100 grad_norm 均值约 0.03790，batch8 last100 grad_norm 均值约 0.05523。

Analysis:
- 观察：batch8 能稳定完成 1000 steps，吞吐符合预期地约为 batch16 的两倍 step 数速度，但因为有效 batch 减半，若按样本数预算对齐，其实际样本吞吐优势不明显。
- 观察：batch8 的 last100 训练均值没有优于 batch16；loss、rec、SSIM loss、LPIPS 均略高，只有 L1 略低约 9.48e-6，差异很小。
- 解释：batch8 的单点 final step 明显差于 batch16，主要可能包含较强 batch 抽样波动；last100 更可靠，显示 batch8 与 batch16 接近但略不占优，且 batch8 loss/rec 标准差和 grad_norm 更高。
- 不确定性：该记录只比较训练日志，不能证明泛化质量；最终应以固定 eval50/view0 的 non-EMA/EMA 重建指标和可视化为准。
- 建议：如果显存允许，当前日志证据更支持继续使用 batch16；若要公平评估 batch8，应补跑固定 eval50/view0，并考虑按样本数预算而不是 step 数预算比较。

Related records:
- CFG-20260717-116
- EXE-20260717-105
- none

## EXP-20260722-001 - SLat mesh decoder 微调前后几何评估

Description:
- 在 FaceScape test 固定 50 个样本上，对比 SLat mesh decoder 微调前权重与 non-EMA step400/600/800/1000 权重的 Stable3DGen 对齐 PLY 几何评估结果。

Experiment:
- 评估目录：`outputs/eval/slat_dec_mesh_fine_tune`。
- 汇总文件：`outputs/eval/slat_dec_mesh_fine_tune/metrics/pretrain_step400_600_800_1000_compare.csv`。
- 评估样本：FaceScape test 固定 50 个样本，所有 checkpoint 使用同一 `eval_samples.json`。
- 输入 latent：`datasets/Facescape/test/latents/dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-6_batch8_step0004000`。
- 对比权重：微调前 `microsoft/trellis-normal-v0-1/ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.pt`，以及 `outputs/train/slat_dec_mesh_fine_tune/ckpts/decoder_step0000400.pt`、`decoder_step0000600.pt`、`decoder_step0000800.pt`、`decoder_step0001000.pt`。
- 推理与导出：使用 Stable3DGen 的 `SLatMeshDecoder` 结构 strict load 权重，导出路径对齐 Stable3DGen 最终推理逻辑，输出 PLY 后再 reload 验证。
- 几何指标：每样本固定 seed 表面采样 50000 点，统计 Chamfer L1/L2、F-score、normal consistency、mesh 顶点/面数、components、watertight、surface area 等。

Results:
- 所有权重 `success_rate=1.0`，50/50 样本均成功导出和评估。
- 微调前权重：`chamfer_l1_mean=0.0096572154`，`chamfer_l2_mean=6.07682877e-05`，`fscore_0p005_mean=0.5735136931`，`fscore_0p01_mean=0.9531871832`，`fscore_0p02_mean=0.9999885985`，`normal_consistency_mean=0.9364577567`，`pred_components_mean=2.26`。
- step400：`chamfer_l1_mean=0.0102062671`，比微调前高 `+5.69%`；`chamfer_l2_mean=7.04540768e-05`，高 `+15.94%`；`fscore_0p01_mean=0.9041122730`，低 `-5.15%`；`normal_consistency_mean=0.9223742495`，低 `-1.50%`；`pred_components_mean=5.02`，高 `+122.12%`。
- step600：`chamfer_l1_mean=0.0102281568`，高 `+5.91%`；`chamfer_l2_mean=7.09439392e-05`，高 `+16.75%`；`fscore_0p01_mean=0.9017152894`，低 `-5.40%`；`normal_consistency_mean=0.9222980512`，低 `-1.51%`；`pred_components_mean=5.28`，高 `+133.63%`。
- step800：`chamfer_l1_mean=0.0102615691`，高 `+6.26%`；`chamfer_l2_mean=7.16350793e-05`，高 `+17.88%`；`fscore_0p01_mean=0.8980332554`，低 `-5.79%`；`normal_consistency_mean=0.9212534838`，低 `-1.62%`；`pred_components_mean=4.04`，高 `+78.76%`。
- step1000：`chamfer_l1_mean=0.0102581382`，高 `+6.22%`；`chamfer_l2_mean=7.14236472e-05`，高 `+17.53%`；`fscore_0p01_mean=0.8988693333`，低 `-5.70%`；`normal_consistency_mean=0.9217675529`，低 `-1.57%`；`pred_components_mean=4.06`，高 `+79.65%`。
- per-sample 检查：50/50 个共同样本上，step400/600/800/1000 相比微调前在 `chamfer_l1`、`chamfer_l2`、`fscore_0p005`、`fscore_0p01`、`normal_consistency` 上均为更差方向，没有发现是少数异常样本拉动均值。

Analysis:
- 当前评估结果明确支持：在这套固定 test latent 与几何指标下，mesh decoder 微调后的 non-EMA 权重没有优于微调前权重，且随着 step 增加整体略有退化。
- 退化具有一致性：不是单个 checkpoint 的随机波动，也不是少数样本异常；多个关键指标和 50 个样本方向一致。
- 微调后 `pred_components_mean` 明显升高，说明预测 mesh 更容易出现碎片/多连通块，这对最终可用 mesh 质量是不利信号。
- `fscore_0p02` 基本接近 1.0，说明粗尺度形状覆盖仍然对齐；问题主要体现在更细阈值、Chamfer、normal consistency 和拓扑碎片上。
- 可靠性限制：该评估使用的是固定 encoder 生成的 latent，因此结论严格对应“当前 step4000 SLat enc/dec latent 输入下的 mesh decoder 表现”；若后续更换 encoder 或 flow 输出分布，应重新评估。
- 训练 loss 曾小幅下降，但 test 几何评估变差，说明当前 mesh decoder 微调可能更贴合训练渲染损失/训练分布，未转化为 test 几何质量收益。
- 当前建议：暂不采用 step400/600/800/1000 微调 mesh decoder 权重替换微调前权重；下一步应优先检查训练/评估坐标系与 GT mesh 对齐、训练 loss 对高精度 FaceScape 表面细节的约束是否与 Chamfer/F-score 目标一致，再考虑更低学习率、更短训练、冻结部分模块或调整 mesh decoder 损失权重。

Related records:
- EXE-20260722-001
- EXE-20260722-002

## EXP-20260720-004 - SLat enc/dec lambda_kl=1e-6 batch8 3000steps 训练结果

Description:
- FaceScape SLat encoder + Gaussian decoder 使用 `lambda_kl=1e-6`、有效 batch8 继续训练到 3000 steps 后的训练日志检查。

Experiment:
- 训练输出目录为 `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8`。
- 配置为 `batch_size_per_gpu=8`、`batch_split=4`、`max_steps=3000`、`dataloader_num_workers=16`、`dataloader_persistent_workers=true`、`prefetch_data=true`、AdamW lr=1e-5、`lambda_kl=1e-6`。
- 本轮从已有 batch8 目录的 latest checkpoint 继续训练，命令记录为 `python train.py --config configs/vae/slat_enc_dec_gs_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8 --load_dir outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8 --ckpt latest --num_gpus 1 --auto_retry 0`。

Results:
- 训练完整跑到 step 3000，`loss` 与 `log` 均有 3000 条记录，step3000 的 encoder、decoder、EMA 和 misc checkpoint 均存在。
- final sample 图片存在且尺寸正常：GT/rec 为 512x512 RGB，multi-view 为 1024x1024 RGB。
- step1801-2000 mean: loss 0.020303295，rec 0.020221451，L1 0.004104527，SSIM loss 0.042541075，LPIPS 0.038043542，KL 9.487722，loss std 0.003100521。
- step2501-3000 mean: loss 0.020334614，rec 0.020252384，L1 0.004141742，SSIM loss 0.042867599，LPIPS 0.037685609，KL 9.369088，loss std 0.003454340。
- step2901-3000 mean: loss 0.020254915，rec 0.020172265，L1 0.004136540，SSIM loss 0.042619569，LPIPS 0.037559052，KL 9.322321，loss std 0.002890484。
- final step 单点: loss 0.023834061，rec 0.023752309，L1 0.004689223，SSIM loss 0.053486988，LPIPS 0.041828441，KL 9.364864。
- step2901-3000 mean step time 约 1.028970 秒；相比 step1801-2000 的约 1.041682 秒，`dataloader_num_workers=16` 未显示出显著吞吐提升。

Analysis:
- 观察：继续训练到 3000 steps 后没有发现训练崩坏或非有限 loss，KL 和 LPIPS 继续下降，最近 100 step 的 loss/rec 也略优于 step1801-2000 窗口。
- 观察：L1 与 SSIM loss 相比 step1801-2000 略差，且 step2501-3000 的 loss std 更高，说明后段收益变小且波动增加。
- 解释：step3000 训练结果仍有一定收益，但主要体现在感知项 LPIPS 和 KL；像素级/结构项没有全面同步改善，因此不能仅凭训练日志决定继续训练。
- 不确定性：final step 单点偏高，说明单个 batch 波动仍明显；最终权重选择必须依赖固定 eval50/view0，而不是最后一行训练 loss。
- 建议：先停止盲目继续训练，优先评估 batch8 step3000 non-EMA/EMA，并与 batch8 step2000、batch16 step1000 的固定 eval 结果比较；若 step3000 fixed eval 仍稳定优于 step2000，再考虑继续到 4000 steps。

Related records:
- CFG-20260717-116
- EXE-20260717-105
- none

## EXP-20260720-003 - SLat enc/dec lambda_kl=1e-6 batch8 2000steps 与 batch16 1000steps 公平对比

Description:
- FaceScape SLat encoder + Gaussian decoder 在 `lambda_kl=1e-6` 下按同样 16000 个训练样本预算对比 batch8 2000steps 与 batch16 1000steps。

Experiment:
- batch16 对照使用 `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6`，配置为 `batch_size_per_gpu=16`、`batch_split=8`、AdamW lr=1e-5、1000 steps、`lambda_kl=1e-6`。
- batch8 试验使用 `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6_batch8`，配置为 `batch_size_per_gpu=8`、`batch_split=4`、AdamW lr=1e-5、2000 steps、`lambda_kl=1e-6`。
- 训练集过滤后共有 6450 个样本；batch16 1000steps 与 batch8 2000steps 都对应 16000 个样本，约 2.48 epochs。
- 公平统计窗口使用 batch16 step901-1000 与 batch8 step1801-2000，两者都覆盖最近约 1600 个训练样本。

Results:
- batch16 公平窗口 mean: loss 0.020489135，rec 0.020408426，L1 0.004176522，SSIM loss 0.042604298，LPIPS 0.038555224，KL 9.662710066。
- batch8 公平窗口 mean: loss 0.020303295，rec 0.020221451，L1 0.004104527，SSIM loss 0.042541075，LPIPS 0.038043542，KL 9.487722297。
- batch8 相对 batch16: loss -0.000185839，rec -0.000186975，L1 -0.000071994，SSIM loss -0.000063223，LPIPS -0.000511683，KL -0.174987769。
- batch16 公平窗口 loss std 0.002411358、rec std 0.002410183；batch8 公平窗口 loss std 0.003100521、rec std 0.003098195。
- batch16 公平窗口 grad_norm mean 0.037900334，batch8 公平窗口 grad_norm mean 0.054160205。
- batch16 公平窗口 step time mean 约 2.000108 秒，batch8 公平窗口 step time mean 约 1.041682 秒。

Analysis:
- 观察：按同样 16000 个样本预算和同样 1600 个样本统计窗口比较，batch8 2000steps 的训练均值已经略优于 batch16 1000steps，重建相关指标和 KL 均更低。
- 观察：batch8 的 loss/rec 标准差和 grad_norm 明显高于 batch16，说明小 batch 带来的梯度噪声与训练波动仍更强。
- 解释：该对比按样本数公平，但不按 optimizer update 次数公平；batch8 做了 2000 次参数更新，batch16 只做了 1000 次，因此 batch8 的优势可能来自更频繁的小步更新，而不只是 batch size 本身。
- 不确定性：该记录只基于训练日志；由于训练数据视角在 `SparseFeat2Render` 中随机采样，batch8 与 batch16 并非完全相同输入序列。
- 建议：下一步用同一固定 eval50/view0 对 batch16 step1000 与 batch8 step2000 的 non-EMA/EMA checkpoint 做重建评估；如果 batch8 在固定 eval 上仍优于 batch16，才可认为 batch8 2000steps 是更好的微调选择。

Related records:
- CFG-20260717-116
- EXE-20260717-105
- none
