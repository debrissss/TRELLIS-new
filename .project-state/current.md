# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

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
- ART-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- 已新增 `eval/` 评估工具：固定 FaceScape eval 子集准备、SLat enc/dec checkpoint 重建评估、多 run summary 对比。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。
- 固定 eval 子集流程可以避免训练 DataLoader 的随机视角和随机 batch 噪声，适合作为不同 KL 权重与 EMA/non-EMA checkpoint 的选择依据。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。

## Next Actions
1. 用固定 test/holdout 样本评估 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 的 step1000 与 EMA step1000。
2. 对比本次 `lambda_kl=1e-7` 与此前 v2 `lambda_kl=1e-6` 的固定样本重建质量，而不只看训练日志均值。
3. 若验证质量稳定，准备 SLat flow 人脸域微调配置，明确使用哪个 encoder/decoder checkpoint 生成或解码 latent。
4. 低配机器测速时继续记录统一口径的稳定段 samples/h、端到端 samples/h 与单位小时成本。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。

## Open Questions
- 本次 step1000 EMA 与 non-EMA 哪个在固定验证集上更好？
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？
