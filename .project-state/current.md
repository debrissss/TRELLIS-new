# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat enc/dec 的 kl1e-6、kl1e-7、kl1e-8 三组 1000-step 训练均已完成，训练产物统一保存到 `outputs/train`。
- 已完成固定 eval8/view0 的 KL 局部梯度贡献诊断，三组 KL 对 encoder 总梯度的范数贡献均很小且基本随 `lambda_kl` 线性缩放。
- SLat flow 固定生成评估已改为默认保存 generated/GT PLY。

## Next Actions
1. 将 KL 梯度贡献诊断从 eval8 扩展到 eval50/view0，以提高结论置信度。
2. 对 kl1e-6/kl1e-7/kl1e-8 的 step1000 non-EMA 与 EMA 跑固定 eval50/view0 重建评估。
3. 根据固定重建评估与梯度贡献诊断共同决定哪个权重进入 flow 阶段。

## Relevant Records
- CFG-20260717-116
- EXP-20260720-001
- EXE-20260720-001
- RUN-20260720-002
- AST-20260720-002
