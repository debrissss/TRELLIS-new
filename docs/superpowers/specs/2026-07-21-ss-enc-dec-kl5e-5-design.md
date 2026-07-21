# SS Encoder/Decoder KL 5e-5 微调配置设计

## 目标

将 `configs/vae/ss_enc_dec_fine_tune.json` 的 `trainer.args.lambda_kl` 从 `1e-4` 调整为 `5e-5`，其余训练配置保持不变，并提供一个使用独立输出目录的单卡微调命令。

## 变更范围

- 只修改 `trainer.args.lambda_kl`。
- 不修改模型、数据集、batch、优化器、学习率、训练步数、EMA、FP16、梯度裁剪或初始化 checkpoint。
- 输出目录使用 `outputs/train/ss_enc_dec_fine_tune_kl5e-5`，避免覆盖现有实验。
- 从配置中的官方 encoder/decoder checkpoint 开始微调，不从旧实验续训。

## 验证

- 解析 JSON，确认配置语法有效。
- 检查 `lambda_kl == 5e-5`。
- 检查 diff，确认配置中只有该数值发生变化。

## 微调命令

使用项目现有的 `train.py` 入口、FaceScape 训练集、单 GPU、禁用自动重试，并明确指定不恢复旧训练状态。
