# 3D ControlNet 部署打包契约

本专用部署配置使用
`ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8` 的 step-2000 encoder 作为
ControlNet 条件编码器。因此稀疏结构 decoder 必须来自同一训练 run、同一步的
`decoder_step0002000.pt`。旧 TRELLIS decoder 即使结构和 shape 相同，也不代表
latent space 等价，不能作为默认替代品。

## 必需产物

部署目录至少包含：

- `pipeline.json`：复制
  `configs/pipelines/trellis_image_to_3d_ControlNet.json`。
- `ckpts/ss_flow_ControlNet.json` 与 `.safetensors`：使用
  `fine_tuning/convert_pt_to_safetensors_ControlNet.py` 转换。JSON 不得包含
  `control_encoder_ckpt`，safetensors 必须包含完整 `control_encoder.*` 权重。
- `ckpts/ss_dec_fine_tune_step2000_ControlNet.json` 与 `.safetensors`：必须由
  同一 run 的 `decoder_step0002000.pt` 转换。仓库内同名 JSON 是明确 latent
  配套关系的部署模板；通用转换器会从 VAE 训练配置写出等价模型结构 JSON。

ControlNet flow 转换示例：

```bash
python fine_tuning/convert_pt_to_safetensors_ControlNet.py \
  --input <完整 ControlNet checkpoint.pt> \
  --train_config configs/generation/ss_flow_finetune_ControlNet.json \
  --output_prefix <部署目录>/ckpts/ss_flow_ControlNet
```

配套 decoder 可继续使用项目原有通用转换器（它不需要 ControlNet 特殊清理）：

```bash
python fine_tuning/convert_pt_to_safetensors.py \
  --input outputs/train/ss_enc_dec_fine_tune_kl1e-4_lr1e-6_batch8/ckpts/decoder_step0002000.pt \
  --train_config configs/vae/ss_enc_dec_fine_tune.json \
  --model_key decoder \
  --output_prefix <部署目录>/ckpts/ss_dec_fine_tune_step2000_ControlNet
```

转换脚本的严格加载只能验证结构和权重完整性；仍需在有资源环境用真实 encoder、
decoder 和 ControlNet artifact 做 latent 配套及端到端采样验证。
