# Experiment Configs

## CFG-20260717-101

Description:
- `cli.py` 的默认图片到 3D 推理参数。
Path:
- `configs/default.yaml`

## CFG-20260717-102

Description:
- 原始 Sparse Structure image-conditioned flow 训练配置。
Path:
- `configs/generation/ss_flow_img_dit_L_16l8_fp16.json`

## CFG-20260717-103

Description:
- FaceScape Sparse Structure Flow 微调配置。
Path:
- `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json`

## CFG-20260717-104

Description:
- FaceScape Sparse Structure Flow 二段微调配置；旧 shell 包装器已删除，需直接通过 `train.py` 或新入口调用。
Path:
- `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape_35000_to_40000.json`

## CFG-20260717-105

Description:
- 原始 SLat image-conditioned flow 训练配置。
Path:
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json`

## CFG-20260717-106

Description:
- FaceScape SLat Flow 微调配置。
Path:
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune_facescape.json`

## CFG-20260717-107

Description:
- FaceScape SLat Flow phase2/from50k 微调配置。
Path:
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune_facescape_phase2_from50k.json`

## CFG-20260717-108

Description:
- FaceScape SS Flow 单样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/ss_flow_facescape_overfit_1.json`

## CFG-20260717-109

Description:
- FaceScape SS Flow 4 样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/ss_flow_facescape_overfit_4.json`

## CFG-20260717-110

Description:
- FaceScape SS Flow 8 样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/ss_flow_facescape_overfit_8.json`

## CFG-20260717-111

Description:
- FaceScape SLat Flow 单样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/slat_flow_facescape_overfit_1.json`

## CFG-20260717-112

Description:
- SLat mesh decoder VAE 配置。
Path:
- `configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json`

## CFG-20260717-113

Description:
- SLat radiance-field decoder VAE 配置。
Path:
- `configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json`

## CFG-20260717-114

Description:
- SLat Gaussian encoder/decoder VAE 配置。
Path:
- `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`

## CFG-20260717-115

Description:
- Sparse Structure VAE 配置。
Path:
- `configs/vae/ss_vae_conv3d_16l8_fp16.json`

## CFG-20260717-116

Description:
- SLat encoder + Gaussian decoder fine-tune 配置；基于原 VAE 配置调整为 1000 step、有效 batch 16、micro-batch 2、lr=1e-5，并使用 8 个 persistent DataLoader worker、启用 trainer prefetch 以减少 GPU 空等。
Path:
- `configs/vae/slat_enc_dec_gs_fine_tune.json`

## CFG-20260718-001

Description:
- FaceScape Sparse Structure encoder + decoder VAE 微调配置；复制自基础 SS VAE 配置并接入官方 SS encoder/decoder `.pt` 初始化权重。
Path:
- `configs/vae/ss_enc_dec_fine_tune.json`

## CFG-20260718-002

Description:
- 固定样本 SS encoder/decoder 重建评估使用的 checkpoint manifest，列出 official 和三个 KL ablation 的 encoder/decoder 权重。
Path:
- `eval/ss_eval_checkpoints.json`

## CFG-20260718-003

Description:
- 固定样本 SS encoder/decoder 重建评估使用的 `lambda_kl=1e-4` step500 和 step1000 checkpoint 对照 manifest。
Path:
- `eval/ss_eval_kl1e-4_steps.json`

## CFG-20260718-004

Description:
- 基于 FaceScape image-conditioned SS flow 微调配置复制的新配置，用于测试 flow 对 `kl1e-4_step1000` SS encoder/decoder latent 分布的适配性。
Path:
- `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`

## CFG-20260722-001

Description:
- TRELLIS image-large 两阶段 image-conditioned pipeline 的组件、采样器和归一化配置。
Path:
- `microsoft/TRELLIS-image-large/pipeline.json`

## CFG-20260722-002

Description:
- TRELLIS normal-conditioned Mesh-only pipeline 的组件、采样器和归一化配置。
Path:
- `microsoft/trellis-normal-v0-1/pipeline.json`
