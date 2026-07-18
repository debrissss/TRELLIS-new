# Experiment Configs

## CFG-20260717-101 - default CLI config

Description:
- `cli.py` 的默认图片到 3D 推理参数。
Path:
- `configs/default.yaml`
Format:
- yaml
Related Entrypoints:
- EXE-20260717-104
Last observed:
- 2026-07-17

## CFG-20260717-102 - ss flow image base config

Description:
- 原始 Sparse Structure image-conditioned flow 训练配置。
Path:
- `configs/generation/ss_flow_img_dit_L_16l8_fp16.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-103 - FaceScape SS flow finetune config

Description:
- FaceScape Sparse Structure Flow 微调配置。
Path:
- `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-104 - FaceScape SS flow 35000_to_40000 config

Description:
- FaceScape Sparse Structure Flow 二段微调配置；旧 shell 包装器已删除，需直接通过 `train.py` 或新入口调用。
Path:
- `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape_35000_to_40000.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-105 - slat flow image base config

Description:
- 原始 SLat image-conditioned flow 训练配置。
Path:
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-106 - FaceScape SLat flow finetune config

Description:
- FaceScape SLat Flow 微调配置。
Path:
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune_facescape.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-107 - FaceScape SLat flow phase2 config

Description:
- FaceScape SLat Flow phase2/from50k 微调配置。
Path:
- `configs/generation/slat_flow_img_dit_L_64l8p2_fp16_finetune_facescape_phase2_from50k.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-108 - SS overfit 1 config

Description:
- FaceScape SS Flow 单样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/ss_flow_facescape_overfit_1.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-109 - SS overfit 4 config

Description:
- FaceScape SS Flow 4 样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/ss_flow_facescape_overfit_4.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-110 - SS overfit 8 config

Description:
- FaceScape SS Flow 8 样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/ss_flow_facescape_overfit_8.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-111 - SLat overfit 1 config

Description:
- FaceScape SLat Flow 单样本 overfit 配置；旧准备/训练包装器已删除。
Path:
- `configs/generation/overfit/slat_flow_facescape_overfit_1.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-112 - slat mesh decoder VAE config

Description:
- SLat mesh decoder VAE 配置。
Path:
- `configs/vae/slat_vae_dec_mesh_swin8_B_64l8_fp16.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-113 - slat radiance-field decoder VAE config

Description:
- SLat radiance-field decoder VAE 配置。
Path:
- `configs/vae/slat_vae_dec_rf_swin8_B_64l8_fp16.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-114 - slat gaussian encoder-decoder VAE config

Description:
- SLat Gaussian encoder/decoder VAE 配置。
Path:
- `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-115 - sparse structure VAE config

Description:
- Sparse Structure VAE 配置。
Path:
- `configs/vae/ss_vae_conv3d_16l8_fp16.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260717-116 - SLat encoder plus GS decoder fine-tune config

Description:
- SLat encoder + Gaussian decoder fine-tune 配置；基于原 VAE 配置调整为 1000 step、有效 batch 16、micro-batch 2、lr=1e-5，并使用 8 个 persistent DataLoader worker、启用 trainer prefetch 以减少 GPU 空等。
Path:
- `configs/vae/slat_enc_dec_gs_fine_tune.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Last observed:
- 2026-07-17

## CFG-20260718-001 - SS encoder plus decoder fine-tune config

Description:
- FaceScape Sparse Structure encoder + decoder VAE 微调配置；复制自基础 SS VAE 配置并接入官方 SS encoder/decoder `.pt` 初始化权重。
Path:
- `configs/vae/ss_enc_dec_fine_tune.json`
Format:
- json
Related Entrypoints:
- EXE-20260717-105
Material Parameter Summary:
- `models.encoder.name: SparseStructureEncoder`
- `models.decoder.name: SparseStructureDecoder`
- `dataset.name: SparseStructure`
- `dataset.args.min_aesthetic_score: 4.5`
- `trainer.name: SparseStructureVaeTrainer`
- `trainer.args.max_steps: 1000`
- `trainer.args.batch_size_per_gpu: 16`
- `trainer.args.batch_split: 4`
- `trainer.args.optimizer.args.lr: 1e-5`
- `trainer.args.lambda_kl: 0.001 -> 5e-4 -> 1e-4`
- `trainer.args.finetune_ckpt.encoder: microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.pt`
- `trainer.args.finetune_ckpt.decoder: microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.pt`
- `trainer.args.i_print: 10`
- `trainer.args.i_save: 500`
Last observed:
- 2026-07-18

## CFG-20260718-002 - SS encoder plus decoder evaluation checkpoint manifest

Description:
- 固定样本 SS encoder/decoder 重建评估使用的 checkpoint manifest，列出 official 和三个 KL ablation 的 encoder/decoder 权重。
Path:
- `eval/ss_eval_checkpoints.json`
Format:
- json
Related Entrypoints:
- EXE-20260718-003
Material Parameter Summary:
- `official.encoder: microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.pt`
- `official.decoder: microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.pt`
- `kl1e-3_step1000.encoder: outputs/ss_enc_dec_fine_tune/ckpts/encoder_step0001000.pt`
- `kl1e-3_step1000.decoder: outputs/ss_enc_dec_fine_tune/ckpts/decoder_step0001000.pt`
- `kl5e-4_step1000.encoder: outputs/ss_enc_dec_fine_tune_kl5e-4/ckpts/encoder_step0001000.pt`
- `kl5e-4_step1000.decoder: outputs/ss_enc_dec_fine_tune_kl5e-4/ckpts/decoder_step0001000.pt`
- `kl1e-4_step1000.encoder: outputs/ss_enc_dec_fine_tune_kl1e-4/ckpts/encoder_step0001000.pt`
- `kl1e-4_step1000.decoder: outputs/ss_enc_dec_fine_tune_kl1e-4/ckpts/decoder_step0001000.pt`
Last observed:
- 2026-07-18
