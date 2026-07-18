# Artifacts

## ART-20260717-001 - FaceScape processed dataset

Description:
- 本地 FaceScape 规范化/预处理数据，是当前微调与审计主数据资源。
Path:
- `datasets/Facescape`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 包含 FaceScape train/test/merged 数据和预处理资源。
Notes:
- `du -sh` 显示约 441G；此前扫描确认 train/test 下有 `features`、`renders`、`renders_cond`、`voxels`。
- 2026-07-17 使用临时脚本扫描 train/test 的 DINOv2 feature `.npz` 后，已在 `datasets/Facescape/train/metadata.csv` 中将两个损坏特征样本的 `feature_dinov2_vitl14_reg` 置为 `False`；test 未发现坏样本。

## ART-20260717-002 - TRELLIS-image-large local model

Description:
- 本地 Hugging Face TRELLIS-image-large 模型目录和 checkpoint 权重。
Path:
- `microsoft/TRELLIS-image-large`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 训练、推理、预处理和重建工具引用的本地预训练模型资源。
Notes:
- `du -sh` 显示约 3.1G；存在 `ckpts/*.json` 和 `ckpts/*.safetensors`。

## ART-20260717-003 - 3D-FUTURE dataset resource

Description:
- 本地 TRELLIS 参考数据集 3D-FUTURE。
Path:
- `datasets/3D-FUTURE`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 9473 行；目录约 8.6M。

## ART-20260717-004 - ABO dataset resource

Description:
- 本地 TRELLIS 参考数据集 ABO。
Path:
- `datasets/ABO`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 4486 行；目录约 97M。

## ART-20260717-005 - HSSD dataset resource

Description:
- 本地 TRELLIS 参考数据集 HSSD。
Path:
- `datasets/HSSD`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 6671 行；目录约 6.0M。

## ART-20260717-006 - ObjaverseXL Sketchfab dataset resource

Description:
- 本地 TRELLIS 参考数据集 ObjaverseXL Sketchfab。
Path:
- `datasets/ObjaverseXL_sketchfab`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 168308 行；目录约 344M。

## ART-20260717-007 - Toys4k dataset resource

Description:
- 本地 TRELLIS 参考数据集 Toys4k。
Path:
- `datasets/Toys4k`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 原始 TRELLIS 数据准备和训练参考资源。
Notes:
- `metadata.csv` 3230 行；目录约 2.9M。

## ART-20260717-008 - SLat encoder safetensors checkpoint

Description:
- TRELLIS-image-large SLat encoder safetensors 权重。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- SLat encoder 预训练权重，作为转换为 PyTorch `.pt` state_dict 的输入。
Notes:
- 源文件约 166M。

## ART-20260717-009 - SLat GS decoder safetensors checkpoint

Description:
- TRELLIS-image-large SLat Gaussian decoder safetensors 权重。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- SLat Gaussian decoder 预训练权重，作为转换为 PyTorch `.pt` state_dict 的输入。
Notes:
- 源文件约 164M。

## ART-20260717-010 - SLat encoder PyTorch checkpoint

Description:
- 由 SLat encoder safetensors 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt`
Origin:
- run-output
Produced by run:
- RUN-20260717-001
Created/Updated: 2026-07-17
Meaning:
- 可用于 `trainer.args.finetune_ckpt.encoder` 的 PyTorch checkpoint。
Notes:
- 约 166M；转换脚本已完成内置验证，另用 `torch.load(..., weights_only=True)` 读取到 100 个 state_dict 条目。

## ART-20260717-011 - SLat GS decoder PyTorch checkpoint

Description:
- 由 SLat Gaussian decoder safetensors 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`
Origin:
- run-output
Produced by run:
- RUN-20260717-002
Created/Updated: 2026-07-17
Meaning:
- 可用于 `trainer.args.finetune_ckpt.decoder` 的 PyTorch checkpoint。
Notes:
- 约 164M；转换脚本已完成内置验证，另用 `torch.load(..., weights_only=True)` 读取到 101 个 state_dict 条目。

## ART-20260717-012 - corrupt FaceScape DINO feature 3ad9da5e

Description:
- FaceScape 训练集中损坏的 DINOv2 patch token `.npz` 特征缓存。
Path:
- `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- SLat encoder + GS decoder fine-tune 训练在读取该样本 `patchtokens.npy` 时触发 `zipfile.BadZipFile`。
Notes:
- 文件大小 36713 bytes；`zipfile.testzip()` 返回 `patchtokens.npy`；zip 条目声明 `patchtokens.npy` 压缩大小 15535914 bytes，远大于实际文件大小。

## ART-20260717-013 - corrupt FaceScape DINO feature 3ad9e565

Description:
- FaceScape 训练集中另一个损坏的 DINOv2 patch token `.npz` 特征缓存。
Path:
- `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-17
Meaning:
- 与已触发训练失败的坏样本具有相同损坏模式，若继续训练可能在后续 batch 中再次触发 `zipfile.BadZipFile`。
Notes:
- 文件大小 31951 bytes；`zipfile.testzip()` 返回 `patchtokens.npy`；zip 条目声明 `patchtokens.npy` 压缩大小 15537879 bytes，远大于实际文件大小。

## ART-20260717-014 - SLat GS fine-tune 1000-step output

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune 1000-step 试验输出目录。
Path:
- `outputs/slat_enc_dec_gs_fine_tune`
Origin:
- run-output
Produced by run:
- RUN-20260717-004
Created/Updated: 2026-07-17
Meaning:
- 包含完成的 1000-step fine-tune 日志、TensorBoard、init/final samples，以及 step 500/1000 的 encoder/decoder/misc checkpoint。
Notes:
- `log.txt` 和 `loss.txt` 各 1000 行；`ckpts` 下保存 step0000500 与 step0001000 的 encoder/decoder/EMA/misc checkpoint；final sample 中 `rec_image_final.jpg` 与 `gt_image_final.jpg` 视觉上高度接近。

## ART-20260718-001 - SLat GS fine-tune v2 batch8 lr1e-5 output

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune v2 输出目录，使用有效 batch 8 和 lr=1e-5。
Path:
- `outputs/slat_enc_dec_gs_fine_tune_v2`
Origin:
- run-output
Produced by run:
- RUN-20260718-001
Created/Updated: 2026-07-18
Meaning:
- 包含 batch 8、micro-batch 2、lr=1e-5 的 1000-step fine-tune 日志、TensorBoard、init/final samples，以及 step 500/1000 checkpoint。
Notes:
- `log.txt` 和 `loss.txt` 各 1000 行；`config.json` 确认 `batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5`；final sample 中 `rec_image_final.jpg` 与 `gt_image_final.jpg` 视觉上高度接近。

## ART-20260718-002 - SLat GS fine-tune v3 batch16 failed output

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune v3 输出目录，batch16/lr1e-5 启动后在进入训练 DataLoader 阶段失败。
Path:
- `outputs/slat_enc_dec_gs_fine_tune_v3`
Origin:
- run-output
Produced by run:
- RUN-20260718-002
Created/Updated: 2026-07-18
Meaning:
- 记录 batch16 对照实验未能进入正式训练日志阶段；用于追踪 DataLoader shared memory 问题。
Notes:
- 目录中写入了 `command.txt`、`config.json`、model summary、`samples/dataset.jpg` 和 `samples/init/*`；没有完整 `log.txt` 或 checkpoint。用户报告终端出现多条 `Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`

## ART-20260718-003 - FaceScape SLat GS 50GB train subset

Description:
- 为低配置机器测速准备的 FaceScape SLat encoder + Gaussian decoder 训练数据子集。
Path:
- `datasets/Facescape_slat_gs_50gb`
Origin:
- run-output
Produced by run:
- RUN-20260718-003
Created/Updated: 2026-07-18
Meaning:
- 允许在另一台机器上用约 50GB 数据测试 SLat encoder + Gaussian decoder fine-tune 速度与成本收益，而无需搬运完整约 441GB FaceScape 数据集。
Notes:
- 目录大小 `51G`；`train/metadata.csv` 为 1178 个样本加表头；包含 1178 个 `train/renders/<sha>/` 目录和 1178 个 `train/features/dinov2_vitl14_reg/<sha>.npz` 文件。轻量一致性检查确认 metadata 中每个样本都有 feature 文件和 `renders/<sha>/transforms.json`。未包含 `voxels/`、`renders_cond/` 或预训练 checkpoint。

## ART-20260718-004 - SS encoder plus decoder fine-tune 1000-step output

Description:
- FaceScape SS encoder + decoder fine-tune 的 1000-step 输出目录，包含日志、loss、checkpoint 和 init/final 重建样本图。
Path:
- `outputs/ss_enc_dec_fine_tune`
Origin:
- run-output
Produced by run:
- RUN-20260718-005
Created/Updated: 2026-07-18
Meaning:
- 用于评估 `configs/vae/ss_enc_dec_fine_tune.json` 在 1000 step、batch16、lr=1e-5 下的初步微调效果。
Notes:
- 包含 `log_ss_enc_dec_fine_tune.txt`、`loss.txt`、step 500/1000 encoder/decoder checkpoint、`samples/init` 和 `samples/final` 可视化。

## ART-20260718-005 - SS encoder plus decoder fine-tune kl5e-4 1000-step output

Description:
- FaceScape SS encoder + decoder fine-tune 的 `lambda_kl=5e-4` 1000-step 输出目录。
Path:
- `outputs/ss_enc_dec_fine_tune_kl5e-4`
Origin:
- run-output
Produced by run:
- RUN-20260718-006
Created/Updated: 2026-07-18
Meaning:
- 用于评估把 SS VAE KL 权重从 `0.001` 降到 `5e-4` 后，FaceScape 高精度人脸 SS 重建是否更容易适配。
Notes:
- 包含 `log_ss_enc_dec_fine_tune_kl5e-4.txt`、`loss_ss_enc_dec_fine_tune_kl5e-4.txt`、step 500/1000 encoder/decoder checkpoint、`samples/init` 和 `samples/final` 可视化；目录中另有用户复制的 `log_ss_enc_dec_fine_tune_kl5e-4-Copy1.txt`。

## ART-20260718-006 - SS encoder plus decoder fine-tune kl1e-4 1000-step output

Description:
- FaceScape SS encoder + decoder fine-tune 的 `lambda_kl=1e-4` 1000-step 输出目录。
Path:
- `outputs/ss_enc_dec_fine_tune_kl1e-4`
Origin:
- run-output
Produced by run:
- RUN-20260718-007
Created/Updated: 2026-07-18
Meaning:
- 用于评估把 SS VAE KL 权重进一步从 `5e-4` 降到 `1e-4` 后，FaceScape 高精度人脸 SS 重建是否继续改善。
Notes:
- 目录约 `6.0G`；包含 `log_ss_enc_dec_fine_tune_kl1e-4.txt`、`loss_ss_enc_dec_fine_tune_kl1e-4.txt`、step 500/1000 encoder/decoder checkpoint、EMA checkpoint、`misc` checkpoint、`samples/init` 和 `samples/final` 可视化。
