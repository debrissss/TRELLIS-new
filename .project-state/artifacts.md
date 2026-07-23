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

## ART-20260718-007 - FaceScape SS fixed evaluation dataset 64

Description:
- FaceScape test split 的固定 64 样本 SparseStructure mini evaluation dataset。
Path:
- `datasets/Facescape_ss_eval_test_64`
Origin:
- run-output
Produced by run:
- RUN-20260718-010
Created/Updated: 2026-07-18
Meaning:
- 用于在同一组 test 样本上比较 official 和 SS encoder/decoder KL ablation checkpoint 的重建指标。
Notes:
- `metadata.csv` 为 64 个样本加表头；`voxels` 是指向 `../Facescape/test/voxels` 的软链接，目录大小约 `16K`。

## ART-20260718-008 - SS encoder plus decoder posterior-mean eval results

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 posterior mean 重建评估结果。
Path:
- `outputs/ss_enc_dec_eval`
Origin:
- run-output
Produced by run:
- RUN-20260718-011
Created/Updated: 2026-07-18
Meaning:
- 用于评估 deterministic posterior mean 重建口径下各 SS encoder/decoder checkpoint 的 hard voxel 指标和 soft Dice loss。
Notes:
- 包含 4 个 `*_per_sample_metrics.csv`、`summary.csv`、`summary.json`；每个 per-sample CSV 为 64 行样本加表头；目录大小约 `60K`。

## ART-20260718-009 - SS encoder plus decoder sample-posterior eval results seed20260718

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 重建评估结果，seed `20260718`。
Path:
- `outputs/ss_enc_dec_eval_sample_posterior`
Origin:
- run-output
Produced by run:
- RUN-20260718-012
Created/Updated: 2026-07-18
Meaning:
- 用于评估 stochastic posterior sampling 口径下各 SS encoder/decoder checkpoint 的重建稳定性。
Notes:
- 包含 4 个 `*_per_sample_metrics.csv`、`summary.csv`、`summary.json`；每个 per-sample CSV 为 64 行样本加表头；目录大小约 `72K`。

## ART-20260718-010 - SS encoder plus decoder sample-posterior eval results seed20260719

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 重建评估结果，seed `20260719`。
Path:
- `outputs/ss_enc_dec_eval_sample_posterior_seed20260719`
Origin:
- run-output
Produced by run:
- RUN-20260718-013
Created/Updated: 2026-07-18
Meaning:
- 用于评估 stochastic posterior sampling 指标的跨 seed 稳定性。
Notes:
- 包含 4 个 `*_per_sample_metrics.csv`、`summary.csv`、`summary.json`；每个 per-sample CSV 为 64 行样本加表头；目录大小约 `72K`。

## ART-20260718-011 - SS encoder plus decoder sample-posterior eval results seed20260720

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 重建评估结果，seed `20260720`。
Path:
- `outputs/ss_enc_dec_eval_sample_posterior_seed20260720`
Origin:
- run-output
Produced by run:
- RUN-20260718-014
Created/Updated: 2026-07-18
Meaning:
- 用于评估 stochastic posterior sampling 指标的跨 seed 稳定性。
Notes:
- 包含 4 个 `*_per_sample_metrics.csv`、`summary.csv`、`summary.json`；每个 per-sample CSV 为 64 行样本加表头；目录大小约 `72K`。

## ART-20260718-012 - SS kl1e-4 step500-vs-step1000 posterior-mean eval results

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 posterior mean 重建对照评估结果。
Path:
- `outputs/ss_enc_dec_eval_kl1e-4_steps`
Origin:
- run-output
Produced by run:
- RUN-20260718-015
Created/Updated: 2026-07-18
Meaning:
- 用于判断 `lambda_kl=1e-4` 训练到 500 step 与 1000 step 在 deterministic reconstruction 口径下是否有差异。
Notes:
- 包含 `kl1e-4_step500_per_sample_metrics.csv`、`kl1e-4_step1000_per_sample_metrics.csv`、`summary.csv`、`summary.json`；每个 per-sample CSV 为 64 行样本加表头。

## ART-20260718-013 - SS kl1e-4 step500-vs-step1000 sample-posterior eval results seed20260718

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 对照评估结果，seed `20260718`。
Path:
- `outputs/ss_enc_dec_eval_kl1e-4_steps_sample_posterior`
Origin:
- run-output
Produced by run:
- RUN-20260718-016
Created/Updated: 2026-07-18
Meaning:
- 用于判断 step500 与 step1000 在 stochastic posterior reconstruction 口径下的稳定性差异。
Notes:
- 包含两个 per-sample CSV、`summary.csv`、`summary.json`。

## ART-20260718-014 - SS kl1e-4 step500-vs-step1000 sample-posterior eval results seed20260719

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 对照评估结果，seed `20260719`。
Path:
- `outputs/ss_enc_dec_eval_kl1e-4_steps_sample_posterior_seed20260719`
Origin:
- run-output
Produced by run:
- RUN-20260718-017
Created/Updated: 2026-07-18
Meaning:
- 用于判断 step500 与 step1000 stochastic posterior 指标的跨 seed 稳定性。
Notes:
- 包含两个 per-sample CSV、`summary.csv`、`summary.json`。

## ART-20260718-015 - SS kl1e-4 step500-vs-step1000 sample-posterior eval results seed20260720

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 对照评估结果，seed `20260720`。
Path:
- `outputs/ss_enc_dec_eval_kl1e-4_steps_sample_posterior_seed20260720`
Origin:
- run-output
Produced by run:
- RUN-20260718-018
Created/Updated: 2026-07-18
Meaning:
- 用于判断 step500 与 step1000 stochastic posterior 指标的跨 seed 稳定性。
Notes:
- 包含两个 per-sample CSV、`summary.csv`、`summary.json`。

## ART-20260718-016 - FaceScape SS latent kl1e-4 step1000 train subset 1024

Description:
- 1024 个 FaceScape train 样本的独立 Sparse Structure latent 数据集，使用 `kl1e-4_step1000` SS encoder 编码，并带独立 metadata。
Path:
- `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024`
Origin:
- run-output
Produced by run:
- RUN-20260718-020
Created/Updated: 2026-07-18
Meaning:
- 用于后续验证 flow 能否适配 `lambda_kl=1e-4`、step1000 SS encoder 产生的 latent 分布。
Notes:
- 目录大小在补齐条件图 symlink 前约 `125M`；`renders_cond` 为 symlink 目录后约 `104K`，不复制图片本体。
- `metadata.csv` 有 1024 行样本，latent 列 `ss_latent_ss_enc_dec_fine_tune_kl1e-4_step0001000` 全为 True。
- latent 文件位于 `ss_latents/ss_enc_dec_fine_tune_kl1e-4_step0001000/*.npz`，共 1024 个。
- 单个 latent `mean` 数组 shape 为 `(8, 16, 16, 16)`、dtype 为 `float32`；全量 finite 检查通过。
- `voxels` 是指向 `../Facescape/train/voxels` 的 symlink。
- 2026-07-18 已补 `renders_cond/` symlink 目录，包含 1023 个 `cond_rendered=True` 样本；1 个 metadata 样本 `cond_rendered=False`，image-conditioned dataset 会过滤掉。
- 使用 `ImageConditionedSparseStructureLatent`、`latent_model=ss_enc_dec_fine_tune_kl1e-4_step0001000`、`ss_dec_path=outputs/ss_enc_dec_fine_tune_kl1e-4`、`ss_dec_ckpt=step0001000` 初始化得到 1023 个样本，首样本 `x_0` shape `(8,16,16,16)`，`cond` shape `(3,518,518)`。

## ART-20260718-017 - FaceScape renders_cond extracted condition images

Description:
- 从 `/root/autodl-fs/Facescape_cond/renders_cond.tar.part000` 到 `part006` 分卷 tar 解压出的 FaceScape 条件渲染图总目录。
Path:
- `datasets/Facescape/renders_cond`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-18
Meaning:
- 为 image-conditioned SS flow fine-tune 提供 `ImageConditionedMixin` 所需的 `renders_cond/<sha>/<view>.png` 条件图。
Notes:
- 使用管道 `cat /root/autodl-fs/Facescape_cond/renders_cond.tar.part* | tar -xpf - -C datasets/Facescape` 解压，未额外合并 134G 临时 tar。
- 解压后目录大小约 `135G`，顶层样本目录数 7173，条件图片文件数 324378。
- 覆盖已有划分：train metadata 中 `cond_rendered=True` 的 6453 个样本全部存在；test metadata 中 720 个样本全部存在。
- 解压后 `/root/autodl-tmp` 剩余空间约 `92G`。

## ART-20260718-018 - FaceScape renders_cond split symlinks

Description:
- 按已有 train/test metadata 划分建立的 `renders_cond` symlink 目录，以及 flow 1024 子集对应的条件图 symlink 目录。
Path:
- `datasets/Facescape/train/renders_cond`
- `datasets/Facescape/test/renders_cond`
- `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024/renders_cond`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-18
Meaning:
- 让 `ImageConditionedSparseStructureLatent` 能在各 split root 下按 metadata sha 直接找到条件图，同时避免复制 135G 图片。
Notes:
- `datasets/Facescape/train/renders_cond` 包含 6453 个 symlink，broken symlink 为 0；3 个 train metadata 样本 `cond_rendered=False` 被跳过。
- `datasets/Facescape/test/renders_cond` 包含 720 个 symlink，broken symlink 为 0。
- `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024/renders_cond` 包含 1023 个 symlink，broken symlink 为 0；1 个子集样本 `cond_rendered=False` 被跳过。

## ART-20260718-019 - TRELLIS image SS flow denoiser PyTorch checkpoint

Description:
- 由官方 `ss_flow_img_dit_L_16l8_fp16.safetensors` 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`
Origin:
- run-output
Produced by run:
- RUN-20260718-021
Created/Updated: 2026-07-18
Meaning:
- 用作 image-conditioned SS flow fine-tune 的 `trainer.args.finetune_ckpt.denoiser` 初始化权重。
Notes:
- 源文件为 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors`，源模型 JSON 为同目录 `ss_flow_img_dit_L_16l8_fp16.json`。
- 转换脚本执行 strict model load，并验证保存后的 `.pt` 可通过 `torch.load(..., weights_only=True)` 重新 strict load。
- 输出文件大小约 `1.1G`；`torch.load` 检查显示类型为 `OrderedDict`，489 个 tensor key，总参数数 559737864，首 key 为 `pos_emb`。

## ART-20260718-020 - TRELLIS normal v0-1 local model copy

Description:
- 从 AutoDL FS 复制到当前项目 `microsoft/` 目录下的 `trellis-normal-v0-1` 权重目录。
Path:
- `microsoft/trellis-normal-v0-1`
Origin:
- existing-resource
Produced by run:
- none
Created/Updated: 2026-07-18
Meaning:
- 提供 normal-conditioned TRELLIS 权重，包括 SS normal flow、SLat normal flow、SS decoder 和 mesh decoder，可用于后续 normal 条件实验或权重对照。
Notes:
- 源路径为 `/root/autodl-fs/trellis-normal-v0-1`，等价于 `/autodl-fs/data/trellis-normal-v0-1`。
- 使用 `rsync -a` 复制到 `/root/autodl-tmp/TRELLIS-new/microsoft/trellis-normal-v0-1`。
- 源/目标目录大小均约 `2.5G`，源/目标文件数均为 24，文件列表 diff 结果一致。
- 主要权重包括 `ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors`、`ckpts/slat_flow_normal_dit_L_64l8p2_fp16.safetensors`、`ckpts/ss_dec_conv3d_16l8_fp16.safetensors`、`ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors`。

## ART-20260718-021 - TRELLIS normal SS flow denoiser PyTorch checkpoint

Description:
- 由 `trellis-normal-v0-1` 的 `ss_flow_normal_dit_L_16l8_fp16.safetensors` 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.pt`
Origin:
- run-output
Produced by run:
- RUN-20260718-022
Created/Updated: 2026-07-18
Meaning:
- 可用作 normal-conditioned SS flow 相关训练或对照实验的 `.pt` denoiser 初始化权重。
Notes:
- 源文件为 `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors`，源模型 JSON 为同目录 `ss_flow_normal_dit_L_16l8_fp16.json`。
- 转换脚本执行 strict model load，并验证保存后的 `.pt` 可通过 `torch.load(..., weights_only=True)` 重新 strict load。
- 输出文件大小约 `1.1G`；`torch.load` 检查显示类型为 `OrderedDict`，489 个 tensor key，总参数数 559737864，首 key 为 `pos_emb`。

## ART-20260718-022 - SS flow kl1e-4 step1000 normal-init fine-tune output

Description:
- 使用 `kl1e-4_step1000` SS encoder/decoder latent 数据、normal SS flow denoiser 初始化训练得到的 1000-step SS flow 输出目录。
Path:
- `outputs/ss_flow_finetune_kl1e-4_step1000`
Origin:
- run-output
Produced by run:
- RUN-20260718-023
Created/Updated: 2026-07-18
Meaning:
- 用于判断 `lambda_kl=1e-4` 的 sparse-structure latent 分布是否能被 SS flow 阶段稳定适配，并作为后续 flow checkpoint/初始化方式对照的基线。
Notes:
- 输出目录约 `17G`。
- 包含 `denoiser_step0000500.pt`、`denoiser_ema0.9999_step0000500.pt`、`misc_step0000500.pt`、`denoiser_step0001000.pt`、`denoiser_ema0.9999_step0001000.pt`、`misc_step0001000.pt`。
- 训练日志为 `log_ss_flow_finetune_kl1e-4_step1000.txt`，loss 文件为 `loss_ss_flow_finetune_kl1e-4_step1000.txt`。
- sample 目录包含 `samples/init/*`、`samples/final/*` 和 `samples/dataset.jpg`；由于配置 `i_sample=2000` 且 `max_steps=1000`，没有中间 step sample。
