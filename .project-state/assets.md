# Assets

## AST-20260717-001

Description:
- 本地 FaceScape 规范化/预处理数据，是当前微调与审计主数据资源。
Path:
- `datasets/Facescape`
Produced by run: 

## AST-20260717-002

Description:
- 本地 Hugging Face TRELLIS-image-large 模型目录和 checkpoint 权重。
Path:
- `microsoft/TRELLIS-image-large`
Produced by run: 

## AST-20260717-008

Description:
- TRELLIS-image-large SLat encoder safetensors 权重。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.safetensors`
Produced by run: 

## AST-20260717-009

Description:
- TRELLIS-image-large SLat Gaussian decoder safetensors 权重。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.safetensors`
Produced by run: 

## AST-20260717-010

Description:
- 由 SLat encoder safetensors 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt`
Produced by run: RUN-20260717-001

## AST-20260717-011

Description:
- 由 SLat Gaussian decoder safetensors 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`
Produced by run: RUN-20260717-002

## AST-20260718-004

Description:
- FaceScape SS encoder + decoder fine-tune 的 1000-step 输出目录，包含日志、loss、checkpoint 和 init/final 重建样本图。
Path:
- `outputs/train/ss_enc_dec_fine_tune`
Produced by run: RUN-20260718-005

## AST-20260718-005

Description:
- FaceScape SS encoder + decoder fine-tune 的 `lambda_kl=5e-4` 1000-step 输出目录。
Path:
- `outputs/train/ss_enc_dec_fine_tune_kl5e-4`
Produced by run: RUN-20260718-006

## AST-20260718-006

Description:
- FaceScape SS encoder + decoder fine-tune 的 `lambda_kl=1e-4` 1000-step 输出目录。
Path:
- `outputs/train/ss_enc_dec_fine_tune_kl1e-4`
Produced by run: RUN-20260718-007

## AST-20260718-007

Description:
- FaceScape test split 的固定 64 样本 SparseStructure mini evaluation dataset。
Path:
- `datasets/Facescape_ss_eval_test_64`
Produced by run: RUN-20260718-010

## AST-20260718-008

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 posterior mean 重建评估结果。
Path:
- `outputs/eval/ss_enc_dec_eval`
Produced by run: RUN-20260718-011

## AST-20260718-009

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 重建评估结果，seed `20260718`。
Path:
- `outputs/eval/ss_enc_dec_eval_sample_posterior`
Produced by run: RUN-20260718-012

## AST-20260718-010

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 重建评估结果，seed `20260719`。
Path:
- `outputs/eval/ss_enc_dec_eval_sample_posterior_seed20260719`
Produced by run: RUN-20260718-013

## AST-20260718-011

Description:
- official、`kl1e-3`、`kl5e-4`、`kl1e-4` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 重建评估结果，seed `20260720`。
Path:
- `outputs/eval/ss_enc_dec_eval_sample_posterior_seed20260720`
Produced by run: RUN-20260718-014

## AST-20260718-012

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 posterior mean 重建对照评估结果。
Path:
- `outputs/eval/ss_enc_dec_eval_kl1e-4_steps`
Produced by run: RUN-20260718-015

## AST-20260718-013

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 对照评估结果，seed `20260718`。
Path:
- `outputs/eval/ss_enc_dec_eval_kl1e-4_steps_sample_posterior`
Produced by run: RUN-20260718-016

## AST-20260718-014

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 对照评估结果，seed `20260719`。
Path:
- `outputs/eval/ss_enc_dec_eval_kl1e-4_steps_sample_posterior_seed20260719`
Produced by run: RUN-20260718-017

## AST-20260718-015

Description:
- `kl1e-4_step500` 与 `kl1e-4_step1000` SS encoder/decoder checkpoint 在固定 64 样本上的 sample posterior 对照评估结果，seed `20260720`。
Path:
- `outputs/eval/ss_enc_dec_eval_kl1e-4_steps_sample_posterior_seed20260720`
Produced by run: RUN-20260718-018

## AST-20260718-016

Description:
- 1024 个 FaceScape train 样本的独立 Sparse Structure latent 数据集，使用 `kl1e-4_step1000` SS encoder 编码，并带独立 metadata。
Path:
- `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024`
Produced by run: RUN-20260718-020

## AST-20260718-017

Description:
- 从 `/root/autodl-fs/Facescape_cond/renders_cond.tar.part000` 到 `part006` 分卷 tar 解压出的 FaceScape 条件渲染图总目录。
Path:
- `datasets/Facescape/renders_cond`
Produced by run: 

## AST-20260718-018

Description:
- 按已有 train/test metadata 划分建立的 `renders_cond` symlink 目录，以及 flow 1024 子集对应的条件图 symlink 目录。 (1/3)
Path:
- `datasets/Facescape/train/renders_cond`
Produced by run: 

## AST-20260718-019

Description:
- 由官方 `ss_flow_img_dit_L_16l8_fp16.safetensors` 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`
Produced by run: RUN-20260718-021

## AST-20260718-020

Description:
- 从 AutoDL FS 复制到当前项目 `microsoft/` 目录下的 `trellis-normal-v0-1` 权重目录。
Path:
- `microsoft/trellis-normal-v0-1`
Produced by run: 

## AST-20260718-021

Description:
- 由 `trellis-normal-v0-1` 的 `ss_flow_normal_dit_L_16l8_fp16.safetensors` 转换得到的 PyTorch `.pt` state_dict。
Path:
- `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.pt`
Produced by run: RUN-20260718-022

## AST-20260718-022

Description:
- 使用 `kl1e-4_step1000` SS encoder/decoder latent 数据、normal SS flow denoiser 初始化训练得到的 1000-step SS flow 输出目录。
Path:
- `outputs/train/ss_flow_finetune_kl1e-4_step1000`
Produced by run: RUN-20260718-023

## AST-20260718-023

Description:
- 按已有 train/test metadata 划分建立的 `renders_cond` symlink 目录，以及 flow 1024 子集对应的条件图 symlink 目录。 (2/3)
Path:
- `datasets/Facescape/test/renders_cond`
Produced by run: 

## AST-20260718-024

Description:
- 按已有 train/test metadata 划分建立的 `renders_cond` symlink 目录，以及 flow 1024 子集对应的条件图 symlink 目录。 (3/3)
Path:
- `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024/renders_cond`
Produced by run: 

## AST-20260720-001

Description:
- SS flow `kl=1e-4` step1000 固定条件采样评估结果，包含指标 CSV/JSON 和每样本 PLY 可视化。
Path:
- `outputs/eval/ss_flow_kl1e-4_step1000`
Produced by run: RUN-20260720-002

## AST-20260722-001

Description:
- 本地 TRELLIS normal-conditioned pipeline 配置与四组件预训练权重目录。
Path:
- `microsoft/trellis-normal-v0-1`
Produced by run:
