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

## AST-20260717-003

Description:
- 本地 TRELLIS 参考数据集 3D-FUTURE。
Path:
- `datasets/3D-FUTURE`
Produced by run:

## AST-20260717-004

Description:
- 本地 TRELLIS 参考数据集 ABO。
Path:
- `datasets/ABO`
Produced by run:

## AST-20260717-005

Description:
- 本地 TRELLIS 参考数据集 HSSD。
Path:
- `datasets/HSSD`
Produced by run:

## AST-20260717-006

Description:
- 本地 TRELLIS 参考数据集 ObjaverseXL Sketchfab。
Path:
- `datasets/ObjaverseXL_sketchfab`
Produced by run:

## AST-20260717-007

Description:
- 本地 TRELLIS 参考数据集 Toys4k。
Path:
- `datasets/Toys4k`
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

## AST-20260717-012

Description:
- FaceScape 训练集中损坏的 DINOv2 patch token `.npz` 特征缓存。
Path:
- `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz`
Produced by run:

## AST-20260717-013

Description:
- FaceScape 训练集中另一个损坏的 DINOv2 patch token `.npz` 特征缓存。
Path:
- `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz`
Produced by run:

## AST-20260717-014

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune 1000-step 试验输出目录。
Path:
- `outputs/slat_enc_dec_gs_fine_tune`
Produced by run: RUN-20260717-004

## AST-20260718-001

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune v2 输出目录，使用有效 batch 8 和 lr=1e-5。
Path:
- `outputs/slat_enc_dec_gs_fine_tune_v2`
Produced by run: RUN-20260718-001

## AST-20260718-002

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune v3 输出目录，batch16/lr1e-5 启动后在进入训练 DataLoader 阶段失败。
Path:
- `outputs/slat_enc_dec_gs_fine_tune_v3`
Produced by run: RUN-20260718-002

## AST-20260718-003

Description:
- 为低配置机器测速准备的 FaceScape SLat encoder + Gaussian decoder 训练数据子集。
Path:
- `datasets/Facescape_slat_gs_50gb`
Produced by run: RUN-20260718-003

## AST-20260718-004

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune 输出目录，使用 `lambda_kl=1e-7`、有效 batch16 和 lr=1e-5。
Path:
- `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-7`
Produced by run: RUN-20260718-005

## AST-20260718-005

Description:
- 从 FaceScape test 中固定抽取的 50 样本 SLat GS checkpoint 评估子集。
Path:
- `datasets/Facescape_eval/slat_gs_eval50`
Produced by run: RUN-20260718-006

## AST-20260718-006

Description:
- `lambda_kl=1e-7` batch16 训练的 step1000 非 EMA checkpoint 在 eval50/view0 上的重建评估输出。
Path:
- `outputs/eval/slat_kl1e-7_step1000_eval50_view0`
Produced by run: RUN-20260718-007

## AST-20260718-007

Description:
- `lambda_kl=1e-7` batch16 训练的 step1000 EMA checkpoint 在 eval50/view0 上的重建评估输出。
Path:
- `outputs/eval/slat_kl1e-7_ema_step1000_eval50_view0`
Produced by run: RUN-20260718-008

## AST-20260718-008

Description:
- step1000 非 EMA 与 EMA checkpoint 的 eval50/view0 横向对比 CSV。
Path:
- `outputs/eval/slat_kl1e-7_step1000_eval50_compare.csv`
Produced by run: RUN-20260718-009

## AST-20260718-009

Description:
- 当前本机可发现的全部 `lambda_kl` 训练最终权重在 eval50/view0 上的横向对比 CSV。
Path:
- `outputs/eval/slat_all_kl_final_eval50_view0_compare.csv`
Produced by run: RUN-20260718-010

## AST-20260718-010

Description:
- 从 FaceScape train 固定抽取并扩展到 1024 个样本，使用 `lambda_kl=1e-7` step1000 非 EMA SLat encoder 编码得到的独立 latent smoke 数据集。
Path:
- `datasets/Facescape_slat_kl1e-7_nonema_smoke`
Produced by run: RUN-20260718-011

## AST-20260718-011

Description:
- 从 `/root/autodl-fs/Facescape_cond` 分卷 tar 包解压得到的 FaceScape 条件图目录，并按既有 train/test 划分建立软链接。
Path:
- `datasets/Facescape/renders_cond`
Produced by run: RUN-20260718-013

## AST-20260719-001

Description:
- SLat flow 使用 kl1e-7 non-EMA latent smoke 数据集完成 1000-step fine-tune 的输出目录。
Path:
- `outputs/train/slat_flow_finetune_kl1e-7_step1000`
Produced by run: RUN-20260719-001

## AST-20260719-002

Description:
- `lambda_kl=1e-7` step1000 非 EMA SLat encoder/decoder 在 eval50/view0 上重新评估的输出目录。
Path:
- `outputs/eval/slat_enc_dec_kl1e-7_step1000_nonema_eval50_view0_rerun`
Produced by run: RUN-20260719-002

## AST-20260719-003

Description:
- `lambda_kl=1e-7` step1000 非 EMA SLat flow 在固定 16 样本上生成的评估输出目录。
Path:
- `outputs/eval/slat_flow_kl1e-7_step1000_nonema_fixed_gen16`
Produced by run: RUN-20260719-003

## AST-20260719-004

Description:
- `lambda_kl=1e-7` step1000 非 EMA SLat flow 固定 16 样本生成结果的指标输出目录。
Path:
- `outputs/eval/slat_flow_kl1e-7_step1000_nonema_metrics16`
Produced by run: RUN-20260719-004

## AST-20260719-005

Description:
- 修复 flow generation metrics 兼容入口后复测生成的固定 16 样本指标输出目录。
Path:
- `outputs/eval/slat_flow_kl1e-7_step1000_nonema_metrics16_wrapper_check`
Produced by run: RUN-20260719-005

## AST-20260719-006

Description:
- 从 SLat flow 训练集固定生成结果中挑选 8 个 FaceScape 条件图样本制作的 cond/generated/GT 对照图。
Path:
- `outputs/eval/slat_flow_kl1e-7_step1000_nonema_fixed_gen16_selected_train_contact_sheet.png`
Produced by run:

## AST-20260719-007

Description:
- 验证 SLat flow 固定生成默认保存 generated/GT PLY 的 1 样本 smoke 输出目录。
Path:
- `outputs/eval/slat_flow_ply_smoke`
Produced by run: RUN-20260719-006

## AST-20260720-001

Description:
- SLat encoder + Gaussian decoder FaceScape fine-tune 输出目录，使用 `lambda_kl=1e-6`、有效 batch16 和 lr=1e-5。
Path:
- `outputs/train/slat_enc_dec_gs_fine_tune_kl1e-6`
Produced by run: RUN-20260720-001

## AST-20260720-002

Description:
- `lambda_kl=1e-6/1e-7/1e-8` step500/step1000 非 EMA SLat enc/dec checkpoint 的固定样本 KL 梯度贡献诊断输出目录。
Path:
- `outputs/eval/slat_enc_dec_grad_contrib_kl_sweep_step500_1000_eval8_view0`
Produced by run: RUN-20260720-002
