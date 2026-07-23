# Runs

## RUN-20260717-001

Description:
- 使用转换脚本将 SLat encoder safetensors 权重转换为 PyTorch `.pt` state_dict。

Time: 2026-07-17 21:53 UTC
Config: 
Assets: AST-20260717-008, AST-20260717-010

## RUN-20260717-002

Description:
- 使用转换脚本将 SLat Gaussian decoder safetensors 权重转换为 PyTorch `.pt` state_dict。

Time: 2026-07-17 21:53 UTC
Config: 
Assets: AST-20260717-009, AST-20260717-011

## RUN-20260717-003

Description:
- 用户报告 SLat encoder + GS decoder fine-tune 训练在保存 step 500 checkpoint 后因 FaceScape 特征缓存损坏中断并重试。

Time: 2026-07-17 22:25 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011

## RUN-20260717-004

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 1000-step fine-tune 试验。

Time: 2026-07-17 23:30 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011

## RUN-20260718-001

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 batch 8、lr=1e-5 的 1000-step fine-tune 试验。

Time: 2026-07-18 00:10 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011

## RUN-20260718-002

Description:
- SLat encoder + Gaussian decoder batch16/lr1e-5 对照训练在 init sampling 后因 DataLoader worker shared memory 问题失败。

Time: 2026-07-18 00:35 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011

## RUN-20260718-003

Description:
- 使用临时 Python 筛样和 rsync，从 FaceScape train 数据中复制约 50GB 的 SLat encoder + Gaussian decoder 训练子集。

Time: 2026-07-18 12:04 CST
Config: 
Assets: AST-20260717-001

## RUN-20260718-004

Description:
- 用户报告 SLat encoder + Gaussian decoder batch16 训练在 step 510-780 区间进入稳定吞吐段。

Time: 2026-07-18 12:12 CST
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011

## RUN-20260718-005

Description:
- 用户提供 SS encoder + decoder FaceScape fine-tune 1000-step 日志，分析 loss 曲线、样本图和下一步调参方向。

Time: 2026-07-18 16:30 UTC
Config: CFG-20260718-001
Assets: AST-20260717-001, AST-20260718-004

## RUN-20260718-006

Description:
- 用户提供 SS encoder + decoder FaceScape `lambda_kl=5e-4` 1000-step 日志，分析是否需要继续降低 KL。

Time: 2026-07-18 18:20 UTC
Config: CFG-20260718-001
Assets: AST-20260717-001, AST-20260718-005

## RUN-20260718-007

Description:
- 用户提供 SS encoder + decoder FaceScape `lambda_kl=1e-4` 1000-step 日志，分析相对 `1e-3` 和 `5e-4` 的变化。

Time: 2026-07-18 19:05 UTC
Config: CFG-20260718-001
Assets: AST-20260717-001, AST-20260718-006

## RUN-20260718-008

Description:
- 使用新建的固定样本评估集生成入口，从真实 FaceScape test metadata 临时抽取 4 个样本验证 mini dataset 生成逻辑。

Time: 2026-07-18 20:10 UTC
Config: 
Assets: AST-20260717-001

## RUN-20260718-009

Description:
- 使用新建的 SS encoder/decoder 重建评估入口，在 4 个临时固定样本上验证 checkpoint manifest、模型加载和指标输出。

Time: 2026-07-18 20:15 UTC
Config: CFG-20260718-002
Assets: 

## RUN-20260718-010

Description:
- 使用 `eval/prepare_ss_eval_dataset.py` 从 FaceScape test split 生成正式 64 样本固定 SS 评估集。

Time: 2026-07-18 21:00 UTC
Config: 
Assets: AST-20260717-001, AST-20260718-007

## RUN-20260718-011

Description:
- 在固定 64 个 FaceScape test 样本上，用 posterior mean 口径评估 official 与三组 KL ablation SS encoder/decoder checkpoint。

Time: 2026-07-18 21:05 UTC
Config: CFG-20260718-002
Assets: AST-20260718-007, AST-20260718-008

## RUN-20260718-012

Description:
- 在固定 64 个 FaceScape test 样本上，用 sample posterior 口径和 seed `20260718` 评估 official 与三组 KL ablation checkpoint。

Time: 2026-07-18 21:10 UTC
Config: CFG-20260718-002
Assets: AST-20260718-007, AST-20260718-009

## RUN-20260718-013

Description:
- 在固定 64 个 FaceScape test 样本上，用 sample posterior 口径和 seed `20260719` 评估 official 与三组 KL ablation checkpoint。

Time: 2026-07-18 21:15 UTC
Config: CFG-20260718-002
Assets: AST-20260718-007, AST-20260718-010

## RUN-20260718-014

Description:
- 在固定 64 个 FaceScape test 样本上，用 sample posterior 口径和 seed `20260720` 评估 official 与三组 KL ablation checkpoint。

Time: 2026-07-18 21:20 UTC
Config: CFG-20260718-002
Assets: AST-20260718-007, AST-20260718-011

## RUN-20260718-015

Description:
- 在固定 64 个 FaceScape test 样本上，用 posterior mean 口径评估 `kl1e-4_step500` 与 `kl1e-4_step1000`。

Time: 2026-07-18 22:00 UTC
Config: CFG-20260718-003
Assets: AST-20260718-007, AST-20260718-012

## RUN-20260718-016

Description:
- 在固定 64 个 FaceScape test 样本上，用 sample posterior 口径和 seed `20260718` 评估 `kl1e-4_step500` 与 `kl1e-4_step1000`。

Time: 2026-07-18 22:05 UTC
Config: CFG-20260718-003
Assets: AST-20260718-007, AST-20260718-013

## RUN-20260718-017

Description:
- 在固定 64 个 FaceScape test 样本上，用 sample posterior 口径和 seed `20260719` 评估 `kl1e-4_step500` 与 `kl1e-4_step1000`。

Time: 2026-07-18 22:10 UTC
Config: CFG-20260718-003
Assets: AST-20260718-007, AST-20260718-014

## RUN-20260718-018

Description:
- 在固定 64 个 FaceScape test 样本上，用 sample posterior 口径和 seed `20260720` 评估 `kl1e-4_step500` 与 `kl1e-4_step1000`。

Time: 2026-07-18 22:15 UTC
Config: CFG-20260718-003
Assets: AST-20260718-007, AST-20260718-015

## RUN-20260718-019

Description:
- 从 FaceScape train split 抽取 1024 个高 aesthetic 样本，构建用于 `kl1e-4_step1000` SS latent 编码的独立 dataset root 与 metadata。

Time: 2026-07-18 18:55 UTC
Config: 
Assets: AST-20260717-001, AST-20260718-016

## RUN-20260718-020

Description:
- 使用 `ss_enc_dec_fine_tune_kl1e-4` step1000 encoder 将 1024 个 FaceScape train 子集样本编码为 Sparse Structure latent，并构建可由 `SparseStructureLatent` 读取的独立 metadata。

Time: 2026-07-18 18:56 UTC
Config: CFG-20260718-001
Assets: AST-20260717-001, AST-20260718-016

## RUN-20260718-021

Description:
- 将官方 image-conditioned SS flow denoiser `safetensors` 权重转换为训练 `finetune_ckpt` 可直接读取的 PyTorch `.pt` state_dict。

Time: 2026-07-18 22:40 UTC
Config: CFG-20260717-102
Assets: AST-20260717-002, AST-20260718-019

## RUN-20260718-022

Description:
- 将 `trellis-normal-v0-1` 的 normal-conditioned SS flow denoiser `safetensors` 权重转换为 PyTorch `.pt` state_dict。

Time: 2026-07-18 22:53 UTC
Config: 
Assets: AST-20260718-020, AST-20260718-021

## RUN-20260718-023

Description:
- 使用 `kl1e-4_step1000` SS encoder/decoder latent 子集和 normal SS flow `.pt` 初始化，完成 1000-step image-conditioned SS flow 微调。

Time: 2026-07-18 23:35 UTC
Config: CFG-20260718-004
Assets: AST-20260718-016, AST-20260718-018, AST-20260718-023, AST-20260718-024, AST-20260718-021, AST-20260718-022

## RUN-20260720-001

Description:
- SS flow step1000 采样评估首次运行在 decoder resolution 探测阶段因输入 dtype 不匹配失败。

Time: 2026-07-20 09:32 UTC
Config: CFG-20260718-004
Assets: AST-20260718-022

## RUN-20260720-002

Description:
- 使用 SS flow step1000 对 16 个固定 test 条件图进行 `kl=1e-4` flow 采样可用性评估。

Time: 2026-07-20 09:45 UTC
Config: CFG-20260718-004
Assets: AST-20260718-022, AST-20260720-001
