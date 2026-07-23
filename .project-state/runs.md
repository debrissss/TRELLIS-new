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
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011, AST-20260717-012

## RUN-20260717-004

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 1000-step fine-tune 试验。

Time: 2026-07-17 23:30 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011, AST-20260717-014

## RUN-20260718-001

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 batch 8、lr=1e-5 的 1000-step fine-tune 试验。

Time: 2026-07-18 00:10 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011, AST-20260718-001

## RUN-20260718-002

Description:
- SLat encoder + Gaussian decoder batch16/lr1e-5 对照训练在 init sampling 后因 DataLoader worker shared memory 问题失败。

Time: 2026-07-18 00:35 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011, AST-20260718-002

## RUN-20260718-003

Description:
- 使用临时 Python 筛样和 rsync，从 FaceScape train 数据中复制约 50GB 的 SLat encoder + Gaussian decoder 训练子集。

Time: 2026-07-18 12:04 CST
Config:
Assets: AST-20260717-001, AST-20260718-003

## RUN-20260718-004

Description:
- 用户报告 SLat encoder + Gaussian decoder batch16 训练在 step 510-780 区间进入稳定吞吐段。

Time: 2026-07-18 12:12 CST
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011

## RUN-20260718-005

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 `lambda_kl=1e-7`、batch16、lr=1e-5 的 1000-step fine-tune 试验。

Time: 2026-07-18 17:33 UTC
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011, AST-20260718-004

## RUN-20260718-006

Description:
- 从 FaceScape test 中固定抽取 50 个样本，生成 SLat GS checkpoint 评估子集。

Time: 2026-07-18 18:00 UTC
Config:
Assets: AST-20260717-002, AST-20260718-005

## RUN-20260718-007

Description:
- 在 FaceScape eval50/view0 上评估 `lambda_kl=1e-7` batch16 训练的 step1000 非 EMA encoder/decoder checkpoint。

Time: 2026-07-18 18:01 UTC
Config: CFG-20260717-116
Assets: AST-20260718-004, AST-20260718-005, AST-20260718-006

## RUN-20260718-008

Description:
- 在 FaceScape eval50/view0 上评估 `lambda_kl=1e-7` batch16 训练的 step1000 EMA encoder/decoder checkpoint。

Time: 2026-07-18 18:03 UTC
Config: CFG-20260717-116
Assets: AST-20260718-004, AST-20260718-005, AST-20260718-007

## RUN-20260718-009

Description:
- 汇总 step1000 非 EMA 与 EMA checkpoint 的 eval50/view0 评估结果，生成横向对比 CSV。

Time: 2026-07-18 18:04 UTC
Config:
Assets: AST-20260718-006, AST-20260718-007, AST-20260718-008

## RUN-20260718-010

Description:
- 汇总当前本机可发现的全部 `lambda_kl` 训练最终权重 eval50/view0 结果，生成统一横向对比 CSV。

Time: 2026-07-18 18:51 CST
Config:
Assets: AST-20260718-006, AST-20260718-007, AST-20260718-009

## RUN-20260718-011

Description:
- 从 FaceScape train 固定抽取 200 个样本，构造独立 metadata，并用 `lambda_kl=1e-7` step1000 非 EMA encoder 生成 SLat latent。

Time: 2026-07-18 19:05 CST
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260718-004, AST-20260718-010

## RUN-20260718-012

Description:
- 将现有 200 样本 `kl1e-7` non-EMA smoke latent 数据集原地扩展到 1024 样本，并只编码新增的 824 个 latent。

Time: 2026-07-18 19:25 CST
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260718-004, AST-20260718-010

## RUN-20260718-013

Description:
- 从 `/root/autodl-fs/Facescape_cond` 分卷 tar 包解压 `renders_cond`，按现有 train/test metadata 划分建立条件图软链接，并定位 flow smoke 子集条件图。

Time: 2026-07-18 19:40 CST
Config:
Assets: AST-20260718-011, AST-20260718-010

## RUN-20260718-014

Description:
- 诊断用户启动 SLat flow 微调时在 `Sampling 1 images...` 后出现的 `Floating point exception (core dumped)`，并验证可用启动修复。

Time: 2026-07-18 23:20 CST
Config: CFG-20260718-001
Assets: AST-20260718-010

## RUN-20260719-001

Description:
- 用户报告 SLat flow 使用 kl1e-7 non-EMA latent smoke 数据集完成 1000-step fine-tune 试验。

Time: 2026-07-19 21:42 CST
Config: CFG-20260718-001
Assets: AST-20260718-010, AST-20260719-001

## RUN-20260719-002

Description:
- 对 `lambda_kl=1e-7` step1000 非 EMA SLat encoder/decoder 在 eval50/view0 上执行固定重建评估。

Time: 2026-07-19 22:19 CST
Config: CFG-20260717-116
Assets: AST-20260718-004, AST-20260718-005, AST-20260719-002

## RUN-20260719-003

Description:
- 对 `lambda_kl=1e-7` step1000 非 EMA SLat flow checkpoint 执行固定 16 样本条件生成评估。

Time: 2026-07-19 22:19 CST
Config: CFG-20260718-001
Assets: AST-20260718-010, AST-20260719-001, AST-20260719-003

## RUN-20260719-004

Description:
- 汇总 `lambda_kl=1e-7` step1000 非 EMA SLat flow 固定 16 样本生成指标。

Time: 2026-07-19 22:19 CST
Config:
Assets: AST-20260719-003, AST-20260719-004

## RUN-20260719-005

Description:
- 修复 flow generation metrics 兼容入口后重新执行固定 16 样本指标汇总。

Time: 2026-07-19 22:19 CST
Config:
Assets: AST-20260719-003, AST-20260719-005

## RUN-20260719-006

Description:
- 验证 SLat flow 固定生成代码会为每个成功样本默认保存 generated/GT PLY。

Time: 2026-07-19 22:30 CST
Config: CFG-20260718-001
Assets: AST-20260719-001, AST-20260719-007

## RUN-20260720-001

Description:
- 用户报告 SLat encoder + Gaussian decoder 使用 `lambda_kl=1e-6` 完成 1000-step fine-tune 试验。

Time: 2026-07-20 09:45 CST
Config: CFG-20260717-116
Assets: AST-20260717-001, AST-20260717-010, AST-20260717-011, AST-20260720-001

## RUN-20260720-002

Description:
- 对 `lambda_kl=1e-6/1e-7/1e-8` 的 step500 与 step1000 非 EMA SLat enc/dec checkpoint 执行固定样本 KL 梯度贡献诊断。

Time: 2026-07-20 14:29 CST
Config: CFG-20260717-116
Assets: AST-20260718-005, AST-20260720-002
