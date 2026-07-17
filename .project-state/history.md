# Current State History


## HST-20260717-204945-01 - current.md snapshot

Description:
- 更新 maintain-project-state skill 后重写 current 前的旧状态快照

# Current State

## Active Goal
维护 TRELLIS-new 项目的持久状态台账，支持后续 FaceScape 数据处理、SS/SLat 微调、overfit 实验和重建审计工作连续推进。

## Current Working Thread
当前项目主线是基于 TRELLIS 的 FaceScape 人脸 3D 数据预处理、Sparse Structure Flow 与 SLat Flow 微调/过拟合实验，以及 GT 重建审计。2026-07-17 已完成一次静态全量扫描并补齐基础 `.project-state`。

## Relevant State
- EXE-20260717-001
- EXE-20260717-012
- CFG-20260717-003
- CFG-20260717-005
- CFG-20260717-007
- CFG-20260717-008
- ART-20260717-001
- ART-20260717-002
- EVT-20260717-000000-03

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前 git 工作区已有用户/既有修改与未跟踪文件，包括 `.project-state/`、FaceScape overfit 配置、新增审计/导出工具等；部分旧预处理、训练包装器和 overfit 准备脚本已按用户要求删除。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；`train/test` 下确认有 `features`、`renders`、`renders_cond`、`voxels`。
- FaceScape metadata 行数为 train 6457、test 721、merged 7177。
- 本地预训练模型目录为 `microsoft/TRELLIS-image-large`，约 3.1G，包含 SS/SLat encoder、decoder、flow 的 safetensors/json checkpoint。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此未创建 RUN 记录。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、5 个 SS 训练 shell 包装器、1 个 SLat 训练 shell 包装器，以及 SS/SLat overfit 准备脚本。

## Interpretations
- 项目已从原始 TRELLIS 扩展为 FaceScape 专用微调工作区，重点在数据预处理、SS/SLat 两阶段 flow 微调和小样本 overfit 验证。
- `configs/generation/overfit` 仍保留，但对应训练 shell 包装器和 overfit 准备脚本已删除；若要训练，需要直接调用 `train.py` 或重新创建入口。
- 根目录两个 `*_mesh_truncated.ply` 已按用户要求删除；截断 mesh GT 重建流程若要继续，需要重新提供输入 mesh。

## Active Hypotheses
- H1: 下一步最可能需要通过 `train.py` 直接验证 SS/SLat 配置，或继续 GT 重建审计工具。
  Evidence: overfit/finetune 配置仍存在；训练 shell 包装器和 overfit 准备脚本已删除；`audit_*_gt_reconstruction.py`、`export_random_train_gt_reconstructions.py` 和 `process_truncated_mesh_gt_reconstructions.py` 仍是相关工具。
  Uncertainty: 未知这些脚本是否已经手动成功跑过；当前没有命令日志或 RUN 记录。
- H2: `cli.py` 的硬编码模型路径 `weights/TRELLIS-image-large` 可能与当前已存在的 `microsoft/TRELLIS-image-large` 不一致。
  Evidence: 静态读取 `cli.py` 看到模型路径拼接为 `weights/TRELLIS-image-large`；扫描确认本地模型在 `microsoft/TRELLIS-image-large`。
  Uncertainty: 未检查是否存在 symlink 或外部 `weights` 目录；未执行推理。

## Current Decision State
- Accepted: `.project-state` 以中文维护；全量扫描只登记静态可确认事实，不臆造历史 RUN。
- Accepted: 以聚合 ART 记录登记大型数据集/模型资源，避免逐样本记录。
- Pending: 是否需要把所有 fine_tuning 小工具进一步拆成单独 EXE 记录，取决于后续具体调用频率。

## Next Actions
1. 若要执行训练实验，优先读取 `EXE-20260717-001`，直接用 `train.py` 搭配对应 CFG 和 ART；已删除的训练包装器不可再调用。
2. 若用户报告某次训练/预处理命令或提供输出目录，补登 RUN，并把输出路径注册为 ART。
3. 若要调试推理 CLI，先确认 `weights/TRELLIS-image-large` 与 `microsoft/TRELLIS-image-large` 的路径关系。
4. 若要继续截断 mesh GT 重建流程，读取 `EXE-20260717-012`，并先重新提供或登记新的截断 mesh 输入 ART。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改文件视为既有项目状态。
- 大型数据目录不做逐文件扫描入账；只记录可复用的聚合资源和关键元数据事实。
- 未经明确证据不把现有数据/输出目录推断为一次 RUN。
- 已删除的预处理/训练包装器/overfit 准备脚本不再作为可用入口推荐。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
