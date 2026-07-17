# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，使其符合更新后的 `maintain-project-state` 合同。当前已按新要求完成全量静态重扫，并将可独立调用入口拆分为一入口一 `EXE`。

## Current Working Thread
项目当前主线仍是基于 TRELLIS 的 FaceScape 数据处理、SS/SLat 配置训练、GT 重建审计和相关资源管理。旧的 FaceScape 预处理阶段脚本、训练 shell 包装器和 overfit 准备脚本已删除，后续训练应直接使用 `train.py` 或新增入口。

## Relevant State
- EXE-20260717-105
- EXE-20260717-130
- EXE-20260717-131
- EXE-20260717-133
- EXE-20260717-141
- EXE-20260717-142
- EXE-20260717-143
- EXE-20260717-144
- CFG-20260717-103
- CFG-20260717-106
- CFG-20260717-108
- CFG-20260717-111
- ART-20260717-001
- ART-20260717-002
- EVT-20260717-000000-04

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。

## Active Hypotheses
- H1: 下一步若要继续 FaceScape 训练，最稳妥路径是直接调用 `train.py` 搭配 `configs/generation/*facescape*.json` 或 overfit JSON。
  Evidence: 训练 shell 包装器已删除，但配置文件和主训练入口仍存在。
  Uncertainty: 当前主要训练环境应使用 `trellis` 还是 `trellis5090` 未从仓库内确认。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若要执行训练实验，读取 `EXE-20260717-105` 和目标 CFG，再登记运行输入/输出 ART 与 RUN。
2. 若要执行 FaceScape 数据处理，读取对应单入口 EXE，例如 `EXE-20260717-130`、`EXE-20260717-131`、`EXE-20260717-133` 或 `EXE-20260717-140`。
3. 若要执行 GT 重建审计，读取 `EXE-20260717-141`、`EXE-20260717-142` 或 `EXE-20260717-143`。
4. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
