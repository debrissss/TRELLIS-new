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


## HST-20260717-212445-01 - current.md snapshot

Description:
- 新增 slat_enc_dec_gs_fine_tune 配置前的 current 状态快照

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


## HST-20260717-215356-01 - current.md snapshot

Description:
- 登记 SLat encoder 和 GS decoder safetensors 转 pt 前的 current 状态快照

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- EVT-20260717-000000-04

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- 新增 `configs/vae/slat_enc_dec_gs_fine_tune.json`，当前内容与 `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json` 完全一致。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。

## Active Hypotheses
- H1: 下一步若要继续 FaceScape 训练，最稳妥路径是直接调用 `train.py` 搭配目标 JSON 配置，包括新建的 SLat encoder + GS decoder fine-tune 配置。
  Evidence: 训练 shell 包装器已删除，但配置文件和主训练入口仍存在；`slat_enc_dec_gs_fine_tune.json` 已复制自原 VAE 配置。
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
1. 若要执行训练实验，读取 `EXE-20260717-105` 和目标 CFG；SLat encoder + GS decoder fine-tune 可从 `CFG-20260717-116` 开始调整。
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


## HST-20260717-215829-01 - current.md snapshot

Description:
- 修改 slat_enc_dec_gs_fine_tune 训练参数和 finetune_ckpt 前的 current 状态快照

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- RUN-20260717-001
- RUN-20260717-002
- EVT-20260717-000000-06

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- 新增 `configs/vae/slat_enc_dec_gs_fine_tune.json`，当前内容与 `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json` 完全一致。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。

## Active Hypotheses
- H1: 下一步若要继续 FaceScape 训练，最稳妥路径是直接调用 `train.py` 搭配目标 JSON 配置，包括新建的 SLat encoder + GS decoder fine-tune 配置。
  Evidence: 训练 shell 包装器已删除，但配置文件和主训练入口仍存在；`slat_enc_dec_gs_fine_tune.json` 已复制自原 VAE 配置；encoder/decoder `.pt` 已由 RUN-20260717-001 和 RUN-20260717-002 生成。
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
1. 若要执行 SLat encoder + GS decoder fine-tune，读取 `CFG-20260717-116`，并将 `trainer.args.finetune_ckpt.encoder` 指向 ART-20260717-010、`trainer.args.finetune_ckpt.decoder` 指向 ART-20260717-011。
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


## HST-20260717-221849-01 - current.md snapshot

Description:
- SLat fine-tune config and checkpoint conversion state before corrupt FaceScape feature investigation

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- RUN-20260717-001
- RUN-20260717-002
- EVT-20260717-000000-06

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。

## Active Hypotheses
- H1: 下一步若要继续 FaceScape 训练，最稳妥路径是直接调用 `train.py` 搭配目标 JSON 配置，包括新建的 SLat encoder + GS decoder fine-tune 配置。
  Evidence: 训练 shell 包装器已删除，但配置文件和主训练入口仍存在；`slat_enc_dec_gs_fine_tune.json` 已设置 1000 step 和 encoder/decoder finetune checkpoint；encoder/decoder `.pt` 已由 RUN-20260717-001 和 RUN-20260717-002 生成。
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
1. 若要执行 SLat encoder + GS decoder fine-tune，读取 `CFG-20260717-116` 并通过 `EXE-20260717-105` 启动训练；该配置已指向 ART-20260717-010 和 ART-20260717-011。
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


## HST-20260717-224509-01 - current.md snapshot

Description:
- Current state before marking corrupt FaceScape feature metadata entries false

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。训练已能启动并保存 step 500 checkpoint，但用户报告 DataLoader 在读取损坏的 FaceScape DINO 特征缓存时失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- EVT-20260717-000000-08

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。

## Active Hypotheses
- H1: 下一步继续 FaceScape 训练前，应先重生成两个损坏的 DINOv2 feature `.npz`。
  Evidence: `zipfile.testzip()` 对 ART-20260717-012 和 ART-20260717-013 均返回 `patchtokens.npy`；两个文件是同目录最小的两个 `.npz`，正常文件至少约 15.6 MB。
  Uncertainty: 是否还有大小正常但内部损坏的 `.npz` 尚未做全量 `testzip()` 扫描。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；未提供的启动命令和输出目录标记为 unknown。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013。
2. 可选：对 `datasets/Facescape/train/features/dinov2_vitl14_reg/*.npz` 做全量 `zipfile.testzip()` 完整性扫描，排除大小正常但内部损坏的缓存。
3. 从 step 500 checkpoint 或最近可用 checkpoint 恢复 SLat encoder + GS decoder fine-tune。
4. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；只给出重生成建议。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260717-230305-01 - current.md snapshot

Description:
- Current state before recording terminal progress logging analysis

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。训练曾在 step 500 后遇到损坏 DINO 特征缓存；已扫描 train/test metadata 并把两个坏 train 样本的特征可用标记置为 `False`。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- EVT-20260717-000000-08
- EVT-20260717-000000-09

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。

## Active Hypotheses
- H1: 下一步可以先尝试恢复 FaceScape fine-tune，验证 metadata 过滤是否足以避开坏样本。
  Evidence: train/test 扫描已把两个坏特征样本的 `feature_dinov2_vitl14_reg` 写为 `False`；test 未发现坏样本。
  Uncertainty: 当前训练 dataset 是否在所有路径上都严格按 metadata 的 feature 列过滤样本；坏 `.npz` 文件本身仍存在。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；未提供的启动命令和输出目录标记为 unknown。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 从 step 500 checkpoint 或最近可用 checkpoint 恢复 SLat encoder + GS decoder fine-tune，验证 metadata 禁用坏样本后是否继续正常。
2. 若训练仍碰到这两个坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
3. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
4. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260717-230439-01 - current.md snapshot

Description:
- Current state before setting fine-tune terminal progress interval to 10 steps

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。训练曾在 step 500 后遇到损坏 DINO 特征缓存；已扫描 train/test metadata 并把两个坏 train 样本的特征可用标记置为 `False`。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置未设置 `i_print`，因此使用默认 `1000`。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，如果不设置 `i_print`，终端几乎看不到中间进度；这不代表 `i_log` 没生效，文件日志可能仍每 100 step 写入。

## Active Hypotheses
- H1: 下一步可以先尝试恢复 FaceScape fine-tune，验证 metadata 过滤是否足以避开坏样本。
  Evidence: train/test 扫描已把两个坏特征样本的 `feature_dinov2_vitl14_reg` 写为 `False`；test 未发现坏样本。
  Uncertainty: 当前训练 dataset 是否在所有路径上都严格按 metadata 的 feature 列过滤样本；坏 `.npz` 文件本身仍存在。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；未提供的启动命令和输出目录标记为 unknown。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 从 step 500 checkpoint 或最近可用 checkpoint 恢复 SLat encoder + GS decoder fine-tune，验证 metadata 禁用坏样本后是否继续正常。
2. 若希望恢复训练时控制台每 100 step 打印进度，在 `CFG-20260717-116` 的 `trainer.args` 中添加 `"i_print": 100`。
3. 若训练仍碰到这两个坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260717-232726-01 - current.md snapshot

Description:
- Current state before recording completed 1000-step SLat fine-tune analysis

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。训练曾在 step 500 后遇到损坏 DINO 特征缓存；已扫描 train/test metadata 并把两个坏 train 样本的特征可用标记置为 `False`。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。

## Active Hypotheses
- H1: 下一步可以先尝试恢复 FaceScape fine-tune，验证 metadata 过滤是否足以避开坏样本。
  Evidence: train/test 扫描已把两个坏特征样本的 `feature_dinov2_vitl14_reg` 写为 `False`；test 未发现坏样本。
  Uncertainty: 当前训练 dataset 是否在所有路径上都严格按 metadata 的 feature 列过滤样本；坏 `.npz` 文件本身仍存在。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；未提供的启动命令和输出目录标记为 unknown。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 从 step 500 checkpoint 或最近可用 checkpoint 恢复 SLat encoder + GS decoder fine-tune，验证 metadata 禁用坏样本后是否继续正常。
2. 观察终端是否每 10 step 打印 `Step/Elapsed/Speed/ETA`，并检查 `log.txt`/`loss.txt` 是否每 100 step 写入。
3. 若训练仍碰到这两个坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260717-233843-01 - current.md snapshot

Description:
- Current state before updating fine-tune batch size and learning rate

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用后，1000-step fine-tune 试验已完整跑完并产出日志、samples 和 step 500/1000 checkpoint。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。

## Active Hypotheses
- H1: 下一步应使用固定验证/可视化流程评估 step1000 和 EMA step1000，再决定是否继续延长 fine-tune。
  Evidence: RUN-20260717-004 已完整跑完，final sample 视觉上与 GT 接近；但无独立 test set 指标，训练 loss 仍有明显波动。
  Uncertainty: step1000 普通 checkpoint 与 EMA checkpoint 的下游重建/泛化质量差异未知。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；完成的 1000-step 训练登记为 RUN-20260717-004。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 对 RUN-20260717-004 的 step1000 和 EMA step1000 checkpoint 做固定 test/holdout 可视化或指标评估。
2. 若 step1000 表现稳定，从 step1000 继续延长训练；若过拟合或波动明显，考虑降低学习率或增加验证监控。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-000820-01 - current.md snapshot

Description:
- Current state before recording batch-8 lr-1e-5 fine-tune result analysis

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用后，1000-step fine-tune 试验已完整跑完并产出日志、samples 和 step 500/1000 checkpoint。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- 新 batch/lr 设置会让每次 optimizer update 看到 8 个样本，同时保持每次前后向 micro-batch 为 2；比原 `4/2` 梯度更稳，显存压力预计接近原设置，但每 step 计算量约增加。

## Active Hypotheses
- H1: 下一步应使用固定验证/可视化流程评估 step1000 和 EMA step1000，再决定是否继续延长 fine-tune。
  Evidence: RUN-20260717-004 已完整跑完，final sample 视觉上与 GT 接近；但无独立 test set 指标，训练 loss 仍有明显波动。
  Uncertainty: step1000 普通 checkpoint 与 EMA checkpoint 的下游重建/泛化质量差异未知。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；完成的 1000-step 训练登记为 RUN-20260717-004。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 对 RUN-20260717-004 的 step1000 和 EMA step1000 checkpoint 做固定 test/holdout 可视化或指标评估。
2. 若 step1000 表现稳定，用当前 `batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5` 从 step1000 继续延长训练；若过拟合或波动明显，增加验证监控或进一步调低学习率。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-002107-01 - current.md snapshot

Description:
- Current state before setting fine-tune ablation to batch16 lr1e-5

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验均已完整跑完并产出日志、samples 和 checkpoint。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 fine-tune 配置：`max_steps=1000`、`batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- 新 batch/lr 设置会让每次 optimizer update 看到 8 个样本，同时保持每次前后向 micro-batch 为 2；比原 `4/2` 梯度更稳，显存压力预计接近原设置，但每 step 计算量约增加。

## Active Hypotheses
- H1: 下一步应优先评估 v2 的 step1000 和 EMA step1000，再决定是否继续延长 fine-tune。
  Evidence: RUN-20260718-001 比 RUN-20260717-004 loss 略低、grad_norm 明显更低、尖峰更小，final sample 视觉上与 GT 接近。
  Uncertainty: v2 仍无独立 test set 指标，EMA 与 non-EMA 的下游重建质量差异未知。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 对 RUN-20260718-001 的 v2 step1000 和 EMA step1000 checkpoint 做固定 test/holdout 可视化或指标评估。
2. 若 v2 step1000 表现稳定，用当前 `batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5` 从 v2 step1000 继续延长训练；若过拟合或波动明显，增加验证监控或进一步调低学习率。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-101328-01 - current.md snapshot

Description:
- Current state before recording batch16 DataLoader shared-memory failure analysis

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验均已完整跑完并产出日志、samples 和 checkpoint。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。

## Active Hypotheses
- H1: 下一步应优先评估 v2 的 step1000 和 EMA step1000，再决定是否继续延长 fine-tune。
  Evidence: RUN-20260718-001 比 RUN-20260717-004 loss 略低、grad_norm 明显更低、尖峰更小，final sample 视觉上与 GT 接近。
  Uncertainty: v2 仍无独立 test set 指标，EMA 与 non-EMA 的下游重建质量差异未知。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 用独立输出目录运行 batch16/lr1e-5 对照实验，例如 `outputs/slat_enc_dec_gs_fine_tune_b16_lr1e-5`。
2. 对比 batch16 run 与 RUN-20260718-001 的最后 100 step loss、grad_norm、step time、final sample 和固定验证样本。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-103828-01 - current.md snapshot

Description:
- Current state before adding safe DataLoader settings for batch16 fine-tune

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。

## Active Hypotheses
- H1: batch16 对照若要继续，必须先降低 DataLoader worker/prefetch 并禁用 auto-retry；否则应回退 batch8/lr1e-5。
  Evidence: RUN-20260718-002 在 batch16 下出现 PyTorch DataLoader shared memory bus error；batch8/lr1e-5 已完整跑通且指标更稳。
  Uncertainty: batch16 在安全 DataLoader 设置下的速度和指标尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，对配置增加 `dataloader_num_workers=0` 或小值、`dataloader_persistent_workers=false`、`prefetch_data=false`，并启动命令加 `--auto_retry 0`。
2. 更稳妥的主线是回退 RUN-20260718-001 的 batch8/lr1e-5 配置，并优先评估 v2 step1000/EMA step1000。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-105453-01 - current.md snapshot

Description:
- Current state before recording batch16 GPU utilization oscillation analysis

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=0`、`dataloader_persistent_workers=false`、`prefetch_data=false`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- batch16 配置已加入低共享内存 DataLoader 设置：禁用 worker、persistent workers 和 trainer 预取。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。

## Active Hypotheses
- H1: batch16 对照现在可用低共享内存 DataLoader 设置再次尝试，但启动命令仍应加 `--auto_retry 0`。
  Evidence: CFG-20260717-116 已设置 `dataloader_num_workers=0`、`dataloader_persistent_workers=false`、`prefetch_data=false`，可避免默认 52 workers 预取完整 batch。
  Uncertainty: batch16 在该设置下的速度和指标尚未验证；单 worker 数据加载可能成为瓶颈。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，用当前低共享内存配置并在启动命令加 `--auto_retry 0`，输出目录使用独立路径。
2. 若 batch16 仍失败或过慢，回退 RUN-20260718-001 的 batch8/lr1e-5 配置，并优先评估 v2 step1000/EMA step1000。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-105944-01 - current.md snapshot

Description:
- Current state before setting batch16 DataLoader workers to 2

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=0`、`dataloader_persistent_workers=false`、`prefetch_data=false`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- batch16 配置已加入低共享内存 DataLoader 设置：禁用 worker、persistent workers 和 trainer 预取。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 安全配置当前主要瓶颈是同步数据加载导致的 GPU starvation。
  Evidence: 用户报告 GPU 利用率在高占用和 0 占用间周期性交替；代码在 `prefetch_data=false` 时每 step 同步加载完整 batch16。
  Uncertainty: 将 `dataloader_num_workers` 调到 2 是否足以平衡 `/dev/shm` 与 GPU 利用率尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，优先试 `dataloader_num_workers=2`、`dataloader_persistent_workers=false`、`prefetch_data=false`，启动命令加 `--auto_retry 0`。
2. 若 batch16 仍出现长时间 GPU starvation 或 shm/bus error，回退 RUN-20260718-001 的 batch8/lr1e-5 配置，并优先评估 v2 step1000/EMA step1000。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-110942-01 - current.md snapshot

Description:
- Current state before increasing batch16 DataLoader workers from 2 to 4

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=2`、`dataloader_persistent_workers=false`、`prefetch_data=false`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- batch16 配置已改为折中 DataLoader 设置：`dataloader_num_workers=2`，禁用 persistent workers 和 trainer 预取。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置现在可用 2 个 DataLoader worker 尝试缓解 GPU starvation，同时保持较低 shm 风险。
  Evidence: CFG-20260717-116 已设置 `dataloader_num_workers=2`、`dataloader_persistent_workers=false`、`prefetch_data=false`。
  Uncertainty: workers=2 是否足以平衡 `/dev/shm` 与 GPU 利用率尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，使用当前 `dataloader_num_workers=2`、`dataloader_persistent_workers=false`、`prefetch_data=false` 配置，启动命令加 `--auto_retry 0`。
2. 若 batch16 仍出现长时间 GPU starvation 或 shm/bus error，回退 RUN-20260718-001 的 batch8/lr1e-5 配置，并优先评估 v2 step1000/EMA step1000。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-112116-01 - current.md snapshot

Description:
- Current state before enabling trainer prefetch for batch16 workers4

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06
- EVT-20260718-000000-07

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=4`、`dataloader_persistent_workers=false`、`prefetch_data=false`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- 用户报告 batch16 配置在 `dataloader_num_workers=2` 后速度比 workers=0 快一倍多；当前已进一步改为 `dataloader_num_workers=4`，仍禁用 persistent workers 和 trainer 预取。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置可继续用 4 个 DataLoader worker 试探吞吐上限，同时保持较低 shm 风险。
  Evidence: 用户报告 workers=2 后速度比 workers=0 快一倍多；CFG-20260717-116 已设置 `dataloader_num_workers=4`、`dataloader_persistent_workers=false`、`prefetch_data=false`。
  Uncertainty: workers=4 是否继续提升速度或重新触发 shm/bus error 尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，使用当前 `dataloader_num_workers=4`、`dataloader_persistent_workers=false`、`prefetch_data=false` 配置，启动命令加 `--auto_retry 0`。
2. 若 workers=4 出现 shm/bus error，回退 workers=2；若 batch16 仍出现长时间 GPU starvation，回退 RUN-20260718-001 的 batch8/lr1e-5 配置。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-113408-01 - current.md snapshot

Description:
- Current state before increasing batch16 DataLoader workers from 4 to 8

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06
- EVT-20260718-000000-07
- EVT-20260718-000000-08

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=4`、`dataloader_persistent_workers=false`、`prefetch_data=true`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- 用户报告 batch16 配置在 `dataloader_num_workers=2` 后速度比 workers=0 快一倍多；workers=4 与 workers=2 差别不大但 GPU 利用率仍会掉到 10% 以下。当前方案 A 已设置 `dataloader_num_workers=4`、`dataloader_persistent_workers=false`、`prefetch_data=true`。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置启用 trainer prefetch 后，GPU 空等应减少，但显存峰值和 shm 风险需要观察。
  Evidence: `prefetch_data=true` 会在训练当前 batch 时提前搬运下一 batch；CFG-20260717-116 已设置 workers=4、persistent=false、prefetch=true。
  Uncertainty: 该设置是否会重新触发 bus error 或显存峰值过高尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，使用当前 `dataloader_num_workers=4`、`dataloader_persistent_workers=false`、`prefetch_data=true` 配置，启动命令加 `--auto_retry 0`。
2. 若 prefetch=true 出现 shm/bus error 或显存峰值过高，回退 `prefetch_data=false` 或 workers=2；若 batch16 仍出现长时间 GPU starvation，回退 RUN-20260718-001 的 batch8/lr1e-5 配置。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-114244-01 - current.md snapshot

Description:
- Current state before enabling persistent DataLoader workers for batch16

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06
- EVT-20260718-000000-07
- EVT-20260718-000000-08
- EVT-20260718-000000-09

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=false`、`prefetch_data=true`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- 用户报告 batch16 配置在 `dataloader_num_workers=2` 后速度比 workers=0 快一倍多；workers=4 与 workers=2 差别不大，且 prefetch=true 后 GPU 低占用频率没有明显减少但显存峰值可接受。当前已设置 `dataloader_num_workers=8`、`dataloader_persistent_workers=false`、`prefetch_data=true`。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置现在通过 workers=8 进一步试探 CPU/I/O 吞吐，目标是减少 GPU 空等。
  Evidence: workers=4 与 workers=2 差别不大，prefetch=true 未明显降低低占用频率；CFG-20260717-116 已设置 workers=8、persistent=false、prefetch=true。
  Uncertainty: workers=8 是否会改善 GPU 利用率或重新触发 bus error 尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，使用当前 `dataloader_num_workers=8`、`dataloader_persistent_workers=false`、`prefetch_data=true` 配置，启动命令加 `--auto_retry 0`。
2. 若 workers=8 出现 shm/bus error，回退 workers=4 或 2；若 batch16 仍出现长时间 GPU starvation，回退 RUN-20260718-001 的 batch8/lr1e-5 配置。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-115156-01 - current.md snapshot

Description:
- Current state before recording persistent workers partial GPU utilization improvement

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06
- EVT-20260718-000000-07
- EVT-20260718-000000-08
- EVT-20260718-000000-09
- EVT-20260718-000000-10

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- 用户报告 batch16 配置在 `dataloader_num_workers=2` 后速度比 workers=0 快一倍多；workers=4 与 workers=2 差别不大，prefetch=true 后 GPU 低占用频率没有明显减少，workers=8/persistent=false 也没有好转。当前已设置 `dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置现在通过 persistent workers 测试是否减少 DataLoader 供应抖动。
  Evidence: workers=8、persistent=false、prefetch=true 没有明显好转；CFG-20260717-116 已设置 workers=8、persistent=true、prefetch=true。
  Uncertainty: persistent workers 是否改善 GPU 利用率或重新触发 shm/bus error 尚未验证。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 若继续 batch16，使用当前 `dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true` 配置，启动命令加 `--auto_retry 0`。
2. 若 persistent=true 出现 shm/bus error，回退 persistent=false 或 workers=4；若 batch16 仍出现长时间 GPU starvation，回退 RUN-20260718-001 的 batch8/lr1e-5 配置。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-120505-01 - current.md snapshot

Description:
- 记录 FaceScape SLat GS fine-tune 的 batch16 DataLoader 调参状态，尚未包含 50GB 可迁移子集。

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并跟踪当前 FaceScape SLat encoder + GS decoder fine-tune 的配置、权重转换和数据质量问题。

## Current Working Thread
项目当前主线是基于 TRELLIS 的 FaceScape SLat encoder + GS decoder fine-tune。坏 DINO 特征缓存已通过 metadata 禁用；batch4/lr1e-4 和 batch8/lr1e-5 两轮 1000-step 试验已完整跑完，batch16/lr1e-5 对照在 init sampling 后因 DataLoader shared memory 问题失败。

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
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-002
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
- ART-20260717-013
- ART-20260717-014
- ART-20260718-001
- ART-20260718-002
- RUN-20260717-001
- RUN-20260717-002
- RUN-20260717-003
- RUN-20260717-004
- RUN-20260718-001
- RUN-20260718-002
- EVT-20260717-000000-08
- EVT-20260717-000000-09
- EVT-20260717-000000-10
- EVT-20260717-000000-11
- EVT-20260717-000000-12
- EVT-20260717-000000-13
- EVT-20260718-000000-01
- EVT-20260718-000000-02
- EVT-20260718-000000-03
- EVT-20260718-000000-04
- EVT-20260718-000000-05
- EVT-20260718-000000-06
- EVT-20260718-000000-07
- EVT-20260718-000000-08
- EVT-20260718-000000-09
- EVT-20260718-000000-10
- EVT-20260718-000000-11

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 更新后的 skill 要求每个可独立调用脚本、模块、命令或任务目标都单独建立一个 `EXE`，不得按目录或用途聚合。
- `executables.md` 已重写为 45 条单入口记录；旧的聚合 EXE 记录已被覆盖。
- `experiment-configs.md` 已重写为 15 条单配置文件记录；`artifacts.md` 已重写为 7 条单资源路径记录。
- `train.py` 是训练主入口；会把 `command.txt` 和 `config.json` 写入 `output_dir`，并在 rank 0 写入模型摘要。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 已调整为 batch16 对照配置：`max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`、`i_print=10`、`i_log=100`、`i_save=500`，并设置 encoder/decoder `.pt` 预训练权重。
- 已将 SLat encoder 和 SLat Gaussian decoder 的 safetensors 权重转换为同目录 `.pt` state_dict，可用于 `trainer.args.finetune_ckpt`。
- 用户报告 SLat fine-tune 训练在 step 500 checkpoint 保存完成后，DataLoader worker 读取 FaceScape instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 时抛出 `zipfile.BadZipFile`。
- 本地验证 `datasets/Facescape/train/features/dinov2_vitl14_reg/3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca.npz` 仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
- 同目录还发现 `3ad9e565cc98b4b189e7a7970c48f0767f4ed6ea427f336b74a8d83079d4ecec.npz` 仅 31951 bytes，具有相同 `patchtokens.npy` 损坏模式。
- 已用临时脚本扫描 `datasets/Facescape/train` 和 `datasets/Facescape/test` metadata：train 共 6456 行，命中 2 个坏样本并写回 `feature_dinov2_vitl14_reg=False`；test 共 720 行，命中 0 个坏样本。
- 临时扫描脚本已删除。
- 已分析训练日志无终端进度的问题：`i_log=100` 只控制 `log.txt`/`loss.txt`/TensorBoard，终端 `Step/Elapsed/Speed/ETA` 由 `i_print` 控制；当前 fine-tune 配置已设置 `i_print=10`。
- 1000-step fine-tune 输出位于 `outputs/slat_enc_dec_gs_fine_tune`；`log.txt` 与 `loss.txt` 均有 1000 行，step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint 均已保存。
- 该次 run 的前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%；最终 step loss 为 0.0250695。
- final sample 的重建图与 GT 图视觉上高度接近。
- 1000-step 试验后，fine-tune 配置已改为有效 batch 8、micro-batch 2、学习率 `1e-5`，用于更稳地继续微调。
- 新一轮 batch8/lr1e-5 结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`；旧路径 `outputs/slat_enc_dec_gs_fine_tune` 仍是 batch4/lr1e-4 结果。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%；最终 step loss 为 0.0209562。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step time 高约 82.5%。
- 当前配置已从 v2 的 batch8/lr1e-5 改为 batch16/lr1e-5 对照实验；micro-batch 仍为 2。
- batch16/lr1e-5 运行到 `Sampling 1 images... Done.` 后失败，用户报告 DataLoader worker 反复输出 shared memory bus error；v3 输出目录没有完整 `log.txt` 或 checkpoint。
- 用户报告 batch16 配置在 `dataloader_num_workers=2` 后速度比 workers=0 快一倍多；workers=4 与 workers=2 差别不大，prefetch=true 后 GPU 低占用频率没有明显减少，workers=8/persistent=false 也没有好转。当前已设置 `dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`。
- 用户报告 workers=8、persistent=true、prefetch=true 后 GPU 低占用次数约减少 30%，但仍然频繁。
- 用户报告 batch16 安全 DataLoader 设置运行后，约 80 steps 起 GPU 利用率在高占用和 0 占用间周期性交替，显存从约 15GB 阶梯升至约 20GB/21.5GB。
- FaceScape 主数据资源位于 `datasets/Facescape`，约 441G；本地预训练模型目录位于 `microsoft/TRELLIS-image-large`，约 3.1G。
- 2026-07-17 已删除 `fine_tuning/preprocess_stage1.py`、`fine_tuning/preprocess_stage2.py`、SS/SLat 训练 shell 包装器、SS/SLat overfit 准备脚本，以及根目录两个截断 mesh PLY 输入。
- 当前扫描没有发现 `outputs/*/command.txt`、训练输出目录或其它足以证明历史运行命令的记录，因此仍不创建 RUN 记录。

## Interpretations
- 旧 `.project-state` 的主要不合规点是聚合 `EXE`、聚合 `CFG`、聚合/多路径 `ART`；这些已直接覆盖为新合同友好的单记录形式。
- FaceScape 训练配置仍存在，但相关 shell 包装器已删除；若要训练，需要直接调用 `EXE-20260717-105`。
- FaceScape 数据处理仍有 batch pipeline、render、extract feature、voxelize、metadata 修复/合并/拆分等独立入口可用。
- 当前训练失败点是数据缓存完整性问题：损坏 `.npz` 的 central directory 仍能列出条目，但 `patchtokens.npy` 数据体被截断或覆盖，因此 `np.load(... )['patchtokens']` 触发 BadZipFile。
- 如果当前 dataset 构建逻辑按 `feature_dinov2_vitl14_reg` 过滤样本，这两个坏样本应被跳过；如果某个路径直接从目录枚举 `.npz`，仍需要重建或移走坏 `.npz`。
- 对 1000-step 短程 fine-tune，`i_print=10` 会让 rank 0 终端每 10 step 打印一次进度；`i_log=100` 仍只负责文件日志和 TensorBoard。
- 本次 1000-step 训练验证了配置、权重和数据过滤链路可跑通；但训练 loss 波动较大且只有训练集日志，不能视作泛化质量已验证。
- batch16/lr1e-5 对照会让每次 optimizer update 看到 16 个样本，同时保持每次前后向 micro-batch 为 2；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
- `batch_split=8` 不会降低 DataLoader 侧共享内存压力，因为代码先取出完整 batch16 并搬到设备，再切成 8 个 micro-batch。
- DataLoader 默认 worker 数过高，本机此前打印为 52；batch16 的完整 batch 经多 worker 预取时容易耗尽 `/dev/shm`。
- `train.py` 默认 `auto_retry=3`，可解释失败后似乎自动重启训练；排查或高风险配置应使用 `--auto_retry 0`。
- batch16 的 DataLoader 调参已有改善但未根治 GPU starvation，继续加 worker 的边际收益可能有限；需要比较总体 wall time 和验证质量，而不只看瞬时 GPU 利用率。
- 在 `dataloader_num_workers=0` 且 `prefetch_data=false` 下，数据加载与搬运完全同步，batch16 的完整 batch 加载会让 GPU 等待 CPU/I/O，表现为 GPU 利用率周期性降到 0。
- 显存阶梯式上升更像 CUDA caching allocator、样本 sparse token 数变化和 elastic memory controller 动态 checkpointing 的共同结果，不一定是泄漏。

## Active Hypotheses
- H1: batch16 配置的 GPU starvation 不是单靠 worker/persistent/prefetch 能完全解决。
  Evidence: workers=8、persistent=true、prefetch=true 只让低占用次数减少约 30%，仍频繁出现。
  Uncertainty: batch16 的验证质量是否足以抵消数据管线成本尚未确定。
- H2: 若要继续 GT 重建审计，仍可使用 `audit_ss_gt_reconstruction.py`、`audit_slat_gt_reconstruction.py` 和 `export_random_train_gt_reconstructions.py`。
  Evidence: 三个脚本仍存在并已登记为独立 EXE。
  Uncertainty: 默认 decoder 路径和外部权重是否全部存在未在本次执行运行验证。

## Current Decision State
- Accepted: `.project-state` 以中文维护。
- Accepted: 全量扫描只登记静态可确认事实，不根据输出形目录臆造 RUN。
- Accepted: 当前 ledger 直接覆盖旧聚合记录以符合新 skill。
- Accepted: 用户报告的训练失败登记为 RUN-20260717-003；两轮完成的 1000-step 训练分别登记为 RUN-20260717-004 和 RUN-20260718-001。
- Accepted: 临时扫描脚本用完即删，不登记为长期 EXE。
- Pending: 是否需要为 `trellis/*/__init__.py` 中的内部 `__main__` 调试块登记 EXE；本次按“直接 CLI/脚本入口”口径排除内部包调试入口。

## Next Actions
1. 记录 batch16 当前配置的每 100 step wall time、最后 100 step loss 和 final/fixed validation sample，再与 RUN-20260718-001 的 batch8/lr1e-5 比较。
2. 若 batch16 没有明显质量优势，回退 RUN-20260718-001 的 batch8/lr1e-5 配置；若质量明显更好，再考虑数据预处理/缓存方案而不是继续盲目加 worker。
3. 若训练仍碰到坏 `.npz`，用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 重生成 ART-20260717-012 和 ART-20260717-013，或直接移走坏缓存。
4. 若出现新的坏样本，再按相同结构校验方式扫描并更新 metadata。
5. 若用户恢复或新增已删除的预处理/训练包装器，登记为新的 EXE，不复用旧聚合 ID。

## Constraints
- 不回滚当前工作区已有修改；`.project-state` 之外的未跟踪/已修改/已删除文件视为既有项目状态。
- 大型数据目录不做逐样本入账；只记录可复用的聚合资源路径。
- 未经明确命令日志、metadata 或用户报告，不把现有数据/输出目录推断为一次 RUN。
- 新合同下不得再新增按目录或功能聚合的 EXE 记录。
- 未经用户确认，不删除或覆盖坏 `.npz` 数据文件；当前只修改 metadata 的 feature 可用标记。

## Open Questions
- FaceScape 数据集和 overfit 子集是否已有人工执行成功记录但未保存在仓库内？
- 当前应使用 `trellis` 还是 `trellis5090` 作为主要训练环境？
- `weights/fine_tune/*` 初始化权重是否位于工作区外部，或尚未同步到本仓库？
- step 500 checkpoint 的具体输出目录是什么，是否可直接用于恢复训练？


## HST-20260718-121257-01 - current.md snapshot

Description:
- 记录低配测速子集已准备完成，尚未包含 batch16 稳定期 1803 steps/h 吞吐。

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持当前 FaceScape SLat encoder + GS decoder fine-tune 的成本/速度对比。

## Current Working Thread
用户正在评估更贵 GPU 的速度收益是否能覆盖成本。当前已提交并推送项目状态，同时准备了一个约 50GB 的 FaceScape SLat GS 训练子集，供低配置机器测试训练吞吐。

## Relevant State
- EXE-20260717-105
- EXE-20260718-001
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
- ART-20260718-001
- ART-20260718-002
- ART-20260718-003
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- EVT-20260718-120400-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/track-untracked-state`。
- 2026-07-18 已提交并推送 commit `837e3f9 Add SLat GS fine-tune config and logs`。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 当前为 batch16 对照配置：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`。
- 用户报告 batch16 在当前 DataLoader 设置下约为 1700 steps/h。
- `outputs/slat_enc_dec_gs_fine_tune_v2` 是已完成的 batch8/lr1e-5 1000-step 对照；最后 100 step 平均 loss 为 0.0208222。
- `outputs/slat_enc_dec_gs_fine_tune_v3` 记录了 batch16 早期因 DataLoader shared memory bus error 失败的输出。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`。
- 该子集的 `train/metadata.csv` 为 1178 个样本加表头，包含 1178 个 `renders/<sha>/` 目录和 1178 个 `features/dinov2_vitl14_reg/<sha>.npz` 文件。
- 一致性检查确认子集 metadata 中每个样本都有 feature 文件和 `renders/<sha>/transforms.json`。
- 该子集不包含 `voxels/`、`renders_cond/` 或预训练 `.pt` checkpoint。

## Interpretations
- SLat encoder + Gaussian decoder 训练数据路径需要 metadata、render 图像/相机 transforms、DINOv2 patch token feature；当前子集覆盖这些必要输入。
- 当前低配测速的关键指标应同时看 `steps/h` 和 `samples/h`：batch16 的 1700 steps/h 约等于 27200 samples/h。
- 若低配机器跑 batch8 或 batch16，需要按有效 batch 统一换算样本吞吐，否则只比较 GPU 利用率或 steps/h 容易误判成本收益。

## Active Hypotheses
- H1: batch16 的吞吐优势主要来自每 step 样本数更大，但样本吞吐与 batch8 可能接近。
  Evidence: 用户报告 batch16 约 1700 steps/h；先前 batch8 约可换算到相近 samples/h 量级。
  Uncertainty: 低配机器上的 CPU/I/O、显存和 `/dev/shm` 瓶颈可能改变这个关系。
- H2: 低配机器若复用 batch16 配置，可能先受显存或 DataLoader 共享内存限制。
  Evidence: 当前机器 batch16 曾触发 DataLoader shm bus error；`batch_split` 不降低 DataLoader 完整 batch 压力。
  Uncertainty: 另一台机器的 `/dev/shm`、CPU 核数、磁盘速度和 PyTorch worker 行为未知。

## Current Decision State
- Accepted: 为低配机器准备约 50GB 子集，而不是搬运完整约 441GB FaceScape 数据集。
- Accepted: 子集只复制当前 SLat GS 训练读取的数据，不复制 `voxels/` 和 `renders_cond/`。
- Accepted: 避免在当前机器实际启动训练或压力检查，以免再次触发内存/共享内存问题影响其它任务。
- Pending: 低配机器应使用 batch8 还是 batch16 做第一轮速度测试，取决于其显存和 `/dev/shm`。

## Next Actions
1. 将 `datasets/Facescape_slat_gs_50gb` 同步到低配置机器。
2. 同步 TRELLIS 代码、`configs/vae/slat_enc_dec_gs_fine_tune.json`、以及 SLat encoder/GS decoder `.pt` 预训练权重。
3. 在低配机器先用 `--auto_retry 0` 跑短程测试，记录 steps/h、samples/h、GPU 利用率、显存峰值和是否出现 DataLoader bus error。
4. 用统一的 samples/h 与单位小时成本比较当前昂贵 GPU 和低配机器的实际性价比。

## Constraints
- 不启动训练或重型数据检查。
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 迁移子集时需要保留 `train/metadata.csv`、`train/renders/` 和 `train/features/dinov2_vitl14_reg/` 的相对路径结构。

## Open Questions
- 低配置机器的 GPU 显存、CPU 核数、磁盘类型和 `/dev/shm` 大小是多少？
- 低配机器上是否已经有 `microsoft/TRELLIS-image-large/ckpts/*.pt` 微调初始化权重？


## HST-20260718-154559-01 - current.md snapshot

Description:
- SS encoder/decoder fine-tune readiness audit supersedes prior SLat GS cost comparison active state

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持当前 FaceScape SLat encoder + GS decoder fine-tune 的成本/速度对比。

## Current Working Thread
用户正在评估更贵 GPU 的速度收益是否能覆盖成本。当前已准备约 50GB 的 FaceScape SLat GS 训练子集，并已记录昂贵 GPU 上 batch16 稳定训练段吞吐作为对比基线。

## Relevant State
- EXE-20260717-105
- EXE-20260718-001
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
- ART-20260718-001
- ART-20260718-002
- ART-20260718-003
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/track-untracked-state`。
- 2026-07-18 已提交并推送 commit `837e3f9 Add SLat GS fine-tune config and logs`。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 当前为 batch16 对照配置：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`。
- 用户报告 batch16 在当前 DataLoader 设置下，step 510-780 稳定段平均速度为 `1803.39 steps/h`，约 `28854 samples/h`，平均每 step 约 `1.996s`。
- `outputs/slat_enc_dec_gs_fine_tune_v2` 是已完成的 batch8/lr1e-5 1000-step 对照；最后 100 step 平均 loss 为 0.0208222。
- `outputs/slat_enc_dec_gs_fine_tune_v3` 记录了 batch16 早期因 DataLoader shared memory bus error 失败的输出。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`。
- 该子集的 `train/metadata.csv` 为 1178 个样本加表头，包含 1178 个 `renders/<sha>/` 目录和 1178 个 `features/dinov2_vitl14_reg/<sha>.npz` 文件。
- 一致性检查确认子集 metadata 中每个样本都有 feature 文件和 `renders/<sha>/transforms.json`。
- 该子集不包含 `voxels/`、`renders_cond/` 或预训练 `.pt` checkpoint。

## Interpretations
- SLat encoder + Gaussian decoder 训练数据路径需要 metadata、render 图像/相机 transforms、DINOv2 patch token feature；当前子集覆盖这些必要输入。
- 当前低配测速的关键指标应同时看 `steps/h` 和 `samples/h`：batch16 稳定段 `1803.39 steps/h` 约等于 `28854 samples/h`。
- 若低配机器跑 batch8 或 batch16，需要按有效 batch 统一换算样本吞吐，否则只比较 GPU 利用率或 steps/h 容易误判成本收益。

## Active Hypotheses
- H1: batch16 的吞吐优势主要来自每 step 样本数更大，但样本吞吐与 batch8 可能接近。
  Evidence: 用户报告 batch16 稳定段约 1803.39 steps/h，即约 28854 samples/h；先前 batch8 约可换算到相近 samples/h 量级。
  Uncertainty: 低配机器上的 CPU/I/O、显存和 `/dev/shm` 瓶颈可能改变这个关系。
- H2: 低配机器若复用 batch16 配置，可能先受显存或 DataLoader 共享内存限制。
  Evidence: 当前机器 batch16 曾触发 DataLoader shm bus error；`batch_split` 不降低 DataLoader 完整 batch 压力。
  Uncertainty: 另一台机器的 `/dev/shm`、CPU 核数、磁盘速度和 PyTorch worker 行为未知。

## Current Decision State
- Accepted: 为低配机器准备约 50GB 子集，而不是搬运完整约 441GB FaceScape 数据集。
- Accepted: 子集只复制当前 SLat GS 训练读取的数据，不复制 `voxels/` 和 `renders_cond/`。
- Accepted: 避免在当前机器实际启动训练或压力检查，以免再次触发内存/共享内存问题影响其它任务。
- Pending: 低配机器应使用 batch8 还是 batch16 做第一轮速度测试，取决于其显存和 `/dev/shm`。

## Next Actions
1. 将 `datasets/Facescape_slat_gs_50gb` 同步到低配置机器。
2. 同步 TRELLIS 代码、`configs/vae/slat_enc_dec_gs_fine_tune.json`、以及 SLat encoder/GS decoder `.pt` 预训练权重。
3. 在低配机器先用 `--auto_retry 0` 跑短程测试，记录 step 500 之后稳定段的 steps/h、samples/h、GPU 利用率、显存峰值和是否出现 DataLoader bus error。
4. 用统一的稳定段 samples/h、端到端 samples/h 与单位小时成本比较当前昂贵 GPU 和低配机器的实际性价比。

## Constraints
- 不启动训练或重型数据检查。
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 迁移子集时需要保留 `train/metadata.csv`、`train/renders/` 和 `train/features/dinov2_vitl14_reg/` 的相对路径结构。

## Open Questions
- 低配置机器的 GPU 显存、CPU 核数、磁盘类型和 `/dev/shm` 大小是多少？
- 低配机器上是否已经有 `microsoft/TRELLIS-image-large/ckpts/*.pt` 微调初始化权重？


## HST-20260718-171834-01 - current.md snapshot

Description:
- 记录 lambda_kl=5e-4 新结果分析前的 SS encoder/decoder 微调状态

# Current State

## Active Goal
分析 `codex/train-ss-enc-dec` 分支上的 FaceScape SS encoder + decoder 1000-step 微调结果，并决定下一轮调参方向。

## Current Working Thread
用户已完成 `outputs/ss_enc_dec_fine_tune` 的 1000-step 训练，并认为曲线图不理想；当前重点是判断是否增大 batch、降低 lr，或改其他训练项。

## Relevant State
- CFG-20260718-001
- RUN-20260718-005
- ART-20260718-004
- CFG-20260717-116
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-ss-enc-dec`。
- `datasets/Facescape/train/metadata.csv` 和 `datasets/Facescape/test/metadata.csv` 已存在。
- `datasets/Facescape/train/metadata.csv` 为 6456 行，其中 `voxelized=True` 且 `aesthetic_score>=4.5` 的可训练样本为 6452 个；抽查前 20 个 sha 均能找到对应 `voxels/<sha>.ply`。
- `datasets/Facescape/test/metadata.csv` 为 720 行，`voxelized=True` 为 720 个；抽查前 20 个 sha 均能找到对应 `voxels/<sha>.ply`。
- `configs/vae/ss_vae_conv3d_16l8_fp16.json` 存在，定义 `SparseStructureEncoder`、`SparseStructureDecoder`、`SparseStructure` dataset 和 `SparseStructureVaeTrainer`。
- 已创建 `configs/vae/ss_enc_dec_fine_tune.json`，复制自 `configs/vae/ss_vae_conv3d_16l8_fp16.json`，并加入 `trainer.args.finetune_ckpt`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前训练参数为 `max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_save=500`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前已将 `lambda_kl` 从 `0.001` 降到 `5e-4`，用于增强高精度人脸 SS 重建适配信号。
- `SparseStructureVaeTrainer` 会同时调用 encoder 与 decoder 计算 SS 重建损失和 KL 项。
- `configs/vae/ss_enc_dec_fine_tune.json` 的 `finetune_ckpt.encoder` 指向 `microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.pt`。
- `configs/vae/ss_enc_dec_fine_tune.json` 的 `finetune_ckpt.decoder` 指向 `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.pt`。
- 本地官方 SS encoder/decoder safetensors 存在：`microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.safetensors` 和 `ss_dec_conv3d_16l8_fp16.safetensors`。
- 官方 SS encoder/decoder safetensors 已转换并持久化为 trainer 可读 `.pt` state_dict：`microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16.pt` 和 `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16.pt`。
- `datasets/Facescape/train` 和 `datasets/Facescape/test` 当前只包含 `voxels/`，合计约 7172 个 `.ply`，目录总大小约 929M。
- `python train.py --config configs/vae/ss_enc_dec_fine_tune.json --data_dir datasets/Facescape/train --output_dir /tmp/ss_enc_dec_fine_tune_tryrun --num_gpus 1 --ckpt none --tryrun --auto_retry 0` 已通过，成功加载 fine-tune encoder/decoder 权重并初始化 trainer。
- `outputs/ss_enc_dec_fine_tune/log_ss_enc_dec_fine_tune.txt` 记录了 1000 step 完整训练结果。
- 本次 1000-step 训练总 loss 全程均值 `0.000414844`，901-1000 step 均值 `0.000410608`，相比 1-100 step 均值 `0.000416079` 仅小幅下降约 1.3%。
- Dice loss 全程均值 `2.2405e-05`，901-1000 step 均值 `2.2831e-05`，没有明显下降趋势。
- KL 全程均值 `0.392439`，乘以 `lambda_kl=0.001` 后贡献约 `0.000392439`，约占总 loss 的主要部分。
- 输出目录包含 step 500/1000 checkpoint 以及 init/final SS 重建样本图。
- 训练器日志输出命名已改为 `log_<output_dir最后一级目录名>.txt` 和 `loss_<output_dir最后一级目录名>.txt`；例如 `outputs/ss_enc_dec_fine_tune` 对应 `log_ss_enc_dec_fine_tune.txt` 和 `loss_ss_enc_dec_fine_tune.txt`。
- `trellis5090` 环境可导入 Torch CUDA、easydict、utils3d、safetensors、spconv、torchvision、pandas；GPU 为 RTX 5090 32GB。

## Interpretations
- 当前代码、环境、数据包装和 fine-tune 初始化权重已满足 SS encoder/decoder 微调的初始化条件。
- 当前 1000-step 曲线没有发散，也没有明显震荡，但有效学习很弱；下降主要来自 KL 项轻微下降，而不是 Dice 重建项改善。
- 当前 effective batch 已为 16，单纯增大 batch 主要会平滑曲线，不太可能解决“学习方向不明显”的核心问题。
- `lr=1e-5` 已偏保守；继续降低 lr 更适合保护预训练权重、防止漂移，但会进一步放慢 FaceScape 适配。

## Active Hypotheses
- H1: 默认 DataLoader `num_workers=32` 在正式训练时可能带来共享内存压力。
  Evidence: `tryrun` 显示当前配置初始化的 DataLoader workers 为 32；先前 SLat 训练中出现过 DataLoader shared memory bus error。
  Uncertainty: SS VAE 每 batch 读取 voxel PLY 并构造 64^3 张量的实际 worker 内存压力尚未实测。
- H2: 当前曲线不理想的主因更可能是 loss 目标权重/训练长度，而不是 batch 太小或 lr 过高。
  Evidence: effective batch=16，loss 标准差约 `1.53e-05`；总 loss 主要由 `lambda_kl * kl` 构成，Dice 项没有稳定下降。
  Uncertainty: 尚未计算逐样本 IoU/F1 或同一批样本 init-vs-final 定量对比。
- H3: 因原始 SS VAE 权重来自通用三维模型，而当前数据是高精度三维人脸，适度降低 `lambda_kl` 可能更利于 FaceScape 重建适配。
  Evidence: 用户明确后续 SS flow 也会微调，降低 VAE KL 导致的 latent 分布偏移可由后续 flow 在新 latent 分布上适配一部分。
  Uncertainty: `lambda_kl` 过低仍可能导致 latent 分布偏离过大，增加 flow 学习难度或破坏采样稳定性。

## Current Decision State
- Accepted: 当前项目已满足 SS enc/dec fine-tune 初始化条件。
- Accepted: 专用 fine-tune config 命名为 `configs/vae/ss_enc_dec_fine_tune.json`。
- Accepted: 不优先通过增大 batch 或降低 lr 解决本次 1000-step 曲线问题。
- Accepted: 因后续 flow 也会微调，可以适度降低 `lambda_kl`，但不建议直接降到 0。
- Accepted: 下一轮首个 ablation 使用 `lambda_kl=5e-4`；`1e-4` 作为后续备选。

## Next Actions
1. 首选保留 batch16/lr1e-5，把训练延长到约 5000 step，并把 `i_sample` 调小以观察中间重建变化。
2. 使用当前 `lambda_kl=5e-4` 配置跑下一轮；必要时再试 `1e-4`，但需要关注 latent 分布兼容性风险。
3. 增加定量评估：固定一批样本计算 init/final/ckpt 的 voxel IoU、Dice/F1、occupancy ratio，而不是只看随机 snapshot 图。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传 `--data_dir datasets/Facescape/train` 或其他有效 root。

## Open Questions
- `lambda_kl=5e-4` 跑完后，Dice/IoU 是否有明显改善？


## HST-20260718-172323-01 - current.md snapshot

Description:
- 记录将 SS encoder/decoder fine-tune lambda_kl 从 5e-4 改到 1e-4 前的状态

# Current State

## Active Goal
分析 `codex/train-ss-enc-dec` 分支上的 FaceScape SS encoder + decoder 微调结果，并决定下一轮 KL 权重调参方向。

## Current Working Thread
用户已完成 `lambda_kl=0.001` 和 `lambda_kl=5e-4` 两个 1000-step SS enc/dec fine-tune 运行；当前重点是判断视觉效果不佳时是否继续降低 KL，以及下一轮如何验证。

## Relevant State
- CFG-20260718-001
- RUN-20260718-005
- RUN-20260718-006
- ART-20260718-004
- ART-20260718-005
- CFG-20260717-116
- ART-20260717-001

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-ss-enc-dec`。
- `datasets/Facescape/train/metadata.csv` 和 `datasets/Facescape/test/metadata.csv` 已存在。
- `datasets/Facescape/train/metadata.csv` 为 6456 行，其中 `voxelized=True` 且 `aesthetic_score>=4.5` 的可训练样本为 6452 个；抽查前 20 个 sha 均能找到对应 `voxels/<sha>.ply`。
- `configs/vae/ss_enc_dec_fine_tune.json` 复制自 `configs/vae/ss_vae_conv3d_16l8_fp16.json`，并加入 `trainer.args.finetune_ckpt`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前训练参数为 `max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_save=500`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前已将 `lambda_kl` 从 `0.001` 降到 `5e-4`。
- `finetune_ckpt.encoder` 和 `finetune_ckpt.decoder` 分别指向本地持久化的 SS encoder/decoder `.pt` state_dict。
- `python train.py --config configs/vae/ss_enc_dec_fine_tune.json --data_dir datasets/Facescape/train --output_dir /tmp/ss_enc_dec_fine_tune_tryrun --num_gpus 1 --ckpt none --tryrun --auto_retry 0` 已通过，成功加载 fine-tune encoder/decoder 权重并初始化 trainer。
- `outputs/ss_enc_dec_fine_tune/log_ss_enc_dec_fine_tune.txt` 记录了 `lambda_kl=0.001` 的 1000-step 完整训练结果。
- `lambda_kl=0.001` 运行中，总 loss 全程均值 `0.000414844`，901-1000 step 均值 `0.000410608`；Dice loss 全程均值 `2.2405e-05`，901-1000 step 均值 `2.2831e-05`；KL 全程均值 `0.392439`。
- `outputs/ss_enc_dec_fine_tune_kl5e-4/log_ss_enc_dec_fine_tune_kl5e-4.txt` 记录了 `lambda_kl=5e-4` 的 1000-step 完整训练结果。
- `lambda_kl=5e-4` 运行中，总 loss 全程均值 `0.000215963`，901-1000 step 均值 `0.000214193`；Dice loss 全程均值 `1.1318e-05`，901-1000 step 均值 `1.0851e-05`；KL 全程均值 `0.409290`。
- `lambda_kl=5e-4` 的 Dice loss 明显低于 `lambda_kl=0.001`，但 Dice 在 1000 step 内没有稳定下降趋势，线性趋势约为每 1000 step 上升 `2.37e-07`。
- `lambda_kl=5e-4` final 重建样本从视觉上看相对上一轮没有明显质变；随机 snapshot 证据较弱。
- 训练器日志输出命名已改为 `log_<output_dir最后一级目录名>.txt` 和 `loss_<output_dir最后一级目录名>.txt`。
- `trellis5090` 环境可导入 Torch CUDA、easydict、utils3d、safetensors、spconv、torchvision、pandas；GPU 为 RTX 5090 32GB。

## Interpretations
- 当前代码、环境、数据包装和 fine-tune 初始化权重已满足 SS encoder/decoder 微调的初始化条件。
- `lambda_kl=0.001` 时有效学习很弱，下降主要来自 KL 项轻微下降，而不是 Dice 重建项改善。
- 把 KL 权重降到 `5e-4` 后，Dice loss 绝对水平约减半，说明 KL 约束确实可能压制了 FaceScape 人脸重建适配。
- 但 `5e-4` 的 Dice 曲线仍没有形成稳定下降，视觉样本也没有明显质变，因此“继续降低 KL”是合理 ablation，不应被当成唯一修复。
- 当前 effective batch 已为 16，单纯增大 batch 主要会平滑曲线，不太可能解决“学习方向不明显”的核心问题。
- `lr=1e-5` 已偏保守；继续降低 lr 更适合保护预训练权重、防止漂移，但会进一步放慢 FaceScape 适配。

## Active Hypotheses
- H1: 默认 DataLoader `num_workers=32` 在正式训练时可能带来共享内存压力。
  Evidence: `tryrun` 显示当前配置初始化的 DataLoader workers 为 32；先前 SLat 训练中出现过 DataLoader shared memory bus error。
  Uncertainty: SS VAE 每 batch 读取 voxel PLY 并构造 64^3 张量的实际 worker 内存压力尚未实测。
- H2: 当前曲线不理想的主因更可能是 loss 目标权重/训练长度/评估口径，而不是 batch 太小或 lr 过高。
  Evidence: effective batch=16；总 loss 主要由 `lambda_kl * kl` 构成；`5e-4` 后 Dice 变低但趋势仍不稳定。
  Uncertainty: 尚未计算逐样本 IoU/F1 或同一批样本 init-vs-final 定量对比。
- H3: 因原始 SS VAE 权重来自通用三维模型，而当前数据是高精度三维人脸，继续适度降低 `lambda_kl` 可能更利于 FaceScape 重建适配。
  Evidence: `5e-4` 相比 `0.001` 的 Dice loss 绝对水平明显更低；用户后续 SS flow 也会微调，可适配一部分 latent 分布变化。
  Uncertainty: `lambda_kl` 过低仍可能导致 latent 分布偏离过大，增加 flow 学习难度或破坏采样稳定性。

## Current Decision State
- Accepted: 当前项目已满足 SS enc/dec fine-tune 初始化条件。
- Accepted: 专用 fine-tune config 命名为 `configs/vae/ss_enc_dec_fine_tune.json`。
- Accepted: 不优先通过增大 batch 或降低 lr 解决本次 1000-step 曲线问题。
- Accepted: 因后续 flow 也会微调，可以继续适度降低 `lambda_kl` 做 ablation，但不建议直接降到 0。
- Pending: 下一轮是否把 `lambda_kl` 从 `5e-4` 改为 `1e-4`，并配套固定样本定量评估。

## Next Actions
1. 建议下一轮保持 batch16/lr1e-5 不变，测试 `lambda_kl=1e-4` 跑 1000 step，输出目录可命名为 `outputs/ss_enc_dec_fine_tune_kl1e-4`。
2. 增加定量评估：固定一批样本计算 init/final/ckpt 的 voxel IoU、Dice/F1、occupancy ratio，而不是只看随机 snapshot 图。
3. 如果 `1e-4` 仍无视觉或定量改善，再排查 voxel 尺度、阈值、数据预处理与采样可视化路径。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传 `--data_dir datasets/Facescape/train` 或其他有效 root。

## Open Questions
- `lambda_kl=1e-4` 是否能在固定样本定量评估上优于 `5e-4`？


## HST-20260718-174913-01 - current.md snapshot

Description:
- 记录分析 SS encoder/decoder lambda_kl=1e-4 训练结果前的状态

# Current State

## Active Goal
分析 `codex/train-ss-enc-dec` 分支上的 FaceScape SS encoder + decoder 微调结果，并决定下一轮 KL 权重调参方向。

## Current Working Thread
用户已完成 `lambda_kl=0.001` 和 `lambda_kl=5e-4` 两个 1000-step SS enc/dec fine-tune 运行；当前已把配置改为 `lambda_kl=1e-4`，准备做下一轮受控 ablation。

## Relevant State
- CFG-20260718-001
- RUN-20260718-005
- RUN-20260718-006
- ART-20260718-004
- ART-20260718-005
- CFG-20260717-116
- ART-20260717-001

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-ss-enc-dec`。
- `datasets/Facescape/train/metadata.csv` 和 `datasets/Facescape/test/metadata.csv` 已存在。
- `datasets/Facescape/train/metadata.csv` 为 6456 行，其中 `voxelized=True` 且 `aesthetic_score>=4.5` 的可训练样本为 6452 个；抽查前 20 个 sha 均能找到对应 `voxels/<sha>.ply`。
- `configs/vae/ss_enc_dec_fine_tune.json` 复制自 `configs/vae/ss_vae_conv3d_16l8_fp16.json`，并加入 `trainer.args.finetune_ckpt`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前训练参数为 `max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_save=500`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前已将 `lambda_kl` 从 `0.001` 降到 `5e-4`，再降到 `1e-4`。
- `finetune_ckpt.encoder` 和 `finetune_ckpt.decoder` 分别指向本地持久化的 SS encoder/decoder `.pt` state_dict。
- `python train.py --config configs/vae/ss_enc_dec_fine_tune.json --data_dir datasets/Facescape/train --output_dir /tmp/ss_enc_dec_fine_tune_tryrun --num_gpus 1 --ckpt none --tryrun --auto_retry 0` 已通过，成功加载 fine-tune encoder/decoder 权重并初始化 trainer。
- `outputs/ss_enc_dec_fine_tune/log_ss_enc_dec_fine_tune.txt` 记录了 `lambda_kl=0.001` 的 1000-step 完整训练结果。
- `lambda_kl=0.001` 运行中，总 loss 全程均值 `0.000414844`，901-1000 step 均值 `0.000410608`；Dice loss 全程均值 `2.2405e-05`，901-1000 step 均值 `2.2831e-05`；KL 全程均值 `0.392439`。
- `outputs/ss_enc_dec_fine_tune_kl5e-4/log_ss_enc_dec_fine_tune_kl5e-4.txt` 记录了 `lambda_kl=5e-4` 的 1000-step 完整训练结果。
- `lambda_kl=5e-4` 运行中，总 loss 全程均值 `0.000215963`，901-1000 step 均值 `0.000214193`；Dice loss 全程均值 `1.1318e-05`，901-1000 step 均值 `1.0851e-05`；KL 全程均值 `0.409290`。
- `lambda_kl=5e-4` 的 Dice loss 明显低于 `lambda_kl=0.001`，但 Dice 在 1000 step 内没有稳定下降趋势，线性趋势约为每 1000 step 上升 `2.37e-07`。
- `lambda_kl=5e-4` final 重建样本从视觉上看相对上一轮没有明显质变；随机 snapshot 证据较弱。
- 训练器日志输出命名已改为 `log_<output_dir最后一级目录名>.txt` 和 `loss_<output_dir最后一级目录名>.txt`。
- `trellis5090` 环境可导入 Torch CUDA、easydict、utils3d、safetensors、spconv、torchvision、pandas；GPU 为 RTX 5090 32GB。

## Interpretations
- 当前代码、环境、数据包装和 fine-tune 初始化权重已满足 SS encoder/decoder 微调的初始化条件。
- `lambda_kl=0.001` 时有效学习很弱，下降主要来自 KL 项轻微下降，而不是 Dice 重建项改善。
- 把 KL 权重降到 `5e-4` 后，Dice loss 绝对水平约减半，说明 KL 约束确实可能压制了 FaceScape 人脸重建适配。
- 但 `5e-4` 的 Dice 曲线仍没有形成稳定下降，视觉样本也没有明显质变，因此“继续降低 KL”是合理 ablation，不应被当成唯一修复。
- 当前 effective batch 已为 16，单纯增大 batch 主要会平滑曲线，不太可能解决“学习方向不明显”的核心问题。
- `lr=1e-5` 已偏保守；继续降低 lr 更适合保护预训练权重、防止漂移，但会进一步放慢 FaceScape 适配。

## Active Hypotheses
- H1: 默认 DataLoader `num_workers=32` 在正式训练时可能带来共享内存压力。
  Evidence: `tryrun` 显示当前配置初始化的 DataLoader workers 为 32；先前 SLat 训练中出现过 DataLoader shared memory bus error。
  Uncertainty: SS VAE 每 batch 读取 voxel PLY 并构造 64^3 张量的实际 worker 内存压力尚未实测。
- H2: 当前曲线不理想的主因更可能是 loss 目标权重/训练长度/评估口径，而不是 batch 太小或 lr 过高。
  Evidence: effective batch=16；总 loss 主要由 `lambda_kl * kl` 构成；`5e-4` 后 Dice 变低但趋势仍不稳定。
  Uncertainty: 尚未计算逐样本 IoU/F1 或同一批样本 init-vs-final 定量对比。
- H3: 因原始 SS VAE 权重来自通用三维模型，而当前数据是高精度三维人脸，继续适度降低 `lambda_kl` 可能更利于 FaceScape 重建适配。
  Evidence: `5e-4` 相比 `0.001` 的 Dice loss 绝对水平明显更低；用户后续 SS flow 也会微调，可适配一部分 latent 分布变化。
  Uncertainty: `lambda_kl` 过低仍可能导致 latent 分布偏离过大，增加 flow 学习难度或破坏采样稳定性。

## Current Decision State
- Accepted: 当前项目已满足 SS enc/dec fine-tune 初始化条件。
- Accepted: 专用 fine-tune config 命名为 `configs/vae/ss_enc_dec_fine_tune.json`。
- Accepted: 不优先通过增大 batch 或降低 lr 解决本次 1000-step 曲线问题。
- Accepted: 因后续 flow 也会微调，可以继续适度降低 `lambda_kl` 做 ablation，但不建议直接降到 0。
- Accepted: 下一轮把 `lambda_kl` 从 `5e-4` 改为 `1e-4`，输出目录建议为 `outputs/ss_enc_dec_fine_tune_kl1e-4`。
- Pending: `lambda_kl=1e-4` 是否能在固定样本定量评估上优于 `5e-4`。

## Next Actions
1. 使用当前 `lambda_kl=1e-4` 配置运行 1000 step，输出目录命名为 `outputs/ss_enc_dec_fine_tune_kl1e-4`。
2. 增加定量评估：固定一批样本计算 init/final/ckpt 的 voxel IoU、Dice/F1、occupancy ratio，而不是只看随机 snapshot 图。
3. 如果 `1e-4` 仍无视觉或定量改善，再排查 voxel 尺度、阈值、数据预处理与采样可视化路径。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传 `--data_dir datasets/Facescape/train` 或其他有效 root。

## Open Questions
- `lambda_kl=1e-4` 是否能在固定样本定量评估上优于 `5e-4`？


## HST-20260718-182713-01 - current.md snapshot

Description:
- 记录新增 SS encoder/decoder 固定样本评估工具前的状态

# Current State

## Active Goal
分析 `codex/train-ss-enc-dec` 分支上的 FaceScape SS encoder + decoder 微调结果，并决定下一轮 KL 权重和 flow 微调方向。

## Current Working Thread
用户已完成 `lambda_kl=0.001`、`5e-4`、`1e-4` 三个 1000-step SS enc/dec fine-tune 运行；当前结论是 `1e-4` 的 SS 重建项最好，但需要固定样本定量评估和后续 SS flow 小实验确认 latent 分布风险。

## Relevant State
- CFG-20260718-001
- RUN-20260718-005
- RUN-20260718-006
- RUN-20260718-007
- ART-20260718-004
- ART-20260718-005
- ART-20260718-006
- CFG-20260717-116
- ART-20260717-001

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-ss-enc-dec`。
- `datasets/Facescape/train/metadata.csv` 和 `datasets/Facescape/test/metadata.csv` 已存在。
- `datasets/Facescape/train/metadata.csv` 为 6456 行，其中 `voxelized=True` 且 `aesthetic_score>=4.5` 的可训练样本为 6452 个；抽查前 20 个 sha 均能找到对应 `voxels/<sha>.ply`。
- `configs/vae/ss_enc_dec_fine_tune.json` 复制自 `configs/vae/ss_vae_conv3d_16l8_fp16.json`，并加入 `trainer.args.finetune_ckpt`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前训练参数为 `max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_save=500`。
- `configs/vae/ss_enc_dec_fine_tune.json` 当前已将 `lambda_kl` 从 `0.001` 降到 `5e-4`，再降到 `1e-4`。
- `finetune_ckpt.encoder` 和 `finetune_ckpt.decoder` 分别指向本地持久化的 SS encoder/decoder `.pt` state_dict。
- `outputs/ss_enc_dec_fine_tune/log_ss_enc_dec_fine_tune.txt` 记录了 `lambda_kl=0.001` 的 1000-step 完整训练结果。
- `lambda_kl=0.001` 运行中，总 loss 全程均值 `0.000414844`，Dice loss 全程均值 `2.2405e-05`，901-1000 step Dice 均值 `2.2831e-05`，KL 全程均值 `0.392439`。
- `outputs/ss_enc_dec_fine_tune_kl5e-4/log_ss_enc_dec_fine_tune_kl5e-4.txt` 记录了 `lambda_kl=5e-4` 的 1000-step 完整训练结果。
- `lambda_kl=5e-4` 运行中，总 loss 全程均值 `0.000215963`，Dice loss 全程均值 `1.1318e-05`，901-1000 step Dice 均值 `1.0851e-05`，KL 全程均值 `0.409290`。
- `outputs/ss_enc_dec_fine_tune_kl1e-4/log_ss_enc_dec_fine_tune_kl1e-4.txt` 和 `loss_ss_enc_dec_fine_tune_kl1e-4.txt` 均为 1000 行，记录了 `lambda_kl=1e-4` 的完整训练结果。
- `lambda_kl=1e-4` 运行中，总 loss 全程均值 `4.7369e-05`，Dice loss 全程均值 `2.4171e-06`，901-1000 step Dice 均值 `2.2334e-06`，KL 全程均值 `0.449517`。
- `lambda_kl=1e-4` 的 Dice loss 线性趋势约为每 1000 step 下降 `2.13e-07`，相对约 `8.8%`；`lambda_kl=5e-4` 的 Dice 趋势则轻微上升。
- `lambda_kl=1e-4` 中有 119 个 step Dice 为 `0.0`，436 个 step Dice 小于 `1e-6`。
- `outputs/ss_enc_dec_fine_tune_kl1e-4` 目录约 `6.0G`，包含 step 500/1000 encoder、decoder、EMA 和 misc checkpoint，以及 init/final SS 重建样本图。
- `lambda_kl=1e-4` final 重建图与 GT 的大轮廓较接近，差异主要在局部边缘、细小突起和薄结构；没有明显崩坏。
- 训练器日志输出命名已改为 `log_<output_dir最后一级目录名>.txt` 和 `loss_<output_dir最后一级目录名>.txt`。
- `trellis5090` 环境可导入 Torch CUDA、easydict、utils3d、safetensors、spconv、torchvision、pandas；GPU 为 RTX 5090 32GB。

## Interpretations
- 当前代码、环境、数据包装和 fine-tune 初始化权重已满足 SS encoder/decoder 微调的初始化条件。
- 三轮对比中，`lambda_kl=1e-4` 是目前 SS 重建项最好的设置：Dice 绝对值最低，后 100 step 也最低，并且 Dice 趋势从 `5e-4` 的轻微变差转为轻微变好。
- KL 权重下降后 Dice loss 明显降低，说明 KL 约束确实压制了 FaceScape 高精度人脸分布适配。
- `lambda_kl=1e-4` 已经让 KL 均值升到 `0.449517`，继续降低 KL 的边际收益可能变小，风险会转向 latent 分布漂移与后续 flow 学习难度。
- 当前更像是 SS VAE 已经能较好拟合 64^3 sparse structure；视觉上剩余问题可能来自 SS 表示分辨率、阈值/occupancy、随机可视化样本或后续 SLat/decoder 阶段。
- 当前 effective batch 已为 16，单纯增大 batch 主要会平滑曲线，不太可能解决核心问题。

## Active Hypotheses
- H1: 默认 DataLoader `num_workers=32` 在正式训练时可能带来共享内存压力。
  Evidence: `tryrun` 显示当前配置初始化的 DataLoader workers 为 32；先前 SLat 训练中出现过 DataLoader shared memory bus error。
  Uncertainty: SS VAE 每 batch 读取 voxel PLY 并构造 64^3 张量的实际 worker 内存压力尚未实测。
- H2: `lambda_kl=1e-4` 已接近当前 SS VAE fine-tune 的合理低 KL 区间。
  Evidence: Dice 均值已从 `2.2405e-05` 降到 `2.4171e-06`，且 436/1000 个 step Dice 小于 `1e-6`；KL 均值升到 `0.449517`。
  Uncertainty: 尚未计算固定验证集 IoU/F1，也未验证该 latent 分布对 flow fine-tune 的影响。
- H3: 视觉剩余差异可能不是继续降低 KL 能完全解决的问题。
  Evidence: final recon 与 GT 大轮廓接近，但局部边缘、细小突起和薄结构仍有差异；SS sparse structure 分辨率和后续模型阶段可能决定最终细节。
  Uncertainty: 需要固定样本可视化和 occupancy ratio 判断是否存在系统性欠占用或过占用。

## Current Decision State
- Accepted: 当前项目已满足 SS enc/dec fine-tune 初始化条件。
- Accepted: 专用 fine-tune config 命名为 `configs/vae/ss_enc_dec_fine_tune.json`。
- Accepted: 不优先通过增大 batch 或降低 lr 解决本次曲线问题。
- Accepted: `lambda_kl=1e-4` 是当前三轮 1000-step ablation 的最佳候选。
- Pending: 是否继续试 `5e-5` 或 0；当前不优先，除非固定定量评估显示 `1e-4` 仍明显欠拟合。
- Pending: `lambda_kl=1e-4` 训练出的 SS latent 是否适合后续 SS flow 微调。

## Next Actions
1. 先把 `lambda_kl=1e-4` checkpoint 作为 SS enc/dec 当前候选，用固定样本计算 voxel IoU、Dice/F1、occupancy ratio。
2. 使用 `lambda_kl=1e-4` 的 step1000 encoder/decoder checkpoint 做一个短程 SS flow fine-tune 小实验，观察 flow loss、采样稳定性和生成 SS occupancy。
3. 只有当固定评估显示 `1e-4` 仍明显欠拟合时，再试 `lambda_kl=5e-5`；不建议现在直接降到 0。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传 `--data_dir datasets/Facescape/train` 或其他有效 root。

## Open Questions
- `lambda_kl=1e-4` 的固定样本 IoU/F1 是否显著优于 `5e-4`？
- 后续 SS flow 在 `lambda_kl=1e-4` latent 上是否稳定收敛？


## HST-20260718-183849-01 - current.md snapshot

Description:
- 记录正式固定样本评估 SS encoder/decoder 权重前的状态

# Current State

## Active Goal
构建可靠的固定样本评估工具，用于比较 FaceScape SS encoder + decoder 微调结果，并决定下一轮 KL 权重和 flow 微调方向。

## Current Working Thread
用户已完成 `lambda_kl=0.001`、`5e-4`、`1e-4` 三个 1000-step SS enc/dec fine-tune 运行；当前已新增 `eval/` 工具，用 mini metadata dataset 复用 `SparseStructure`，并输出固定样本 IoU、Dice/F1、occupancy ratio、voxel count 和 trainer-style soft Dice loss。

## Relevant State
- CFG-20260718-001
- CFG-20260718-002
- EXE-20260718-002
- EXE-20260718-003
- RUN-20260718-005
- RUN-20260718-006
- RUN-20260718-007
- RUN-20260718-008
- RUN-20260718-009
- ART-20260718-004
- ART-20260718-005
- ART-20260718-006
- ART-20260717-001

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-ss-enc-dec`。
- 新增 `eval/prepare_ss_eval_dataset.py`：从源 dataset root 的 `metadata.csv` 和 `voxels/` 生成固定样本 mini dataset root。
- 新增 `eval/evaluate_ss_enc_dec_reconstruction.py`：在固定 mini dataset 上评估 SS encoder/decoder checkpoint。
- 新增 `eval/ss_eval_checkpoints.json`：列出 official、`kl1e-3_step1000`、`kl5e-4_step1000`、`kl1e-4_step1000` 四组 encoder/decoder checkpoint。
- 新增 `eval/README.md`：记录固定评估集准备、posterior mean 评估和 sample posterior 评估命令。
- 评估脚本直接复用 `trellis.datasets.SparseStructure`，不重复实现 PLY 到 voxel tensor 的转换。
- 评估脚本默认使用 posterior mean，即 `encoder(ss.float(), sample_posterior=False)`；可通过 `--sample_posterior --seed <seed>` 使用 stochastic posterior。
- 每个样本输出 `iou`、`dice_f1`、`occupancy_ratio`、`gt_occupied_voxels`、`predicted_occupied_voxels`、`intersection_voxels`、`union_voxels`、`soft_dice_loss`。
- `soft_dice_loss` 使用 trainer 中同口径的 sigmoid logits Dice loss 公式，带 `+1` 平滑项。
- 单元测试覆盖 mini dataset 生成、样本不足报错、hard 指标公式、空 GT 边界、summary 忽略 NaN、posterior sampling 开关传递。
- 4 样本 smoke test 验证了 mini dataset 生成、四组 checkpoint deterministic 评估和 official sample posterior 评估路径。
- deterministic 4 样本 smoke 中四组 checkpoint 指标均饱和，hard IoU/Dice 为 `1.0`、`soft_dice_loss=0.0`。
- official sample posterior 4 样本 smoke 中 `iou_mean=0.999956`、`dice_f1_mean=0.999978`、`soft_dice_loss_mean=2.2113e-05`。

## Interpretations
- mini metadata dataset 方案能最大化复用 TRELLIS 现有 `SparseStructure` 数据路径，减少评估代码和训练数据读取逻辑不一致的风险。
- posterior mean 评估可能在 SS VAE 上指标饱和；sample posterior 模式更接近训练 loss 口径，也更容易暴露 latent 分布放松后的随机重建稳定性差异。
- 当前正式模型选择不能只靠 4 样本 smoke；需要生成 64 或更大固定 test mini dataset 后，分别运行 posterior mean 和 sample posterior 两种评估。
- 如果 hard IoU/Dice 在正式集上仍饱和，应优先比较 `soft_dice_loss` 和 sample posterior 模式下的 occupancy/IoU 稳定性。

## Active Hypotheses
- H1: `lambda_kl=1e-4` 已接近当前 SS VAE fine-tune 的合理低 KL 区间。
  Evidence: 训练日志 Dice 均值已从 `2.2405e-05` 降到 `2.4171e-06`，且 436/1000 个 step Dice 小于 `1e-6`；KL 均值升到 `0.449517`。
  Uncertainty: 尚未完成固定 test mini dataset 上的正式 posterior mean 和 sample posterior 评估。
- H2: posterior mean 的 hard voxel 指标可能对当前 SS VAE checkpoint 区分度不足。
  Evidence: 4 样本 smoke 中四组 checkpoint 的 hard IoU/Dice 均为 `1.0`。
  Uncertainty: 4 样本太小，正式 64/256 样本可能仍能暴露差异。
- H3: sample posterior 评估可能更适合判断低 KL 是否破坏 latent 稳定性。
  Evidence: official 4 样本 sample posterior smoke 产生非零 `soft_dice_loss_mean=2.2113e-05`，而 posterior mean 为 `0.0`。
  Uncertainty: 需要四组 checkpoint 在同一固定正式样本集上对比。

## Current Decision State
- Accepted: 固定样本评估使用 mini dataset root，而不是纯 sha list。
- Accepted: 评估代码保留用户指定 hard 指标，并额外输出 trainer-style `soft_dice_loss`。
- Accepted: `lambda_kl=1e-4` 是当前训练日志层面的最佳候选。
- Pending: 正式固定样本评估是否确认 `lambda_kl=1e-4` 优于 `5e-4`。
- Pending: 后续 SS flow 在 `lambda_kl=1e-4` latent 上是否稳定收敛。

## Next Actions
1. 生成正式固定评估集：`datasets/Facescape_ss_eval_test_64`。
2. 运行 posterior mean 评估输出到 `outputs/ss_enc_dec_eval`。
3. 运行 sample posterior 评估输出到 `outputs/ss_enc_dec_eval_sample_posterior`。
4. 根据正式 summary 判断是否保留 `lambda_kl=1e-4` 或继续试 `5e-5`。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。

## Open Questions
- 正式 64 样本 posterior mean hard 指标是否仍全部饱和？
- sample posterior 模式下 `lambda_kl=1e-4` 是否比 `5e-4` 更稳？


## HST-20260718-184945-01 - current.md snapshot

Description:
- 记录评估 kl1e-4 step500 vs step1000 前的状态

# Current State

## Active Goal
评估当前 FaceScape SS encoder + decoder fine-tuned weights，并决定是否把 `lambda_kl=1e-4` checkpoint 作为后续 SS flow fine-tune 的基础。

## Current Working Thread
已使用固定 64 个 FaceScape test 样本，对 official、`kl1e-3_step1000`、`kl5e-4_step1000`、`kl1e-4_step1000` 做 posterior mean 和 3 个 seed 的 sample posterior 重建评估；当前结果支持 `kl1e-4_step1000` 作为 SS enc/dec 当前候选。

## Relevant State
- CFG-20260718-001
- CFG-20260718-002
- EXE-20260718-002
- EXE-20260718-003
- RUN-20260718-010
- RUN-20260718-011
- RUN-20260718-012
- RUN-20260718-013
- RUN-20260718-014
- ART-20260718-007
- ART-20260718-008
- ART-20260718-009
- ART-20260718-010
- ART-20260718-011
- ART-20260718-006

## Facts
- 固定评估集路径为 `datasets/Facescape_ss_eval_test_64`，包含 64 个 FaceScape test 样本加表头，`voxels` 为指向 `../Facescape/test/voxels` 的软链接。
- posterior mean 评估结果路径为 `outputs/ss_enc_dec_eval`。
- sample posterior seed `20260718` 结果路径为 `outputs/ss_enc_dec_eval_sample_posterior`。
- sample posterior seed `20260719` 结果路径为 `outputs/ss_enc_dec_eval_sample_posterior_seed20260719`。
- sample posterior seed `20260720` 结果路径为 `outputs/ss_enc_dec_eval_sample_posterior_seed20260720`。
- 所有 per-sample CSV 都是 64 行样本加表头，未发现 NaN/Inf。
- posterior mean 口径下，official、`kl1e-3_step1000`、`kl5e-4_step1000`、`kl1e-4_step1000` 的 hard IoU、Dice/F1、occupancy ratio 均为 `1.0`，`soft_dice_loss` 均为 `0.0`。
- sample posterior 三个 seed 下，`kl1e-4_step1000` 每次都按 `iou_mean` 和 `soft_dice_loss_mean` 排第一。
- sample posterior 跨 seed `soft_dice_loss_mean` 均值：official `7.0073e-06`，`kl1e-3_step1000` `2.5233e-05`，`kl5e-4_step1000` `1.1981e-05`，`kl1e-4_step1000` `2.5878e-06`。
- sample posterior 跨 seed `iou_mean` 均值：official `0.9999856`，`kl1e-3_step1000` `0.9999516`，`kl5e-4_step1000` `0.9999764`，`kl1e-4_step1000` `0.9999955`。
- sample posterior 跨 seed occupancy ratio 均值都接近 `1.0`：official `0.9999968`，`kl1e-3_step1000` `1.0000038`，`kl5e-4_step1000` `0.9999923`，`kl1e-4_step1000` `1.0000001`。

## Interpretations
- posterior mean hard 指标在固定 64 样本上完全饱和，不能区分 checkpoint 优劣；这说明 SS VAE 的 deterministic mean 重建已经非常强。
- sample posterior 口径更能暴露随机 latent 重建稳定性；在该口径下，`kl1e-4_step1000` 的优势跨 3 个 seed 一致。
- `kl1e-4_step1000` 的优势主要体现在更低 `soft_dice_loss` 和略高 hard IoU；绝对差异很小，但排序稳定。
- 评估结果与训练日志一致：降低 KL 到 `1e-4` 后，SS 重建项更好，且没有出现 occupancy ratio 系统性偏胖或偏瘦。
- 该结论只覆盖 SS VAE 重建，不直接证明后续 SS flow 或完整 3D 生成质量更好。

## Active Hypotheses
- H1: `lambda_kl=1e-4` 是当前 SS VAE fine-tune 的最佳候选。
  Evidence: 训练日志 Dice 最低；固定 64 样本 sample posterior 三个 seed 下 `kl1e-4_step1000` 均排第一。
  Uncertainty: 尚未在完整 test split 或更多 sampling seed 上验证。
- H2: 继续降低 KL 的边际收益可能小于 latent 分布漂移风险。
  Evidence: posterior mean 已完全饱和，sample posterior 差异也非常小；`kl1e-4` 训练日志中 KL 均值已升到 `0.449517`。
  Uncertainty: 尚未跑 `5e-5` 或 `0` 的对照。
- H3: 下一步更应该验证 flow 适配，而不是继续只优化 SS VAE 重建。
  Evidence: SS 重建指标已接近天花板；后续生成质量仍取决于 SS flow 对新 latent 分布的学习。
  Uncertainty: 尚未启动 SS flow fine-tune 小实验。

## Current Decision State
- Accepted: 固定 64 样本评估确认 `kl1e-4_step1000` 是当前 SS enc/dec 最优候选。
- Accepted: 不建议基于当前证据继续直接降低 KL 到 `5e-5` 或 `0`。
- Pending: 使用 `kl1e-4_step1000` encoder/decoder checkpoint 启动短程 SS flow fine-tune 小实验。

## Next Actions
1. 固定使用 `outputs/ss_enc_dec_fine_tune_kl1e-4/ckpts/encoder_step0001000.pt` 和 `decoder_step0001000.pt` 作为 SS enc/dec 当前候选。
2. 配置并启动 SS flow fine-tune 小实验，观察 flow loss、采样稳定性和生成 SS occupancy。
3. 若 flow 对 `kl1e-4` latent 表现不稳，再回退比较 `kl5e-4` 或增加中间 KL 值。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练和评估命令必须显式传有效 dataset root。

## Open Questions
- SS flow 在 `kl1e-4` latent 上是否稳定收敛？
- 完整 test split 是否会暴露 posterior mean hard 指标的非饱和样本？


## HST-20260718-185650-01 - current.md snapshot

Description:
- 记录构建 kl1e-4 step1000 SS latent 子集前的状态

# Current State

## Active Goal
比较 FaceScape SS encoder + decoder `lambda_kl=1e-4` 的 step500 与 step1000 checkpoint，并判断 1000 steps 是否确实优于 500 steps。

## Current Working Thread
已使用固定 64 个 FaceScape test 样本，对 `kl1e-4_step500` 和 `kl1e-4_step1000` 做 posterior mean 与 3 个 seed 的 sample posterior 重建评估；当前结论是两者固定样本重建几乎等价，1000 step 不是显著优于 500 step。

## Relevant State
- CFG-20260718-001
- CFG-20260718-003
- EXE-20260718-003
- RUN-20260718-015
- RUN-20260718-016
- RUN-20260718-017
- RUN-20260718-018
- ART-20260718-007
- ART-20260718-012
- ART-20260718-013
- ART-20260718-014
- ART-20260718-015
- ART-20260718-006

## Facts
- `eval/ss_eval_kl1e-4_steps.json` 记录了 `kl1e-4_step500` 与 `kl1e-4_step1000` 的 encoder/decoder checkpoint 路径。
- posterior mean 结果路径为 `outputs/ss_enc_dec_eval_kl1e-4_steps`。
- sample posterior seed `20260718` 结果路径为 `outputs/ss_enc_dec_eval_kl1e-4_steps_sample_posterior`。
- sample posterior seed `20260719` 结果路径为 `outputs/ss_enc_dec_eval_kl1e-4_steps_sample_posterior_seed20260719`。
- sample posterior seed `20260720` 结果路径为 `outputs/ss_enc_dec_eval_kl1e-4_steps_sample_posterior_seed20260720`。
- 所有 per-sample CSV 都是 64 行样本加表头，未发现 NaN/Inf。
- posterior mean 口径下，step500 与 step1000 的 hard IoU、Dice/F1、occupancy ratio 均为 `1.0`，`soft_dice_loss` 均为 `0.0`。
- sample posterior 跨三个 seed，step500 `iou_mean` 聚合均值 `0.9999938`，step1000 `0.9999942`。
- sample posterior 跨三个 seed，step500 `dice_f1_mean` 聚合均值 `0.9999969`，step1000 `0.9999971`。
- sample posterior 跨三个 seed，step500 `soft_dice_loss_mean` 聚合均值 `3.0644e-06`，step1000 `3.1830e-06`。
- sample posterior 跨三个 seed，step500 `occupancy_ratio_mean` 聚合均值 `0.9999997`，step1000 `1.0000010`。
- seed `20260718` 与 `20260719` 下 step1000 略优，seed `20260720` 下 step500 略优。
- 固定 64 样本 latent 统计：step500 `kl_mean=0.4504665`、`z_std=0.5102527`；step1000 `kl_mean=0.4514627`、`z_std=0.5126375`。

## Interpretations
- posterior mean 口径完全饱和，无法区分 step500 与 step1000。
- sample posterior 口径下，step1000 的 IoU/Dice 平均略高，但 step500 的 soft Dice loss 平均略低；差异非常小且 seed 间有反转。
- 500 到 1000 step 的固定样本重建收益已经很小，当前评估不能证明 step1000 显著优于 step500。
- step1000 的 latent KL 只比 step500 高约 `0.0010`，没有明显 latent 分布漂移。
- 从工程选择上，step1000 仍可作为默认候选，因为它是训练结束 checkpoint，且没有显示更差；但如果后续 flow 对 latent 很敏感，step500 值得作为备选对照。

## Active Hypotheses
- H1: `lambda_kl=1e-4` 在 500 step 左右已经基本完成当前 SS 重建适配。
  Evidence: step500 与 step1000 posterior mean 均饱和；sample posterior 差异极小。
  Uncertainty: 未评估 step250、step750 或完整 test split。
- H2: step1000 相对 step500 没有明显过拟合或 latent 漂移。
  Evidence: step1000 `kl_mean=0.4514627`，step500 `0.4504665`，差异约 `0.0010`；occupancy ratio 都接近 1。
  Uncertainty: flow 对该细微 latent 差异的敏感性未知。
- H3: 后续判断应转向 flow 小实验，而不是继续只比较 SS VAE checkpoint。
  Evidence: SS VAE 固定样本指标接近天花板。
  Uncertainty: 尚未启动 flow 对照。

## Current Decision State
- Accepted: step1000 不是被固定样本评估显著证明优于 step500。
- Accepted: step1000 可以保留为默认 SS enc/dec 候选。
- Pending: 若 flow 对 step1000 latent 不稳，需要用 step500 做 flow 对照。

## Next Actions
1. 默认使用 `kl1e-4_step1000` 进入 SS flow fine-tune 小实验。
2. 若 flow loss 或 sampling occupancy 不稳定，再用 `kl1e-4_step500` 做同配置 flow 对照。
3. 暂不继续 SS VAE 训练到更长步数，除非 flow 或完整 test split 评估暴露问题。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练和评估命令必须显式传有效 dataset root。

## Open Questions
- SS flow 对 `kl1e-4_step1000` latent 是否稳定？
- 完整 test split 上 step500 与 step1000 是否仍然几乎等价？


## HST-20260718-190519-01 - current.md snapshot

Description:
- 记录检查 image-conditioned SS flow 配置适配 kl1e-4 latent 子集前的状态

# Current State

## Active Goal
为后续 SS flow 适配实验准备一份由 `kl1e-4_step1000` SS encoder 编码的 FaceScape latent 子集，并保持独立 metadata。

## Current Working Thread
已从 FaceScape train split 抽取 1024 个高 aesthetic 样本，使用 `ss_enc_dec_fine_tune_kl1e-4` step1000 encoder 编码 SS latent，并构建了可由 `SparseStructureLatent` 读取的独立 dataset root。

## Relevant State
- CFG-20260718-001
- EXE-20260717-121
- EXE-20260718-002
- RUN-20260718-019
- RUN-20260718-020
- ART-20260717-001
- ART-20260718-016
- RUN-20260718-015
- RUN-20260718-016
- RUN-20260718-017
- RUN-20260718-018

## Facts
- 独立 latent 数据集路径为 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024`。
- 子集来自 `datasets/Facescape/train`，抽样参数为 `--num_samples 1024 --seed 20260718 --min_aesthetic_score 4.5`。
- `metadata.csv` 有 1024 行样本。
- latent metadata 列名为 `ss_latent_ss_enc_dec_fine_tune_kl1e-4_step0001000`，1024/1024 行为 True。
- latent 文件位于 `ss_latents/ss_enc_dec_fine_tune_kl1e-4_step0001000/*.npz`，共 1024 个。
- `dataset_toolkits/encode_ss_latent.py` 加载的 encoder checkpoint 是 `outputs/ss_enc_dec_fine_tune_kl1e-4/ckpts/encoder_step0001000.pt`。
- latent `mean` 数组 shape 为 `(8, 16, 16, 16)`、dtype 为 `float32`，抽样及全量 finite 检查通过。
- 全量 latent `mean_std` 均值为 `0.5090935`，范围为 `0.4601546` 到 `0.6296225`。
- `SparseStructureLatent` smoke check 通过：dataset length 1024，首样本 `x_0` shape `(8, 16, 16, 16)` 且 finite。
- 目录大小约 `125M`，`voxels` 指向 `../Facescape/train/voxels`。
- 该目录当前没有 `renders_cond/` 或 image feature 资源。

## Interpretations
- 这份数据已经满足 SS latent-only 数据集读取条件，适合用来快速检查 flow 对 `kl1e-4_step1000` latent 分布的适配性。
- 它还不满足现有 image-conditioned SS flow 配置的完整输入条件，因为当前 TRELLIS image-conditioned dataset 通常还需要 `renders_cond/` 或相应条件特征。
- 保留独立目录和独立 metadata 可以避免污染原始 FaceScape metadata，同时让后续 flow 对照更可复现。

## Active Hypotheses
- H1: `kl1e-4_step1000` latent 子集足够用于 flow 小规模稳定性试验。
  Evidence: 1024 个 latent 全量编码成功，文件完整，`SparseStructureLatent` 可正常加载。
  Uncertainty: 尚未运行 flow fine-tune，无法确认 loss 稳定性或采样质量。
- H2: 若继续使用 image-conditioned flow，当前目录需要补齐条件数据。
  Evidence: 数据目录只有 `metadata.csv`、`voxels` symlink、`ss_latents/` 和编码记录 CSV，没有 `renders_cond/`。
  Uncertainty: 后续 flow 配置是否坚持 image-conditioned 口径尚未决定。

## Current Decision State
- Accepted: `kl1e-4_step1000` SS latent 1024 样本子集已构建完成并通过基础可靠性检查。
- Pending: 下一步 flow 实验使用 image-conditioned 还是 latent-only 配置。

## Next Actions
1. 若使用现有 image-conditioned SS flow 配置，给 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024` 补 `renders_cond/` 或所需特征资源。
2. 若只先验证 latent 分布适配性，新增或调整 flow 配置使用 `SparseStructureLatent`。
3. 用该 1024 样本子集启动短步数 flow smoke fine-tune，观察 loss、latent scale、occupancy 和采样稳定性。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练和评估命令必须显式传有效 dataset root。
- 当前 latent 子集为 SS latent-only 数据，不应误认为已经 image-conditioned flow ready。

## Open Questions
- 后续 flow 微调是否需要保持图片条件输入？
- 1024 样本是否足够暴露 flow 对 `kl1e-4_step1000` latent 的稳定性问题？


## HST-20260718-193842-01 - current.md snapshot

Description:
- 记录解压并划分 FaceScape renders_cond 前的 flow 准备状态

# Current State

## Active Goal
判断 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json` 是否能直接用于测试 SS flow 对 `kl1e-4_step1000` SS encoder/decoder 的适配性。

## Current Working Thread
已核对 flow 配置、`ImageConditionedSparseStructureLatent` 读取逻辑，以及刚构建的 1024 样本 latent 子集；当前判断是不应直接用原配置开训，需要先补配置和条件数据准备。

## Relevant State
- CFG-20260718-001
- EXE-20260717-121
- EXE-20260718-002
- RUN-20260718-019
- RUN-20260718-020
- ART-20260717-001
- ART-20260718-016
- CFG-20260717-116
- RUN-20260718-015
- RUN-20260718-016
- RUN-20260718-017
- RUN-20260718-018

## Facts
- `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json` 的 dataset 为 `ImageConditionedSparseStructureLatent`。
- 该配置当前 `dataset.args.latent_model` 是 `ss_enc_conv3d_16l8_fp16`，不是新编码的 `ss_enc_dec_fine_tune_kl1e-4_step0001000`。
- 该配置当前 `dataset.args.pretrained_ss_dec` 指向官方 decoder `microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16`，不是 `outputs/ss_enc_dec_fine_tune_kl1e-4/ckpts/decoder_step0001000.pt`。
- 独立 latent 数据集路径为 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024`。
- 子集来自 `datasets/Facescape/train`，抽样参数为 `--num_samples 1024 --seed 20260718 --min_aesthetic_score 4.5`。
- `metadata.csv` 有 1024 行样本。
- latent metadata 列名为 `ss_latent_ss_enc_dec_fine_tune_kl1e-4_step0001000`，1024/1024 行为 True。
- latent 文件位于 `ss_latents/ss_enc_dec_fine_tune_kl1e-4_step0001000/*.npz`，共 1024 个。
- `dataset_toolkits/encode_ss_latent.py` 加载的 encoder checkpoint 是 `outputs/ss_enc_dec_fine_tune_kl1e-4/ckpts/encoder_step0001000.pt`。
- latent `mean` 数组 shape 为 `(8, 16, 16, 16)`、dtype 为 `float32`，抽样及全量 finite 检查通过。
- 全量 latent `mean_std` 均值为 `0.5090935`，范围为 `0.4601546` 到 `0.6296225`。
- `SparseStructureLatent` smoke check 通过：dataset length 1024，首样本 `x_0` shape `(8, 16, 16, 16)` 且 finite。
- 目录大小约 `125M`，`voxels` 指向 `../Facescape/train/voxels`。
- 该目录当前没有 `renders_cond/` 或 image feature 资源。
- 用原 flow 配置初始化新 latent 数据目录会因缺少 `ss_latent_ss_enc_conv3d_16l8_fp16` metadata 列报 `KeyError`。
- 将 dataset args 手动改为 `latent_model=ss_enc_dec_fine_tune_kl1e-4_step0001000` 且 decoder 指向 `outputs/ss_enc_dec_fine_tune_kl1e-4` step1000 后，`ImageConditionedSparseStructureLatent` 初始化得到 0 个样本，因为 `Cond image dirs present: 0`。
- 新 latent 子集 metadata 中 `cond_rendered=True` 的样本数为 1023，但实际 `renders_cond/<sha>/` 条件图像目录数为 0。

## Interpretations
- 原 flow 配置不能直接用于刚编码好的 `kl1e-4_step1000` latent 子集。
- 如果目标是测试“SS flow 是否适配刚微调的 SS encoder/decoder latent 分布”，最小准备是复制一份 flow 配置并改 `latent_model`、`ss_dec_path`、`ss_dec_ckpt`，同时补齐 image-conditioned 所需的 `renders_cond/`。
- 如果短期只想测 latent 分布适配，不测图像条件能力，则可以考虑改为非 image-conditioned dataset/trainer 路线；但这不等价于现有 image-conditioned SS flow fine-tune。

## Active Hypotheses
- H1: `kl1e-4_step1000` latent 子集足够用于 flow 小规模稳定性试验。
  Evidence: 1024 个 latent 全量编码成功，文件完整，`SparseStructureLatent` 可正常加载。
  Uncertainty: 尚未运行 flow fine-tune，无法确认 loss 稳定性或采样质量。
- H2: 继续使用 `ImageConditionedSparseStructureLatent` 是更贴近现有 TRELLIS image-conditioned SS flow 的测试方式。
  Evidence: 目标配置和 trainer 都是 image-conditioned；直接改成 latent-only 会改变任务定义。
  Uncertainty: 当前本地 FaceScape 条件图像资源是否在别的路径，尚未定位到可直接 symlink 的 `renders_cond/`。

## Current Decision State
- Accepted: 不应直接使用原 `ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json` 开训新 latent 子集。
- Accepted: 原配置至少需要改 latent model 和 decoder 指向微调后的 `kl1e-4_step1000` 产物。
- Pending: 是否补齐 `renders_cond/` 后保持 image-conditioned flow，还是临时改 latent-only flow 做更窄的分布适配 smoke test。

## Next Actions
1. 优先定位原始 FaceScape `renders_cond/` 资源；若存在，给 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024` 建 symlink。
2. 复制一份 flow fine-tune 配置，改为 `latent_model=ss_enc_dec_fine_tune_kl1e-4_step0001000`、`ss_dec_path=outputs/ss_enc_dec_fine_tune_kl1e-4`、`ss_dec_ckpt=step0001000`。
3. 将 flow smoke test 步数先降到小范围，例如 200-500 step，并确认 dataset 初始化样本数大于 0 后再启动。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练和评估命令必须显式传有效 dataset root。
- 当前 latent 子集为 SS latent-only 数据，不应误认为已经 image-conditioned flow ready。
- 原 flow 配置使用官方 latent/decoder，不能代表刚微调好的 SS encoder/decoder。

## Open Questions
- 后续 flow 微调是否需要保持图片条件输入？
- 1024 样本是否足够暴露 flow 对 `kl1e-4_step1000` latent 的稳定性问题？
- 原始 FaceScape 条件图像是否仍保存在其他持久路径？


## HST-20260718-194513-01 - current.md snapshot

Description:
- 记录新增 kl1e-4 step1000 image-conditioned SS flow 配置前的状态

# Current State

## Active Goal
为 image-conditioned SS flow 适配实验补齐 FaceScape 条件渲染图，并让 `kl1e-4_step1000` latent 子集可被 flow dataset 读取。

## Current Working Thread
已将 `/root/autodl-fs/Facescape_cond` 下的 `renders_cond` 分卷 tar 解压到 `datasets/Facescape/renders_cond`，并按已有 train/test metadata 与 1024 flow 子集建立 `renders_cond` symlink 目录；当前 flow 子集可用样本数为 1023。

## Relevant State
- CFG-20260718-001
- CFG-20260717-116
- EXE-20260717-121
- EXE-20260718-002
- RUN-20260718-019
- RUN-20260718-020
- ART-20260717-001
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- 条件图分卷源为 `/root/autodl-fs/Facescape_cond/renders_cond.tar.part000` 到 `renders_cond.tar.part006`，总大小约 `134G`。
- 已通过管道直接解压到 `datasets/Facescape/renders_cond`，没有创建额外完整 tar 临时文件。
- 解压后的 `datasets/Facescape/renders_cond` 大小约 `135G`，包含 7173 个样本目录、324378 个图片文件。
- train split：`datasets/Facescape/train/metadata.csv` 有 6456 行，其中 `cond_rendered=True` 6453 行；`datasets/Facescape/train/renders_cond` 建立 6453 个 symlink，broken symlink 为 0。
- test split：`datasets/Facescape/test/metadata.csv` 有 720 行；`datasets/Facescape/test/renders_cond` 建立 720 个 symlink，broken symlink 为 0。
- flow 子集：`datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024/metadata.csv` 有 1024 行，其中 `cond_rendered=True` 1023 行；`renders_cond` 建立 1023 个 symlink，broken symlink 为 0。
- flow 子集 latent 列 `ss_latent_ss_enc_dec_fine_tune_kl1e-4_step0001000` 仍为 1024/1024 True。
- 使用 `ImageConditionedSparseStructureLatent` 初始化 flow 子集并设置 `latent_model=ss_enc_dec_fine_tune_kl1e-4_step0001000`、`ss_dec_path=outputs/ss_enc_dec_fine_tune_kl1e-4`、`ss_dec_ckpt=step0001000` 后，dataset length 为 1023。
- flow 子集首样本加载检查通过：`x_0` shape `(8,16,16,16)`，`cond` shape `(3,518,518)`，`x_0` finite，`cond` 值域为 `[0.0, 1.0]`。
- 解压后 `/root/autodl-tmp` 剩余空间约 `92G`。

## Interpretations
- FaceScape image-conditioned 条件数据已经补齐，train/test 按原 metadata split 可读取，且没有复制图片本体造成额外 135G 开销。
- 刚才单独编码的 `kl1e-4_step1000` flow 子集已经从 latent-only 状态变成 image-conditioned dataset 可读状态。
- flow 子集可用样本数是 1023 而不是 1024，这是因为原 metadata 里有 1 个样本 `cond_rendered=False`，属于预期过滤。
- 还不能直接使用原 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json`，因为该配置仍指向官方 latent model 和官方 decoder。

## Active Hypotheses
- H1: 使用 1023 个 image-conditioned flow 子集样本可以做短步数 smoke fine-tune，初步判断 flow 是否适配 `kl1e-4_step1000` latent 分布。
  Evidence: 条件图、latent、metadata 三者已对齐，dataset 初始化和单样本读取通过。
  Uncertainty: 尚未启动 flow 训练，loss 稳定性未知。
- H2: 原 train/test 条件图覆盖完整，可以支持后续扩大样本数。
  Evidence: train `cond_rendered=True` 6453/6453 覆盖，test 720/720 覆盖。
  Uncertainty: 尚未逐张校验 PNG 是否可解码；当前只做了目录、symlink 和首样本读取检查。

## Current Decision State
- Accepted: `renders_cond` 已解压并按 train/test/flow 子集建立入口。
- Accepted: flow 子集用修改后的 image-conditioned dataset args 可初始化为 1023 个样本。
- Pending: 复制并修改 flow fine-tune 配置后，启动短步数 smoke test。

## Next Actions
1. 复制 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json` 为新的 flow smoke/fine-tune 配置。
2. 将新配置的 `latent_model` 改为 `ss_enc_dec_fine_tune_kl1e-4_step0001000`，并将 decoder 指向 `outputs/ss_enc_dec_fine_tune_kl1e-4` 的 `step0001000`。
3. 先用 `--tryrun` 或短步数训练确认模型、dataset、条件图和输出路径都正常。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。
- 原 flow 配置使用官方 latent/decoder，不能代表刚微调好的 SS encoder/decoder。

## Open Questions
- flow smoke test 的步数、输出目录和是否保持 batch_size=16/batch_split=8 是否需要调整？
- 是否需要对 1023 个 flow 子集条件 PNG 做更完整的可解码校验？


## HST-20260718-223006-01 - current.md snapshot

Description:
- 记录 SS flow 配置重命名前的状态

# Current State

## Active Goal
准备 image-conditioned SS flow 配置，用于测试 flow 对 `kl1e-4_step1000` SS encoder/decoder latent 分布的适配性。

## Current Working Thread
已复制原 FaceScape SS flow 配置并改成绑定 `ss_enc_dec_fine_tune_kl1e-4_step0001000` latent 与本地 `step0001000` SS decoder；新配置通过 JSON 解析和 dataset 初始化验证。

## Relevant State
- CFG-20260718-001
- CFG-20260718-004
- CFG-20260717-103
- EXE-20260717-105
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- 新配置路径为 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape_kl1e-4_step1000.json`。
- 新配置复制自 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json`。
- 新配置保留模型结构：`SparseStructureFlowModel`、resolution `16`、in/out channels `8`、model channels `1024`、cond channels `1024`、24 blocks、16 heads、fp16。
- 新配置保留 trainer：`ImageConditionedFlowMatchingCFGTrainer`、`max_steps=40000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`i_print=10`、`i_log=10`、`i_save=500`、`i_sample=2000`。
- 新配置保留 denoiser 初始化：`weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`。
- 新配置将 `dataset.args.latent_model` 改为 `ss_enc_dec_fine_tune_kl1e-4_step0001000`。
- 新配置移除官方 `pretrained_ss_dec`，改为 `ss_dec_path=outputs/ss_enc_dec_fine_tune_kl1e-4`、`ss_dec_ckpt=step0001000`。
- JSON 解析验证通过。
- 使用新配置 dataset args 和 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024` 初始化 `ImageConditionedSparseStructureLatent`，dataset length 为 1023。
- 单样本读取验证通过：`x_0` shape `(8,16,16,16)`，`cond` shape `(3,518,518)`，`x_0` finite，`cond` 值域 `[0.0,1.0]`。

## Interpretations
- 新配置现在能代表“用官方 SS flow denoiser 初始化，微调到 `kl1e-4_step1000` SS latent 分布”的测试口径。
- 当前配置仍是完整 40000-step 微调参数，不是短步数 smoke 配置；如只做适配性验证，建议运行时先用 `--tryrun` 或再复制一个短步数 smoke 配置。
- flow 子集可用样本数 1023 是预期结果，因为原 1024 子集中 1 个样本 `cond_rendered=False`。

## Active Hypotheses
- H1: 新配置配合 1023 个 image-conditioned flow 子集可以启动 flow 适配 smoke test。
  Evidence: JSON 和 dataset 初始化均通过，首样本 latent/cond 可读取。
  Uncertainty: 尚未加载完整 denoiser 或执行训练 step。
- H2: 保留原 flow trainer 和 denoiser 初始化可以最大化隔离变量，让主要变化集中在 SS latent/decoder。
  Evidence: 本次只修改 dataset latent/decoder 指向，未改变优化器、batch、flow 模型结构。
  Uncertainty: `kl1e-4_step1000` latent scale 是否需要 normalization 或更低 lr 仍需训练曲线验证。

## Current Decision State
- Accepted: 已新增并验证 `kl1e-4_step1000` image-conditioned SS flow 配置。
- Pending: 是否直接用该 40000-step 配置启动，还是再派生 200-500 step smoke 配置。

## Next Actions
1. 如只做适配性检查，优先运行新配置的 `--tryrun`。
2. 若 `--tryrun` 通过，建议先用短步数输出目录跑 200-500 step 观察 loss 和 sample。
3. 如果短测稳定，再决定是否按 40000 step 继续正式 flow 微调。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。
- 新配置不是短步数 smoke 配置，`max_steps` 仍为 40000。

## Open Questions
- 是否需要单独复制一个 `max_steps=500` 的 smoke 配置？
- flow 适配测试输出目录命名是否使用 `ss_flow_img_finetune_facescape_kl1e-4_step1000_smoke`？


## HST-20260718-223400-01 - current.md snapshot

Description:
- 记录 SS flow 配置 max_steps 改为 1000 前的状态

# Current State

## Active Goal
准备 image-conditioned SS flow 配置，用于测试 flow 对 `kl1e-4_step1000` SS encoder/decoder latent 分布的适配性。

## Current Working Thread
已复制原 FaceScape SS flow 配置并改成绑定 `ss_enc_dec_fine_tune_kl1e-4_step0001000` latent 与本地 `step0001000` SS decoder；配置已重命名为短文件名，并通过 JSON 解析和 dataset 初始化验证。

## Relevant State
- CFG-20260718-001
- CFG-20260718-004
- CFG-20260717-103
- EXE-20260717-105
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- 新配置路径为 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`。
- 新配置复制自 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json`。
- 新配置保留模型结构：`SparseStructureFlowModel`、resolution `16`、in/out channels `8`、model channels `1024`、cond channels `1024`、24 blocks、16 heads、fp16。
- 新配置保留 trainer：`ImageConditionedFlowMatchingCFGTrainer`、`max_steps=40000`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`i_print=10`、`i_log=10`、`i_save=500`、`i_sample=2000`。
- 新配置保留 denoiser 初始化：`weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`。
- 新配置将 `dataset.args.latent_model` 改为 `ss_enc_dec_fine_tune_kl1e-4_step0001000`。
- 新配置移除官方 `pretrained_ss_dec`，改为 `ss_dec_path=outputs/ss_enc_dec_fine_tune_kl1e-4`、`ss_dec_ckpt=step0001000`。
- JSON 解析验证通过。
- 使用新配置 dataset args 和 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024` 初始化 `ImageConditionedSparseStructureLatent`，dataset length 为 1023。
- 单样本读取验证通过：`x_0` shape `(8,16,16,16)`，`cond` shape `(3,518,518)`，`x_0` finite，`cond` 值域 `[0.0,1.0]`。

## Interpretations
- 新配置现在能代表“用官方 SS flow denoiser 初始化，微调到 `kl1e-4_step1000` SS latent 分布”的测试口径。
- 当前配置仍是完整 40000-step 微调参数，不是短步数 smoke 配置；如只做适配性验证，建议运行时先用 `--tryrun` 或再复制一个短步数 smoke 配置。
- flow 子集可用样本数 1023 是预期结果，因为原 1024 子集中 1 个样本 `cond_rendered=False`。

## Active Hypotheses
- H1: 新配置配合 1023 个 image-conditioned flow 子集可以启动 flow 适配 smoke test。
  Evidence: JSON 和 dataset 初始化均通过，首样本 latent/cond 可读取。
  Uncertainty: 尚未加载完整 denoiser 或执行训练 step。
- H2: 保留原 flow trainer 和 denoiser 初始化可以最大化隔离变量，让主要变化集中在 SS latent/decoder。
  Evidence: 本次只修改 dataset latent/decoder 指向，未改变优化器、batch、flow 模型结构。
  Uncertainty: `kl1e-4_step1000` latent scale 是否需要 normalization 或更低 lr 仍需训练曲线验证。

## Current Decision State
- Accepted: 已新增并验证 `kl1e-4_step1000` image-conditioned SS flow 配置。
- Pending: 是否直接用该 40000-step 配置启动，还是再派生 200-500 step smoke 配置。

## Next Actions
1. 如只做适配性检查，优先运行新配置的 `--tryrun`。
2. 若 `--tryrun` 通过，建议先用短步数输出目录跑 200-500 step 观察 loss 和 sample。
3. 如果短测稳定，再决定是否按 40000 step 继续正式 flow 微调。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。
- 新配置不是短步数 smoke 配置，`max_steps` 仍为 40000。

## Open Questions
- 是否需要单独复制一个 `max_steps=500` 的 smoke 配置？
- flow 适配测试输出目录命名是否使用 `ss_flow_img_finetune_facescape_kl1e-4_step1000_smoke`？


## HST-20260718-224106-01 - current.md snapshot

Description:
- 记录 SS flow safetensors 转 pt 前的状态

# Current State

## Active Goal
准备 image-conditioned SS flow 配置，用于测试 flow 对 `kl1e-4_step1000` SS encoder/decoder latent 分布的适配性。

## Current Working Thread
已复制原 FaceScape SS flow 配置并改成绑定 `ss_enc_dec_fine_tune_kl1e-4_step0001000` latent 与本地 `step0001000` SS decoder；配置已重命名为短文件名，并将 `max_steps` 改为 1000。

## Relevant State
- CFG-20260718-001
- CFG-20260718-004
- CFG-20260717-103
- EXE-20260717-105
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- 新配置路径为 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`。
- 新配置复制自 `configs/generation/ss_flow_img_dit_L_16l8_fp16_finetune_facescape.json`。
- 新配置保留模型结构：`SparseStructureFlowModel`、resolution `16`、in/out channels `8`、model channels `1024`、cond channels `1024`、24 blocks、16 heads、fp16。
- 新配置保留 trainer：`ImageConditionedFlowMatchingCFGTrainer`、`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`、`i_print=10`、`i_log=10`、`i_save=500`、`i_sample=2000`。
- 新配置将 `trainer.args.max_steps` 从 40000 改为 1000。
- 新配置保留 denoiser 初始化：`weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`。
- 新配置将 `dataset.args.latent_model` 改为 `ss_enc_dec_fine_tune_kl1e-4_step0001000`。
- 新配置移除官方 `pretrained_ss_dec`，改为 `ss_dec_path=outputs/ss_enc_dec_fine_tune_kl1e-4`、`ss_dec_ckpt=step0001000`。
- JSON 解析验证通过。
- 使用新配置 dataset args 和 `datasets/Facescape_ss_latent_kl1e-4_step1000_train_1024` 初始化 `ImageConditionedSparseStructureLatent`，dataset length 为 1023。
- 单样本读取验证通过：`x_0` shape `(8,16,16,16)`，`cond` shape `(3,518,518)`，`x_0` finite，`cond` 值域 `[0.0,1.0]`。

## Interpretations
- 新配置现在能代表“用官方 SS flow denoiser 初始化，微调到 `kl1e-4_step1000` SS latent 分布”的测试口径。
- 当前配置已变成 1000-step 适配性测试配置，更适合先观察 flow 对 `kl1e-4_step1000` latent 的初期稳定性。
- flow 子集可用样本数 1023 是预期结果，因为原 1024 子集中 1 个样本 `cond_rendered=False`。

## Active Hypotheses
- H1: 新配置配合 1023 个 image-conditioned flow 子集可以启动 flow 适配 smoke test。
  Evidence: JSON 和 dataset 初始化均通过，首样本 latent/cond 可读取。
  Uncertainty: 尚未加载完整 denoiser 或执行训练 step。
- H2: 保留原 flow trainer 和 denoiser 初始化可以最大化隔离变量，让主要变化集中在 SS latent/decoder。
  Evidence: 本次只修改 dataset latent/decoder 指向，未改变优化器、batch、flow 模型结构。
  Uncertainty: `kl1e-4_step1000` latent scale 是否需要 normalization 或更低 lr 仍需训练曲线验证。

## Current Decision State
- Accepted: 已新增并验证 `kl1e-4_step1000` image-conditioned SS flow 配置。
- Accepted: 已将新配置 `max_steps` 改为 1000。
- Pending: 是否先执行 `--tryrun`，还是直接启动 1000-step flow 训练。

## Next Actions
1. 优先运行新配置的 `--tryrun`，确认完整模型初始化、denoiser 权重加载和 dataset 读取都正常。
2. 若 `--tryrun` 通过，启动 1000-step flow 训练并观察 loss、grad clip、sample 与 occupancy。
3. 如果 1000-step 稳定，再决定是否扩大样本或延长正式 flow 微调。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。
- 新配置是 1000-step 适配性测试配置，不是完整 40000-step 正式微调配置。

## Open Questions
- flow 适配测试输出目录命名是否使用 `ss_flow_finetune_kl1e-4_step1000`？


## HST-20260718-225142-01 - current.md snapshot

Description:
- 记录复制 trellis-normal-v0-1 权重前的状态

# Current State

## Active Goal
准备 image-conditioned SS flow 适配训练所需的官方 denoiser `.pt` 初始化权重，以及 `kl1e-4_step1000` latent 对应的 flow 配置。

## Current Working Thread
已将 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors` 转换为同目录 `.pt` state_dict，并通过脚本内 strict reload 与额外 `torch.load` 检查。

## Relevant State
- CFG-20260717-102
- CFG-20260718-004
- EXE-20260717-105
- EXE-20260717-128
- RUN-20260718-021
- ART-20260717-002
- ART-20260718-019
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- 转换源 safetensors 路径为 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.safetensors`。
- 转换输出 `.pt` 路径为 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`。
- 转换命令使用 `fine_tuning/convert_safetensors_to_pt.py`，并指定 `--train_config configs/generation/ss_flow_img_dit_L_16l8_fp16.json --model_key denoiser`。
- 转换脚本 strict 加载 safetensors，并验证保存后的 `.pt` 可通过 `torch.load(..., weights_only=True)` strict reload。
- 额外检查显示 `.pt` 为 `OrderedDict`，489 个 tensor key，总参数数 559737864，首 key 为 `pos_emb`。
- `.pt` 文件大小约 `1.1G`。
- 当前 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 denoiser 初始化仍是 `weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`。

## Interpretations
- 新 `.pt` 权重满足训练器 `finetune_ckpt.denoiser` 的 state_dict 格式要求。
- 如果目标是直接从官方 image-conditioned SS flow 初始化，应将 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 `trainer.args.finetune_ckpt.denoiser` 改为 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`。
- 如果 `weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt` 是之前已有的人工处理权重，则需要确认它与刚转换出的官方 `.pt` 是否一致或哪个更适合作为起点。

## Active Hypotheses
- H1: 使用刚转换的官方 `.pt` 作为 denoiser 初始化，可以避免训练器读取 safetensors 格式不兼容的问题。
  Evidence: 训练器通过 `torch.load(...weights_only=True)` 加载 finetune ckpt；转换后 `.pt` 已验证可读。
  Uncertainty: 尚未实际启动 flow 训练验证。
- H2: 直接使用官方 image-conditioned SS flow denoiser初始化，是评估 `kl1e-4_step1000` latent 适配性的合理起点。
  Evidence: flow 配置模型结构与官方 `ss_flow_img_dit_L_16l8_fp16` 对齐。
  Uncertainty: 当前配置仍指向 `weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`，是否替换需要用户确认或下一步修改。

## Current Decision State
- Accepted: 官方 `ss_flow_img_dit_L_16l8_fp16.safetensors` 已成功转换为同目录 `.pt`。
- Pending: 是否把 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 denoiser 初始化路径改为新 `.pt`。

## Next Actions
1. 决定 flow 配置中的 `finetune_ckpt.denoiser` 使用新 `.pt` 还是现有 `weights/fine_tune` 权重。
2. 若使用新 `.pt`，修改 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 并运行 `--tryrun`。
3. `--tryrun` 通过后启动 1000-step flow 适配训练。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。

## Open Questions
- 是否将 flow 配置的 denoiser 初始化路径改为 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`？


## HST-20260718-225406-01 - current.md snapshot

Description:
- 记录 normal SS flow safetensors 转 pt 前的状态

# Current State

## Active Goal
准备 image-conditioned SS flow 适配训练所需的权重资源，并补充 normal-conditioned TRELLIS 权重目录到当前项目。

## Current Working Thread
已将官方 image SS flow safetensors 转换为 `.pt`，并将 `/root/autodl-fs/trellis-normal-v0-1` 复制到当前项目 `microsoft/trellis-normal-v0-1`；源/目标文件列表一致。

## Relevant State
- CFG-20260717-102
- CFG-20260718-004
- EXE-20260717-105
- EXE-20260717-128
- RUN-20260718-021
- ART-20260717-002
- ART-20260718-019
- ART-20260718-020
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt` 已由官方 safetensors 转换得到，大小约 `1.1G`，489 个 tensor key，总参数数 559737864。
- `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 当前 `max_steps=1000`，dataset 指向 `ss_enc_dec_fine_tune_kl1e-4_step0001000` latent 和本地 step1000 SS decoder。
- `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 denoiser 初始化仍是 `weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`。
- 已找到 AutoDL FS 中的 normal 权重目录：`/root/autodl-fs/trellis-normal-v0-1`。
- 已复制到当前项目：`microsoft/trellis-normal-v0-1`。
- 源/目标 normal 权重目录大小均约 `2.5G`，文件数均为 24，文件列表 diff 一致。
- normal 权重目录包含 `ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors`、`ckpts/slat_flow_normal_dit_L_64l8p2_fp16.safetensors`、`ckpts/ss_dec_conv3d_16l8_fp16.safetensors`、`ckpts/slat_dec_mesh_swin8_B_64l8m256c_fp16.safetensors`。

## Interpretations
- 当前项目同时具备官方 image SS flow `.pt` 初始化权重和 normal-conditioned TRELLIS safetensors 权重目录。
- 若要运行当前 `ss_flow_finetune_kl1e-4_step1000.json`，需要确认 denoiser 初始化应使用官方 image flow `.pt`，还是 normal flow 权重。
- 如果目标是 image-conditioned flow 适配 `kl1e-4_step1000` latent，配置中的 trainer 是 `ImageConditionedFlowMatchingCFGTrainer`，更自然的初始化是官方 image SS flow `.pt`。
- 如果目标是 normal-conditioned flow，则还需要检查/新增 normal-conditioned dataset 和 trainer 配置，不能只替换 denoiser 路径。

## Active Hypotheses
- H1: 当前 image-conditioned flow 适配实验应优先使用 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt` 初始化。
  Evidence: 配置 trainer 和 dataset 都是 image-conditioned；该 `.pt` 与 image SS flow 模型结构 strict 验证通过。
  Uncertainty: 用户是否希望改用 normal-conditioned 预训练权重尚未确认。
- H2: `microsoft/trellis-normal-v0-1` 可作为后续 normal 条件实验的本地模型目录。
  Evidence: 目录复制完整，包含 normal flow 和 decoder safetensors/json。
  Uncertainty: 尚未转换 normal safetensors 为 `.pt` 或接入训练配置。

## Current Decision State
- Accepted: `trellis-normal-v0-1` 已复制到当前项目 `microsoft/` 下。
- Pending: 当前 flow 微调配置的 denoiser 初始化路径是否切换到官方 image flow `.pt`。
- Pending: 是否需要把 normal flow safetensors 也转换为 `.pt`。

## Next Actions
1. 决定 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 `finetune_ckpt.denoiser` 应指向 image flow `.pt` 还是 normal flow 权重。
2. 如果使用 normal flow 权重，先将 `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors` 转换为 `.pt` 并验证模型结构匹配。
3. 确认权重路径后运行 `train.py --tryrun`。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。

## Open Questions
- 当前 flow 适配实验是继续 image-conditioned，还是要切到 normal-conditioned 权重路线？


## HST-20260718-225704-01 - current.md snapshot

Description:
- 记录 SS flow 配置 denoiser 路径改为 normal pt 前的状态

# Current State

## Active Goal
准备 SS flow 适配训练所需的 image 与 normal denoiser `.pt` 初始化权重，并维护 `kl1e-4_step1000` latent 对应的 flow 配置。

## Current Working Thread
已将官方 image SS flow safetensors 与 `trellis-normal-v0-1` normal SS flow safetensors 分别转换为 `.pt` state_dict；两个 `.pt` 都通过 strict reload 与额外 `torch.load` 检查。

## Relevant State
- CFG-20260717-102
- CFG-20260718-004
- EXE-20260717-105
- EXE-20260717-128
- RUN-20260718-021
- RUN-20260718-022
- ART-20260717-002
- ART-20260718-019
- ART-20260718-020
- ART-20260718-021
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- image SS flow `.pt` 路径为 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt`。
- normal SS flow `.pt` 路径为 `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.pt`。
- normal SS flow 转换源为 `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors`。
- normal SS flow 转换使用同目录 `ss_flow_normal_dit_L_16l8_fp16.json` 构建模型，未额外指定训练配置。
- normal SS flow `.pt` 文件大小约 `1.1G`。
- normal SS flow `.pt` 额外检查显示为 `OrderedDict`，489 个 tensor key，总参数数 559737864，首 key 为 `pos_emb`。
- `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 当前 `max_steps=1000`，dataset 指向 `ss_enc_dec_fine_tune_kl1e-4_step0001000` latent 和本地 step1000 SS decoder。
- `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 denoiser 初始化仍是 `weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt`。

## Interpretations
- 当前项目已有两个可供 SS flow 初始化的 `.pt` denoiser：官方 image flow 和 normal flow。
- 如果继续跑 image-conditioned flow 适配，官方 image flow `.pt` 是语义上最匹配的初始化。
- normal flow `.pt` 虽然模型结构兼容，但条件语义不同；如果用于 image-conditioned trainer，需要把它视为一个对照实验，而不是默认选择。

## Active Hypotheses
- H1: image-conditioned `kl1e-4_step1000` flow 适配实验应优先用 `microsoft/TRELLIS-image-large/ckpts/ss_flow_img_dit_L_16l8_fp16.pt` 初始化。
  Evidence: 当前 dataset/trainer 是 image-conditioned，image `.pt` 与该训练语义一致。
  Uncertainty: 用户可能希望用 normal flow 做迁移对照。
- H2: normal SS flow `.pt` 可作为额外初始化对照，测试 normal 预训练权重是否更适合高精度三维人脸数据。
  Evidence: normal `.pt` 已严格转换并与模型结构兼容。
  Uncertainty: 条件输入分布和 image-conditioned trainer 的匹配度未知。

## Current Decision State
- Accepted: normal SS flow safetensors 已成功转换为同目录 `.pt`。
- Pending: 当前 1000-step flow 配置的 `finetune_ckpt.denoiser` 最终指向 image `.pt`、normal `.pt`，还是现有 `weights/fine_tune` 路径。

## Next Actions
1. 决定 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 `finetune_ckpt.denoiser` 路径。
2. 路径确认后运行 `train.py --tryrun`。
3. `--tryrun` 通过后启动 1000-step flow 适配训练。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。

## Open Questions
- 当前 flow 适配实验是否要使用 normal SS flow `.pt` 作为初始化，还是使用官方 image SS flow `.pt`？


## HST-20260718-234725-01 - current.md snapshot

Description:
- flow 配置已指向 normal SS flow pt 权重，准备运行 1000-step 适配训练

# Current State

## Active Goal
准备使用 normal SS flow denoiser 初始化的 `kl1e-4_step1000` image-conditioned SS flow 适配训练配置。

## Current Working Thread
已将 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 的 `trainer.args.finetune_ckpt.denoiser` 指向刚转换完成的 normal SS flow `.pt` 权重。

## Relevant State
- CFG-20260718-004
- EXE-20260717-105
- EXE-20260717-128
- RUN-20260718-022
- ART-20260718-021
- ART-20260718-016
- ART-20260718-017
- ART-20260718-018

## Facts
- flow 配置路径为 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`。
- 该配置 `max_steps=1000`。
- 该配置 dataset 指向 `ss_enc_dec_fine_tune_kl1e-4_step0001000` latent 和本地 step1000 SS decoder。
- 该配置 `trainer.args.finetune_ckpt.denoiser` 已从 `weights/fine_tune/ss_flow_normal_dit_L_16l8_fp16.pt` 改为 `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.pt`。
- 新 denoiser `.pt` 由 `microsoft/trellis-normal-v0-1/ckpts/ss_flow_normal_dit_L_16l8_fp16.safetensors` 转换得到。
- 新 denoiser `.pt` 已验证为 `OrderedDict`，489 个 tensor key，总参数数 559737864。

## Interpretations
- 当前 1000-step flow 适配配置现在会从 normal-conditioned SS flow 预训练 denoiser 初始化。
- 该权重模型结构与 SS flow denoiser 兼容，但条件语义与当前 image-conditioned dataset/trainer 可能不同；这次实验应理解为 normal 权重迁移对照。

## Active Hypotheses
- H1: normal SS flow denoiser 可能对高精度三维人脸模型的 sparse structure 先验更有帮助。
  Evidence: 用户选择将配置指向 normal flow `.pt`；normal 权重已转换并结构兼容。
  Uncertainty: 条件输入语义不同，训练初期 loss 和 sample 稳定性需要实测。
- H2: 如果 normal 初始化不稳定，官方 image flow `.pt` 仍是备用初始化。
  Evidence: 官方 image flow `.pt` 已转换并验证通过。
  Uncertainty: 尚未跑对照实验。

## Current Decision State
- Accepted: 1000-step flow 配置已改为使用 normal SS flow `.pt` 初始化。
- Pending: 运行 `train.py --tryrun` 验证完整模型加载和 trainer 初始化。

## Next Actions
1. 运行新配置的 `--tryrun`。
2. `--tryrun` 通过后启动 1000-step flow 训练。
3. 训练后分析 loss、sample、occupancy，并决定是否与官方 image flow 初始化做对照。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前 `data/` 默认目录不存在，训练命令必须显式传有效 dataset root。

## Open Questions
- 是否立即运行 `--tryrun`？


## HST-20260720-084648-01 - current.md snapshot

Description:
- kl1e-4 flow 训练稳定，下一步待做固定条件采样评估

# Current State

## Active Goal
评估 `kl1e-4_step1000` SS encoder/decoder latent 在 SS flow 阶段的适配效果。

## Current Working Thread
用户已完成 `outputs/ss_flow_finetune_kl1e-4_step1000` 的 1000-step flow 微调；当前结论是训练数值稳定、loss 有效下降，但还需要固定条件采样评估来判断 checkpoint 优劣和 flow 质量。

## Relevant State
- RUN-20260718-023
- CFG-20260718-004
- EXE-20260717-105
- ART-20260718-022
- ART-20260718-021
- ART-20260718-016
- ART-20260718-018

## Facts
- flow 训练命令记录在输出目录 `command.txt`，使用 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`、1024 latent 子集、`--num_gpus 1 --auto_retry 0`。
- 输出目录包含 step500 和 step1000 的 denoiser、EMA denoiser、misc checkpoint。
- loss 覆盖 1000 steps，未发现 NaN/Inf；first100 mean `0.2824327543`，last100 mean `0.2503342265`，下降约 `11.365%`。
- grad_norm 与 log_scale 均 finite，log 中未检出 traceback/error/retry。
- final sample 非空、非满体，能形成头颈粗结构，但只代表少量可视化样本。

## Interpretations
- `kl1e-4` latent 分布对 SS flow 训练没有表现出明显数值不适配。
- 1000-step 结果目前应视为稳定性通过和可继续评估的基线，而不是最终最优 flow checkpoint。

## Active Hypotheses
- H1: step1000 可能优于 step500，但优势需要采样定量确认。
  Evidence: 训练 loss 到最后仍在缓慢下降；step500/step1000 都已保存 checkpoint。
  Uncertainty: 训练 loss 不等价于生成质量，可能存在过拟合或条件漂移。
- H2: normal SS flow 初始化没有引发训练发散，但是否优于 image SS flow 初始化仍未知。
  Evidence: 本次 normal-init run 数值稳定，sample 未崩。
  Uncertainty: 缺少同配置 image-init 对照。

## Current Decision State
- Accepted: `kl1e-4_step1000` latent 可以进入 flow 阶段继续评估。
- Pending: 用固定条件采样和定量指标判断 step500 vs step1000、normal-init vs image-init。

## Next Actions
1. 编写或复用 flow 固定条件采样评估脚本。
2. 对 step500/step1000 EMA checkpoint 固定条件采样并统计 occupancy ratio、非空/满体比例和可用的 GT 对齐指标。
3. 决定是否延长训练、切换 image-flow 初始化对照，或调整 KL/学习率。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前训练 sample 只保存 init/final 图片，没有中间 step 图片。

## Open Questions
- 是否先做 step500 vs step1000 的固定条件 flow 采样评估？


## HST-20260720-085247-01 - current.md snapshot

Description:
- 按更新后的 maintain-project-state skill 对齐 ledger 前的当前状态

# Current State

## Active Goal
评估 `kl1e-4_step1000` SS encoder/decoder latent 在 SS flow 阶段的可用性，重点从训练稳定性推进到采样可用性。

## Current Working Thread
用户已完成 `outputs/ss_flow_finetune_kl1e-4_step1000` 的 1000-step flow 微调；当前结论是训练数值稳定、loss 有效下降。TRELLIS 已有端到端推理入口可复用，但还缺一个面向本实验的薄评估 wrapper，用于固定样本/seed、替换微调 SS flow 与 SS decoder，并输出 voxel 级指标。

## Relevant State
- RUN-20260718-023
- CFG-20260718-004
- EXE-20260717-105
- ART-20260718-022
- ART-20260718-021
- ART-20260718-016
- ART-20260718-018

## Facts
- flow 训练命令记录在输出目录 `command.txt`，使用 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`、1024 latent 子集、`--num_gpus 1 --auto_retry 0`。
- 输出目录包含 step500 和 step1000 的 denoiser、EMA denoiser、misc checkpoint。
- loss 覆盖 1000 steps，未发现 NaN/Inf；first100 mean `0.2824327543`，last100 mean `0.2503342265`，下降约 `11.365%`。
- grad_norm 与 log_scale 均 finite，log 中未检出 traceback/error/retry。
- final sample 非空、非满体，能形成头颈粗结构，但只代表少量可视化样本。
- 现有 TRELLIS 推理入口包括 `example.py`、`app.py` 和 `trellis/pipelines/trellis_image_to_3d.py`。
- `TrellisImageTo3DPipeline.run()` 可完成 image condition、SS flow、SS decoder、SLAT flow 到最终 3D 表征的端到端生成。
- 本实验需要截取 SS 阶段中间结果，即 condition image -> SS flow latent -> fine-tuned SS decoder -> binary voxel。

## Interpretations
- `kl1e-4` latent 分布对 SS flow 训练没有表现出明显数值不适配。
- 1000-step 结果目前应视为稳定性通过和可继续评估的基线，而不是最终最优 flow checkpoint。
- 不需要重写 TRELLIS 推理链路；应复用 pipeline、sampler、image condition encoder、SS decoder 和现有 metric 函数。
- 官方 demo 推理脚本偏最终资产生成，不能直接保证固定样本/固定 seed/多 checkpoint 对比/voxel 指标落盘，因此还需要评估 wrapper。

## Active Hypotheses
- H1: step1000 可能优于 step500，但优势需要采样定量确认。
  Evidence: 训练 loss 到最后仍在缓慢下降；step500/step1000 都已保存 checkpoint。
  Uncertainty: 训练 loss 不等价于生成质量，可能存在过拟合或条件漂移。
- H2: normal SS flow 初始化没有引发训练发散，但是否优于 image SS flow 初始化仍未知。
  Evidence: 本次 normal-init run 数值稳定，sample 未崩。
  Uncertainty: 缺少同配置 image-init 对照。

## Current Decision State
- Accepted: `kl1e-4_step1000` latent 可以进入 flow 阶段继续评估。
- Accepted: 评估实现应优先复用 TRELLIS 现成推理模块，而不是重新实现 cond-to-voxel 的核心模型逻辑。
- Pending: 用固定条件采样和定量指标判断 step500 vs step1000、normal-init vs image-init。

## Next Actions
1. 新增薄的 flow 固定条件采样评估 wrapper，复用 `TrellisImageTo3DPipeline`/sampler/decoder/metrics。
2. 对 step500/step1000 EMA checkpoint 固定条件采样并统计 occupancy ratio、非空/满体比例和可用的 GT 对齐指标。
3. 决定是否延长训练、切换 image-flow 初始化对照，或调整 KL/学习率。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前训练 sample 只保存 init/final 图片，没有中间 step 图片。

## Open Questions
- 是否先实现轻量版，仅评估 16/32 个固定 test 样本的 step500 vs step1000？


## HST-20260720-090408-01 - current.md snapshot

Description:
- 迁移到新版 maintain-project-state 精简 current schema 前的状态

# Current State

## Active Goal
评估 `kl1e-4_step1000` SS encoder/decoder latent 在 SS flow 阶段的可用性，重点从训练稳定性推进到采样可用性。

## Current Working Thread
用户已完成 `outputs/ss_flow_finetune_kl1e-4_step1000` 的 1000-step flow 微调；当前结论是训练数值稳定、loss 有效下降。已按更新后的 `maintain-project-state` skill 对 `.project-state/` 做规则对齐检查：当前关键配置、训练 run 和输出 artifact 均已有详细记录。TRELLIS 已有端到端推理入口可复用，但还缺一个面向本实验的薄评估 wrapper，用于固定样本/seed、替换微调 SS flow 与 SS decoder，并输出 voxel 级指标。

## Relevant State
- EVT-20260720-085252-01
- RUN-20260718-023
- CFG-20260718-004
- EXE-20260717-105
- ART-20260718-022
- ART-20260718-021
- ART-20260718-016
- ART-20260718-018

## Facts
- flow 训练命令记录在输出目录 `command.txt`，使用 `configs/generation/ss_flow_finetune_kl1e-4_step1000.json`、1024 latent 子集、`--num_gpus 1 --auto_retry 0`。
- 输出目录包含 step500 和 step1000 的 denoiser、EMA denoiser、misc checkpoint。
- loss 覆盖 1000 steps，未发现 NaN/Inf；first100 mean `0.2824327543`，last100 mean `0.2503342265`，下降约 `11.365%`。
- grad_norm 与 log_scale 均 finite，log 中未检出 traceback/error/retry。
- final sample 非空、非满体，能形成头颈粗结构，但只代表少量可视化样本。
- 现有 TRELLIS 推理入口包括 `example.py`、`app.py` 和 `trellis/pipelines/trellis_image_to_3d.py`。
- `TrellisImageTo3DPipeline.run()` 可完成 image condition、SS flow、SS decoder、SLAT flow 到最终 3D 表征的端到端生成。
- 本实验需要截取 SS 阶段中间结果，即 condition image -> SS flow latent -> fine-tuned SS decoder -> binary voxel。
- 2026-07-20 按更新后的 project-state skill 规则核对后，未发现当前 flow 工作链路缺少必须的 RUN/CFG/ART 记录。
- `configs/generation/ss_flow_finetune_kl1e-4_step1000.json` 与 `eval/ss_eval_kl1e-4_steps.json` JSON 格式校验通过。

## Interpretations
- `kl1e-4` latent 分布对 SS flow 训练没有表现出明显数值不适配。
- 1000-step 结果目前应视为稳定性通过和可继续评估的基线，而不是最终最优 flow checkpoint。
- 不需要重写 TRELLIS 推理链路；应复用 pipeline、sampler、image condition encoder、SS decoder 和现有 metric 函数。
- 官方 demo 推理脚本偏最终资产生成，不能直接保证固定样本/固定 seed/多 checkpoint 对比/voxel 指标落盘，因此还需要评估 wrapper。

## Active Hypotheses
- H1: step1000 可能优于 step500，但优势需要采样定量确认。
  Evidence: 训练 loss 到最后仍在缓慢下降；step500/step1000 都已保存 checkpoint。
  Uncertainty: 训练 loss 不等价于生成质量，可能存在过拟合或条件漂移。
- H2: normal SS flow 初始化没有引发训练发散，但是否优于 image SS flow 初始化仍未知。
  Evidence: 本次 normal-init run 数值稳定，sample 未崩。
  Uncertainty: 缺少同配置 image-init 对照。

## Current Decision State
- Accepted: `kl1e-4_step1000` latent 可以进入 flow 阶段继续评估。
- Accepted: 评估实现应优先复用 TRELLIS 现成推理模块，而不是重新实现 cond-to-voxel 的核心模型逻辑。
- Accepted: 当前 ledger 对本工作链路已经满足详细记录优先原则；后续新增脚本/评估结果时再追加 EXE/RUN/ART。
- Pending: 用固定条件采样和定量指标判断 step500 vs step1000、normal-init vs image-init。

## Next Actions
1. 新增薄的 flow 固定条件采样评估 wrapper，复用 `TrellisImageTo3DPipeline`/sampler/decoder/metrics。
2. 对 step500/step1000 EMA checkpoint 固定条件采样并统计 occupancy ratio、非空/满体比例和可用的 GT 对齐指标。
3. 决定是否延长训练、切换 image-flow 初始化对照，或调整 KL/学习率。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录和权重文件不提交到 git。
- 当前训练 sample 只保存 init/final 图片，没有中间 step 图片。

## Open Questions
- 是否先实现轻量版，仅评估 16/32 个固定 test 样本的 step500 vs step1000？


## HST-20260720-092000-01 - current.md snapshot

Description:
- 将 flow 评估目标从 checkpoint 比较收窄为 kl1e-4 可用性验证前的状态

# Current State

## Goal
评估 `kl1e-4_step1000` SS encoder/decoder latent 在 SS flow 阶段的采样可用性。

## Key State
- `kl1e-4_step1000` flow 1000-step 训练数值稳定，但还缺固定条件采样评估来判断采样质量。
- 评估实现应复用 TRELLIS 现成推理模块，只新增薄的批量评估 wrapper。
- `.project-state/` 已迁移到新版 skill 的精简记录 schema，资产记录使用 `AST`。

## Next Actions
1. 新增固定条件 SS flow 采样评估 wrapper。
2. 对 step500/step1000 EMA checkpoint 做固定样本、固定 seed 的 voxel 指标评估。
3. 根据采样可用性决定是否延长训练、切换 image-flow 初始化对照或调整 KL。

## Relevant Records
- RUN-20260718-023
- CFG-20260718-004
- EXE-20260717-105
- AST-20260718-022
- AST-20260718-016


## HST-20260720-092721-01 - current.md snapshot

Description:
- 新增 SS flow 采样评估脚本前的当前状态

# Current State

## Goal
判断 `kl=1e-4` 的 SS encoder/decoder latent 分布是否适合后续 SS flow 阶段。

## Key State
- 已有 1000-step flow 训练显示 `kl=1e-4` 数值可训练，但还需要确认采样出的 voxel 是否整体合理。
- 不需要评估 EMA 权重，也不需要比较 step500 和 step1000 谁更好。
- 评估只服务于一个问题：`kl=1e-4` 是否适合 flow 阶段继续使用。
- `.project-state/` 已迁移到新版 skill 的精简记录 schema，资产记录使用 `AST`。

## Next Actions
1. 新增或复用最小化 SS flow 采样检查脚本，只加载当前 `kl=1e-4` flow 结果的一个代表 checkpoint。
2. 用固定少量 test 条件图采样并统计空体/满体比例、occupancy ratio 和基础 voxel 合理性。
3. 根据结果给出 `kl=1e-4` 是否适合继续进入后续 flow 微调链路的结论。

## Relevant Records
- RUN-20260718-023
- CFG-20260718-004
- EXE-20260717-105
- AST-20260718-022
- AST-20260718-016


## HST-20260720-093235-01 - current.md snapshot

Description:
- 记录 SS flow 采样评估脚本 dtype 修复前的当前状态

# Current State

## Goal
判断 `kl=1e-4` 的 SS encoder/decoder latent 分布是否适合后续 SS flow 阶段。

## Key State
- 已有 1000-step flow 训练显示 `kl=1e-4` 数值可训练，但还需要确认采样出的 voxel 是否整体合理。
- 已新增最小化 SS flow step1000 采样评估脚本，评估不涉及 EMA 或 step500/step1000 对比。
- 评估只服务于一个问题：`kl=1e-4` 是否适合 flow 阶段继续使用。

## Next Actions
1. 运行 `eval/evaluate_ss_flow_sparse_structure.py` 评估 SS flow step1000。
2. 检查 per-sample 指标、summary 和导出的 PLY 可视化。
3. 根据结果给出 `kl=1e-4` 是否适合继续进入后续 flow 微调链路的结论。

## Relevant Records
- RUN-20260718-023
- CFG-20260718-004
- EXE-20260720-001
- AST-20260718-022
- AST-20260718-016


## HST-20260720-093648-01 - current.md snapshot

Description:
- 记录 SS flow kl1e-4 step1000 采样评估结果前的当前状态

# Current State

## Goal
判断 `kl=1e-4` 的 SS encoder/decoder latent 分布是否适合后续 SS flow 阶段。

## Key State
- 已有 1000-step flow 训练显示 `kl=1e-4` 数值可训练，但还需要确认采样出的 voxel 是否整体合理。
- SS flow step1000 采样评估脚本的 decoder dtype 探测问题已修复，改用显式 `--resolution 64`。
- 评估只服务于一个问题：`kl=1e-4` 是否适合 flow 阶段继续使用。

## Next Actions
1. 运行 `eval/evaluate_ss_flow_sparse_structure.py` 评估 SS flow step1000。
2. 检查 per-sample 指标、summary 和导出的 PLY 可视化。
3. 根据结果给出 `kl=1e-4` 是否适合继续进入后续 flow 微调链路的结论。

## Relevant Records
- RUN-20260720-001
- CFG-20260718-004
- EXE-20260720-001
- AST-20260718-022
- AST-20260718-016
