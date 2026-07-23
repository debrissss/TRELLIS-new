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


## HST-20260718-175250-01 - current.md snapshot

Description:
- before recording completed kl1e-7 SLat GS fine-tune analysis

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


## HST-20260718-183857-01 - current.md snapshot

Description:
- before recording fixed SLat eval tooling implementation

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

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
- ART-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。

## Next Actions
1. 用固定 test/holdout 样本评估 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 的 step1000 与 EMA step1000。
2. 对比本次 `lambda_kl=1e-7` 与此前 v2 `lambda_kl=1e-6` 的固定样本重建质量，而不只看训练日志均值。
3. 若验证质量稳定，准备 SLat flow 人脸域微调配置，明确使用哪个 encoder/decoder checkpoint 生成或解码 latent。
4. 低配机器测速时继续记录统一口径的稳定段 samples/h、端到端 samples/h 与单位小时成本。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。

## Open Questions
- 本次 step1000 EMA 与 non-EMA 哪个在固定验证集上更好？
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？


## HST-20260718-184527-01 - current.md snapshot

Description:
- before recording kl1e-7 fixed eval results

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

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
- ART-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- 已新增 `eval/` 评估工具：固定 FaceScape eval 子集准备、SLat enc/dec checkpoint 重建评估、多 run summary 对比。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。
- 固定 eval 子集流程可以避免训练 DataLoader 的随机视角和随机 batch 噪声，适合作为不同 KL 权重与 EMA/non-EMA checkpoint 的选择依据。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。

## Next Actions
1. 用固定 test/holdout 样本评估 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 的 step1000 与 EMA step1000。
2. 对比本次 `lambda_kl=1e-7` 与此前 v2 `lambda_kl=1e-6` 的固定样本重建质量，而不只看训练日志均值。
3. 若验证质量稳定，准备 SLat flow 人脸域微调配置，明确使用哪个 encoder/decoder checkpoint 生成或解码 latent。
4. 低配机器测速时继续记录统一口径的稳定段 samples/h、端到端 samples/h 与单位小时成本。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。

## Open Questions
- 本次 step1000 EMA 与 non-EMA 哪个在固定验证集上更好？
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？


## HST-20260718-194902-01 - current.md snapshot

Description:
- 记录 flow smoke 配置创建前的当前状态

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

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
- ART-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- 已新增 `eval/` 评估工具：固定 FaceScape eval 子集准备、SLat enc/dec checkpoint 重建评估、多 run summary 对比。
- 已生成固定评估集 `datasets/Facescape_eval/slat_gs_eval50`，从 FaceScape test 固定抽取 50 个样本。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 非 EMA step1000 在 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.0253610、mean L1 0.00512148、mean PSNR 33.1676、mean SSIM loss 0.0522815、mean LPIPS 0.0489114、mean KL 9.89067。
- EMA step1000 在同一 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.111183、mean L1 0.0403950、mean PSNR 20.7630、mean LPIPS 0.212343、mean KL 0.0541608。
- 当前本机只发现 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 这一组带 KL 命名的 SLat enc/dec 微调输出；没有发现其它 KL 值的最终 checkpoint 可纳入横评。
- 已生成全可用 KL 终权重横评 CSV `eval_outputs/slat_all_kl_final_eval50_view0_compare.csv`，当前覆盖 `kl1e-7` 非 EMA 与 `kl1e-7_ema`。
- 已生成并原地扩展独立 smoke latent 数据集 `datasets/Facescape_slat_kl1e-7_nonema_smoke`：从 train 固定抽取 1024 个样本，使用 `kl1e-7` step1000 非 EMA encoder 编码，metadata 中 `latent_dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000` 列 1024 条均为 True。
- 已从 `/root/autodl-fs/Facescape_cond` 分卷 tar 包解压 `renders_cond` 到 `datasets/Facescape/renders_cond`，并按现有 train/test metadata 建立 `train/renders_cond` 与 `test/renders_cond` 软链接。
- `datasets/Facescape/train` 的 6456 条中 6453 条有条件图，缺 3 条；`datasets/Facescape/test` 的 720 条全部有条件图。
- `datasets/Facescape_slat_kl1e-7_nonema_smoke` 的 1024 条中 1023 条已链接条件图，缺 1 条 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。
- 固定 eval 子集流程可以避免训练 DataLoader 的随机视角和随机 batch 噪声，适合作为不同 KL 权重与 EMA/non-EMA checkpoint 的选择依据。
- 当前已保存的 EMA checkpoint 显著差于非 EMA，视觉样图也糊坏；可能与训练器在 `finetune_from` 前创建 EMA 参数而未同步到微调初始化权重有关。
- 当前“所有 KL 值”横评只能说明本机可用候选中非 EMA `kl1e-7` 最优；不能外推为 `1e-7` 一定优于未参与评估的 `1e-6` 或其它 KL 权重。
- 新 smoke latent 数据集可由 `trellis.datasets.structured_latent.SLat` 读取 1024 条；`ImageConditionedSLat` 在条件图过滤后可读取 1023 条，`cond` shape 为 `(3, 518, 518)`。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。
- Accepted: 当前后续 SLat flow 微调/评估优先使用 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt` 和 `decoder_step0001000.pt`，不要使用本次 EMA checkpoint。

## Next Actions
1. 决定 flow smoke 是否接受 1023 条条件图样本，或补齐/替换缺失的 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
2. 若时间允许，将 view0 扩展为 `0,4,8,12` 多视角平均，确认结论不依赖单视角。
3. 检查并修复训练器 EMA 初始化逻辑，避免未来 finetune EMA checkpoint 从错误初始状态累积。
4. 准备 SLat flow 人脸域微调配置，明确使用非 EMA step1000 encoder/decoder checkpoint。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。
- 当前 eval 输出目录和 eval 数据集是实验产物，不应直接提交到 git。

## Open Questions
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？


## HST-20260718-231113-01 - current.md snapshot

Description:
- 记录 SLat flow spconv FPE 诊断前的当前状态

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

## Relevant State
- EXE-20260717-105
- EXE-20260718-001
- CFG-20260717-116
- CFG-20260718-001
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
- ART-20260718-001
- ART-20260718-002
- ART-20260718-003
- ART-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- 已新增 `eval/` 评估工具：固定 FaceScape eval 子集准备、SLat enc/dec checkpoint 重建评估、多 run summary 对比。
- 已生成固定评估集 `datasets/Facescape_eval/slat_gs_eval50`，从 FaceScape test 固定抽取 50 个样本。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 非 EMA step1000 在 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.0253610、mean L1 0.00512148、mean PSNR 33.1676、mean SSIM loss 0.0522815、mean LPIPS 0.0489114、mean KL 9.89067。
- EMA step1000 在同一 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.111183、mean L1 0.0403950、mean PSNR 20.7630、mean LPIPS 0.212343、mean KL 0.0541608。
- 当前本机只发现 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 这一组带 KL 命名的 SLat enc/dec 微调输出；没有发现其它 KL 值的最终 checkpoint 可纳入横评。
- 已生成全可用 KL 终权重横评 CSV `eval_outputs/slat_all_kl_final_eval50_view0_compare.csv`，当前覆盖 `kl1e-7` 非 EMA 与 `kl1e-7_ema`。
- 已生成并原地扩展独立 smoke latent 数据集 `datasets/Facescape_slat_kl1e-7_nonema_smoke`：从 train 固定抽取 1024 个样本，使用 `kl1e-7` step1000 非 EMA encoder 编码，metadata 中 `latent_dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000` 列 1024 条均为 True。
- 已从 `/root/autodl-fs/Facescape_cond` 分卷 tar 包解压 `renders_cond` 到 `datasets/Facescape/renders_cond`，并按现有 train/test metadata 建立 `train/renders_cond` 与 `test/renders_cond` 软链接。
- `datasets/Facescape/train` 的 6456 条中 6453 条有条件图，缺 3 条；`datasets/Facescape/test` 的 720 条全部有条件图。
- `datasets/Facescape_slat_kl1e-7_nonema_smoke` 的 1024 条中 1023 条已链接条件图，缺 1 条 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。
- 已新增 SLat flow smoke 微调配置 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json`：使用 `ImageConditionedSLat`、latent model `dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000`、decoder `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 的 `step0001000`，训练参数为 1000 step、batch16、batch_split4、lr=1e-5、i_sample=20000、i_save=500。
- 该 smoke 配置 JSON 语法校验通过，且按 `train.py` 的真实 dataset 加载方式可读取 1023 条样本；首样本包含 `cond` `(3, 518, 518)`、`coords` `(10886, 3)`、`feats` `(10886, 8)`。
- 该 smoke 配置当前指向的 SLat flow 原始权重为 `microsoft/trellis-normal-v0-1/ckpts/slat_flow_normal_dit_L_64l8p2_fp16.pt`，该文件由同目录 `.safetensors` 转换得到且本机存在。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。
- 固定 eval 子集流程可以避免训练 DataLoader 的随机视角和随机 batch 噪声，适合作为不同 KL 权重与 EMA/non-EMA checkpoint 的选择依据。
- 当前已保存的 EMA checkpoint 显著差于非 EMA，视觉样图也糊坏；可能与训练器在 `finetune_from` 前创建 EMA 参数而未同步到微调初始化权重有关。
- 当前“所有 KL 值”横评只能说明本机可用候选中非 EMA `kl1e-7` 最优；不能外推为 `1e-7` 一定优于未参与评估的 `1e-6` 或其它 KL 权重。
- 新 smoke latent 数据集可由 `trellis.datasets.structured_latent.SLat` 读取 1024 条；`ImageConditionedSLat` 在条件图过滤后可读取 1023 条，`cond` shape 为 `(3, 518, 518)`。
- SLat flow smoke 配置本身已准备好用于小规模测试；当前 `finetune_ckpt.denoiser` 已指向本机可用的 `.pt` 权重。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。
- Accepted: 当前后续 SLat flow 微调/评估优先使用 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt` 和 `decoder_step0001000.pt`，不要使用本次 EMA checkpoint。

## Next Actions
1. 决定 flow smoke 是否接受 1023 条条件图样本，或补齐/替换缺失的 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
2. 若时间允许，将 view0 扩展为 `0,4,8,12` 多视角平均，确认结论不依赖单视角。
3. 检查并修复训练器 EMA 初始化逻辑，避免未来 finetune EMA checkpoint 从错误初始状态累积。
4. 用 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json` 启动 SLat flow 短程微调测试。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。
- 当前 eval 输出目录和 eval 数据集是实验产物，不应直接提交到 git。
- SLat flow smoke 配置依赖本地复制的 `microsoft/trellis-normal-v0-1` 权重目录；迁移环境时需一并复制该目录或改配置路径。

## Open Questions
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？


## HST-20260719-205258-01 - current.md snapshot

Description:
- 迁移 .project-state 到新版 maintain-project-state schema 前的当前状态

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

## Relevant State
- EXE-20260717-105
- EXE-20260718-001
- CFG-20260717-116
- CFG-20260718-001
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
- ART-20260718-001
- ART-20260718-002
- ART-20260718-003
- ART-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005
- EVT-20260718-120400-01
- EVT-20260718-121200-01

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- 已新增 `eval/` 评估工具：固定 FaceScape eval 子集准备、SLat enc/dec checkpoint 重建评估、多 run summary 对比。
- 已生成固定评估集 `datasets/Facescape_eval/slat_gs_eval50`，从 FaceScape test 固定抽取 50 个样本。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 非 EMA step1000 在 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.0253610、mean L1 0.00512148、mean PSNR 33.1676、mean SSIM loss 0.0522815、mean LPIPS 0.0489114、mean KL 9.89067。
- EMA step1000 在同一 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.111183、mean L1 0.0403950、mean PSNR 20.7630、mean LPIPS 0.212343、mean KL 0.0541608。
- 当前本机只发现 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 这一组带 KL 命名的 SLat enc/dec 微调输出；没有发现其它 KL 值的最终 checkpoint 可纳入横评。
- 已生成全可用 KL 终权重横评 CSV `eval_outputs/slat_all_kl_final_eval50_view0_compare.csv`，当前覆盖 `kl1e-7` 非 EMA 与 `kl1e-7_ema`。
- 已生成并原地扩展独立 smoke latent 数据集 `datasets/Facescape_slat_kl1e-7_nonema_smoke`：从 train 固定抽取 1024 个样本，使用 `kl1e-7` step1000 非 EMA encoder 编码，metadata 中 `latent_dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000` 列 1024 条均为 True。
- 已从 `/root/autodl-fs/Facescape_cond` 分卷 tar 包解压 `renders_cond` 到 `datasets/Facescape/renders_cond`，并按现有 train/test metadata 建立 `train/renders_cond` 与 `test/renders_cond` 软链接。
- `datasets/Facescape/train` 的 6456 条中 6453 条有条件图，缺 3 条；`datasets/Facescape/test` 的 720 条全部有条件图。
- `datasets/Facescape_slat_kl1e-7_nonema_smoke` 的 1024 条中 1023 条已链接条件图，缺 1 条 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。
- 已新增 SLat flow smoke 微调配置 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json`：使用 `ImageConditionedSLat`、latent model `dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000`、decoder `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 的 `step0001000`，训练参数为 1000 step、batch16、batch_split4、lr=1e-5、i_sample=20000、i_save=500。
- 该 smoke 配置 JSON 语法校验通过，且按 `train.py` 的真实 dataset 加载方式可读取 1023 条样本；首样本包含 `cond` `(3, 518, 518)`、`coords` `(10886, 3)`、`feats` `(10886, 8)`。
- 该 smoke 配置当前指向的 SLat flow 原始权重为 `microsoft/trellis-normal-v0-1/ckpts/slat_flow_normal_dit_L_64l8p2_fp16.pt`，该文件由同目录 `.safetensors` 转换得到且本机存在。
- 用户按该配置启动 SLat flow 微调时在 `Sampling 1 images...` 后遇到 `Floating point exception (core dumped)`；诊断确认 FPE 发生在 `ElasticSLatFlowModel.input_blocks.0.conv1` 的 spconv `SparseConv3d` kernel。
- 当前 RTX 5090 / PyTorch 2.7.1+cu128 / spconv 2.3.6 环境下，spconv 默认 `SPCONV_ALGO=auto` 会触发该 FPE；设置 `SPCONV_ALGO=native` 后 denoiser forward、完整 `run_step` 和初始 `snapshot(init)` 均通过。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。
- 固定 eval 子集流程可以避免训练 DataLoader 的随机视角和随机 batch 噪声，适合作为不同 KL 权重与 EMA/non-EMA checkpoint 的选择依据。
- 当前已保存的 EMA checkpoint 显著差于非 EMA，视觉样图也糊坏；可能与训练器在 `finetune_from` 前创建 EMA 参数而未同步到微调初始化权重有关。
- 当前“所有 KL 值”横评只能说明本机可用候选中非 EMA `kl1e-7` 最优；不能外推为 `1e-7` 一定优于未参与评估的 `1e-6` 或其它 KL 权重。
- 新 smoke latent 数据集可由 `trellis.datasets.structured_latent.SLat` 读取 1024 条；`ImageConditionedSLat` 在条件图过滤后可读取 1023 条，`cond` shape 为 `(3, 518, 518)`。
- SLat flow smoke 配置本身已准备好用于小规模测试；当前 `finetune_ckpt.denoiser` 已指向本机可用的 `.pt` 权重。
- SLat flow 的 FPE 不是 sample 跳过问题，也不是 DINO 条件编码问题；是 spconv `auto` 算法在当前环境下选到的不兼容 sparse conv kernel。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。
- Accepted: 当前后续 SLat flow 微调/评估优先使用 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt` 和 `decoder_step0001000.pt`，不要使用本次 EMA checkpoint。

## Next Actions
1. 决定 flow smoke 是否接受 1023 条条件图样本，或补齐/替换缺失的 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
2. 若时间允许，将 view0 扩展为 `0,4,8,12` 多视角平均，确认结论不依赖单视角。
3. 检查并修复训练器 EMA 初始化逻辑，避免未来 finetune EMA checkpoint 从错误初始状态累积。
4. 用带 `SPCONV_ALGO=native` 的命令启动 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json` 的 SLat flow 短程微调测试。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。
- 当前 eval 输出目录和 eval 数据集是实验产物，不应直接提交到 git。
- SLat flow smoke 配置依赖本地复制的 `microsoft/trellis-normal-v0-1` 权重目录；迁移环境时需一并复制该目录或改配置路径。
- 在当前机器上运行 SLat flow 时必须显式设置 `SPCONV_ALGO=native`，否则 spconv `auto` 可能在第一个 sparse conv 处触发 FPE。

## Open Questions
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？


## HST-20260719-211834-01 - current.md snapshot

Description:
- 按新版 current.md schema 精简当前状态前的快照

# Current State

## Active Goal
维护 TRELLIS-new 的 `.project-state`，并支持 FaceScape SLat encoder + GS decoder fine-tune 结果评估与后续 SLat flow 微调准备。

## Current Working Thread
用户已完成 `lambda_kl=1e-7` 的 SLat encoder + Gaussian decoder 1000-step 微调。当前重点是判断该 checkpoint 是否适合作为人脸域后续 SLat flow 微调的初始化，并继续用固定验证集补充证据。

## Relevant State
- EXE-20260717-105
- CFG-20260717-116
- CFG-20260718-001
- AST-20260717-001
- AST-20260717-010
- AST-20260717-011
- AST-20260718-001
- AST-20260718-002
- AST-20260718-003
- AST-20260718-004
- RUN-20260718-001
- RUN-20260718-002
- RUN-20260718-003
- RUN-20260718-004
- RUN-20260718-005

## Facts
- 仓库根目录为 `/root/autodl-tmp/TRELLIS-new`。
- 当前分支为 `codex/train-slat-enc-dec`。
- `.project-state/` 已按新版 `maintain-project-state` schema 迁移；当前维护 `current.md`、`history.md`、`runs.md`、`assets.md`、`events.md`、`executables.md` 和 `experiment-configs.md`，资源 ID 使用 `AST-*` 前缀。
- 当前微调配置 `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置 `lambda_kl=1e-7`。
- 已新增 `eval/` 评估工具：固定 FaceScape eval 子集准备、SLat enc/dec checkpoint 重建评估、多 run summary 对比。
- 已生成固定评估集 `datasets/Facescape_eval/slat_gs_eval50`，从 FaceScape test 固定抽取 50 个样本。
- `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 是已完成的 batch16/lr1e-5/`lambda_kl=1e-7` 1000-step 微调结果。
- 本次输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 本次日志文件为 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/log_slat_enc_dec_gs_fine_tune_kl1e-7.txt`，共有 1000 行。
- 本次最终 step loss 为 0.0208777；最后 100 step 平均 loss 为 0.0204838，较前 100 step 下降约 8.51%。
- 本次最后 100 step 平均 LPIPS 为 0.0385662，较前 100 step 下降约 13.75%；最后 100 step 平均 grad_norm 为 0.0376774，较前 100 step 下降约 39.67%。
- 本次总 elapsed 为 2050.82 秒，端到端约 34.18 分钟；最后 100 step 平均 step_time 为 1.97847 秒。
- 非 EMA step1000 在 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.0253610、mean L1 0.00512148、mean PSNR 33.1676、mean SSIM loss 0.0522815、mean LPIPS 0.0489114、mean KL 9.89067。
- EMA step1000 在同一 eval50/view0 上 `num_records=50`、`failed_count=0`、mean loss 0.111183、mean L1 0.0403950、mean PSNR 20.7630、mean LPIPS 0.212343、mean KL 0.0541608。
- 当前本机只发现 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 这一组带 KL 命名的 SLat enc/dec 微调输出；没有发现其它 KL 值的最终 checkpoint 可纳入横评。
- 已生成全可用 KL 终权重横评 CSV `eval_outputs/slat_all_kl_final_eval50_view0_compare.csv`，当前覆盖 `kl1e-7` 非 EMA 与 `kl1e-7_ema`。
- 已生成并原地扩展独立 smoke latent 数据集 `datasets/Facescape_slat_kl1e-7_nonema_smoke`：从 train 固定抽取 1024 个样本，使用 `kl1e-7` step1000 非 EMA encoder 编码，metadata 中 `latent_dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000` 列 1024 条均为 True。
- 已从 `/root/autodl-fs/Facescape_cond` 分卷 tar 包解压 `renders_cond` 到 `datasets/Facescape/renders_cond`，并按现有 train/test metadata 建立 `train/renders_cond` 与 `test/renders_cond` 软链接。
- `datasets/Facescape/train` 的 6456 条中 6453 条有条件图，缺 3 条；`datasets/Facescape/test` 的 720 条全部有条件图。
- `datasets/Facescape_slat_kl1e-7_nonema_smoke` 的 1024 条中 1023 条已链接条件图，缺 1 条 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
- 已创建 `datasets/Facescape_slat_gs_50gb`，大小 `51G`，用于低配置机器测速。
- 已新增 SLat flow smoke 微调配置 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json`：使用 `ImageConditionedSLat`、latent model `dinov2_vitl14_reg_slat_enc_dec_gs_fine_tune_kl1e-7_step0001000`、decoder `outputs/slat_enc_dec_gs_fine_tune_kl1e-7` 的 `step0001000`，训练参数为 1000 step、batch16、batch_split4、lr=1e-5、i_sample=20000、i_save=500。
- 该 smoke 配置 JSON 语法校验通过，且按 `train.py` 的真实 dataset 加载方式可读取 1023 条样本；首样本包含 `cond` `(3, 518, 518)`、`coords` `(10886, 3)`、`feats` `(10886, 8)`。
- 该 smoke 配置当前指向的 SLat flow 原始权重为 `microsoft/trellis-normal-v0-1/ckpts/slat_flow_normal_dit_L_64l8p2_fp16.pt`，该文件由同目录 `.safetensors` 转换得到且本机存在。
- 用户按该配置启动 SLat flow 微调时在 `Sampling 1 images...` 后遇到 `Floating point exception (core dumped)`；诊断确认 FPE 发生在 `ElasticSLatFlowModel.input_blocks.0.conv1` 的 spconv `SparseConv3d` kernel。
- 当前 RTX 5090 / PyTorch 2.7.1+cu128 / spconv 2.3.6 环境下，spconv 默认 `SPCONV_ALGO=auto` 会触发该 FPE；设置 `SPCONV_ALGO=native` 后 denoiser forward、完整 `run_step` 和初始 `snapshot(init)` 均通过。

## Interpretations
- `lambda_kl=1e-7` 下 KL 原始值没有明显暴涨，说明 1000-step 短程训练中 latent 正则没有失控；但加权 KL 贡献约为 `1e-6` 量级，对总 loss 已非常弱。
- 本次最后 100 step 平均 loss 略低于 RUN-20260718-001 的 0.0208222，但本次同时改变了 batch size 和 KL 权重，不能单独归因于 `lambda_kl=1e-7`。
- 这次 checkpoint 可以作为后续 SLat flow 人脸域微调候选，但需要固定验证集重建指标和 EMA/non-EMA 对比来降低风险。
- 固定 eval 子集流程可以避免训练 DataLoader 的随机视角和随机 batch 噪声，适合作为不同 KL 权重与 EMA/non-EMA checkpoint 的选择依据。
- 当前已保存的 EMA checkpoint 显著差于非 EMA，视觉样图也糊坏；可能与训练器在 `finetune_from` 前创建 EMA 参数而未同步到微调初始化权重有关。
- 当前“所有 KL 值”横评只能说明本机可用候选中非 EMA `kl1e-7` 最优；不能外推为 `1e-7` 一定优于未参与评估的 `1e-6` 或其它 KL 权重。
- 新 smoke latent 数据集可由 `trellis.datasets.structured_latent.SLat` 读取 1024 条；`ImageConditionedSLat` 在条件图过滤后可读取 1023 条，`cond` shape 为 `(3, 518, 518)`。
- SLat flow smoke 配置本身已准备好用于小规模测试；当前 `finetune_ckpt.denoiser` 已指向本机可用的 `.pt` 权重。
- SLat flow 的 FPE 不是 sample 跳过问题，也不是 DINO 条件编码问题；是 spconv `auto` 算法在当前环境下选到的不兼容 sparse conv kernel。

## Active Hypotheses
- H1: 降低 `lambda_kl` 到 `1e-7` 对人脸域重建有轻微正向作用。
  Evidence: 本次最后 100 step 平均 loss 为 0.0204838，低于此前 batch8/lr1e-5/`lambda_kl=1e-6` 的 0.0208222。
  Uncertainty: 有效 batch 从 8 增到 16，无法隔离 KL 权重影响；也缺少固定验证集结果。
- H2: 本次 SLat enc/dec checkpoint 适合进入 SLat flow 微调前的候选池。
  Evidence: 训练完整结束，checkpoint 齐全，loss 与 LPIPS 有下降，final sample 未见明显崩坏。
  Uncertainty: 未验证生成链路、holdout 重建质量和 EMA/non-EMA 差异。

## Current Decision State
- Accepted: SLAT enc/dec 人脸域微调配置使用 `lambda_kl=1e-7` 做一轮激进实验。
- Accepted: 后续 SLAT diffusion/flow 也会做微调，因此可接受 latent 分布较原始通用 3D 模型有一定偏移。
- Pending: 是否采用本次 step1000 EMA checkpoint 还是 non-EMA checkpoint 作为后续 flow 微调/评估输入。
- Accepted: 当前后续 SLat flow 微调/评估优先使用 `outputs/slat_enc_dec_gs_fine_tune_kl1e-7/ckpts/encoder_step0001000.pt` 和 `decoder_step0001000.pt`，不要使用本次 EMA checkpoint。

## Next Actions
1. 决定 flow smoke 是否接受 1023 条条件图样本，或补齐/替换缺失的 `8ad92a2a586548b93d6fb1e809c67fff9537e03de244dd969f4ab5436afe8be6`。
2. 若时间允许，将 view0 扩展为 `0,4,8,12` 多视角平均，确认结论不依赖单视角。
3. 检查并修复训练器 EMA 初始化逻辑，避免未来 finetune EMA checkpoint 从错误初始状态累积。
4. 用带 `SPCONV_ALGO=native` 的命令启动 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json` 的 SLat flow 短程微调测试。

## Constraints
- 不回滚用户或环境中的既有修改。
- 大型数据目录不提交到 git。
- 训练日志分析不能替代独立验证集评估。
- 比较不同实验时需要注意 batch size、学习率、KL 权重是否同时变化。
- 当前 eval 输出目录和 eval 数据集是实验产物，不应直接提交到 git。
- SLat flow smoke 配置依赖本地复制的 `microsoft/trellis-normal-v0-1` 权重目录；迁移环境时需一并复制该目录或改配置路径。
- 在当前机器上运行 SLat flow 时必须显式设置 `SPCONV_ALGO=native`，否则 spconv `auto` 可能在第一个 sparse conv 处触发 FPE。

## Open Questions
- 后续 SLat flow 微调应使用完整 FaceScape train 还是先用 50GB 子集做流程 smoke test？


## HST-20260719-214214-01 - current.md snapshot

Description:
- 记录 SLat flow smoke 训练完成前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat encoder + Gaussian decoder 的 `lambda_kl=1e-7` step1000 non-EMA checkpoint 是当前后续 flow 微调优先候选，EMA checkpoint 不建议使用。
- SLat flow 测试配置已准备好并指向本地可用的 flow `.pt` 权重与 kl1e-7 non-EMA latent smoke 数据集。
- 当前机器运行 SLat flow 必须显式设置 `SPCONV_ALGO=native`，否则 spconv `auto` 会在第一个 sparse conv 处触发 FPE。

## Next Actions
1. 用带 `SPCONV_ALGO=native` 的命令启动 `configs/generation/slat_flow_finetune_kl1e-7_step1000.json` 的短程 SLat flow 微调测试。
2. 训练完成后检查 log、loss、checkpoint 和初始/最终 sample，判断是否进入更大规模 flow 微调。
3. 如需严格 1024 条 flow smoke 样本，补齐或替换缺失条件图样本。

## Relevant Records
- CFG-20260718-001
- RUN-20260718-014
- AST-20260718-010
- AST-20260718-004
- EXE-20260717-105


## HST-20260719-220157-01 - current.md snapshot

Description:
- 记录新增 kl1e-7 可靠性评估代码前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat flow 的 kl1e-7 step1000 smoke fine-tune 已完整跑到 step 1000，并产出 500/1000 checkpoint、log/loss 和 init/final samples。
- 训练 loss 从前 100 step 均值约 0.0907 降到后 100 step 均值约 0.0645，未发现缺步或非有限 loss。
- final sample 已能生成与 GT 姿态/轮廓接近的人脸形状，但细节仍偏粗，下一步需要定量和更多样本评估。

## Next Actions
1. 对 step1000 non-EMA 与 EMA flow checkpoint 做固定样本生成评估，并和 init/pretrained 结果对比。
2. 检查最终 checkpoint 的更多条件样本，判断 1000-step smoke 是否已足够进入更大规模 flow 微调。
3. 如要扩展训练，继续保留 `SPCONV_ALGO=native` 并决定是否补齐缺失条件图样本。

## Relevant Records
- RUN-20260719-001
- AST-20260719-001
- CFG-20260718-001
- AST-20260718-010
- EXE-20260717-105


## HST-20260719-221238-01 - current.md snapshot

Description:
- 记录 outputs/train 与 outputs/eval 目录规范建立前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat flow 的 kl1e-7 step1000 smoke fine-tune 已完整跑到 step 1000，并产出 500/1000 checkpoint、log/loss 和 init/final samples。
- 已新增 kl1e-7 可靠性评估代码，覆盖 latent 分布统计、固定 flow 生成和生成结果指标对比。
- 新评估代码的轻量测试、py_compile、真实 latent stats 小样本 smoke、1 样本 flow generation smoke 和 generation metrics smoke 均已通过。

## Next Actions
1. 对完整 1024 latent smoke 数据集运行 latent 分布统计。
2. 用固定样本分别生成 pretrained、step1000 non-EMA 和 step1000 EMA flow 结果。
3. 汇总固定生成指标，判断 kl1e-7 non-EMA flow 是否相对 pretrained/EMA 更可靠。

## Relevant Records
- EXE-20260719-001
- EXE-20260719-003
- EXE-20260719-004
- RUN-20260719-001
- AST-20260719-001


## HST-20260719-222021-01 - current.md snapshot

Description:
- 记录 kl1e-7 encdec 与 flow 评估完成前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat flow 的 kl1e-7 step1000 smoke fine-tune 已完整跑到 step 1000，并产出 500/1000 checkpoint、log/loss 和 init/final samples。
- 已新增 kl1e-7 可靠性评估代码，覆盖 latent 分布统计、固定 flow 生成和生成结果指标对比。
- 当前训练产物已统一迁入 `outputs/train`，已有和后续评估产物统一放在 `outputs/eval`。

## Next Actions
1. 对完整 1024 latent smoke 数据集运行 latent 分布统计。
2. 用固定样本分别生成 pretrained、step1000 non-EMA 和 step1000 EMA flow 结果。
3. 汇总固定生成指标，判断 kl1e-7 non-EMA flow 是否相对 pretrained/EMA 更可靠。

## Relevant Records
- EXE-20260719-001
- EXE-20260719-003
- EXE-20260719-004
- RUN-20260719-001
- AST-20260719-001


## HST-20260719-223136-01 - current.md snapshot

Description:
- 记录 flow 生成默认保存 PLY 前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat flow 的 kl1e-7 step1000 smoke fine-tune 已完整跑到 step 1000，并产出 500/1000 checkpoint、log/loss 和 init/final samples。
- 已完成 kl1e-7 step1000 非 EMA SLat enc/dec 的 eval50/view0 重建评估。
- 已完成 kl1e-7 step1000 非 EMA SLat flow 的固定 16 样本生成评估和指标汇总。
- 当前训练产物已统一迁入 `outputs/train`，已有和后续评估产物统一放在 `outputs/eval`。

## Next Actions
1. 用同一固定 16 样本补跑 pretrained 或 EMA flow 结果，建立相对基线。
2. 若要提高结论可靠性，将 flow 固定生成样本数从 16 扩到 50 或 128。
3. 决定是否继续加长 kl1e-7 flow 训练或调整 flow 微调配置。

## Relevant Records
- EXE-20260719-003
- EXE-20260719-004
- RUN-20260719-002
- RUN-20260719-003
- RUN-20260719-004


## HST-20260719-223747-01 - current.md snapshot

Description:
- 记录准备 kl1e-8 SLat encdec 训练前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat flow 的 kl1e-7 step1000 smoke fine-tune 已完整跑到 step 1000，并产出 500/1000 checkpoint、log/loss 和 init/final samples。
- 已完成 kl1e-7 step1000 非 EMA SLat enc/dec eval50/view0 重建评估和 SLat flow 固定 16 样本生成评估。
- SLat flow 固定生成评估已改为默认保存 generated/GT PLY，训练产物在 `outputs/train`，评估产物在 `outputs/eval`。

## Next Actions
1. 用同一固定 16 样本补跑 pretrained 或 EMA flow 结果，建立相对基线。
2. 若要提高结论可靠性，将 flow 固定生成样本数从 16 扩到 50 或 128。
3. 决定是否继续加长 kl1e-7 flow 训练或调整 flow 微调配置。

## Relevant Records
- EXE-20260719-003
- EXE-20260719-004
- RUN-20260719-003
- RUN-20260719-006
- AST-20260719-007


## HST-20260720-090657-01 - current.md snapshot

Description:
- 记录准备 kl1e-6 SLat encdec 训练前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat enc/dec 微调配置已从 `lambda_kl=1e-7` 改为 `lambda_kl=1e-8`，其余 1000-step/batch16/lr1e-5 设置保持不变。
- 训练产物统一保存到 `outputs/train`，本轮 kl1e-8 输出目录应使用新目录避免覆盖 kl1e-7。
- SLat flow 固定生成评估已改为默认保存 generated/GT PLY。

## Next Actions
1. 启动 SLat enc/dec kl1e-8 训练并检查 init sampling 是否正常。
2. 训练完成后用固定 eval50/view0 跑 non-EMA 与 EMA 重建评估。
3. 与 kl1e-7 的固定评估结果横向比较后再决定是否进入 flow 阶段。

## Relevant Records
- EXE-20260717-105
- CFG-20260717-116
- AST-20260718-004
- RUN-20260719-002


## HST-20260720-094620-01 - current.md snapshot

Description:
- 记录 kl1e-6 训练结果分析完成前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat enc/dec 微调配置已改为 `lambda_kl=1e-6`，其余 1000-step/batch16/lr1e-5 设置保持不变。
- 已完成并记录 kl1e-8 训练日志实验，训练产物统一保存到 `outputs/train`。
- SLat flow 固定生成评估已改为默认保存 generated/GT PLY。

## Next Actions
1. 启动 SLat enc/dec kl1e-6 训练并检查 init sampling 是否正常。
2. 训练完成后用固定 eval50/view0 跑 non-EMA 与 EMA 重建评估。
3. 与 kl1e-7/kl1e-8 的固定评估结果横向比较后再决定哪个权重进入 flow 阶段。

## Relevant Records
- EXE-20260717-105
- CFG-20260717-116
- EXP-20260720-001


## HST-20260720-142955-01 - current.md snapshot

Description:
- 记录 KL 梯度贡献诊断完成前的当前状态

# Current State

## Goal
支持 FaceScape 人脸域 SLat encoder/decoder 与后续 SLat flow 微调，保持项目状态记录符合最新版 `maintain-project-state` schema。

## Key State
- SLat enc/dec 的 kl1e-6、kl1e-7、kl1e-8 三组 1000-step 训练均已完成，训练产物统一保存到 `outputs/train`。
- 训练日志横向看三组重建项几乎重合，单靠训练日志不能证明哪一个 KL 权重更优。
- SLat flow 固定生成评估已改为默认保存 generated/GT PLY。

## Next Actions
1. 对 kl1e-6 的 step1000 non-EMA 与 EMA 跑固定 eval50/view0 重建评估。
2. 对 kl1e-7/kl1e-8/kl1e-6 的固定评估结果做横向汇总。
3. 根据固定评估而非训练日志决定哪个权重进入 flow 阶段。

## Relevant Records
- EXE-20260717-105
- CFG-20260717-116
- EXP-20260720-001
- RUN-20260720-001
- AST-20260720-001
