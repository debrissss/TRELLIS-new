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
