# Timeline

## EVT-20260717-000000-01 - full project-state scan

Description:
- 完成 TRELLIS-new 首次 `.project-state` 全量静态扫描并建立基础台账。

Type: finding
Source: file-inspected
Related records:
- none
Facts:
- 已登记训练、推理、FaceScape 预处理、overfit 准备、训练包装器和重建审计入口。
- 已登记 FaceScape 数据、TRELLIS-image-large 本地模型、参考数据集和截断 mesh 输入资源。
- 未发现可证明历史运行命令的输出记录，因此未创建 RUN。
Evidence:
- 静态读取 `README.md`、`fine_tuning/5-stages.md`、`train.py`、`cli.py`、`fine_tuning/train_*.sh`、配置目录、数据目录和模型目录。
Interpretation:
- 项目当前核心上下文是 FaceScape 微调和过拟合实验准备，而不是原始 TRELLIS demo 本身。
Uncertainty:
- 历史训练/预处理可能由用户手动执行过，但仓库内未发现足以入账的命令日志。
Next:
- 后续执行或用户报告运行时，按 RUN 合同补登命令、输入/输出 ART 和结果事实。

## EVT-20260717-000000-02 - truncated mesh inputs deleted

Description:
- 删除仓库根目录两个截断 mesh PLY 输入文件。

Type: artifact
Source: agent-run
Related records:
- none
Facts:
- `2020bc_mesh_truncated.ply` 和 `f674d4_mesh_truncated.ply` 已按用户要求删除。
Evidence:
- 删除前 `ls -lh` 确认两个文件存在且各约 35M；删除后 `test ! -e` 验证两个路径不存在。
Interpretation:
- 截断 mesh GT 重建流程不再有这两个根目录输入资源可用。
Uncertainty:
- 是否存在其它副本未知。
Next:
- 如需继续截断 mesh 处理，先提供新的输入 mesh 并登记为新的 ART。

## EVT-20260717-000000-03 - fine_tuning entrypoints deleted

Description:
- 删除 FaceScape 预处理、训练包装器和 overfit 准备入口文件。

Type: executable-interface
Source: agent-run
Related records:
- none
Facts:
- 已删除 `fine_tuning/preprocess_stage1.py` 和 `fine_tuning/preprocess_stage2.py`。
- 已删除 `fine_tuning/train_ss_flow_facescape_finetune.sh`、`fine_tuning/train_ss_flow_facescape_35000_to_40000.sh`、`fine_tuning/train_ss_flow_facescape_overfit_1.sh`、`fine_tuning/train_ss_flow_facescape_overfit_4.sh`、`fine_tuning/train_ss_flow_facescape_overfit_8.sh`。
- 已删除 `fine_tuning/train_slat_flow_facescape_overfit_1.sh`、`fine_tuning/prepare_ss_overfit_experiments.py`、`fine_tuning/prepare_slat_overfit_experiments.py`。
Evidence:
- 删除前 `ls -l` 确认 10 个路径存在；删除后 `test ! -e` 验证 10 个路径均不存在。
Interpretation:
- 后续 FaceScape 训练不能再依赖这些 shell 包装器或 overfit 准备脚本；如配置仍要训练，应直接使用 `train.py` 或重新建立入口。
Uncertainty:
- 是否存在外部副本未知。
Next:
- 若需要恢复相关流程，从 git 或外部备份恢复文件，或基于 `train.py` 重新创建新的入口并登记为新 EXE。

## EVT-20260717-000000-04 - ledger rewritten for updated skill

Description:
- 按更新后的 `maintain-project-state` 合同重写 `.project-state`，拆分聚合记录。

Type: finding
Source: agent-run
Related records:
- EXE-20260717-105
- EXE-20260717-130
- CFG-20260717-103
- CFG-20260717-106
- ART-20260717-001
- ART-20260717-002
Facts:
- `executables.md` 已覆盖为 45 条单入口 EXE 记录。
- `experiment-configs.md` 已覆盖为 15 条单配置文件 CFG 记录。
- `artifacts.md` 已覆盖为 7 条单资源路径 ART 记录。
- 重写 `current.md` 前已按合同追加 `history.md` 快照。
Evidence:
- 静态扫描 `if __name__ == "__main__"`、argparse 入口、shell 脚本、配置文件和主要数据/模型资源后重写 ledger。
Interpretation:
- 旧 ledger 中按目录/用途聚合的入口和多路径资源记录不再符合新 skill，已直接覆盖。
Uncertainty:
- 未执行入口命令，只做静态接口盘点；内部包 `__main__` 调试块未作为用户级直接入口登记。
Next:
- 后续新增或恢复脚本时保持一脚本一 EXE；执行命令时再登记 RUN。
