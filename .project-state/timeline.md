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

## EVT-20260717-000000-05 - slat enc dec gs fine tune config added

Description:
- 新增 SLat encoder + Gaussian decoder fine-tune 配置文件。

Type: artifact
Source: agent-run
Related records:
- CFG-20260717-116
- EXE-20260717-105
Facts:
- 创建 `configs/vae/slat_enc_dec_gs_fine_tune.json`。
- 新文件当前完整复制自 `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`。
Evidence:
- 使用 `cmp -s` 验证源配置与新配置内容完全一致。
Interpretation:
- 后续可在新文件中针对 FaceScape 或 fine-tune 目标修改参数，而不直接改动原始 VAE 配置。
Uncertainty:
- 具体 fine-tune 参数尚未调整。
Next:
- 根据目标数据和训练策略修改 `CFG-20260717-116`，再通过 `EXE-20260717-105` 启动训练并登记 RUN。

## EVT-20260717-000000-06 - SLat safetensors converted to pt

Description:
- 将 SLat encoder 和 SLat Gaussian decoder safetensors 权重转换为 PyTorch `.pt`。

Type: run
Source: agent-run
Related records:
- RUN-20260717-001
- RUN-20260717-002
- ART-20260717-010
- ART-20260717-011
Facts:
- 已生成 `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt`。
- 已生成 `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`。
Evidence:
- `fine_tuning/convert_safetensors_to_pt.py` 对两个转换均输出 `Conversion completed and verified.`；额外 `torch.load(..., weights_only=True)` 分别读取到 100 和 101 个 state_dict 条目。
Interpretation:
- 新建的 SLat encoder + GS decoder fine-tune 配置现在可以引用 `.pt` 格式预训练权重。
Uncertainty:
- 尚未运行实际 fine-tune 训练。
Next:
- 更新 `CFG-20260717-116` 的 `trainer.args.finetune_ckpt` 后启动训练。

## EVT-20260717-000000-07 - SLat fine-tune config updated

Description:
- 更新 SLat encoder + Gaussian decoder fine-tune 配置的训练步数、日志/保存间隔和预训练权重。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- ART-20260717-010
- ART-20260717-011
Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.max_steps` 改为 1000。
- `trainer.args.i_log` 改为 100，`trainer.args.i_save` 改为 500。
- 新增 `trainer.args.finetune_ckpt.encoder` 指向 `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt`。
- 新增 `trainer.args.finetune_ckpt.decoder` 指向 `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`。
Evidence:
- `python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 验证 JSON 格式有效。
Interpretation:
- 该配置现在可作为短程 SLat encoder + GS decoder fine-tune 入口配置。
Uncertainty:
- 尚未执行训练验证数据和 checkpoint 加载链路。
Next:
- 使用 `train.py --config configs/vae/slat_enc_dec_gs_fine_tune.json` 启动试跑或训练，并登记 RUN。

## EVT-20260717-000000-08 - corrupt FaceScape DINO feature cache found

Description:
- SLat fine-tune 训练日志中的 BadZipFile 被定位为 FaceScape DINO 特征 `.npz` 缓存损坏。

Type: finding
Source: user-reported
Related records:
- RUN-20260717-003
- ART-20260717-012
- ART-20260717-013
Facts:
- 用户报告训练在 step 500 checkpoint 保存完成后，读取 `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 的 `patchtokens.npy` 时失败。
- 本地验证 `ART-20260717-012` 和 `ART-20260717-013` 均为几十 KB，而同目录正常 `.npz` 至少约 15.6 MB。
- `zipfile.testzip()` 对两个文件均返回 `patchtokens.npy`。
Evidence:
- `trellis/datasets/sparse_feat2render.py` 在 `_get_feat` 中通过 `np.load(... )['patchtokens']` 读取特征。
- `find datasets/Facescape/train/features/dinov2_vitl14_reg -name '*.npz' -printf '%s %f\n' | sort -n | head` 显示两个异常小文件。
Interpretation:
- 这两个缓存大概率是在特征提取或写盘过程中被中断，留下了 zip central directory 可见但 `patchtokens.npy` 数据不完整的截断文件。
Uncertainty:
- 未验证是否还有内容大小正常但内部损坏的 `.npz`；当前只确认了两个明显异常小且 `testzip()` 失败的文件。
Next:
- 重生成 ART-20260717-012 和 ART-20260717-013，或先运行完整特征完整性扫描再恢复训练。

## EVT-20260717-000000-09 - corrupt FaceScape feature metadata disabled

Description:
- 临时扫描 train/test 的 DINOv2 特征缓存，并在 metadata 中禁用坏样本特征标记。

Type: artifact
Source: agent-run
Related records:
- ART-20260717-001
- ART-20260717-012
- ART-20260717-013
Facts:
- 临时脚本扫描 `datasets/Facescape/train/metadata.csv` 的 6456 行和 `datasets/Facescape/test/metadata.csv` 的 720 行。
- train 中发现 2 个坏样本，均为 `patchtokens.npy` 条目截断：ART-20260717-012 和 ART-20260717-013。
- test 中发现 0 个坏样本。
- 已将两个 train 样本的 `feature_dinov2_vitl14_reg` 写为 `False`。
- 临时脚本已删除。
Evidence:
- 正式运行输出 `train: scanned=6456 changed=2 action=updated`、`test: scanned=720 changed=0 action=updated`。
- 后续 pandas 复查显示两个 train 行的 `feature_dinov2_vitl14_reg` 均为 `False`。
Interpretation:
- 后续 DataLoader 按 metadata 过滤特征可用性时，应跳过这两个损坏样本，避免再次因同一坏缓存触发 BadZipFile。
Uncertainty:
- 未重生成两个坏 `.npz`；如果训练数据管线不按 `feature_dinov2_vitl14_reg` 过滤，仍可能需要删除或重建坏文件。
Next:
- 重新启动或恢复 fine-tune，观察是否还出现其它坏特征；必要时重生成 ART-20260717-012 和 ART-20260717-013。

## EVT-20260717-000000-10 - terminal progress controlled by i_print

Description:
- 分析训练日志无进度打印问题，确认终端进度由 `i_print` 控制而不是 `i_log`。

Type: finding
Source: file-inspected
Related records:
- CFG-20260717-116
- EXE-20260717-105
Facts:
- `trellis/trainers/base.py` 中终端进度打印条件为 `self.is_master and self.step % self.i_print == 0`。
- `i_log` 只控制写入 `log.txt`、`loss.txt` 和 TensorBoard scalar，不控制终端进度行。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置了 `i_log=100`，但没有设置 `i_print`，因此使用 `Trainer.__init__` 默认值 `i_print=1000`。
- 当前 fine-tune 配置 `max_steps=1000`，所以正常只会在第 1000 步打印一次 `Step/Elapsed/Speed/ETA` 进度行。
Evidence:
- `trellis/trainers/base.py` lines 63-64 define `i_print=1000` and `i_log=500` defaults.
- `trellis/trainers/base.py` lines 407-417 print terminal progress by `i_print`。
- `trellis/trainers/base.py` lines 449-470 write log files and TensorBoard by `i_log`。
Interpretation:
- 训练日志中只看到 step 500 checkpoint 保存而没有进度行，是因为 `i_print` 未在短程 1000-step fine-tune 配置中调小。
Uncertainty:
- 未检查当前输出目录中的 `log.txt`/`loss.txt` 是否已有每 100 step 的记录。
Next:
- 若希望控制台每 100 step 打印进度，在 `CFG-20260717-116` 的 `trainer.args` 中添加 `"i_print": 100`。

## EVT-20260717-000000-11 - SLat fine-tune terminal progress interval set

Description:
- 将 SLat encoder + GS decoder fine-tune 配置的终端进度打印间隔设置为 10 step。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- EXE-20260717-105
Facts:
- 在 `configs/vae/slat_enc_dec_gs_fine_tune.json` 的 `trainer.args` 中新增 `"i_print": 10`。
- 当前关键训练间隔为 `max_steps=1000`、`i_print=10`、`i_log=100`、`i_save=500`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `trainer.args.i_print` 为 `10`。
Interpretation:
- 之后使用该配置训练时，rank 0 终端应每 10 step 打印一次 `Step/Elapsed/Speed/ETA` 进度行。
Uncertainty:
- 尚未重新启动训练验证终端输出。
Next:
- 恢复或重新启动 fine-tune，观察终端进度和坏样本过滤是否正常。

## EVT-20260717-000000-12 - SLat GS fine-tune 1000-step completed

Description:
- FaceScape SLat encoder + GS decoder 1000-step fine-tune 完成并完成日志分析。

Type: run
Source: user-reported
Related records:
- RUN-20260717-004
- CFG-20260717-116
- ART-20260717-014
Facts:
- `outputs/slat_enc_dec_gs_fine_tune/log.txt` 和 `loss.txt` 均包含 1000 行。
- step 500 与 step 1000 checkpoint 均存在。
- 最后 100 step 平均 loss 为 0.0212889，相比前 100 step 平均 loss 0.0226317 下降约 5.93%。
- final sample 的重建图与 GT 图视觉上高度接近。
Evidence:
- `outputs/slat_enc_dec_gs_fine_tune/command.txt` 记录完整启动命令。
- 本地解析 `log.txt` 汇总 loss、rec、l1、ssim、lpips、kl、grad_norm 和耗时。
Interpretation:
- 试验已经完成并验证训练链路可跑通；1000 step 有小幅指标改善，但曲线波动较大，仍偏短程验证。
Uncertainty:
- 尚无独立 test set 指标；无法仅凭训练日志判断泛化质量。
Next:
- 使用固定验证/可视化流程比较 step1000 普通 checkpoint 与 EMA checkpoint，决定是否继续延长 fine-tune。

## EVT-20260717-000000-13 - SLat fine-tune batch and lr adjusted

Description:
- 调整 SLat encoder + GS decoder fine-tune 配置的有效 batch 和学习率。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260717-004
Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.batch_size_per_gpu` 从 4 改为 8。
- `trainer.args.batch_split` 从 2 改为 4，因此单卡有效 batch 为 8，micro-batch 仍为 2。
- `trainer.args.optimizer.args.lr` 从 `1e-4` 改为 `1e-5`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-05`。
Interpretation:
- 该配置更适合在 1000-step 试验后继续低学习率微调：每次参数更新的样本数翻倍，显存压力基本保持在 micro-batch 2 附近。
Uncertainty:
- 尚未用新配置恢复训练验证显存占用、速度和 loss 稳定性。
Next:
- 从 RUN-20260717-004 的 step1000 checkpoint 继续训练时，观察新 batch/lr 下的显存、step time 和 loss 波动。

## EVT-20260718-000000-01 - SLat GS fine-tune v2 analyzed

Description:
- 分析 batch 8、lr=1e-5 的 SLat encoder + GS decoder 1000-step fine-tune v2 结果。

Type: run
Source: user-reported
Related records:
- RUN-20260718-001
- CFG-20260717-116
- ART-20260718-001
Facts:
- 新结果实际在 `outputs/slat_enc_dec_gs_fine_tune_v2`，不是旧的 `outputs/slat_enc_dec_gs_fine_tune`。
- v2 `log.txt` 和 `loss.txt` 均包含 1000 行，step 500 与 step 1000 checkpoint 均存在。
- v2 最后 100 step 平均 loss 为 0.0208222，相比前 100 step 0.0227392 下降约 8.43%。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%。
- v2 final sample 的重建图与 GT 图视觉上高度接近。
Evidence:
- `outputs/slat_enc_dec_gs_fine_tune_v2/config.json` 确认 `batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5`。
- 本地解析 v2 `log.txt` 并与 `outputs/slat_enc_dec_gs_fine_tune/log.txt` 做同口径比较。
Interpretation:
- v2 是更稳的设置：loss 略低、梯度范数明显更低、尖峰较小，但每 step 时间显著增加。
Uncertainty:
- 仍缺少独立 test/holdout 指标和 EMA/non-EMA 对比。
Next:
- 对 v2 step1000 和 EMA step1000 做固定验证/可视化；若质量稳定，再继续延长训练。

## EVT-20260718-000000-02 - SLat fine-tune batch16 ablation configured

Description:
- 将 SLat encoder + GS decoder fine-tune 配置改为 batch16/lr1e-5 对照实验。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-001
Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.batch_size_per_gpu` 从 8 改为 16。
- `trainer.args.batch_split` 从 4 改为 8，因此单卡有效 batch 为 16，micro-batch 仍为 2。
- `trainer.args.optimizer.args.lr` 保持 `1e-5`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `batch_size_per_gpu=16`、`batch_split=8`、`micro_batch=2`、`lr=1e-05`。
Interpretation:
- 该配置用于固定学习率下测试更大有效 batch 的影响；显存压力预计接近 batch8 配置，但每 step 计算量约翻倍。
Uncertainty:
- 尚未运行 batch16/lr1e-5 训练验证速度、loss 和可视化结果。
Next:
- 用独立输出目录运行 batch16/lr1e-5 对照，例如 `outputs/slat_enc_dec_gs_fine_tune_b16_lr1e-5`。

## EVT-20260718-000000-03 - batch16 DataLoader shm failure diagnosed

Description:
- batch16/lr1e-5 对照训练在 init sampling 后触发 DataLoader worker shared memory bus error。

Type: issue
Source: user-reported
Related records:
- RUN-20260718-002
- ART-20260718-002
- CFG-20260717-116
Facts:
- 用户报告终端多次输出 `Unexpected bus error encountered in worker... insufficient shared memory (shm)`。
- 代码默认 DataLoader worker 数按 CPU 数计算，本机此前打印为 52。
- batch16 数据在 DataLoader 侧作为完整 batch collate 和预取，`batch_split=8` 只在完整 batch 取出后切分。
Evidence:
- `trellis/trainers/base.py` 中 `prepare_dataloader` 设置 `num_workers`、`pin_memory=True`、`persistent_workers`。
- `trellis/trainers/base.py` 中 `load_data` 先 `next(self.data_iterator)` 和 `recursive_to_device`，再按 `batch_split` 切分。
- `train.py` 默认 `auto_retry=3`，异常时会 retry。
Interpretation:
- 这是 DataLoader shared memory/worker prefetch 问题，不是普通采样阶段错误；增大有效 batch 到 16 后触发。
Uncertainty:
- 未记录 `/dev/shm` 实际容量；不再进行运行时压力检查。
Next:
- 为 batch16 对照禁用或大幅减少 DataLoader worker/prefetch，并用 `--auto_retry 0` 防止失败后自动重试。

## EVT-20260718-000000-04 - batch16 safe DataLoader config applied

Description:
- 为 batch16/lr1e-5 对照配置加入低共享内存 DataLoader 设置。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- 在 `configs/vae/slat_enc_dec_gs_fine_tune.json` 的 `trainer.args` 中新增 `dataloader_num_workers=0`。
- 新增 `dataloader_persistent_workers=false`。
- 新增 `prefetch_data=false`。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `dataloader_num_workers=0`、`dataloader_persistent_workers=False`、`prefetch_data=False`。
Interpretation:
- 新配置避免 DataLoader 使用 52 workers 和预取完整 batch，可降低 `/dev/shm` 压力；训练可能变慢。
Uncertainty:
- 尚未运行验证 batch16 在该安全 DataLoader 设置下是否能完整训练。
Next:
- 若运行 batch16 对照，启动命令应加 `--auto_retry 0`，并使用独立输出目录。

## EVT-20260718-000000-05 - batch16 safe DataLoader GPU starvation observed

Description:
- batch16 安全 DataLoader 配置运行后出现 GPU 高占用与 0 占用周期性交替。

Type: finding
Source: user-reported
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- 用户报告训练初期 GPU 利用率 85%-100%，显存约 15GB。
- 约 80 steps、启动约 5 分钟后，GPU 利用率降到 0，显存升到约 20GB，并持续约 2 分钟。
- 之后 GPU 利用率在高占用约 1 分钟与 0 占用约 1 分钟之间循环，显存两次循环后升到约 21.5GB。
- 当前配置为 `batch_size_per_gpu=16`、`batch_split=8`、`dataloader_num_workers=0`、`prefetch_data=false`。
Evidence:
- `trellis/trainers/base.py` 中 `load_data` 在 `prefetch_data=false` 时同步执行 `next(self.data_iterator)` 并搬到 GPU。
- `trellis/datasets/sparse_feat2render.py` 每样本读取随机视角图像并解压/读取 DINO `patchtokens.npy`。
- `trellis/utils/elastic_utils.py` 的 `LinearMemoryController` 会记录峰值显存，并让 sparse transformer 按动态 `mem_ratio` 选择 checkpointing 策略。
Interpretation:
- GPU 0% 周期大概率是同步 DataLoader/CPU I/O 阶段造成的 GPU starvation；batch16 完整 batch 加载和 collate 时间较长，`batch_split` 不影响这部分。
- 显存阶梯式上升可能来自不同样本 sparse token 数导致的更高峰值、elastic controller 的动态 checkpoint 策略，以及 PyTorch CUDA caching allocator 保留已分配显存。
Uncertainty:
- 未做运行时压力复现；按用户要求不再实际启动训练排查。
Next:
- 在不恢复 52 workers 的前提下，可试 `dataloader_num_workers=2`、`dataloader_persistent_workers=false`、`prefetch_data=false`、`--auto_retry 0`；若仍卡顿，回退 batch8/lr1e-5。

## EVT-20260718-000000-06 - batch16 DataLoader workers set to 2

Description:
- 将 batch16/lr1e-5 对照配置的 DataLoader worker 数调整为 2。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_num_workers` 从 0 改为 2。
- `dataloader_persistent_workers=false` 和 `prefetch_data=false` 保持不变。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `dataloader_num_workers=2`、`dataloader_persistent_workers=False`、`prefetch_data=False`。
Interpretation:
- 该设置试图在低 `/dev/shm` 风险和减少 GPU starvation 之间折中；仍应配合 `--auto_retry 0`。
Uncertainty:
- 尚未运行验证是否消除 GPU 周期性空等或是否再次触发 shm/bus error。
Next:
- 用独立输出目录和 `--auto_retry 0` 运行 batch16 对照；若仍不稳，回退 batch8/lr1e-5。

## EVT-20260718-000000-07 - batch16 DataLoader workers set to 4

Description:
- 将 batch16/lr1e-5 对照配置的 DataLoader worker 数从 2 提升到 4。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- 用户报告 `dataloader_num_workers=2` 后训练速度比 workers=0 快一倍多。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_num_workers` 从 2 改为 4。
- `dataloader_persistent_workers=false` 和 `prefetch_data=false` 保持不变。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `dataloader_num_workers=4`、`dataloader_persistent_workers=False`、`prefetch_data=False`。
Interpretation:
- worker 数 4 是在 workers=2 已显著改善吞吐后的进一步试探；仍保留禁用 persistent/prefetch 来控制 shm 风险。
Uncertainty:
- 尚未验证 workers=4 是否继续提升速度或重新引入 shm/bus error。
Next:
- 用独立输出目录和 `--auto_retry 0` 运行 batch16 workers=4 对照；若出现 bus error 或 GPU 等待恶化，回退 workers=2。

## EVT-20260718-000000-08 - batch16 trainer prefetch enabled

Description:
- 为 batch16/lr1e-5、workers=4 对照配置启用 trainer 侧数据预取。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.prefetch_data` 从 `false` 改为 `true`。
- `dataloader_num_workers=4` 和 `dataloader_persistent_workers=false` 保持不变。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `dataloader_num_workers=4`、`dataloader_persistent_workers=False`、`prefetch_data=True`。
Interpretation:
- 该设置让 trainer 在训练当前 batch 时预取下一 batch，目标是减少 batch16 下 GPU 等待数据导致的低利用率。
Uncertainty:
- 尚未验证显存是否足够，以及是否重新触发 shm/bus error。
Next:
- 运行时加 `--auto_retry 0`；观察 GPU 利用率、显存峰值、step time 和是否出现 bus error。

## EVT-20260718-000000-09 - batch16 DataLoader workers set to 8

Description:
- 将 batch16/lr1e-5 对照配置的 DataLoader worker 数从 4 提升到 8。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- 用户报告 workers=4、persistent=false、prefetch=true 后，GPU 低占用频率没有明显减少，但显存峰值可接受。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_num_workers` 从 4 改为 8。
- `dataloader_persistent_workers=false` 和 `prefetch_data=true` 保持不变。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `dataloader_num_workers=8`、`dataloader_persistent_workers=False`、`prefetch_data=True`。
Interpretation:
- 该设置尝试用更多 DataLoader worker 提升 CPU/I/O 侧吞吐，同时仍禁用 persistent workers 控制 shm 风险。
Uncertainty:
- 尚未验证 workers=8 是否减少 GPU 空等或重新触发 shm/bus error。
Next:
- 用 `--auto_retry 0` 运行并观察 GPU 利用率、step time、显存峰值和 bus error。

## EVT-20260718-000000-10 - batch16 persistent workers enabled

Description:
- 将 batch16/lr1e-5 对照配置改为 workers=8、persistent=true、prefetch=true。

Type: code-change
Source: agent-run
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- 用户报告 workers=8、persistent=false、prefetch=true 后 GPU 低占用没有好转。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_persistent_workers` 从 `false` 改为 `true`。
- `dataloader_num_workers=8` 和 `prefetch_data=true` 保持不变。
Evidence:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python -m json.tool configs/vae/slat_enc_dec_gs_fine_tune.json` 校验通过。
- 读取 JSON 确认 `dataloader_num_workers=8`、`dataloader_persistent_workers=True`、`prefetch_data=True`。
Interpretation:
- 该设置测试 worker 常驻是否能减少 worker 初始化/批次供应抖动，但可能增加长期 `/dev/shm` 占用。
Uncertainty:
- 尚未验证是否改善 GPU 利用率或重新触发 shm/bus error。
Next:
- 用 `--auto_retry 0` 运行并观察 GPU 利用率、step time、显存峰值和 bus error；若不稳，回退 persistent=false 或 batch8/lr1e-5。

## EVT-20260718-000000-11 - persistent workers partially improved GPU utilization

Description:
- batch16 配置启用 workers=8、persistent=true、prefetch=true 后 GPU 低占用次数约减少 30%，但仍频繁出现。

Type: finding
Source: user-reported
Related records:
- CFG-20260717-116
- RUN-20260718-002
Facts:
- 用户报告 GPU 占用降低到低位的次数大约减少 30%。
- 用户同时报告低占用仍然非常多。
Evidence:
- 当前配置为 `dataloader_num_workers=8`、`dataloader_persistent_workers=true`、`prefetch_data=true`、`batch_size_per_gpu=16`、`batch_split=8`。
Interpretation:
- persistent workers 有帮助，但瓶颈仍主要来自每 batch 的数据读取/解压/resize/collate 或 batch16 数据量本身；继续只调 worker 参数可能边际收益有限。
Uncertainty:
- 未记录精确 step time 分布和 CPU/IO 利用率。
Next:
- 优先比较 batch16 当前配置与 batch8/lr1e-5 的每 100 step wall time 和验证质量；若 batch16 质量没有明显优势，回退 batch8。
