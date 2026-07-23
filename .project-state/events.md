# Events

## EVT-20260717-000000-01 - full project-state scan

Description:
- 完成 TRELLIS-new 首次 `.project-state` 全量静态扫描并建立基础台账。

Facts:
- 已登记训练、推理、FaceScape 预处理、overfit 准备、训练包装器和重建审计入口。
- 已登记 FaceScape 数据、TRELLIS-image-large 本地模型、参考数据集和截断 mesh 输入资源。
- 未发现可证明历史运行命令的输出记录，因此未创建 RUN。
Related records:
- none

## EVT-20260722-060807-01 - mesh decoder dataloader workers32 prefetch1 stable

Description:
- 用户报告 mesh decoder 微调使用 `dataloader_num_workers=32`、`dataloader_prefetch_factor=1` 后训练明显更稳定，速度稳定在约 `1600 steps/h`。

Facts:
- 用户报告配置项为 `"dataloader_num_workers": 32`。
- 用户报告配置项为 `"dataloader_prefetch_factor": 1`。
- 用户报告该配置下训练稳定性明显提升。
- 用户报告训练速度稳定在约 `1600 steps/h`。
Related records:
- none

## EVT-20260722-030307-01 - dataloader prefetch factor configurable

Description:
- 为训练 DataLoader 增加 `dataloader_prefetch_factor` 配置项，并在 mesh decoder 微调配置中设为 `1`。

Facts:
- 修改代码文件：`trellis/trainers/base.py`。
- 修改配置文件：`configs/vae/slat_dec_mesh_fine_tune.json`。
- 改动前：`BaseTrainer.__init__` 支持 `dataloader_num_workers`、`dataloader_drop_last`、`dataloader_persistent_workers`，但不支持从配置文件控制 PyTorch DataLoader 的 `prefetch_factor`。
- 改动前：`prepare_dataloader()` 构造 DataLoader 时未传入 `prefetch_factor`，当 `num_workers > 0` 时使用 PyTorch 默认值 `2`。
- 改动后：`BaseTrainer.__init__` 新增 `dataloader_prefetch_factor=None`，并保存为 `self.dataloader_prefetch_factor`。
- 改动后：`prepare_dataloader()` 在 `num_workers > 0` 且 `dataloader_prefetch_factor` 非空时，将其作为 `prefetch_factor` 传给 DataLoader；未配置时保持原默认行为不变。
- 改动后：`configs/vae/slat_dec_mesh_fine_tune.json` 增加 `"dataloader_prefetch_factor": 1`。
- 修改原因：mesh decoder 训练样本包含高精度大 mesh，默认 `prefetch_factor=2` 会让 `num_workers * 2 * batch_size` 个样本在 CPU 侧排队，容易造成内存从十几 GB 膨胀到数十 GB；将当前配置设为 `1` 可减少预取队列中的完整 batch 数，提高训练启动和运行稳定性。
- 验证：已对 `trellis/trainers/base.py` 执行 `py_compile`；已用 `python -m json.tool` 验证 `configs/vae/slat_dec_mesh_fine_tune.json` 是合法 JSON。
Related records:
- none

## EVT-20260717-000000-02 - truncated mesh inputs deleted

Description:
- 删除仓库根目录两个截断 mesh PLY 输入文件。

Facts:
- `2020bc_mesh_truncated.ply` 和 `f674d4_mesh_truncated.ply` 已按用户要求删除。
Related records:
- none

## EVT-20260717-000000-03 - fine_tuning entrypoints deleted

Description:
- 删除 FaceScape 预处理、训练包装器和 overfit 准备入口文件。

Facts:
- 已删除 `fine_tuning/preprocess_stage1.py` 和 `fine_tuning/preprocess_stage2.py`。
- 已删除 `fine_tuning/train_ss_flow_facescape_finetune.sh`、`fine_tuning/train_ss_flow_facescape_35000_to_40000.sh`、`fine_tuning/train_ss_flow_facescape_overfit_1.sh`、`fine_tuning/train_ss_flow_facescape_overfit_4.sh`、`fine_tuning/train_ss_flow_facescape_overfit_8.sh`。
- 已删除 `fine_tuning/train_slat_flow_facescape_overfit_1.sh`、`fine_tuning/prepare_ss_overfit_experiments.py`、`fine_tuning/prepare_slat_overfit_experiments.py`。
Related records:
- none

## EVT-20260717-000000-04 - ledger rewritten for updated skill

Description:
- 按更新后的 `maintain-project-state` 合同重写 `.project-state`，拆分聚合记录。

Facts:
- `executables.md` 已覆盖为 45 条单入口 EXE 记录。
- `experiment-configs.md` 已覆盖为 15 条单配置文件 CFG 记录。
- `assets.md` 已覆盖为 7 条单资源路径 AST 记录。
- 重写 `current.md` 前已按合同追加 `history.md` 快照。
Related records:
- EXE-20260717-105
- EXE-20260717-130
- CFG-20260717-103
- CFG-20260717-106
- AST-20260717-001
- AST-20260717-002

## EVT-20260717-000000-05 - slat enc dec gs fine tune config added

Description:
- 新增 SLat encoder + Gaussian decoder fine-tune 配置文件。

Facts:
- 创建 `configs/vae/slat_enc_dec_gs_fine_tune.json`。
- 新文件当前完整复制自 `configs/vae/slat_vae_enc_dec_gs_swin8_B_64l8_fp16.json`。
Related records:
- CFG-20260717-116
- EXE-20260717-105

## EVT-20260717-000000-06 - SLat safetensors converted to pt

Description:
- 将 SLat encoder 和 SLat Gaussian decoder safetensors 权重转换为 PyTorch `.pt`。

Facts:
- 已生成 `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt`。
- 已生成 `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`。
Related records:
- RUN-20260717-001
- RUN-20260717-002
- AST-20260717-010
- AST-20260717-011

## EVT-20260717-000000-07 - SLat fine-tune config updated

Description:
- 更新 SLat encoder + Gaussian decoder fine-tune 配置的训练步数、日志/保存间隔和预训练权重。

Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.max_steps` 改为 1000。
- `trainer.args.i_log` 改为 100，`trainer.args.i_save` 改为 500。
- 新增 `trainer.args.finetune_ckpt.encoder` 指向 `microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt`。
- 新增 `trainer.args.finetune_ckpt.decoder` 指向 `microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`。
Related records:
- CFG-20260717-116
- AST-20260717-010
- AST-20260717-011

## EVT-20260717-000000-08 - corrupt FaceScape DINO feature cache found

Description:
- SLat fine-tune 训练日志中的 BadZipFile 被定位为 FaceScape DINO 特征 `.npz` 缓存损坏。

Facts:
- 用户报告训练在 step 500 checkpoint 保存完成后，读取 `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 的 `patchtokens.npy` 时失败。
- 本地验证 `AST-20260717-012` 和 `AST-20260717-013` 均为几十 KB，而同目录正常 `.npz` 至少约 15.6 MB。
- `zipfile.testzip()` 对两个文件均返回 `patchtokens.npy`。
Related records:
- RUN-20260717-003
- AST-20260717-012
- AST-20260717-013

## EVT-20260717-000000-09 - corrupt FaceScape feature metadata disabled

Description:
- 临时扫描 train/test 的 DINOv2 特征缓存，并在 metadata 中禁用坏样本特征标记。

Facts:
- 临时脚本扫描 `datasets/Facescape/train/metadata.csv` 的 6456 行和 `datasets/Facescape/test/metadata.csv` 的 720 行。
- train 中发现 2 个坏样本，均为 `patchtokens.npy` 条目截断：AST-20260717-012 和 AST-20260717-013。
- test 中发现 0 个坏样本。
- 已将两个 train 样本的 `feature_dinov2_vitl14_reg` 写为 `False`。
- 临时脚本已删除。
Related records:
- AST-20260717-001
- AST-20260717-012
- AST-20260717-013

## EVT-20260717-000000-10 - terminal progress controlled by i_print

Description:
- 分析训练日志无进度打印问题，确认终端进度由 `i_print` 控制而不是 `i_log`。

Facts:
- `trellis/trainers/base.py` 中终端进度打印条件为 `self.is_master and self.step % self.i_print == 0`。
- `i_log` 只控制写入 `log.txt`、`loss.txt` 和 TensorBoard scalar，不控制终端进度行。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 设置了 `i_log=100`，但没有设置 `i_print`，因此使用 `Trainer.__init__` 默认值 `i_print=1000`。
- 当前 fine-tune 配置 `max_steps=1000`，所以正常只会在第 1000 步打印一次 `Step/Elapsed/Speed/ETA` 进度行。
Related records:
- CFG-20260717-116
- EXE-20260717-105

## EVT-20260717-000000-11 - SLat fine-tune terminal progress interval set

Description:
- 将 SLat encoder + GS decoder fine-tune 配置的终端进度打印间隔设置为 10 step。

Facts:
- 在 `configs/vae/slat_enc_dec_gs_fine_tune.json` 的 `trainer.args` 中新增 `"i_print": 10`。
- 当前关键训练间隔为 `max_steps=1000`、`i_print=10`、`i_log=100`、`i_save=500`。
Related records:
- CFG-20260717-116
- EXE-20260717-105

## EVT-20260717-000000-12 - SLat GS fine-tune 1000-step completed

Description:
- FaceScape SLat encoder + GS decoder 1000-step fine-tune 完成并完成日志分析。

Facts:
- `outputs/slat_enc_dec_gs_fine_tune/log.txt` 和 `loss.txt` 均包含 1000 行。
- step 500 与 step 1000 checkpoint 均存在。
- 最后 100 step 平均 loss 为 0.0212889，相比前 100 step 平均 loss 0.0226317 下降约 5.93%。
- final sample 的重建图与 GT 图视觉上高度接近。
Related records:
- RUN-20260717-004
- CFG-20260717-116
- AST-20260717-014

## EVT-20260717-000000-13 - SLat fine-tune batch and lr adjusted

Description:
- 调整 SLat encoder + GS decoder fine-tune 配置的有效 batch 和学习率。

Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.batch_size_per_gpu` 从 4 改为 8。
- `trainer.args.batch_split` 从 2 改为 4，因此单卡有效 batch 为 8，micro-batch 仍为 2。
- `trainer.args.optimizer.args.lr` 从 `1e-4` 改为 `1e-5`。
Related records:
- CFG-20260717-116
- RUN-20260717-004

## EVT-20260718-000000-01 - SLat GS fine-tune v2 analyzed

Description:
- 分析 batch 8、lr=1e-5 的 SLat encoder + GS decoder 1000-step fine-tune v2 结果。

Facts:
- 新结果实际在 `outputs/slat_enc_dec_gs_fine_tune_v2`，不是旧的 `outputs/slat_enc_dec_gs_fine_tune`。
- v2 `log.txt` 和 `loss.txt` 均包含 1000 行，step 500 与 step 1000 checkpoint 均存在。
- v2 最后 100 step 平均 loss 为 0.0208222，相比前 100 step 0.0227392 下降约 8.43%。
- v2 最后 100 step 平均 loss 比上一轮 batch4/lr1e-4 低约 2.19%。
- v2 final sample 的重建图与 GT 图视觉上高度接近。
Related records:
- RUN-20260718-001
- CFG-20260717-116
- AST-20260718-001

## EVT-20260718-000000-02 - SLat fine-tune batch16 ablation configured

Description:
- 将 SLat encoder + GS decoder fine-tune 配置改为 batch16/lr1e-5 对照实验。

Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.batch_size_per_gpu` 从 8 改为 16。
- `trainer.args.batch_split` 从 4 改为 8，因此单卡有效 batch 为 16，micro-batch 仍为 2。
- `trainer.args.optimizer.args.lr` 保持 `1e-5`。
Related records:
- CFG-20260717-116
- RUN-20260718-001

## EVT-20260718-000000-03 - batch16 DataLoader shm failure diagnosed

Description:
- batch16/lr1e-5 对照训练在 init sampling 后触发 DataLoader worker shared memory bus error。

Facts:
- 用户报告终端多次输出 `Unexpected bus error encountered in worker... insufficient shared memory (shm)`。
- 代码默认 DataLoader worker 数按 CPU 数计算，本机此前打印为 52。
- batch16 数据在 DataLoader 侧作为完整 batch collate 和预取，`batch_split=8` 只在完整 batch 取出后切分。
Related records:
- RUN-20260718-002
- AST-20260718-002
- CFG-20260717-116

## EVT-20260718-000000-04 - batch16 safe DataLoader config applied

Description:
- 为 batch16/lr1e-5 对照配置加入低共享内存 DataLoader 设置。

Facts:
- 在 `configs/vae/slat_enc_dec_gs_fine_tune.json` 的 `trainer.args` 中新增 `dataloader_num_workers=0`。
- 新增 `dataloader_persistent_workers=false`。
- 新增 `prefetch_data=false`。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-05 - batch16 safe DataLoader GPU starvation observed

Description:
- batch16 安全 DataLoader 配置运行后出现 GPU 高占用与 0 占用周期性交替。

Facts:
- 用户报告训练初期 GPU 利用率 85%-100%，显存约 15GB。
- 约 80 steps、启动约 5 分钟后，GPU 利用率降到 0，显存升到约 20GB，并持续约 2 分钟。
- 之后 GPU 利用率在高占用约 1 分钟与 0 占用约 1 分钟之间循环，显存两次循环后升到约 21.5GB。
- 当前配置为 `batch_size_per_gpu=16`、`batch_split=8`、`dataloader_num_workers=0`、`prefetch_data=false`。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-06 - batch16 DataLoader workers set to 2

Description:
- 将 batch16/lr1e-5 对照配置的 DataLoader worker 数调整为 2。

Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_num_workers` 从 0 改为 2。
- `dataloader_persistent_workers=false` 和 `prefetch_data=false` 保持不变。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-07 - batch16 DataLoader workers set to 4

Description:
- 将 batch16/lr1e-5 对照配置的 DataLoader worker 数从 2 提升到 4。

Facts:
- 用户报告 `dataloader_num_workers=2` 后训练速度比 workers=0 快一倍多。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_num_workers` 从 2 改为 4。
- `dataloader_persistent_workers=false` 和 `prefetch_data=false` 保持不变。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-08 - batch16 trainer prefetch enabled

Description:
- 为 batch16/lr1e-5、workers=4 对照配置启用 trainer 侧数据预取。

Facts:
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.prefetch_data` 从 `false` 改为 `true`。
- `dataloader_num_workers=4` 和 `dataloader_persistent_workers=false` 保持不变。
- batch16/lr1e-5 设置保持不变：`batch_size_per_gpu=16`、`batch_split=8`、`lr=1e-5`。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-09 - batch16 DataLoader workers set to 8

Description:
- 将 batch16/lr1e-5 对照配置的 DataLoader worker 数从 4 提升到 8。

Facts:
- 用户报告 workers=4、persistent=false、prefetch=true 后，GPU 低占用频率没有明显减少，但显存峰值可接受。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_num_workers` 从 4 改为 8。
- `dataloader_persistent_workers=false` 和 `prefetch_data=true` 保持不变。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-10 - batch16 persistent workers enabled

Description:
- 将 batch16/lr1e-5 对照配置改为 workers=8、persistent=true、prefetch=true。

Facts:
- 用户报告 workers=8、persistent=false、prefetch=true 后 GPU 低占用没有好转。
- `configs/vae/slat_enc_dec_gs_fine_tune.json` 中 `trainer.args.dataloader_persistent_workers` 从 `false` 改为 `true`。
- `dataloader_num_workers=8` 和 `prefetch_data=true` 保持不变。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-000000-11 - persistent workers partially improved GPU utilization

Description:
- batch16 配置启用 workers=8、persistent=true、prefetch=true 后 GPU 低占用次数约减少 30%，但仍频繁出现。

Facts:
- 用户报告 GPU 占用降低到低位的次数大约减少 30%。
- 用户同时报告低占用仍然非常多。
Related records:
- CFG-20260717-116
- RUN-20260718-002

## EVT-20260718-120400-01 - FaceScape SLat GS 50GB subset prepared

Description:
- 已从 FaceScape train 数据中复制一个约 51G 的 SLat encoder + GS decoder 训练子集，用于低配置机器测速。

Facts:
- 子集路径为 `datasets/Facescape_slat_gs_50gb`。
- 包含 1178 个样本、1178 个 DINO feature `.npz` 和 1178 个 render 实例目录。
- 一致性检查未发现 metadata 样本缺失 feature 文件或 `transforms.json`。
Related records:
- RUN-20260718-003
- AST-20260718-003

## EVT-20260718-121200-01 - batch16 stable throughput baseline recorded

Description:
- 用户报告当前昂贵 GPU 上 batch16 训练稳定段约为 1803 steps/h。

Facts:
- 稳定段来自 step 510-780 的 28 个进度打印。
- 平均速度为 `1803.39 steps/h`，约等于 batch16 下 `28854 samples/h`。
- 平均每 step 约 `1.996s`。
Related records:
- RUN-20260718-004
- CFG-20260717-116
- EXE-20260717-105

## EVT-20260722-093222-01 - mesh decoder snapshot sample-count mismatch fixed

Description:
- 修复 SLat mesh decoder 微调启动时 init snapshot 在 `num_samples=1` 下的 camera batch 越界问题。

Facts:
- 修改代码文件：`trellis/trainers/vae/structured_latent_vae_mesh_dec.py`。
- 改动前：`run_snapshot(num_samples=1, batch_size=4)` 内部 DataLoader 仍使用 `batch_size=4` 读取样本，可能产生 4 个 `reps`。
- 改动前：多视角渲染 camera 使用 `extrinsics.unsqueeze(0).expand(num_samples, -1, -1)` 和 `intrinsics.unsqueeze(0).expand(num_samples, -1, -1)`，只按请求样本数构造 camera batch。
- 触发错误：当实际 `reps` 数量大于 `num_samples` 时，`_render_batch` 遍历 reps 访问 `extrinsics[i]`，在 `num_samples=1` 时触发 `IndexError: index 1 is out of bounds for dimension 0 with size 1`。
- 改动后：新增 `snapshot_batch_size = min(batch_size, num_samples)`，并让 snapshot DataLoader 使用 `batch_size=snapshot_batch_size`，避免请求 1 个样本时实际读取 4 个样本。
- 改动后：多视角 camera batch 改为 `expand(len(reps), -1, -1)`，按实际 decoder 输出数量对齐 camera 数量。
- 修改原因：mesh decoder 的启动 snapshot 默认请求 1 张图，但原实现混用了请求样本数和 DataLoader batch size；修复后 init snapshot 的样本数量和 camera 数量一致，可避免正式训练前因 snapshot 越界中断。
- 验证：已对 `trellis/trainers/vae/structured_latent_vae_mesh_dec.py` 执行 `py_compile`；并用 `SPCONV_ALGO=native --tryrun` 验证配置、dataset、模型、finetune 权重和 trainer 初始化可通过。
Related records:
- none
