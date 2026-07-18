# Runs

Full scan note, 2026-07-17: static inventory found no explicit training/preprocessing execution evidence such as `outputs/*/command.txt` or `.project-state` RUN records. Existing datasets, model directories, and mesh files are registered as ART records instead of inferred historical runs.

## RUN-20260717-001 - convert SLat encoder safetensors to pt

Description:
- 使用转换脚本将 SLat encoder safetensors 权重转换为 PyTorch `.pt` state_dict。

Time: 2026-07-17 21:53 UTC
Execution source: agent-run
Entrypoints:
- EXE-20260717-128
Command:
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/mamba_envs/trellis5090/bin/python fine_tuning/convert_safetensors_to_pt.py --model_prefix microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16 --output microsoft/TRELLIS-image-large/ckpts/slat_enc_swin8_B_64l8_fp16.pt --overwrite`
Config file:
- none
Input Artifacts:
- ART-20260717-008
Output Path:
- ART-20260717-010
Facts:
- 转换脚本输出 `Conversion completed and verified.`。
- 输出 `.pt` 可用 `torch.load(..., weights_only=True)` 读取，包含 100 个 state_dict 条目。
Analysis / Evaluation:
- Source: agent
- 输出可作为 SLat encoder fine-tune checkpoint。
Uncertainty:
- 未启动实际 fine-tune 训练验证该 checkpoint。
Next:
- 在 fine-tune 配置中将 `trainer.args.finetune_ckpt.encoder` 指向 ART-20260717-010。

## RUN-20260717-002 - convert SLat GS decoder safetensors to pt

Description:
- 使用转换脚本将 SLat Gaussian decoder safetensors 权重转换为 PyTorch `.pt` state_dict。

Time: 2026-07-17 21:53 UTC
Execution source: agent-run
Entrypoints:
- EXE-20260717-128
Command:
- `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 /root/autodl-tmp/mamba_envs/trellis5090/bin/python fine_tuning/convert_safetensors_to_pt.py --model_prefix microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16 --output microsoft/TRELLIS-image-large/ckpts/slat_dec_gs_swin8_B_64l8gs32_fp16.pt`
Config file:
- none
Input Artifacts:
- ART-20260717-009
Output Path:
- ART-20260717-011
Facts:
- 转换脚本输出 `Conversion completed and verified.`。
- 输出 `.pt` 可用 `torch.load(..., weights_only=True)` 读取，包含 101 个 state_dict 条目。
Analysis / Evaluation:
- Source: agent
- 输出可作为 SLat Gaussian decoder fine-tune checkpoint。
Uncertainty:
- 未启动实际 fine-tune 训练验证该 checkpoint。
Next:
- 在 fine-tune 配置中将 `trainer.args.finetune_ckpt.decoder` 指向 ART-20260717-011。

## RUN-20260717-003 - user-reported SLat GS fine-tune hit corrupt feature cache

Description:
- 用户报告 SLat encoder + GS decoder fine-tune 训练在保存 step 500 checkpoint 后因 FaceScape 特征缓存损坏中断并重试。

Time: 2026-07-17 22:25 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- unknown
Config file:
- CFG-20260717-116
Input Artifacts:
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
- ART-20260717-012
Output Path:
- unknown
Facts:
- 日志显示 VGG16 权重首次下载到 `/root/.cache/torch/hub/checkpoints/vgg16-397923af.pth`，LPIPS 从环境包内加载 `vgg.pth`。
- 日志显示 `Saving checkpoint at step 500... Done.`。
- 随后 DataLoader worker 26 在 index 304、instance `3ad9da5e876ef8f20a92f5fc71769b91ac983f91aa83c7ead853ddb8e815d0ca` 处抛出 `zipfile.BadZipFile: Overlapped entries: 'patchtokens.npy' (possible zip bomb)`。
- 本地复查发现对应 `.npz` 文件仅 36713 bytes，`zipfile.testzip()` 返回 `patchtokens.npy`。
Analysis / Evaluation:
- Source: agent
- 失败原因是 FaceScape DINOv2 特征缓存 `.npz` 被截断或压缩包目录损坏，不是 checkpoint 转换或训练模型结构本身的直接错误。
- 同目录还发现 ART-20260717-013 具有相同损坏模式，训练继续随机采样时可能再次失败。
Uncertainty:
- 用户未提供完整启动命令和输出目录；step 500 checkpoint 的具体路径未知。
- 两个 `.npz` 损坏的产生时间和原始生成进程未知。
Next:
- 使用 `fine_tuning/facescape_extract_feature.py --instances <bad-list> --overwrite` 只重生成坏样本的 DINO 特征，然后从已保存 checkpoint 继续训练。

## RUN-20260717-004 - SLat GS fine-tune 1000-step completed

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 1000-step fine-tune 试验。

Time: 2026-07-17 23:30 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- `python train.py --config configs/vae/slat_enc_dec_gs_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/slat_enc_dec_gs_fine_tune --num_gpus 1 --ckpt none`
Config file:
- CFG-20260717-116
Input Artifacts:
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
Output Path:
- ART-20260717-014
Facts:
- `log.txt` 和 `loss.txt` 各有 1000 行，记录 step 1 到 step 1000。
- `config.json` 显示关键参数为 `max_steps=1000`、`batch_size_per_gpu=4`、`batch_split=2`、`lr=0.0001`、`i_print=10`、`i_log=100`、`i_save=500`。
- 输出保存了 step 500 和 step 1000 的 encoder、decoder、EMA 和 misc checkpoint。
- 最终 step loss 为 0.0250695，rec 为 0.0249654，l1 为 0.00525688，ssim loss 为 0.0538422，lpips 为 0.0447002，kl 为 9.49770。
- 前 100 step 平均 loss 为 0.0226317，最后 100 step 平均 loss 为 0.0212889，约下降 5.93%。
- 总 elapsed 为 592.29 秒，除首步初始化外，后期单 step 约 0.56 秒。
Analysis / Evaluation:
- Source: agent
- 训练完整结束且产物齐全；禁用坏样本 metadata 后未再出现 BadZipFile 失败。
- 指标有小幅改善，尤其 LPIPS 最后 100 step 均值较前 100 step 下降约 11.4%，KL 下降约 9.8%，grad_norm 下降约 28.2%。
- loss 曲线存在明显随机样本波动和少数尖峰，最大 loss 在 step 499 为 0.06497；1000 step 更像短程 smoke/fine-tune 验证，不足以证明充分收敛。
- final sample 的 `rec_image_final.jpg` 与 `gt_image_final.jpg` 视觉上高度接近，但样本数量有限，不能替代固定验证集评估。
Uncertainty:
- 未运行独立 test/validation set 指标。
- 未比较 step 500 与 step 1000 checkpoint 的固定样本重建质量。
- 未确认 EMA checkpoint 与 non-EMA checkpoint 哪个在下游推理中更好。
Next:
- 用固定 test/holdout 样本评估 step1000 和 EMA step1000；若质量稳定，再考虑继续从 step1000 延长训练或调整学习率。

## RUN-20260718-001 - SLat GS fine-tune v2 batch8 lr1e-5 completed

Description:
- SLat encoder + Gaussian decoder 使用 FaceScape train 数据完成 batch 8、lr=1e-5 的 1000-step fine-tune 试验。

Time: 2026-07-18 00:10 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- `python train.py --config configs/vae/slat_enc_dec_gs_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/slat_enc_dec_gs_fine_tune_v2 --num_gpus 1 --ckpt none`
Config file:
- CFG-20260717-116
Input Artifacts:
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
Output Path:
- ART-20260718-001
Facts:
- `outputs/slat_enc_dec_gs_fine_tune/log.txt` 仍是上一轮旧结果，`config.json` 为 `batch_size_per_gpu=4`、`batch_split=2`、`lr=1e-4`。
- 新结果实际位于 `outputs/slat_enc_dec_gs_fine_tune_v2`，`log.txt` 和 `loss.txt` 各有 1000 行。
- v2 `config.json` 显示 `batch_size_per_gpu=8`、`batch_split=4`、`lr=1e-5`，micro-batch 仍为 2。
- v2 最终 step loss 为 0.0209562，rec 为 0.0208766，l1 为 0.00414328，ssim loss 为 0.0427032，lpips 为 0.0409636，kl 为 9.77622。
- v2 前 100 step 平均 loss 为 0.0227392，最后 100 step 平均 loss 为 0.0208222，约下降 8.43%。
- v2 最后 100 step 平均 loss 比 RUN-20260717-004 最后 100 step 低约 2.19%，平均 grad_norm 低约 43.4%，平均 step_time 高约 82.5%。
Analysis / Evaluation:
- Source: agent
- v2 指标比上一轮略好且梯度范数更低，符合有效 batch 增大、学习率降低后训练更稳的预期。
- v2 单 step 约 1.02 秒，比上一轮约 0.56 秒慢，主要来自有效 batch 从 4 增到 8。
- v2 loss 仍有随机样本尖峰，但最大 loss 0.03889 明显低于上一轮最大 0.06497。
- final sample 的 `rec_image_final.jpg` 与 `gt_image_final.jpg` 视觉上高度接近，但仍只有训练样本可视化，不能替代独立验证集评估。
Uncertainty:
- 用户提到的路径与实际新结果目录不一致；本记录按本地时间戳和配置确认的 v2 目录登记。
- 尚未运行独立 test/validation set 指标。
- 未比较 v2 step 1000 EMA 与 non-EMA checkpoint 的下游质量。
Next:
- 用固定 test/holdout 样本评估 v2 step1000 和 EMA step1000；若稳定，可从 v2 step1000 继续延长训练。

## RUN-20260718-002 - SLat GS batch16 failed with DataLoader shm bus error

Description:
- SLat encoder + Gaussian decoder batch16/lr1e-5 对照训练在 init sampling 后因 DataLoader worker shared memory 问题失败。

Time: 2026-07-18 00:35 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python train.py --config configs/vae/slat_enc_dec_gs_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/slat_enc_dec_gs_fine_tune_v3 --num_gpus 1 --ckpt none`
Config file:
- CFG-20260717-116
Input Artifacts:
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
Output Path:
- ART-20260718-002
Facts:
- 用户报告运行打印 `Sampling 1 images... Done.` 后进程被系统 kill，并随后似乎被自动唤起继续执行。
- 随后终端反复输出 `ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).`
- 静态代码显示 `train.py` 默认 `auto_retry=3`，捕获异常后会重跑 `main`。
- `Trainer.prepare_dataloader` 在未配置 `dataloader_num_workers` 时使用 `max(1, ceil(os.cpu_count()/torch.cuda.device_count())//4)`，本机此前打印为 52 workers。
- `Trainer.load_data` 会先取完整 batch 并 `recursive_to_device`，再按 `batch_split` 切成 micro-batch；因此 `batch_split=8` 不降低 DataLoader 共享内存压力。
Analysis / Evaluation:
- Source: agent
- 失败主因是 batch16 下 DataLoader 多 worker 预取完整大 batch，worker 到主进程的 tensor 传输占用大量 `/dev/shm`；PyTorch worker bus error 与该模式吻合。
- `batch_split` 只降低训练前后向 micro-batch 显存，不降低 DataLoader collate、worker prefetch 或共享内存中的完整 batch 大小。
- `auto_retry=3` 解释了用户观察到的“被 kill 后又被唤起”现象；DataLoader worker 异常由父进程捕获后触发 retry。
Uncertainty:
- 未再运行压力复现；后续按用户要求避免实际启动训练检查。
- `/dev/shm` 实际大小未记录。
Next:
- 若继续 batch16 对照，使用 `--auto_retry 0` 并在配置中设置 `dataloader_num_workers=0` 或小值、`dataloader_persistent_workers=false`、`prefetch_data=false`；否则回退 batch8/lr1e-5。

## RUN-20260718-003 - FaceScape SLat GS 50GB subset prepared

Description:
- 使用临时 Python 筛样和 rsync，从 FaceScape train 数据中复制约 50GB 的 SLat encoder + Gaussian decoder 训练子集。

Time: 2026-07-18 12:04 CST
Execution source: agent-run
Entrypoints:
- EXE-20260718-001
Command:
- `python - <<'PY' ... PY && rsync -a --info=progress2 --files-from=/tmp/facescape_slat_gs_50gb_files.txt datasets/Facescape/train/ datasets/Facescape_slat_gs_50gb/train/`
Config file:
- none
Input Artifacts:
- ART-20260717-001
Output Path:
- ART-20260718-003
Facts:
- Python 筛样阶段从 train metadata 中选择 1178 个有效样本，按 `renders/<sha>` 与 `features/dinov2_vitl14_reg/<sha>.npz` 累计估算约 50.020 GiB。
- rsync 复制完成，退出码为 0，最终传输 53,708,809,800 bytes。
- `du -sh datasets/Facescape_slat_gs_50gb` 显示 `51G`。
- `metadata.csv` 为 1179 行，即 1178 个样本加表头。
- 复制后统计有 1178 个 DINO feature `.npz` 文件和 1178 个 render 实例目录。
- 一致性检查显示 `rows=1178`、`missing_required_paths=0`。
Analysis / Evaluation:
- Source: agent
- 该子集包含 `SparseFeat2Render` 训练 SLat encoder + GS decoder 所需的数据面：metadata、render 图像/相机 transforms、DINOv2 patch token 特征。
- 该子集不包含 `voxels/` 或 `renders_cond/`，因为当前训练路径不读取这些资源；也不包含预训练 `.pt` checkpoint，迁移到低配机器时仍需同步代码和模型权重。
Uncertainty:
- 未在子集上实际启动训练；按用户约束避免运行会占用 GPU/内存的检查。
Next:
- 将 `datasets/Facescape_slat_gs_50gb` 传到低配置机器，并用相同 config 或受显存限制调整后的 config 记录 step/samples throughput。

## RUN-20260718-004 - SLat GS batch16 stable throughput observed

Description:
- 用户报告 SLat encoder + Gaussian decoder batch16 训练在 step 510-780 区间进入稳定吞吐段。

Time: 2026-07-18 12:12 CST
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- unknown
Config file:
- CFG-20260717-116
Input Artifacts:
- ART-20260717-001
- ART-20260717-010
- ART-20260717-011
Output Path:
- unknown
Facts:
- 用户提供了 step 510 到 step 780、每 10 step 一次的训练进度打印，共 28 个速度观测值。
- 该区间速度均值为 `1803.39 steps/h`，中位数为 `1803.89 steps/h`。
- 该区间速度最小值为 `1757.12 steps/h`，最大值为 `1841.48 steps/h`，总体标准差为 `18.45 steps/h`。
- 按当前 `batch_size_per_gpu=16` 估算，该稳定段平均样本吞吐约为 `28854 samples/h`。
- 平均每 step wall time 约为 `1.996s`。
Analysis / Evaluation:
- Source: agent
- 该区间位于训练 51%-78% 进度，已避开启动采样和早期缓存/worker 初始化噪声，适合作为当前昂贵 GPU 的 batch16 稳定速度基线。
- 速度波动约为均值的 1% 左右，说明从进度打印看整体吞吐已经比较稳定；瞬时 GPU 利用率仍可能波动，但对成本比较更应优先看 samples/h 与单位小时价格。
- 与此前粗略 `1700 steps/h` 相比，稳定段更接近 `1800 steps/h`，batch16 样本吞吐约 `28.9k samples/h`。
Uncertainty:
- 用户未提供本次运行的完整命令、输出目录、最终 loss、checkpoint 或训练完成状态。
- 该速度只覆盖 step 510-780，不代表完整 1000-step 含初始化、采样和保存 checkpoint 的端到端平均速度。
Next:
- 训练完成后记录最终 `log.txt`、loss 汇总和端到端 wall time；低配机器测速时也记录同口径的稳定段 samples/h 与端到端 samples/h。

## RUN-20260718-005 - SS encoder plus decoder fine-tune 1000 steps analyzed

Description:
- 用户提供 SS encoder + decoder FaceScape fine-tune 1000-step 日志，分析 loss 曲线、样本图和下一步调参方向。

Time: 2026-07-18 16:30 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python train.py --config configs/vae/ss_enc_dec_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/ss_enc_dec_fine_tune --num_gpus 1 --ckpt none --auto_retry 0`
Config file:
- CFG-20260718-001
Input Artifacts:
- ART-20260717-001
Output Path:
- ART-20260718-004
Facts:
- `outputs/ss_enc_dec_fine_tune/log_ss_enc_dec_fine_tune.txt` 有 1000 行，step 1-1000 完成。
- 总 loss 全程均值 `0.000414844`，标准差 `0.000015321`，最小 `0.000372609`，最大 `0.000465461`。
- 1-100 step 总 loss 均值 `0.000416079`；901-1000 step 总 loss 均值 `0.000410608`，约下降 1.3%。
- Dice loss 全程均值 `2.2405e-05`，901-1000 step 均值 `2.2831e-05`，没有稳定下降趋势。
- KL 全程均值 `0.392439`；乘以 `lambda_kl=0.001` 后贡献约 `0.000392439`，是总 loss 的主要部分。
- 901-1000 step 中 Dice 均值约 `2.2831e-05`，KL 加权贡献约 `0.000387777`，总 loss 均值约 `0.000410608`。
- 平均 step time 约 `0.785s`，前 100 step 之后稳定在约 `0.78s/step`。
- 输出包含 step 500/1000 checkpoint 和 init/final SS 重建样本图。
Analysis / Evaluation:
- Source: agent
- 曲线没有发散，也没有明显过拟合震荡；但 1000 step 的下降非常小，主要来自 KL 项轻微下降，而不是 Dice 重建项改善。
- 当前 effective batch 已为 16；单纯增大 batch 只能进一步平滑曲线，不太可能解决“没有学习方向”的核心问题。
- lr=1e-5 已偏保守；继续降低 lr 更适合“保护预训练权重、防止漂移”，但会让 FaceScape 适配更慢，不适合作为首要改进手段。
- 若目标是更明显改善 FaceScape SS 重建，优先考虑延长训练并增加中间评估，或做 `lambda_kl` 小幅 ablation；但降低 KL 可能破坏 latent 分布，对后续 flow/下游兼容性有风险。
Uncertainty:
- 尚未计算逐样本 IoU/F1 或同一批样本的 init-vs-final 定量对比。
- snapshot init/final 可能抽到不同样本，视觉对比只能作为粗略参考。
Next:
- 不优先增大 batch 或降低 lr；建议先保留 batch16/lr1e-5，延长到 5000 step 并增加 `i_sample`/定量评估，或另开小实验测试 `lambda_kl=1e-4` 与 `5e-4`。

## RUN-20260718-006 - SS encoder plus decoder fine-tune kl5e-4 1000 steps analyzed

Description:
- 用户提供 SS encoder + decoder FaceScape `lambda_kl=5e-4` 1000-step 日志，分析是否需要继续降低 KL。

Time: 2026-07-18 18:20 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python train.py --config configs/vae/ss_enc_dec_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/ss_enc_dec_fine_tune_kl5e-4 --num_gpus 1 --ckpt none --auto_retry 0`
Config file:
- CFG-20260718-001
Input Artifacts:
- ART-20260717-001
Output Path:
- ART-20260718-005
Facts:
- `outputs/ss_enc_dec_fine_tune_kl5e-4/log_ss_enc_dec_fine_tune_kl5e-4.txt` 有 1000 行，step 1-1000 完成。
- 本次配置使用 `lambda_kl=5e-4`，其他关键参数保持 `max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_save=500`。
- 总 loss 全程均值 `0.000215963`，901-1000 step 均值 `0.000214193`；因为 KL 权重改变，该总 loss 不能直接和 `lambda_kl=0.001` 的总 loss 做优劣比较。
- Dice loss 全程均值 `1.1318e-05`，901-1000 step 均值 `1.0851e-05`；相比 `lambda_kl=0.001` 的全程均值 `2.2405e-05` 明显更低。
- Dice loss 的线性趋势约为每 1000 step 上升 `2.37e-07`，相对约 `2.1%`，没有形成稳定下降趋势。
- KL 全程均值 `0.409290`，901-1000 step 均值 `0.406685`；加权 KL 贡献约为 `0.000204645`，仍然是总 loss 主体。
- Grad norm 全程均值约 `0.002286`，低于 `lambda_kl=0.001` 运行的 `0.003241`。
- final 重建样本从视觉上看相对上一轮没有明显质变，随机 snapshot 证据较弱。
Analysis / Evaluation:
- Source: agent
- 降到 `5e-4` 确实降低了 Dice loss 的绝对水平，但 1000 step 内 Dice 没有持续改善，说明“KL 太强”可能是问题之一，但不是唯一瓶颈。
- 当前 evidence 支持继续做一次更低 KL 的受控 ablation，例如 `lambda_kl=1e-4`；但不建议立刻降到 0，因为后续 flow 虽会微调，latent 分布过散仍可能增加 flow 训练难度。
- 因为 total loss 的尺度主要随 KL 权重变化而变化，后续判断应优先看固定样本的 Dice/IoU/occupancy ratio，以及同一批样本的 init-vs-final-vs-checkpoint 对比。
Uncertainty:
- 尚未计算固定验证集或固定 batch 的 voxel IoU、F1/Dice、occupancy ratio。
- 训练 snapshot 可能不是同一样本，视觉对比不能单独作为调参依据。
Next:
- 建议下一轮保持 batch16/lr1e-5 不变，测试 `lambda_kl=1e-4` 跑 1000 step，并同步做固定样本定量评估；若仍无改善，再排查 voxel 尺度、阈值、数据预处理与采样可视化路径。

## RUN-20260718-007 - SS encoder plus decoder fine-tune kl1e-4 1000 steps analyzed

Description:
- 用户提供 SS encoder + decoder FaceScape `lambda_kl=1e-4` 1000-step 日志，分析相对 `1e-3` 和 `5e-4` 的变化。

Time: 2026-07-18 19:05 UTC
Execution source: user-reported
Entrypoints:
- EXE-20260717-105
Command:
- `python train.py --config configs/vae/ss_enc_dec_fine_tune.json --data_dir datasets/Facescape/train --output_dir outputs/ss_enc_dec_fine_tune_kl1e-4 --num_gpus 1 --ckpt none --auto_retry 0`
Config file:
- CFG-20260718-001
Input Artifacts:
- ART-20260717-001
Output Path:
- ART-20260718-006
Facts:
- `outputs/ss_enc_dec_fine_tune_kl1e-4/log_ss_enc_dec_fine_tune_kl1e-4.txt` 有 1000 行，`loss_ss_enc_dec_fine_tune_kl1e-4.txt` 也有 1000 行，step 1-1000 完成。
- 输出目录包含 step 500/1000 的 encoder、decoder、EMA 和 misc checkpoint，目录大小约 `6.0G`。
- 本次配置使用 `lambda_kl=1e-4`，其他关键参数保持 `max_steps=1000`、`batch_size_per_gpu=16`、`batch_split=4`、`lr=1e-5`、`i_print=10`、`i_save=500`。
- 总 loss 全程均值 `4.7369e-05`，901-1000 step 均值 `4.7108e-05`；因为 KL 权重改变，该总 loss 不能直接和更高 KL 权重运行做优劣比较。
- Dice loss 全程均值 `2.4171e-06`，901-1000 step 均值 `2.2334e-06`；相比 `lambda_kl=5e-4` 的全程均值 `1.1318e-05` 进一步明显降低。
- Dice loss 的线性趋势约为每 1000 step 下降 `2.13e-07`，相对约 `8.8%`，比 `lambda_kl=5e-4` 的轻微上升趋势更好。
- Dice loss 中有 119 个 step 为 `0.0`，436 个 step 小于 `1e-6`，说明部分 batch 的 SS 重建项已经接近饱和。
- KL 全程均值 `0.449517`，901-1000 step 均值 `0.448742`；比 `lambda_kl=5e-4` 的 `0.409290` 更高，说明 latent 分布约束被明显放松。
- 加权 KL 贡献全程均值约 `4.4952e-05`，仍然是总 loss 主体；Dice 均值约占总 loss 的 5% 左右。
- Grad norm 全程均值约 `0.000890`，低于 `lambda_kl=5e-4` 的 `0.002286`。
- final 重建图与 GT 的大轮廓较接近，差异主要在局部边缘、细小突起和薄结构；没有看到明显崩坏。
Analysis / Evaluation:
- Source: agent
- 三轮对比中，`lambda_kl=1e-4` 是目前 SS 重建项最好的设置：Dice 绝对值最低，后 100 step 也最低，并且趋势从 `5e-4` 的轻微变差转为轻微变好。
- 但 `1e-4` 已经让 KL 均值明显升高，且大量 batch Dice 接近 0，继续单纯降低 KL 的边际收益可能变小，风险会转向 latent 分布漂移与后续 flow 学习难度。
- 当前更像是“SS VAE 已经能较好拟合 64^3 sparse structure”，而不是“还需要大幅继续压低 KL”；视觉上剩余问题可能来自 SS 表示分辨率、阈值/occupancy、随机可视化样本或后续 SLat/decoder 阶段。
Uncertainty:
- 尚未计算固定验证集或固定 batch 的 voxel IoU、F1/Dice、occupancy ratio。
- 训练 snapshot 可能不是同一样本，视觉对比不能单独证明泛化效果。
- 尚未验证 `lambda_kl=1e-4` 训练出的 latent 对后续 SS flow 微调是否更容易或更困难。
Next:
- 建议暂时把 `lambda_kl=1e-4` 作为当前最佳候选，不急于继续降到 `5e-5` 或 0；下一步优先做固定样本定量评估，并用该 checkpoint 启动后续 flow 微调小实验。

## RUN-20260718-008 - SS eval dataset preparation smoke test

Description:
- 使用新建的固定样本评估集生成入口，从真实 FaceScape test metadata 临时抽取 4 个样本验证 mini dataset 生成逻辑。

Time: 2026-07-18 20:10 UTC
Execution source: agent-run
Entrypoints:
- EXE-20260718-002
Command:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/prepare_ss_eval_dataset.py --source_root datasets/Facescape/test --output_root /tmp/trellis_ss_eval_test_4 --num_samples 4 --seed 20260718 --min_aesthetic_score 4.5 --replace`
Config file:
- none
Input Artifacts:
- ART-20260717-001
Output Path:
- none
Facts:
- 命令成功写出 4 个样本的临时 mini dataset metadata，并创建 `voxels` symlink。
- 临时输出路径在 `/tmp/trellis_ss_eval_test_4`，不作为持久 artifact 记录。
Analysis / Evaluation:
- Source: agent
- 该 smoke test 验证了脚本能读取真实 metadata、筛选存在 voxel 的样本，并生成可供 `SparseStructure` 复用的数据 root。
Uncertainty:
- 未在本次 smoke 中生成正式 64 样本评估集。
Next:
- 使用同一入口生成 `datasets/Facescape_ss_eval_test_64` 作为正式固定评估集。

## RUN-20260718-009 - SS encoder plus decoder evaluation smoke test

Description:
- 使用新建的 SS encoder/decoder 重建评估入口，在 4 个临时固定样本上验证 checkpoint manifest、模型加载和指标输出。

Time: 2026-07-18 20:15 UTC
Execution source: agent-run
Entrypoints:
- EXE-20260718-003
Command:
- `/root/autodl-tmp/mamba_envs/trellis5090/bin/python eval/evaluate_ss_enc_dec_reconstruction.py --config configs/vae/ss_enc_dec_fine_tune.json --data_root /tmp/trellis_ss_eval_test_4 --checkpoints eval/ss_eval_checkpoints.json --output_dir /tmp/trellis_ss_enc_dec_eval_smoke_all --batch_size 2`
Config file:
- CFG-20260718-002
Input Artifacts:
- none
Output Path:
- none
Facts:
- official、`kl1e-3_step1000`、`kl5e-4_step1000`、`kl1e-4_step1000` 四组 checkpoint 都成功加载并完成 4 样本 deterministic posterior-mean 评估。
- deterministic smoke 中四组 checkpoint 的 hard IoU/Dice 均为 `1.0`，`soft_dice_loss` 均为 `0.0`，提示该小样本和 posterior mean 口径下指标饱和。
- 额外使用 `--sample_posterior --seed 20260718 --checkpoint_names official` 验证 stochastic posterior 评估路径；official 在 4 样本上 `iou_mean=0.999956`、`dice_f1_mean=0.999978`、`soft_dice_loss_mean=2.2113e-05`。
Analysis / Evaluation:
- Source: agent
- 评估入口能在真实模型和真实数据上跑通；posterior sampling 模式比 posterior mean 更能暴露随机重建稳定性差异，因此正式评估建议两种模式都跑。
Uncertainty:
- 临时 4 样本 smoke 只验证管线，不代表正式 64 样本或更大验证集上的模型优劣。
Next:
- 生成正式 64 样本 mini dataset，并分别运行 posterior mean 与 sample posterior 两种评估。
