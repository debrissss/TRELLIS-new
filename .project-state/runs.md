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
