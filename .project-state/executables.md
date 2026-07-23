# Executables

## EXE-20260717-101

Description:
- TRELLIS 环境安装脚本，可安装基础、训练、demo 和加速依赖。
Path:
- one `setup.sh`

## EXE-20260717-102

Description:
- TRELLIS image-to-3D Gradio Web Demo。
Path:
- one `app.py`

## EXE-20260717-103

Description:
- TRELLIS text-to-3D Gradio Web Demo。
Path:
- one `app_text.py`

## EXE-20260717-104

Description:
- 本地图片到 3D 资产的命令行推理入口。
Path:
- one `cli.py`

## EXE-20260717-105

Description:
- TRELLIS 训练主入口，按 JSON 配置加载数据集、模型和 trainer。
Path:
- one `train.py`

## EXE-20260717-106

Description:
- TRELLIS image-to-3D 最小示例脚本。
Path:
- one `example.py`

## EXE-20260717-107

Description:
- TRELLIS 多图片条件推理示例脚本。
Path:
- one `example_multi_image.py`

## EXE-20260717-108

Description:
- TRELLIS text-to-3D 示例脚本。
Path:
- one `example_text.py`

## EXE-20260717-109

Description:
- TRELLIS 资产变体生成示例脚本。
Path:
- one `example_variant.py`

## EXE-20260717-110

Description:
- dataset_toolkits 依赖安装脚本。
Path:
- one `dataset_toolkits/setup.sh`

## EXE-20260717-111

Description:
- 单对象 Blender 渲染脚本。
Path:
- one `dataset_toolkits/blender_script/render.py`

## EXE-20260717-112

Description:
- Blender 批量渲染多个对象入口。
Path:
- one `dataset_toolkits/blender_script/render_batch.py`

## EXE-20260717-113

Description:
- 为 TRELLIS 风格数据集构建或补全 metadata.csv。
Path:
- one `dataset_toolkits/build_metadata.py`

## EXE-20260717-114

Description:
- 为数据集图片计算 aesthetic score。
Path:
- one `dataset_toolkits/calculate_aesthetic_scores.py`

## EXE-20260717-115

Description:
- 下载或筛选 TRELLIS 数据集资源。
Path:
- one `dataset_toolkits/download.py`

## EXE-20260717-116

Description:
- 对 TRELLIS 数据集对象执行多视角渲染。
Path:
- one `dataset_toolkits/render.py`

## EXE-20260717-117

Description:
- 生成条件视图渲染资源。
Path:
- one `dataset_toolkits/render_cond.py`

## EXE-20260717-118

Description:
- 将数据集对象体素化为点云。
Path:
- one `dataset_toolkits/voxelize.py`

## EXE-20260717-119

Description:
- 从多视角渲染提取并融合图像特征。
Path:
- one `dataset_toolkits/extract_feature.py`

## EXE-20260717-120

Description:
- 编码 SLat latent。
Path:
- one `dataset_toolkits/encode_latent.py`

## EXE-20260717-121

Description:
- 编码 Sparse Structure latent。
Path:
- one `dataset_toolkits/encode_ss_latent.py`

## EXE-20260717-122

Description:
- 统计 latent 均值/方差等训练归一化信息。
Path:
- one `dataset_toolkits/stat_latent.py`

## EXE-20260717-123

Description:
- 为 metadata 添加或更新 aesthetic score。
Path:
- one `fine_tuning/add_aesthetic_score.py`

## EXE-20260717-124

Description:
- 自动构建或更新 FaceScape 适配 metadata.csv。
Path:
- one `fine_tuning/build_facescape_metadata.py`

## EXE-20260717-125

Description:
- 按相机角度/姿态过滤 FaceScape 视角。
Path:
- one `fine_tuning/camera_view_filter.py`

## EXE-20260717-126

Description:
- 清理 alpha 为空的条件渲染图。
Path:
- one `fine_tuning/clean_empty_alpha_renders_cond.py`

## EXE-20260717-127

Description:
- 将 PyTorch checkpoint 转换为 safetensors。
Path:
- one `fine_tuning/convert_pt_to_safetensors.py`

## EXE-20260717-128

Description:
- 将 safetensors 权重转换为 PyTorch checkpoint。
Path:
- one `fine_tuning/convert_safetensors_to_pt.py`

## EXE-20260717-129

Description:
- 对训练 log 按 step 去重。
Path:
- one `fine_tuning/deduplicate_log_steps.py`

## EXE-20260717-130

Description:
- FaceScape 分片批处理预处理管线。
Path:
- one `fine_tuning/facescape_batch_pipeline.py`

## EXE-20260717-131

Description:
- FaceScape 专用特征提取脚本。
Path:
- one `fine_tuning/facescape_extract_feature.py`

## EXE-20260717-132

Description:
- 过滤 FaceScape 原始相机视角用于条件法线图。
Path:
- one `fine_tuning/facescape_filter_views.py`

## EXE-20260717-133

Description:
- FaceScape mesh 多视角渲染脚本。
Path:
- one `fine_tuning/facescape_render.py`

## EXE-20260717-134

Description:
- 修复 FaceScape metadata 与输出资源的一致性。
Path:
- one `fine_tuning/facescape_repair_metadata.py`

## EXE-20260717-135

Description:
- 从预处理结果生成 profiling 子集。
Path:
- one `fine_tuning/make_profile_subset.py`

## EXE-20260717-136

Description:
- 合并多分片 FaceScape TRELLIS 预处理输出。
Path:
- one `fine_tuning/merge_facescape_outputs.py`

## EXE-20260717-137

Description:
- 绘制训练日志曲线。
Path:
- one `fine_tuning/plot_log_curves.py`

## EXE-20260717-138

Description:
- 从 FaceScape shard 输出重建 metadata。
Path:
- one `fine_tuning/rebuild_facescape_metadata_from_outputs.py`

## EXE-20260717-139

Description:
- 将 FaceScape 数据集拆分为 train/test 等分区。
Path:
- one `fine_tuning/split_facescape_dataset.py`

## EXE-20260717-140

Description:
- FaceScape/TRELLIS mesh 体素化脚本，带超时保护。
Path:
- one `fine_tuning/voxelize.py`

## EXE-20260717-141

Description:
- 审计 Sparse Structure GT 重建结果。
Path:
- one `fine_tuning/audit_ss_gt_reconstruction.py`

## EXE-20260717-142

Description:
- 审计 SLat GT 重建结果。
Path:
- one `fine_tuning/audit_slat_gt_reconstruction.py`

## EXE-20260717-143

Description:
- 从 FaceScape train 随机导出 GT 重建样本。
Path:
- one `fine_tuning/export_random_train_gt_reconstructions.py`

## EXE-20260717-144

Description:
- 处理截断 mesh 并生成 GT 重建相关输出。
Path:
- one `fine_tuning/process_truncated_mesh_gt_reconstructions.py`

## EXE-20260717-145

Description:
- FlexiCubes optimization 示例入口。
Path:
- one `trellis/representations/mesh/flexicubes/examples/optimize.py`

## EXE-20260718-002

Description:
- 从 FaceScape test 数据中抽取固定样本，生成用于 checkpoint 对比的 eval 子集 metadata 和资源链接。
Path:
- one `eval/prepare_facescape_eval_subset.py`

## EXE-20260718-003

Description:
- 在固定 FaceScape eval 子集上评估一组 SLat encoder/GS decoder checkpoint 的重建指标。
Path:
- one `eval/slat_enc_dec_reconstruction.py`

## EXE-20260718-004

Description:
- 汇总多个 SLat enc/dec eval 输出目录的 `summary.json`，生成 checkpoint 对比 CSV。
Path:
- one `eval/compare_slat_metrics.py`

## EXE-20260719-001

Description:
- 统计 SLat latent `.npz` 文件的 token 数、feature 分布和有限值比例。
Path:
- one `eval/latent_stats.py`

## EXE-20260719-002

Description:
- `eval/latent_stats.py` 的兼容入口，用于运行 SLat latent 分布统计。
Path:
- one `eval/analyze_slat_latent_stats.py`

## EXE-20260719-003

Description:
- 对固定样本和固定 seed 执行 SLat flow checkpoint 条件生成，并保存 cond、GT、generated 图像和 PLY 模型。
Path:
- `eval/slat_flow_fixed_generation.py`

## EXE-20260719-004

Description:
- 对固定 SLat flow 生成结果计算 L1、MSE、PSNR、SSIM、LPIPS 和 silhouette IoU，并汇总多个 run。
Path:
- one `eval/flow_generation_metrics.py`

## EXE-20260719-005

Description:
- `eval/flow_generation_metrics.py` 的兼容入口，用于对比固定 SLat flow 生成结果指标。
Path:
- one `eval/compare_flow_generation_metrics.py`

## EXE-20260720-001

Description:
- 在固定 FaceScape eval 样本上计算 SLat encoder 训练中 weighted KL 项的局部梯度贡献。
Path:
- `eval/slat_enc_dec_gradient_contrib.py`

## EXE-20260722-001

Description:
- 在固定 FaceScape latent/test 样本上评估 SLat mesh decoder checkpoint，按 Stable3DGen mesh 后处理导出 PLY 并计算几何指标。
Path:
- `eval/mesh_decoder_reconstruction.py`

## EXE-20260722-002

Description:
- 合并并排序多个 SLat mesh decoder eval summary，生成 checkpoint 横向对比 CSV。
Path:
- `eval/compare_mesh_decoder_metrics.py`
