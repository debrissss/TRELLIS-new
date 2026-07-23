# Executables

## EXE-20260717-101

Description:
- TRELLIS 环境安装脚本，可安装基础、训练、demo 和加速依赖。
Path:
- `setup.sh`

## EXE-20260717-102

Description:
- TRELLIS image-to-3D Gradio Web Demo。
Path:
- `app.py`

## EXE-20260718-001

Description:
- 临时 Python 筛样加 rsync 复制命令，用于从 FaceScape train 数据中准备约 50GB 的 SLat encoder + Gaussian decoder 训练子集。
Path:
- `ad hoc shell command in repository root`

## EXE-20260717-103

Description:
- TRELLIS text-to-3D Gradio Web Demo。
Path:
- `app_text.py`

## EXE-20260717-104

Description:
- 本地图片到 3D 资产的命令行推理入口。
Path:
- `cli.py`

## EXE-20260717-105

Description:
- TRELLIS 训练主入口，按 JSON 配置加载数据集、模型和 trainer。
Path:
- `train.py`

## EXE-20260717-106

Description:
- TRELLIS image-to-3D 最小示例脚本。
Path:
- `example.py`

## EXE-20260717-107

Description:
- TRELLIS 多图片条件推理示例脚本。
Path:
- `example_multi_image.py`

## EXE-20260717-108

Description:
- TRELLIS text-to-3D 示例脚本。
Path:
- `example_text.py`

## EXE-20260717-109

Description:
- TRELLIS 资产变体生成示例脚本。
Path:
- `example_variant.py`

## EXE-20260717-110

Description:
- dataset_toolkits 依赖安装脚本。
Path:
- `dataset_toolkits/setup.sh`

## EXE-20260717-111

Description:
- 单对象 Blender 渲染脚本。
Path:
- `dataset_toolkits/blender_script/render.py`

## EXE-20260717-112

Description:
- Blender 批量渲染多个对象入口。
Path:
- `dataset_toolkits/blender_script/render_batch.py`

## EXE-20260717-113

Description:
- 为 TRELLIS 风格数据集构建或补全 metadata.csv。
Path:
- `dataset_toolkits/build_metadata.py`

## EXE-20260717-114

Description:
- 为数据集图片计算 aesthetic score。
Path:
- `dataset_toolkits/calculate_aesthetic_scores.py`

## EXE-20260717-115

Description:
- 下载或筛选 TRELLIS 数据集资源。
Path:
- `dataset_toolkits/download.py`

## EXE-20260717-116

Description:
- 对 TRELLIS 数据集对象执行多视角渲染。
Path:
- `dataset_toolkits/render.py`

## EXE-20260717-117

Description:
- 生成条件视图渲染资源。
Path:
- `dataset_toolkits/render_cond.py`

## EXE-20260717-118

Description:
- 将数据集对象体素化为点云。
Path:
- `dataset_toolkits/voxelize.py`

## EXE-20260717-119

Description:
- 从多视角渲染提取并融合图像特征。
Path:
- `dataset_toolkits/extract_feature.py`

## EXE-20260717-120

Description:
- 编码 SLat latent。
Path:
- `dataset_toolkits/encode_latent.py`

## EXE-20260717-121

Description:
- 编码 Sparse Structure latent。
Path:
- `dataset_toolkits/encode_ss_latent.py`

## EXE-20260717-122

Description:
- 统计 latent 均值/方差等训练归一化信息。
Path:
- `dataset_toolkits/stat_latent.py`

## EXE-20260717-123

Description:
- 为 metadata 添加或更新 aesthetic score。
Path:
- `fine_tuning/add_aesthetic_score.py`

## EXE-20260717-124

Description:
- 自动构建或更新 FaceScape 适配 metadata.csv。
Path:
- `fine_tuning/build_facescape_metadata.py`

## EXE-20260717-125

Description:
- 按相机角度/姿态过滤 FaceScape 视角。
Path:
- `fine_tuning/camera_view_filter.py`

## EXE-20260717-126

Description:
- 清理 alpha 为空的条件渲染图。
Path:
- `fine_tuning/clean_empty_alpha_renders_cond.py`

## EXE-20260717-127

Description:
- 将 PyTorch checkpoint 转换为 safetensors。
Path:
- `fine_tuning/convert_pt_to_safetensors.py`

## EXE-20260717-128

Description:
- 将本地 TRELLIS `.json + .safetensors` 模型权重严格校验后转换为 PyTorch `.pt` state_dict。
Path:
- `fine_tuning/convert_safetensors_to_pt.py`

## EXE-20260717-129

Description:
- 对训练 log 按 step 去重。
Path:
- `fine_tuning/deduplicate_log_steps.py`

## EXE-20260717-130

Description:
- FaceScape 分片批处理预处理管线。
Path:
- `fine_tuning/facescape_batch_pipeline.py`

## EXE-20260717-131

Description:
- FaceScape 专用特征提取脚本。
Path:
- `fine_tuning/facescape_extract_feature.py`

## EXE-20260717-132

Description:
- 过滤 FaceScape 原始相机视角用于条件法线图。
Path:
- `fine_tuning/facescape_filter_views.py`

## EXE-20260717-133

Description:
- FaceScape mesh 多视角渲染脚本。
Path:
- `fine_tuning/facescape_render.py`

## EXE-20260717-134

Description:
- 修复 FaceScape metadata 与输出资源的一致性。
Path:
- `fine_tuning/facescape_repair_metadata.py`

## EXE-20260717-135

Description:
- 从预处理结果生成 profiling 子集。
Path:
- `fine_tuning/make_profile_subset.py`

## EXE-20260717-136

Description:
- 合并多分片 FaceScape TRELLIS 预处理输出。
Path:
- `fine_tuning/merge_facescape_outputs.py`

## EXE-20260717-137

Description:
- 绘制训练日志曲线。
Path:
- `fine_tuning/plot_log_curves.py`

## EXE-20260717-138

Description:
- 从 FaceScape shard 输出重建 metadata。
Path:
- `fine_tuning/rebuild_facescape_metadata_from_outputs.py`

## EXE-20260717-139

Description:
- 将 FaceScape 数据集拆分为 train/test 等分区。
Path:
- `fine_tuning/split_facescape_dataset.py`

## EXE-20260717-140

Description:
- FaceScape/TRELLIS mesh 体素化脚本，带超时保护。
Path:
- `fine_tuning/voxelize.py`

## EXE-20260717-141

Description:
- 审计 Sparse Structure GT 重建结果。
Path:
- `fine_tuning/audit_ss_gt_reconstruction.py`

## EXE-20260717-142

Description:
- 审计 SLat GT 重建结果。
Path:
- `fine_tuning/audit_slat_gt_reconstruction.py`

## EXE-20260717-143

Description:
- 从 FaceScape train 随机导出 GT 重建样本。
Path:
- `fine_tuning/export_random_train_gt_reconstructions.py`

## EXE-20260717-144

Description:
- 处理截断 mesh 并生成 GT 重建相关输出。
Path:
- `fine_tuning/process_truncated_mesh_gt_reconstructions.py`

## EXE-20260717-145

Description:
- FlexiCubes optimization 示例入口。
Path:
- `trellis/representations/mesh/flexicubes/examples/optimize.py`

## EXE-20260718-002

Description:
- 从 FaceScape SparseStructure 数据集生成固定样本 mini evaluation dataset。
Path:
- `eval/prepare_ss_eval_dataset.py`

## EXE-20260718-003

Description:
- 在固定 SparseStructure mini dataset 上评估 SS encoder/decoder checkpoint 的 voxel 重建指标。
Path:
- `eval/evaluate_ss_enc_dec_reconstruction.py`

## EXE-20260720-001

Description:
- 对 SS flow step1000 进行固定条件 voxel 采样评估并导出 PLY 可视化。
Path:
- `eval/evaluate_ss_flow_sparse_structure.py`
