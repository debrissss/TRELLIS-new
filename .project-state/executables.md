# Executables

## EXE-20260717-101 - setup.sh

Description:
- TRELLIS 环境安装脚本，可安装基础、训练、demo 和加速依赖。
Path / Declaration:
- one `setup.sh`
Kind:
- shell-script
Invocation:
- `. ./setup.sh --basic --train --xformers --flash-attn --spconv`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--new-env` | no | false | 创建 conda 环境 |
| `--basic` | no | false | 安装基础依赖 |
| `--train` | no | false | 安装训练依赖 |
| `--demo` | no | false | 安装 demo 依赖 |
| acceleration flags | no | false | 安装 xformers/flash-attn/spconv/kaolin 等可选依赖 |
Inputs:
- 当前 shell、conda/mamba、CUDA Toolkit 和网络包源。
Outputs:
- Python/conda 运行环境。
Side effects:
- 安装软件包并可能修改当前 shell 环境。
Prerequisites:
- Linux、conda/mamba、匹配的 CUDA Toolkit。
Environment:
- `PATH`: 应指向目标 CUDA 版本。
Failure / Exit behavior:
- 安装命令失败时返回非零。
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-102 - app.py

Description:
- TRELLIS image-to-3D Gradio Web Demo。
Path / Declaration:
- one `app.py`
Kind:
- python-script
Invocation:
- `python app.py`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | none | Web UI 参数在 Gradio 界面中设置 |
Inputs:
- 用户通过 Web UI 上传的图片。
Outputs:
- 交互式预览、GLB、Gaussian PLY、mesh 等导出文件。
Side effects:
- 启动本地 Gradio 服务并加载 GPU 模型。
Prerequisites:
- Demo 依赖、CUDA、TRELLIS image 模型权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/pipelines/trellis_image_to_3d.py`
Last verified:
- 2026-07-17

## EXE-20260718-001 - ad hoc FaceScape SLat GS 50GB subset preparation

Description:
- 临时 Python 筛样加 rsync 复制命令，用于从 FaceScape train 数据中准备约 50GB 的 SLat encoder + Gaussian decoder 训练子集。
Path / Declaration:
- one `ad hoc shell command in repository root`
Kind:
- other
Invocation:
- `python - <<'PY' ... PY && rsync -a --info=progress2 --files-from=/tmp/facescape_slat_gs_50gb_files.txt datasets/Facescape/train/ datasets/Facescape_slat_gs_50gb/train/`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| source train root | yes | `datasets/Facescape/train` | 原始 FaceScape train 数据根目录 |
| destination train root | yes | `datasets/Facescape_slat_gs_50gb/train` | 小份可迁移训练数据输出目录 |
| target size | no | about 50 GiB | 按 `renders/<sha>` 加 `features/dinov2_vitl14_reg/<sha>.npz` 估算累计大小 |
Inputs:
- FaceScape train `metadata.csv`、`renders/<sha>/`、`features/dinov2_vitl14_reg/<sha>.npz`。
Outputs:
- 只含 SLat encoder + Gaussian decoder 训练所需字段与文件的 FaceScape train 子集。
Side effects:
- 创建或覆盖 `datasets/Facescape_slat_gs_50gb/train`；写临时 `/tmp/facescape_slat_gs_50gb_files.txt` 和 `/tmp/facescape_slat_gs_50gb_ids.txt`。
Prerequisites:
- 原始 FaceScape train 数据存在，且目标磁盘至少有约 50GiB 可用空间。
Environment:
- none: shell/Python/rsync.
Failure / Exit behavior:
- Python 阶段生成文件列表；rsync 失败时返回非零退出码并留下部分复制数据。
Related Code:
- none
Last verified:
- 2026-07-18

## EXE-20260717-103 - app_text.py

Description:
- TRELLIS text-to-3D Gradio Web Demo。
Path / Declaration:
- one `app_text.py`
Kind:
- python-script
Invocation:
- `python app_text.py`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | none | Web UI 参数在 Gradio 界面中设置 |
Inputs:
- 用户通过 Web UI 输入的文本提示。
Outputs:
- 交互式预览和 3D 资产导出。
Side effects:
- 启动本地 Gradio 服务并加载 GPU 模型。
Prerequisites:
- Demo 依赖、CUDA、TRELLIS text 模型权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/pipelines/trellis_text_to_3d.py`
Last verified:
- 2026-07-17

## EXE-20260717-104 - cli.py

Description:
- 本地图片到 3D 资产的命令行推理入口。
Path / Declaration:
- one `cli.py`
Kind:
- python-script
Invocation:
- `python cli.py --config configs/default.yaml --input_dir <images> --output_dir <output>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--config` | no | `configs/default.yaml` | YAML/JSON 默认参数文件 |
| `--input_dir` | yes if config missing | config value | 输入图片目录 |
| `--output_dir` | yes if config missing | config value | 输出产物目录 |
| sampler flags | no | config/defaults | seed、CFG 强度、采样步数、多图算法等 |
| export flags | no | config/defaults | mesh 简化比例和纹理尺寸 |
Inputs:
- 图片目录和本地 TRELLIS image 模型。
Outputs:
- `sample.mp4`、`sample.glb`、`sample.ply`、`sample_mesh.ply`。
Side effects:
- 创建输出目录；加载 CUDA 模型；使用临时 session 后清理。
Prerequisites:
- 推理依赖、CUDA、本地权重路径可用。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- 缺少输入/输出参数时打印帮助并返回；模型缺失会加载失败。
Related Code:
- `app.py`
Last verified:
- 2026-07-17

## EXE-20260717-105 - train.py

Description:
- TRELLIS 训练主入口，按 JSON 配置加载数据集、模型和 trainer。
Path / Declaration:
- one `train.py`
Kind:
- python-script
Invocation:
- `python train.py --config <config.json> --data_dir <dataset_dir> --output_dir <output_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--config` | yes | none | 实验 JSON 配置 |
| `--output_dir` | yes | none | 日志、模型摘要、checkpoint 输出目录 |
| `--data_dir` | no | `./data/` | 数据目录 |
| `--load_dir` | no | `output_dir` | 恢复训练读取目录 |
| `--ckpt` | no | `latest` | `latest`、`none` 或步数 |
| distributed flags | no | local defaults | 节点、GPU、master 地址和端口 |
| debug flags | no | false | `--tryrun`、`--profile` |
Inputs:
- JSON 训练配置、数据目录、可选 checkpoint。
Outputs:
- `output_dir` 下 `command.txt`、`config.json`、模型摘要、日志、采样和 checkpoint。
Side effects:
- 创建/更新输出目录；占用 GPU；可能长时间运行。
Prerequisites:
- CUDA、PyTorch、TRELLIS 训练依赖、数据集和权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPARSE_ATTN_BACKEND`: 稀疏 attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- `auto_retry > 0` 时异常后重试；最终失败行为取决于最后一次异常。
Related Code:
- `trellis/trainers/`
- `trellis/datasets/`
Last verified:
- 2026-07-17

## EXE-20260717-106 - example.py

Description:
- TRELLIS image-to-3D 最小示例脚本。
Path / Declaration:
- one `example.py`
Kind:
- python-script
Invocation:
- `python example.py`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | hardcoded | 示例图片、seed 和导出路径在脚本中定义 |
Inputs:
- 示例图片和 TRELLIS image 模型。
Outputs:
- 示例视频、GLB 和 PLY。
Side effects:
- 加载 CUDA 模型并在当前目录写示例产物。
Prerequisites:
- 推理依赖、CUDA、模型权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/pipelines/trellis_image_to_3d.py`
Last verified:
- 2026-07-17

## EXE-20260717-107 - example_multi_image.py

Description:
- TRELLIS 多图片条件推理示例脚本。
Path / Declaration:
- one `example_multi_image.py`
Kind:
- python-script
Invocation:
- `python example_multi_image.py`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | hardcoded | 示例图片和推理参数在脚本中定义 |
Inputs:
- 多张示例图片和 TRELLIS image 模型。
Outputs:
- 示例 3D 资产和渲染产物。
Side effects:
- 加载 CUDA 模型并写示例产物。
Prerequisites:
- 推理依赖、CUDA、模型权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/pipelines/trellis_image_to_3d.py`
Last verified:
- 2026-07-17

## EXE-20260717-108 - example_text.py

Description:
- TRELLIS text-to-3D 示例脚本。
Path / Declaration:
- one `example_text.py`
Kind:
- python-script
Invocation:
- `python example_text.py`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | hardcoded | 文本提示和导出路径在脚本中定义 |
Inputs:
- 文本提示和 TRELLIS text 模型。
Outputs:
- 示例 3D 资产和渲染产物。
Side effects:
- 加载 CUDA 模型并写示例产物。
Prerequisites:
- 推理依赖、CUDA、text 模型权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/pipelines/trellis_text_to_3d.py`
Last verified:
- 2026-07-17

## EXE-20260717-109 - example_variant.py

Description:
- TRELLIS 资产变体生成示例脚本。
Path / Declaration:
- one `example_variant.py`
Kind:
- python-script
Invocation:
- `python example_variant.py`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | hardcoded | 示例输入和变体参数在脚本中定义 |
Inputs:
- 示例 3D/条件输入和模型权重。
Outputs:
- 示例变体资产。
Side effects:
- 加载 CUDA 模型并写示例产物。
Prerequisites:
- 推理依赖、CUDA、模型权重。
Environment:
- `ATTN_BACKEND`: attention 后端。
- `SPCONV_ALGO`: spconv 算法。
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/pipelines/`
Last verified:
- 2026-07-17

## EXE-20260717-110 - dataset_toolkits/setup.sh

Description:
- dataset_toolkits 依赖安装脚本。
Path / Declaration:
- one `dataset_toolkits/setup.sh`
Kind:
- shell-script
Invocation:
- `. ./dataset_toolkits/setup.sh`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| none | no | script-defined | 安装行为由脚本定义 |
Inputs:
- 当前 Python/conda 环境。
Outputs:
- 数据集处理依赖。
Side effects:
- 安装软件包。
Prerequisites:
- shell、Python 环境和网络包源。
Environment:
- unknown
Failure / Exit behavior:
- 安装命令失败时返回非零。
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-111 - dataset_toolkits/blender_script/render.py

Description:
- 单对象 Blender 渲染脚本。
Path / Declaration:
- one `dataset_toolkits/blender_script/render.py`
Kind:
- python-script
Invocation:
- `blender -b -P dataset_toolkits/blender_script/render.py -- --object <mesh> --output_folder <dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| blender/script args | varies | varies | 对象路径、输出目录、视角/渲染参数 |
Inputs:
- 3D 对象文件。
Outputs:
- 多视角渲染图和 transforms。
Side effects:
- 启动 Blender 并写渲染目录。
Prerequisites:
- Blender 和渲染依赖。
Environment:
- CUDA/Blender GPU settings: optional.
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-112 - dataset_toolkits/blender_script/render_batch.py

Description:
- Blender 批量渲染多个对象入口。
Path / Declaration:
- one `dataset_toolkits/blender_script/render_batch.py`
Kind:
- python-script
Invocation:
- `blender -b -P dataset_toolkits/blender_script/render_batch.py -- <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| batch render args | varies | varies | 批量对象、输出目录和渲染参数 |
Inputs:
- 多个 3D 对象文件。
Outputs:
- 批量渲染图和 transforms。
Side effects:
- 启动 Blender 并写多个输出目录。
Prerequisites:
- Blender 和渲染依赖。
Environment:
- CUDA/Blender GPU settings: optional.
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/blender_script/render.py`
Last verified:
- 2026-07-17

## EXE-20260717-113 - dataset_toolkits/build_metadata.py

Description:
- 为 TRELLIS 风格数据集构建或补全 metadata.csv。
Path / Declaration:
- one `dataset_toolkits/build_metadata.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/build_metadata.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集输出目录 |
| `--field` | no | `all` | 需要构建的字段 |
| `--from_file` | no | false | 从文件读取实例列表 |
Inputs:
- 数据集目录和对象/中间资源。
Outputs:
- `metadata.csv`。
Side effects:
- 创建或更新 metadata。
Prerequisites:
- 数据集目录结构存在。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-114 - dataset_toolkits/calculate_aesthetic_scores.py

Description:
- 为数据集图片计算 aesthetic score。
Path / Declaration:
- one `dataset_toolkits/calculate_aesthetic_scores.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/calculate_aesthetic_scores.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--clip_model` | no | `vit_l_14` | CLIP 模型 |
| `--rank` | no | `0` | 分片 rank |
| `--world_size` | no | `1` | 分片总数 |
Inputs:
- 数据集渲染/图片资源。
Outputs:
- metadata 中 aesthetic score 字段或相关输出。
Side effects:
- 读取图片并写数据集元数据。
Prerequisites:
- CLIP/aesthetic 模型依赖。
Environment:
- CUDA: optional/likely.
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-115 - dataset_toolkits/download.py

Description:
- 下载或筛选 TRELLIS 数据集资源。
Path / Declaration:
- one `dataset_toolkits/download.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/download.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 输出目录 |
| `--filter_low_aesthetic_score` | no | none | aesthetic 分数过滤阈值 |
| `--instances` | no | none | 实例列表 |
| `--limit` | no | none | 最大下载数 |
| `--rank` / `--world_size` | no | `0`/`1` | 分片参数 |
Inputs:
- 远程数据源和实例列表。
Outputs:
- 下载到数据集目录的资源。
Side effects:
- 网络下载和磁盘写入。
Prerequisites:
- 网络访问、数据集依赖。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-116 - dataset_toolkits/render.py

Description:
- 对 TRELLIS 数据集对象执行多视角渲染。
Path / Declaration:
- one `dataset_toolkits/render.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/render.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--num_views` | no | `150` | 渲影视角数 |
| `--instances` | no | none | 实例列表 |
| `--rank` / `--world_size` | no | `0`/`1` | 分片参数 |
| `--max_workers` | no | `8` | 并行 worker 数 |
Inputs:
- 数据集对象/mesh。
Outputs:
- `renders/<sha256>/`。
Side effects:
- 调用渲染流程并写大量图像。
Prerequisites:
- Blender/渲染依赖。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/blender_script/render.py`
Last verified:
- 2026-07-17

## EXE-20260717-117 - dataset_toolkits/render_cond.py

Description:
- 生成条件视图渲染资源。
Path / Declaration:
- one `dataset_toolkits/render_cond.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/render_cond.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--num_views` | no | `24` | 条件视角数 |
| `--instances` | no | none | 实例列表 |
| `--rank` / `--world_size` | no | `0`/`1` | 分片参数 |
| `--max_workers` | no | `8` | 并行 worker 数 |
Inputs:
- 数据集对象/mesh。
Outputs:
- `renders_cond/<sha256>/`。
Side effects:
- 调用渲染流程并写条件图像。
Prerequisites:
- Blender/渲染依赖。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/blender_script/render.py`
Last verified:
- 2026-07-17

## EXE-20260717-118 - dataset_toolkits/voxelize.py

Description:
- 将数据集对象体素化为点云。
Path / Declaration:
- one `dataset_toolkits/voxelize.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/voxelize.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--num_views` | no | `150` | 视角/采样相关参数 |
| `--instances` | no | none | 实例列表 |
| `--rank` / `--world_size` | no | `0`/`1` | 分片参数 |
| `--max_workers` | no | none | 并行 worker 数 |
Inputs:
- mesh 或渲染资源。
Outputs:
- `voxels/<sha256>.ply`。
Side effects:
- 写 voxel 点云。
Prerequisites:
- 体素化依赖。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-119 - dataset_toolkits/extract_feature.py

Description:
- 从多视角渲染提取并融合图像特征。
Path / Declaration:
- one `dataset_toolkits/extract_feature.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/extract_feature.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--model` | no | `dinov2_vitl14_reg` | 特征模型 |
| `--batch_size` | no | `16` | 批大小 |
| `--instances` | no | none | 实例列表 |
| `--rank` / `--world_size` | no | `0`/`1` | 分片参数 |
Inputs:
- `renders`、`voxels` 和 metadata。
Outputs:
- `features/<model>/<sha256>.npz`。
Side effects:
- GPU 推理并写特征文件。
Prerequisites:
- CUDA、DINOv2/特征模型依赖。
Environment:
- CUDA: required for normal use.
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-120 - dataset_toolkits/encode_latent.py

Description:
- 编码 SLat latent。
Path / Declaration:
- one `dataset_toolkits/encode_latent.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/encode_latent.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--feat_model` | no | `dinov2_vitl14_reg` | 输入特征模型 |
| `--enc_pretrained` | no | TRELLIS slat encoder | 预训练 encoder |
| `--enc_model` / `--ckpt` | no | none | 自定义模型/ckpt |
| `--instances` / shard flags | no | none/`0`/`1` | 实例与分片 |
Inputs:
- feature NPZ 和 SLat encoder。
Outputs:
- `latents/<latent_model>/<sha256>.npz`。
Side effects:
- GPU 编码并写 latent。
Prerequisites:
- CUDA、TRELLIS VAE encoder 权重。
Environment:
- CUDA: required for normal use.
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/models/structured_latent_vae/`
Last verified:
- 2026-07-17

## EXE-20260717-121 - dataset_toolkits/encode_ss_latent.py

Description:
- 编码 Sparse Structure latent。
Path / Declaration:
- one `dataset_toolkits/encode_ss_latent.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/encode_ss_latent.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--enc_pretrained` | no | TRELLIS ss encoder | 预训练 encoder |
| `--resolution` | no | `64` | 体素/latent 分辨率 |
| `--enc_model` / `--ckpt` | no | none | 自定义模型/ckpt |
| `--instances` / shard flags | no | none/`0`/`1` | 实例与分片 |
Inputs:
- voxel 点云和 SS encoder。
Outputs:
- `latents/<ss_latent_model>/<sha256>.npz`。
Side effects:
- GPU 编码并写 latent。
Prerequisites:
- CUDA、TRELLIS SS encoder 权重。
Environment:
- CUDA: required for normal use.
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/models/sparse_structure_vae.py`
Last verified:
- 2026-07-17

## EXE-20260717-122 - dataset_toolkits/stat_latent.py

Description:
- 统计 latent 均值/方差等训练归一化信息。
Path / Declaration:
- one `dataset_toolkits/stat_latent.py`
Kind:
- python-script
Invocation:
- `python dataset_toolkits/stat_latent.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--model` | no | SLat latent model | latent 模型名 |
| `--num_samples` | no | `50000` | 采样数量 |
Inputs:
- latent NPZ 文件。
Outputs:
- latent 统计信息。
Side effects:
- 读取大量 latent 并写统计结果。
Prerequisites:
- 已完成 latent 编码。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-123 - fine_tuning/add_aesthetic_score.py

Description:
- 为 metadata 添加或更新 aesthetic score。
Path / Declaration:
- one `fine_tuning/add_aesthetic_score.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/add_aesthetic_score.py <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| script flags | varies | varies | 输入 metadata、输出路径和 score 规则 |
Inputs:
- FaceScape/TRELLIS metadata。
Outputs:
- 更新后的 metadata。
Side effects:
- 写 metadata 或副本。
Prerequisites:
- pandas 等数据处理依赖。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-124 - fine_tuning/build_facescape_metadata.py

Description:
- 自动构建或更新 FaceScape 适配 metadata.csv。
Path / Declaration:
- one `fine_tuning/build_facescape_metadata.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/build_facescape_metadata.py --dataset_root <raw_facescape> --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--dataset_root` | yes | none | 原始 FaceScape 根目录 |
| `--output_dir` | yes | none | 预处理输出目录 |
Inputs:
- 原始 FaceScape 和预处理输出目录。
Outputs:
- FaceScape `metadata.csv`。
Side effects:
- 创建或更新 metadata。
Prerequisites:
- FaceScape 目录结构。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `fine_tuning/utils/facescape_utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-125 - fine_tuning/camera_view_filter.py

Description:
- 按相机角度/姿态过滤 FaceScape 视角。
Path / Declaration:
- one `fine_tuning/camera_view_filter.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/camera_view_filter.py <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| input/output flags | varies | varies | 数据集、输出、视角阈值和并行参数 |
Inputs:
- FaceScape 相机参数。
Outputs:
- 过滤后的视角清单或 transforms。
Side effects:
- 写输出文件。
Prerequisites:
- FaceScape 相机参数可读。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `fine_tuning/utils/facescape_utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-126 - fine_tuning/clean_empty_alpha_renders_cond.py

Description:
- 清理 alpha 为空的条件渲染图。
Path / Declaration:
- one `fine_tuning/clean_empty_alpha_renders_cond.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/clean_empty_alpha_renders_cond.py <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| script flags | varies | varies | 数据集目录、删除/报告模式和过滤参数 |
Inputs:
- `renders_cond` 图像。
Outputs:
- 清理报告或修改后的条件渲染目录。
Side effects:
- 可能删除/修改图像文件。
Prerequisites:
- PIL/OpenCV 等图像依赖。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-127 - fine_tuning/convert_pt_to_safetensors.py

Description:
- 将 PyTorch checkpoint 转换为 safetensors。
Path / Declaration:
- one `fine_tuning/convert_pt_to_safetensors.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/convert_pt_to_safetensors.py <input.pt> <output.safetensors>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| input/output flags | varies | varies | 输入 pt、输出 safetensors 和 key 处理选项 |
Inputs:
- PyTorch checkpoint。
Outputs:
- safetensors 文件。
Side effects:
- 写转换后的权重文件。
Prerequisites:
- torch、safetensors。
Environment:
- unknown
Failure / Exit behavior:
- 转换或校验失败返回非零。
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-128 - fine_tuning/convert_safetensors_to_pt.py

Description:
- 将 safetensors 权重转换为 PyTorch checkpoint。
Path / Declaration:
- one `fine_tuning/convert_safetensors_to_pt.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/convert_safetensors_to_pt.py <input.safetensors> <output.pt>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| input/output flags | varies | varies | 输入 safetensors、输出 pt 和 key 处理选项 |
Inputs:
- safetensors 权重。
Outputs:
- PyTorch checkpoint。
Side effects:
- 写转换后的权重文件。
Prerequisites:
- torch、safetensors。
Environment:
- unknown
Failure / Exit behavior:
- 转换或校验失败返回非零。
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-129 - fine_tuning/deduplicate_log_steps.py

Description:
- 对训练 log 按 step 去重。
Path / Declaration:
- one `fine_tuning/deduplicate_log_steps.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/deduplicate_log_steps.py <log_path>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `log_path` | yes | none | 输入 log.txt |
| output/backup flags | no | script defaults | 输出和备份行为 |
Inputs:
- 训练日志。
Outputs:
- 去重后的日志或副本。
Side effects:
- 可能覆盖日志。
Prerequisites:
- 文本日志格式符合脚本预期。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-130 - fine_tuning/facescape_batch_pipeline.py

Description:
- FaceScape 分片批处理预处理管线。
Path / Declaration:
- one `fine_tuning/facescape_batch_pipeline.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/facescape_batch_pipeline.py <machine_index>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `machine_index` | yes | none | 当前机器/分片索引 |
| `--dataset_root` | no | `/root/autodl-tmp/facescape` | 原始 FaceScape |
| `--work_root` | no | `/root/autodl-tmp` | 工作根目录 |
| batch/render/feature flags | no | script defaults | 批大小、视角、worker、超时和重试参数 |
Inputs:
- 原始 FaceScape 数据。
Outputs:
- 预处理 shard 数据目录。
Side effects:
- 大量读写；可能启动 Blender 和 GPU 特征提取。
Prerequisites:
- FaceScape、Blender、CUDA、DINOv2/TRELLIS 依赖。
Environment:
- unknown
Failure / Exit behavior:
- 支持 continue/retry 参数；具体失败行为由脚本实现。
Related Code:
- `fine_tuning/facescape_render.py`
- `fine_tuning/voxelize.py`
- `fine_tuning/facescape_extract_feature.py`
Last verified:
- 2026-07-17

## EXE-20260717-131 - fine_tuning/facescape_extract_feature.py

Description:
- FaceScape 专用特征提取脚本。
Path / Declaration:
- one `fine_tuning/facescape_extract_feature.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/facescape_extract_feature.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | FaceScape 预处理目录 |
| `--model` | no | `dinov2_vitl14_reg` | 特征模型 |
| `--batch_size` | no | `16` | 批大小 |
| `--overwrite` | no | false | 覆盖已有特征 |
| shard flags | no | `0`/`1` | 分片参数 |
Inputs:
- FaceScape renders/voxels。
Outputs:
- FaceScape feature NPZ。
Side effects:
- GPU 推理并写特征。
Prerequisites:
- CUDA、DINOv2、预处理数据。
Environment:
- CUDA: required for normal use.
Failure / Exit behavior:
- unknown
Related Code:
- `dataset_toolkits/extract_feature.py`
Last verified:
- 2026-07-17

## EXE-20260717-132 - fine_tuning/facescape_filter_views.py

Description:
- 过滤 FaceScape 原始相机视角用于条件法线图。
Path / Declaration:
- one `fine_tuning/facescape_filter_views.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/facescape_filter_views.py --dataset_root <raw_facescape> --output_dir <out>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--dataset_root` | yes | none | 原始 FaceScape |
| `--output_dir` | yes | none | 输出目录 |
| view threshold flags | no | script defaults | yaw/pitch/center 过滤阈值 |
| shard flags | no | `0`/`1` | 分片参数 |
Inputs:
- FaceScape 相机参数。
Outputs:
- 过滤后的视角/transforms。
Side effects:
- 写输出目录。
Prerequisites:
- FaceScape 原始相机参数。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `fine_tuning/utils/facescape_utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-133 - fine_tuning/facescape_render.py

Description:
- FaceScape mesh 多视角渲染脚本。
Path / Declaration:
- one `fine_tuning/facescape_render.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/facescape_render.py --dataset_root <raw_facescape> --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--dataset_root` | yes | none | FaceScape 原始根目录 |
| `--output_dir` | yes | none | 含 metadata.csv 的输出目录 |
| `--num_views` | no | `150` | 渲影视角数 |
| blender/profile flags | no | script defaults | Blender 批量、日志、profiling、超时参数 |
| shard flags | no | `0`/`1` | 分片参数 |
Inputs:
- FaceScape mesh 和 metadata。
Outputs:
- `renders/<sha256>/`。
Side effects:
- 启动 Blender；写大量渲染图。
Prerequisites:
- Blender、FaceScape 数据、渲染依赖。
Environment:
- Blender/CUDA GPU settings: optional.
Failure / Exit behavior:
- timeout 参数控制单项渲染超时。
Related Code:
- `dataset_toolkits/blender_script/render_batch.py`
Last verified:
- 2026-07-17

## EXE-20260717-134 - fine_tuning/facescape_repair_metadata.py

Description:
- 修复 FaceScape metadata 与输出资源的一致性。
Path / Declaration:
- one `fine_tuning/facescape_repair_metadata.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/facescape_repair_metadata.py --dataset_root <raw_facescape> --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--dataset_root` | yes | none | 原始 FaceScape |
| `--output_dir` | yes | none | 预处理输出目录 |
| `--temp_dir` | no | none | 临时目录 |
| `--keep_temp` | no | false | 保留临时目录 |
Inputs:
- FaceScape 数据和 metadata。
Outputs:
- 修复后的 metadata。
Side effects:
- 可能写临时目录和覆盖 metadata。
Prerequisites:
- FaceScape 数据。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- `fine_tuning/utils/facescape_utils.py`
Last verified:
- 2026-07-17

## EXE-20260717-135 - fine_tuning/make_profile_subset.py

Description:
- 从预处理结果生成 profiling 子集。
Path / Declaration:
- one `fine_tuning/make_profile_subset.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/make_profile_subset.py <count> --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `count` | yes | none | 样本数 |
| `--output_dir` | yes | none | 输出目录 |
| `--profile_dir` | no | `/root/autodl-tmp` | profiling 根目录 |
| `--only_feature_done` | no | false | 只选特征已完成样本 |
Inputs:
- FaceScape/预处理 profiling 目录。
Outputs:
- profile 子集清单或 metadata。
Side effects:
- 写输出目录。
Prerequisites:
- profiling 目录存在。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-136 - fine_tuning/merge_facescape_outputs.py

Description:
- 合并多分片 FaceScape TRELLIS 预处理输出。
Path / Declaration:
- one `fine_tuning/merge_facescape_outputs.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/merge_facescape_outputs.py --output_dir <merged_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 合并输出目录 |
| `--source_root` | no | `/root/autodl-tmp` | 分片根目录 |
| `--indices` | no | `1-5` | 分片索引 |
| pattern flags | no | script defaults | 分片目录/metadata/latents/renders_cond 命名模式 |
| `--overwrite` | no | `skip` | 冲突处理 |
Inputs:
- 多个 FaceScape 预处理 shard。
Outputs:
- 合并后的数据集目录。
Side effects:
- 复制/合并大量文件。
Prerequisites:
- shard 输出目录存在。
Environment:
- unknown
Failure / Exit behavior:
- overwrite=`error` 时冲突失败。
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-137 - fine_tuning/plot_log_curves.py

Description:
- 绘制训练日志曲线。
Path / Declaration:
- one `fine_tuning/plot_log_curves.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/plot_log_curves.py <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| log/input flags | varies | varies | 日志路径、输出图片和曲线字段 |
Inputs:
- 训练日志。
Outputs:
- 曲线图或统计输出。
Side effects:
- 写图片/报告。
Prerequisites:
- matplotlib/pandas。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-138 - fine_tuning/rebuild_facescape_metadata_from_outputs.py

Description:
- 从 FaceScape shard 输出重建 metadata。
Path / Declaration:
- one `fine_tuning/rebuild_facescape_metadata_from_outputs.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/rebuild_facescape_metadata_from_outputs.py --shard_dir <dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--shard_dir` | no | `/root/autodl-tmp/preprocess_facescape_1` | shard 目录 |
| `--shard_index` | no | `1` | shard 编号 |
| `--output` | no | none | 输出 metadata 路径 |
| model/reference flags | no | script defaults | latent/feature 模型和参考 metadata |
| `--overwrite` | no | false | 覆盖输出 |
Inputs:
- FaceScape 预处理 shard 输出。
Outputs:
- 重建后的 metadata。
Side effects:
- 写 metadata。
Prerequisites:
- shard 输出结构存在。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-139 - fine_tuning/split_facescape_dataset.py

Description:
- 将 FaceScape 数据集拆分为 train/test 等分区。
Path / Declaration:
- one `fine_tuning/split_facescape_dataset.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/split_facescape_dataset.py <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| split flags | varies | varies | 输入/输出目录、比例、subject/front-back 分割规则 |
Inputs:
- FaceScape merged metadata 和资源目录。
Outputs:
- train/test 分区目录与 metadata。
Side effects:
- 复制/链接文件并写 split manifest。
Prerequisites:
- merged 数据集完整。
Environment:
- unknown
Failure / Exit behavior:
- unknown
Related Code:
- none
Last verified:
- 2026-07-17

## EXE-20260717-140 - fine_tuning/voxelize.py

Description:
- FaceScape/TRELLIS mesh 体素化脚本，带超时保护。
Path / Declaration:
- one `fine_tuning/voxelize.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/voxelize.py --output_dir <dataset_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--output_dir` | yes | none | 数据集目录 |
| `--num_views` | no | `150` | 视角/采样相关参数 |
| `--max_workers` | no | none | 并行 worker 数 |
| `--timeout` | no | `5.0` | 单样本超时 |
| shard/instance flags | no | defaults | 实例和分片参数 |
Inputs:
- mesh/renders。
Outputs:
- `voxels/<sha256>.ply`。
Side effects:
- 写 voxel 点云。
Prerequisites:
- 体素化依赖。
Environment:
- unknown
Failure / Exit behavior:
- timeout 后跳过或失败，取决于脚本实现。
Related Code:
- `dataset_toolkits/voxelize.py`
Last verified:
- 2026-07-17

## EXE-20260717-141 - fine_tuning/audit_ss_gt_reconstruction.py

Description:
- 审计 Sparse Structure GT 重建结果。
Path / Declaration:
- one `fine_tuning/audit_ss_gt_reconstruction.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/audit_ss_gt_reconstruction.py --sha256 <sample>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--sha256` | no | script default | 审计样本 |
| dataset/model/output flags | no | script defaults | 数据集、decoder、输出、设备等 |
Inputs:
- FaceScape SS latent/voxel 数据和 decoder。
Outputs:
- 审计重建产物。
Side effects:
- GPU 解码并写输出。
Prerequisites:
- CUDA、SS decoder、FaceScape latent。
Environment:
- CUDA: required for normal use.
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/models/sparse_structure_vae.py`
Last verified:
- 2026-07-17

## EXE-20260717-142 - fine_tuning/audit_slat_gt_reconstruction.py

Description:
- 审计 SLat GT 重建结果。
Path / Declaration:
- one `fine_tuning/audit_slat_gt_reconstruction.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/audit_slat_gt_reconstruction.py --sha256 <sample>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--sha256` | no | script default | 审计样本 |
| dataset/model/output flags | no | script defaults | 数据集、decoder、输出、设备等 |
Inputs:
- FaceScape SLat latent/feature 数据和 decoder。
Outputs:
- 审计重建产物。
Side effects:
- GPU 解码并写输出。
Prerequisites:
- CUDA、SLat decoder、FaceScape latent。
Environment:
- CUDA: required for normal use.
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/models/structured_latent_vae/`
Last verified:
- 2026-07-17

## EXE-20260717-143 - fine_tuning/export_random_train_gt_reconstructions.py

Description:
- 从 FaceScape train 随机导出 GT 重建样本。
Path / Declaration:
- one `fine_tuning/export_random_train_gt_reconstructions.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/export_random_train_gt_reconstructions.py --num_samples 100`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--dataset_root` | no | script default | FaceScape train 根目录 |
| `--output_dir` | no | script default | 输出目录 |
| `--num_samples` | no | `100` | 样本数 |
| `--seed` | no | `20260701` | 随机种子 |
| decoder/device flags | no | script defaults | SS/SLat decoder、阈值和设备 |
Inputs:
- FaceScape train latent 数据和 decoder。
Outputs:
- 随机样本重建导出目录。
Side effects:
- GPU 解码并写多个样本。
Prerequisites:
- CUDA、FaceScape train 数据、decoder 权重。
Environment:
- CUDA: default device.
Failure / Exit behavior:
- unknown
Related Code:
- `fine_tuning/audit_ss_gt_reconstruction.py`
- `fine_tuning/audit_slat_gt_reconstruction.py`
Last verified:
- 2026-07-17

## EXE-20260717-144 - fine_tuning/process_truncated_mesh_gt_reconstructions.py

Description:
- 处理截断 mesh 并生成 GT 重建相关输出。
Path / Declaration:
- one `fine_tuning/process_truncated_mesh_gt_reconstructions.py`
Kind:
- python-script
Invocation:
- `python fine_tuning/process_truncated_mesh_gt_reconstructions.py --source_root <mesh_dir>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| `--source_root` | no | script default | 截断 mesh 输入目录 |
| `--processing_dir` | no | none | 中间处理目录 |
| `--num_views` | no | `150` | 渲影视角 |
| render/voxel/feature flags | no | script defaults | worker、超时、批大小 |
| decoder/device flags | no | script defaults | SS/SLat decoder、阈值和设备 |
Inputs:
- 截断 mesh 文件。
Outputs:
- 处理中间数据、特征、latent 和重建导出。
Side effects:
- 创建处理目录；可能启动 Blender、体素化、GPU 特征和解码流程。
Prerequisites:
- 输入 mesh、Blender、CUDA、TRELLIS 预处理和 decoder 权重。
Environment:
- CUDA: default device.
Failure / Exit behavior:
- timeout 参数控制部分子步骤。
Related Code:
- `fine_tuning/facescape_render.py`
- `fine_tuning/voxelize.py`
- `fine_tuning/facescape_extract_feature.py`
Last verified:
- 2026-07-17

## EXE-20260717-145 - trellis/representations/mesh/flexicubes/examples/optimize.py

Description:
- FlexiCubes optimization 示例入口。
Path / Declaration:
- one `trellis/representations/mesh/flexicubes/examples/optimize.py`
Kind:
- python-script
Invocation:
- `python trellis/representations/mesh/flexicubes/examples/optimize.py <args>`
Parameters:

| Parameter | Required | Default | Meaning |
|---|---:|---|---|
| example args | varies | varies | FlexiCubes 优化示例参数 |
Inputs:
- 示例 mesh/SDF 数据。
Outputs:
- 优化结果示例。
Side effects:
- 写示例输出。
Prerequisites:
- FlexiCubes 示例依赖。
Environment:
- CUDA: optional/likely.
Failure / Exit behavior:
- unknown
Related Code:
- `trellis/representations/mesh/flexicubes/`
Last verified:
- 2026-07-17
